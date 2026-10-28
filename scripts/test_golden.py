#!/usr/bin/env python3
"""Export a temporary F32 model and check C++ logits against independent NumPy."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GQA_OVERRIDES = {"n_kv_heads": 2, "rms_norm_eps": 3e-4, "rope_theta": 37.0}
MODES = (("native", 1), ("threaded", 4), ("simd", 1), ("threaded-simd", 4))


def run_command(args, *, success=True):
    # Keep the independent reference deterministic without changing C++ threads.
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [str(arg) for arg in args], cwd=ROOT, env=environment,
        text=True, capture_output=True, timeout=60)
    if (result.returncode == 0) != success:
        raise AssertionError(
            f"unexpected command result ({result.returncode}): {args!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def export_fixture(directory, overrides=None):
    spec = importlib.util.spec_from_file_location(
        "golden_tiny_exporter", ROOT / "model/tinymodel/export_model.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with contextlib.redirect_stdout(io.StringIO()):
        exporter.export_model(directory, 42, json_vocab=True,
                              config_overrides=overrides)
    return json.loads((directory / "model.json").read_text(encoding="utf-8"))


def expected_contract(manifest, prompt, requested):
    if any(ord(char) > 127 for char in prompt):
        raise ValueError("golden prompt must be ASCII")
    config = manifest["config"]
    bos_id = manifest["tokenizer"]["bos_id"]
    tokens = [bos_id] + [ord(char) for char in prompt]
    if len(tokens) > config["max_seq_len"]:
        raise ValueError("golden prompt exceeds context")
    generated = min(requested, config["max_seq_len"] - len(tokens))
    return {
        "format_version": 1,
        "vocab_size": config["vocab_size"],
        "bos_id": bos_id,
        "prompt_tokens": tokens,
        "requested_generation": requested,
        "max_seq_len": config["max_seq_len"],
        "generated_count": generated,
        "step_count": len(tokens) + max(generated - 1, 0),
    }


def cpp_command(binary, model, prompt, requested, directory, mode, threads):
    return [binary, "--config", model / "model.json",
            "--model", model / "model.bin", "--prompt", prompt,
            "--n-predict", requested, "--dump-logits", directory,
            "--linear-mode", mode, "--threads", threads]


def python_command(model, prompt, requested, directory):
    return [sys.executable, ROOT / "scripts/make_golden.py",
            "--config", model / "model.json", "--weights", model / "model.bin",
            "--prompt", prompt, "-n", requested, "--output-dir", directory]


def check_cpp_contract(directory, expected, mode, threads):
    metadata = json.loads((directory / "cpp_run.json").read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError(
                f"C++ metadata {key}: expected {value!r}, got {metadata.get(key)!r}")
    if metadata.get("linear_mode") != mode or metadata.get("threads") != threads:
        raise AssertionError(
            f"C++ did not record requested mode/thread configuration: {metadata}")


def binary_argument(parser):
    parser.add_argument("--binary", default=os.environ.get(
        "MINI_LLAMA_BIN", str(ROOT / "build/mini-llama")))
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    if not binary.is_file():
        parser.error(f"C++ executable does not exist: {binary}")
    return binary


def main():
    from compare_logits import compare_directories

    binary = binary_argument(argparse.ArgumentParser(description=__doc__))
    work = Path(tempfile.mkdtemp(prefix="yan-golden-logits-"))
    cases = 0
    max_abs = 0.0
    max_rel = 0.0

    def compare_case(name, model, manifest, prompt, requested, modes):
        nonlocal cases, max_abs, max_rel
        expected = expected_contract(manifest, prompt, requested)
        py_dir = work / name / "python"
        run_command(python_command(model, prompt, requested, py_dir))
        for mode, threads in modes:
            cpp_dir = work / name / f"cpp-{mode}-{threads}"
            cpp_result = run_command(cpp_command(
                binary, model, prompt, requested, cpp_dir, mode, threads))
            summaries = set(cpp_result.stdout.splitlines())
            if (f"linear_mode:{mode}" not in summaries or
                    f"threads:{threads}" not in summaries or
                    not summaries.intersection({"compiled_simd:NEON",
                                                "compiled_simd:AVX2/FMA",
                                                "compiled_simd:scalar fallback"})):
                raise AssertionError(f"Missing execution summary: {cpp_result.stdout}")
            check_cpp_contract(cpp_dir, expected, mode, threads)
            stats = compare_directories(
                cpp_dir, py_dir, vocab_size=expected["vocab_size"],
                expected_steps=expected["step_count"], verbose=False)
            max_abs = max(max_abs, stats["max_abs_error"])
            max_rel = max(max_rel, stats["max_rel_error"])
            cases += 1
            print(f"PASS {name}/{mode}: steps={expected['step_count']} "
                  f"max_abs={stats['max_abs_error']:.3e} "
                  f"max_rel={stats['max_rel_error']:.3e}")
        return py_dir

    try:
        tiny = work / "tiny"
        manifest = export_fixture(tiny)
        for name, prompt, count in (
                ("empty-0", "", 0), ("empty-1", "", 1),
                ("hello-0", "hello", 0), ("hello-1", "hello", 1),
                ("repeat", "abca" * 10, 16),
                ("context-clip", "a" * 125, 4),
                ("context-full", "a" * 127, 4)):
            compare_case(name, tiny, manifest, prompt, count, (("native", 1),))
        py_hello = compare_case("hello-16", tiny, manifest, "hello", 16, MODES)

        gqa = work / "gqa"
        gqa_manifest = export_fixture(gqa, GQA_OVERRIDES)
        compare_case("gqa-16", gqa, gqa_manifest, "hello", 16, MODES)

        # Equal output rows force exact greedy ties, independently of RNG values.
        tied = work / "tied"
        tied_manifest = export_fixture(tied)
        output = next(tensor for tensor in tied_manifest["tensors"]
                      if tensor["name"] == "lm_head")
        with (tied / "model.bin").open("r+b") as weights:
            weights.seek(output["offset"])
            weights.write(bytes(output["byte_size"]))
        compare_case("greedy-ties", tied, tied_manifest, "hello", 4, MODES)

        for index, prompt in enumerate(("你好", "a" * 128)):
            run_command(cpp_command(binary, tiny, prompt, 0,
                                    work / f"invalid-cpp-{index}", "native", 1),
                        success=False)
            run_command(python_command(tiny, prompt, 0,
                                       work / f"invalid-python-{index}"),
                        success=False)
        # Existing producer files must not silently mix with a second run.
        run_command(cpp_command(binary, tiny, "hello", 16,
                                work / "hello-16/cpp-native-1", "native", 1),
                    success=False)
        run_command(python_command(tiny, "hello", 16, py_hello), success=False)
        run_command(cpp_command(binary, tiny, "hello", 1,
                                work / "invalid-mode", "unknown", 1),
                    success=False)

        print(f"golden cases: {cases}; max_abs_error={max_abs:.9g}; "
              f"max_rel_error={max_rel:.9g}")
    except Exception as error:
        print(f"golden artifacts retained: {work}", file=sys.stderr)
        print(f"RESULT: FAILED: {error}", file=sys.stderr)
        return 1
    shutil.rmtree(work)
    print("RESULT: ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
