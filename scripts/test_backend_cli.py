#!/usr/bin/env python3
"""Black-box backend contract; numeric correctness lives in the C++ layers."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build/mini-llama")).resolve()
TINY = ROOT / "models/tiny"
FIELDS = (
    "uploaded weights", "cuda f32 linear weights", "cuda q8_0 linear weights",
    "cuda q4_0 linear weights", "cuda q4_1 linear weights", "cuda linear calls",
    "cuda activation calls", "cuda attention calls", "cpu linear fallback calls",
    "cpu attention fallback calls", "host->device copies", "host->device bytes",
    "device->host copies", "device->host bytes",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(*args, binary=BINARY, timeout=300):
    return subprocess.run([str(binary), *map(str, args)], cwd=ROOT, text=True,
                          capture_output=True, timeout=timeout)


def output(result):
    return result.stdout + result.stderr


def success(result):
    require(result.returncode == 0, output(result))


def failure(result, message):
    require(result.returncode != 0, output(result))
    require(message in output(result), output(result))


def trace(result, status):
    matches = re.findall(r"^trace summary: trace_id=(\S+) status=(\w+)$",
                         output(result), re.M)
    require(len(matches) == 1 and matches[0][1] == status, output(result))
    return matches[0][0]


def field(text, name):
    match = re.search(r"^" + re.escape(name) + r":\s*(\d+)\s*$", text, re.M)
    require(match is not None, f"Missing integer {name}:\n{text}")
    return int(match[1])


def summary(result, backend):
    require(f"backend: {backend}" in result.stdout, output(result))
    require("cuda runtime:" in result.stdout and "cuda device count:" in result.stdout, output(result))
    for name in FIELDS:
        field(result.stdout, name)


def unavailable(result):
    failure(result, "Backend setup failed:")
    require("CUDA backend was not built" in output(result) or
            "CUDA device" in output(result), output(result))
    trace(result, "error")


def test_modes(cuda_available, cuda_built):
    for part in ("runtime", "matmul", "linear", "quant", "ops", "kv_cache",
                 "attention", "forward", "forward_buffer", "forward_linear"):
        binary = BINARY.parent / "tests" / f"test_cuda_{part}"
        require(binary.is_file(), f"Missing C++ test executable: {binary}")
        result = run("unknown-mode", binary=binary)
        failure(result, "unknown mode: unknown-mode")
        require("PASS" not in output(result), output(result))
        if cuda_available or not cuda_built:
            result = run("all", binary=binary)
            success(result)
            expected = 4 if part == "quant" else 3 if part == "attention" else 2 if part == "forward_buffer" else 1
            require(len(re.findall(r"^PASS ", result.stdout, re.M)) == expected, output(result))
        print(f"PASS modes {part}")


def test_generate_stopping(cuda_available):
    from test_chat_backend_cli import FIXTURE

    with tempfile.TemporaryDirectory(prefix="yan-generate-eos-") as temp:
        model = Path(temp) / "tiny.gguf"
        for backend in (["cpu", "cuda"] if cuda_available else ["cpu"]):
            # Test termination behavior with deterministic GGUF fixtures. These
            # output assertions do not replace CPU/CUDA logits comparisons.
            for eos, count, expected, nan_logits in (
                (True, 4, "", False),
                (False, 4, "xxxx", False),
                (False, 1, "x", False),
                # NaN logits would make SamplerO fail if count=0 sampled at all.
                (True, 0, "", True),
            ):
                FIXTURE.model_file(model, q8=True, eos=eos, nan_logits=nan_logits)
                result = run("generate", model, "--backend", backend,
                             "-p", "hello", "-n", count, "--seed", 1)
                success(result)
                trace(result, "ok")
                summary(result, backend)
                generated = result.stdout.split(f"backend: {backend}\n", 1)[0]
                require(generated == "GGUF model loaded successfully\n" + expected + "\n",
                        output(result))
                require("<eos>" not in result.stdout, output(result))
                if backend == "cuda":
                    # Fixture has one layer and eight linears/activations per
                    # forward. Immediate EOS permits only the five prompt steps.
                    steps = len("hello") + (0 if eos or count == 0 else count - 1)
                    for name, expected_count in (
                        ("cuda linear calls", 8 * steps),
                        ("cuda activation calls", 8 * steps),
                        ("cuda attention calls", steps),
                        ("device->host copies", steps),
                        ("device->host bytes", 257 * 4 * steps),
                        ("host->device copies", 0),
                        ("host->device bytes", 0),
                        ("cpu linear fallback calls", 0),
                        ("cpu attention fallback calls", 0),
                    ):
                        require(field(result.stdout, name) == expected_count,
                                f"{name} expected {expected_count}:\n{output(result)}")
                print(f"PASS generate {backend} eos={eos} limit={count}")

            # Bench intentionally retains a fixed workload, even for EOS.
            FIXTURE.model_file(model, q8=True, eos=True)
            result = run("bench", model, "--backend", backend, "-p", "hello", "-n", 4)
            success(result)
            trace(result, "ok")
            require(field(result.stdout, "n_generate_tokens") == 4, output(result))
            print(f"PASS bench {backend} fixed workload with EOS")


def main():
    result = run("generate", TINY, "--backend", "cpu", "-p", "hello", "-n", 1, "--seed", 1)
    success(result)
    summary(result, "cpu")
    require("compute: cpu" in result.stdout, output(result))
    for name in FIELDS:
        require(field(result.stdout, name) == 0, output(result))
    generate_trace = trace(result, "ok")
    print("PASS generate CPU summary")

    result = run("bench", TINY, "--backend", "cpu", "-p", "hello", "-n", 1)
    success(result)
    require("tokens/s (total):" in result.stdout, output(result))
    require(trace(result, "ok") != generate_trace, "trace_id reused")
    print("PASS bench CPU throughput")

    for args, message in (
        (("--backend", "metal"), "Invalid --backend value"),
        (("--device", "-1"), "Invalid --device value"),
        (("--backend", "cpu", "--device", "0"), "--device can only be used with --backend cuda"),
    ):
        result = run("generate", "/missing/model", *args)
        failure(result, message)
        trace(result, "error")
        print(f"PASS {message}")

    result = run("generate", TINY, "--backend", "cpu", "--n-predict", 129)
    failure(result, "context window")
    trace(result, "error")
    print("PASS context error retains trace_id")

    probe = run("--cuda-info")
    available = probe.returncode == 0
    built = "CUDA backend was not built" not in output(probe)
    if available:
        for name in ("device name:", "compute capability:", "total memory:",
                     "driver version:", "runtime version:"):
            require(name in probe.stdout, output(probe))
        require(field(probe.stdout, "cuda device count") > 0, output(probe))
        require(field(probe.stdout, "total memory") > 0, output(probe))
        require(field(probe.stdout, "driver version") > 0 and
                field(probe.stdout, "runtime version") > 0, output(probe))
        trace(probe, "ok")
    else:
        unavailable(probe)
    print("PASS cuda-info" + (" available" if available else " unavailable contract"))

    result = run("bench", TINY, "--backend", "cuda", "-p", "hello", "-n", 1, "--seed", 1)
    if available:
        success(result)
        summary(result, "cuda")
        for text in ("compute: cuda linear", "CUDA attention over GPU KV cache", "CPU sampler"):
            require(text in result.stdout, output(result))
        for name, expected in (("uploaded weights", 21), ("cuda f32 linear weights", 15),
                               ("cuda linear calls", 90), ("cuda activation calls", 90),
                               ("cuda attention calls", 12), ("cpu attention fallback calls", 0),
                               ("cpu linear fallback calls", 0), ("host->device copies", 0),
                               ("host->device bytes", 0), ("device->host copies", 6),
                               ("device->host bytes", 6*524)):
            require(field(result.stdout, name) == expected, f"{name} expected {expected}:\n{output(result)}")
        trace(result, "ok")
        print(result.stdout, end="")
        inspected = run("inspect", TINY, "--backend", "cuda")
        success(inspected)
        summary(inspected, "cuda")
        for name in FIELDS[5:]:
            require(field(inspected.stdout, name) == 0, output(inspected))
    else:
        unavailable(result)
    print("PASS CUDA backend" + (" execution" if available else " unavailable contract"))
    test_generate_stopping(available)
    test_modes(available, built)
    from test_chat_backend_cli import test_chat_backend
    test_chat_backend(BINARY, available)
    print("PASS backend-cli-smoke")


if __name__ == "__main__":
    main()
