#!/usr/bin/env python3
"""Exercise the existing interactive loop with CPU/CUDA backend selection."""
from pathlib import Path
import importlib.util
import io
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gguf_interactive_fixture", ROOT / "test/gguf_run_test.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def values(text, field):
    return [int(value) for value in re.findall(
        r"^" + re.escape(field) + r":\s*(\d+)\s*$", text, re.M)]


def run(runner, model, *args, text="/exit\n"):
    return subprocess.run([str(runner), str(model), *map(str, args)],
                          input=text, capture_output=True, text=True,
                          cwd=ROOT, timeout=300)


def legacy_cpu_cases(runner):
    # Reuse the CPU conversation regressions without requiring the optional
    # gguf_generate executable or registering another CTest case.
    FIXTURE.RUNNER = str(runner)
    names = unittest.defaultTestLoader.getTestCaseNames(FIXTURE.InteractiveTest)
    suite = unittest.TestSuite(FIXTURE.InteractiveTest(name) for name in names
                              if name != "test_q8_forward_and_single_turn_regression")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    require(result.wasSuccessful(), log.getvalue())
    print(f"PASS existing CPU conversation regressions ({result.testsRun})")


def multiturn(runner, model, backend, q8):
    FIXTURE.model_file(model, q8=q8)
    text = "hello\n/stats\nb\n/stats\n/clear\n/stats\nc\n/stats\n/exit\n"
    args = ["2", "--backend", backend]
    if backend == "cuda":
        args += ["--device", "0"]
    result = run(runner, model, *args, text=text)
    combined = result.stdout + result.stderr
    require(result.returncode == 0 and "FAIL" not in combined, combined)
    require(result.stderr.count("Loading model:") == 1, combined)
    require("Chat ready." in result.stdout, combined)
    # Header once, then only /stats displays backend counters.
    require(result.stdout.count(f"backend: {backend}\n") == 5, combined)
    require(FIXTURE.InteractiveTest().counts(result.stdout, "messages") == [2, 4, 0, 2], combined)
    prompts = [len(FIXTURE.prompt([("user", "hello")])),
               len(FIXTURE.prompt([("user", "hello"), ("assistant", "xx"), ("user", "b")])),
               len(FIXTURE.prompt([("user", "c")]))]
    actual_prompts = FIXTURE.InteractiveTest().counts(result.stderr, "Prompt tokens")
    require(actual_prompts == prompts, combined)
    require(values(result.stdout, "cpu attention fallback calls") == [0]*5, combined)
    require(values(result.stdout, "cpu linear fallback calls") == [0]*5, combined)
    require(values(result.stdout, "host->device copies") == [0]*5, combined)
    require(values(result.stdout, "host->device bytes") == [0]*5, combined)
    if backend == "cpu":
        for name in ("uploaded weights", "cuda linear calls", "cuda activation calls",
                     "cuda attention calls", "device->host copies", "device->host bytes"):
            require(values(result.stdout, name) == [0]*5, combined)
    else:
        # This fixture has one layer, eight linears and eight activation ops per
        # forward. Two generated tokens require one decode after prompt prefill.
        steps = [0, prompts[0]+1, prompts[1]+1, 0, prompts[2]+1]
        require(values(result.stdout, "uploaded weights") == [12]*5, combined)
        kind = "q8_0" if q8 else "f32"
        require(values(result.stdout, f"cuda {kind} linear weights") == [8]*5, combined)
        for name in ("cuda linear calls", "cuda activation calls"):
            require(values(result.stdout, name) == [8*n for n in steps], combined)
        require(values(result.stdout, "cuda attention calls") == steps, combined)
        require(values(result.stdout, "device->host copies") == steps, combined)
        require(values(result.stdout, "device->host bytes") == [257*4*n for n in steps], combined)
        require("CUDA attention over GPU KV cache" in result.stdout, combined)
        print("cpu attention fallback calls: 0\nhost->device copies: 0")
    print(f"PASS interactive {backend} {'q8_0' if q8 else 'f32'} history/clear/stats")

    # No per-turn summary unless /stats was requested; final line without a
    # newline is still processed before EOF.
    eof = run(runner, model, "2", "--backend", backend, text="hello")
    combined = eof.stdout + eof.stderr
    require(eof.returncode == 0 and "FAIL" not in combined, combined)
    require(eof.stdout.count(f"backend: {backend}\n") == 1, combined)
    require(eof.stderr.count("Turn complete:") == 1, combined)


def test_chat_backend(binary, cuda_available):
    runner = Path(binary).parent / "test/gguf_run"
    require(runner.is_file(), f"Missing interactive executable: {runner}")
    legacy_cpu_cases(runner)
    with tempfile.TemporaryDirectory(prefix="yan-chat-backend-") as temp:
        model = Path(temp) / "tiny.gguf"
        for q8 in (False, True):
            multiturn(runner, model, "cpu", q8)
            if cuda_available:
                multiturn(runner, model, "cuda", q8)
        for args, message in (
            (("--backend", "metal"), "Invalid --backend value"),
            (("--backend",), "Missing value for --backend"),
            (("--device",), "Missing value for --device"),
            (("--backend", "cuda", "--device", "-1"), "Invalid --device value"),
            (("--device", "1x"), "Invalid --device value"),
            (("--backend", "cpu", "--device", "0"), "--device can only be used with --backend cuda"),
            (("--unknown",), "Unknown option"),
            (("2", "3"), "Unexpected argument"),
        ):
            result = run(runner, model, *args)
            combined = result.stdout + result.stderr
            require(result.returncode != 0 and message in combined, combined)
            require("Loading model:" not in result.stderr, combined)
        if not cuda_available:
            result = run(runner, model, "2", "--backend", "cuda")
            combined = result.stdout + result.stderr
            require(result.returncode != 0 and "Backend setup failed:" in combined, combined)
            require("CUDA backend was not built" in combined or "CUDA device" in combined, combined)
            require("Loading model:" not in result.stderr, combined)
            print("PASS interactive CUDA unavailable contract")
    print("PASS interactive backend argument validation")
