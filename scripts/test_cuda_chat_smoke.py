#!/usr/bin/env python3
"""Run the local Qwen model; a missing model is an explicit successful SKIP."""
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("MINI_LLAMA_CHAT_MODEL", ROOT / "tools/model/chat/Qwen2-0.5B-Instruct-Q8_0.gguf")).resolve()
BINARY = Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build/mini-llama")).resolve()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(*args, binary=BINARY, input_text=None):
    return subprocess.run([str(binary), *map(str, args)], cwd=ROOT,
                          input=input_text, capture_output=True, text=True, timeout=300)


def interactive(available):
    runner = BINARY.parent / "test/gguf_run"
    require(runner.is_file(), f"Missing interactive executable: {runner}")
    result = run(MODEL, "1", "--backend", "cuda", binary=runner,
                 input_text="hello\n/stats\nhi\n/stats\n/clear\n/stats\n/exit\n")
    text = result.stdout + result.stderr
    if not available:
        require(result.returncode != 0 and "Backend setup failed:" in text, text)
        require("CUDA backend was not built" in text or "CUDA device" in text, text)
        print("PASS interactive CUDA unavailable contract")
        return
    require(result.returncode == 0 and "FAIL" not in text, text)
    require(result.stderr.count("Loading model:") == 1, text)
    require(result.stderr.count("Turn complete:") == 2, text)
    require(result.stdout.count("backend: cuda\n") == 4, text)
    messages = re.findall(r"messages\s*:\s*(\d+)", result.stdout)
    require(messages == ["2", "4", "0"], text)

    def values(name):
        return [int(n) for n in re.findall(
            r"^" + re.escape(name) + r":\s*(\d+)\s*$", result.stdout, re.M)]

    uploaded = values("uploaded weights")
    require(len(uploaded) == 4 and uploaded[0] > 0 and len(set(uploaded)) == 1, text)
    for name in ("cuda linear calls", "cuda activation calls", "cuda attention calls",
                 "device->host copies", "device->host bytes"):
        counts = values(name)
        require(len(counts) == 4 and counts[0] == counts[3] == 0 and
                counts[1] > 0 and counts[2] > 0, text)
    for name in ("cpu attention fallback calls", "cpu linear fallback calls",
                 "host->device copies", "host->device bytes"):
        require(values(name) == [0]*4, text)
    print(result.stdout, end="")
    print("PASS interactive CUDA history/clear/stats")


def main():
    if not MODEL.is_file():
        print(f"SKIP cuda-chat-smoke missing model: {MODEL}")
        return
    print(f"chat model: {MODEL}")
    probe = run("--cuda-info")
    available = probe.returncode == 0
    if not available:
        text = probe.stdout + probe.stderr
        require("Backend setup failed:" in text, text)
        require("CUDA backend was not built" in text or "CUDA device" in text, text)
    for command in ("inspect", "bench"):
        args = [command, MODEL, "--backend", "cuda"]
        if command == "bench":
            args += ["-p", "hello", "-n", "1", "--seed", "1"]
        result = run(*args)
        text = result.stdout + result.stderr
        if not available:
            require(result.returncode != 0, text)
            require("Backend setup failed:" in text, text)
            require("CUDA backend was not built" in text or "CUDA device" in text, text)
            if command == "bench":
                require(result.stdout == "", text)
            else:
                require(re.search(r"trace summary: trace_id=\S+ status=error", text), text)
            print(f"PASS {command} CUDA unavailable contract")
            continue
        require(result.returncode == 0, text)
        if command == "bench":
            from bench import parse_stdout
            metrics = parse_stdout(result.stdout, "cuda", True, 1)
            q8 = metrics["cuda_q8_0_linear_weights"]
            require(q8["uploaded"] == q8["total"] and q8["total"] > 0, text)
            for key in ("cpu_attention_fallback", "h2d_copies"):
                require(metrics[key] == 0, text)
            for key in ("cuda_linear_calls", "cuda_attention_calls"):
                require(metrics[key] > 0, text)
            print(result.stdout, end="")
            print("PASS bench CUDA execution")
            continue
        for field in ("backend: cuda", "GGUF model loaded successfully", "compute: cuda linear",
                      "CUDA attention over GPU KV cache", "CPU sampler", "uploaded weights:",
                      "cuda f32 linear weights:", "cuda q8_0 linear weights:",
                      "cuda q4_0 linear weights:", "cuda q4_1 linear weights:", "cuda runtime:",
                      "cuda linear calls:", "cuda activation calls:", "cuda attention calls:",
                      "cpu linear fallback calls:", "cpu attention fallback calls:",
                      "host->device copies:", "host->device bytes:",
                      "device->host copies:", "device->host bytes:"):
            require(field in result.stdout, text)
        for field in ("uploaded weights", "cuda q8_0 linear weights"):
            require(re.search(r"^" + re.escape(field) + r": [1-9]\d*$", result.stdout, re.M), text)
        for field in ("cpu attention fallback calls", "cpu linear fallback calls", "host->device copies", "host->device bytes"):
            require(re.search(r"^" + re.escape(field) + r": 0$", result.stdout, re.M), text)
        require(re.search(r"trace summary: trace_id=\S+ status=ok", text), text)
        print(result.stdout, end="")
        print(f"PASS {command} CUDA execution")
    interactive(available)
    print("PASS cuda-chat-smoke")


if __name__ == "__main__":
    main()
