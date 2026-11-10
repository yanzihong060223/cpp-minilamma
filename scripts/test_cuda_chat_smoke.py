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


def run(*args):
    return subprocess.run([str(BINARY), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


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
            require(re.search(r"trace summary: trace_id=\S+ status=error", text), text)
            print(f"PASS {command} CUDA unavailable contract")
            continue
        require(result.returncode == 0, text)
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
        if command == "bench":
            require("Benchmark:" in result.stdout and "tokens/s (total):" in result.stdout, text)
            for field in ("cuda linear calls", "cuda attention calls"):
                require(re.search(r"^" + field + r": [1-9]\d*$", result.stdout, re.M), text)
        for field in ("cpu attention fallback calls", "cpu linear fallback calls", "host->device copies", "host->device bytes"):
            require(re.search(r"^" + re.escape(field) + r": 0$", result.stdout, re.M), text)
        require(re.search(r"trace summary: trace_id=\S+ status=ok", text), text)
        print(result.stdout, end="")
        print(f"PASS {command} CUDA execution")
    print("PASS cuda-chat-smoke")


if __name__ == "__main__":
    main()
