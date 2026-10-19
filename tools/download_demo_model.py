#!/usr/bin/env python3
"""Download the Qwen2 GGUF demo model using only the Python standard library.

Based on the download defaults and CLI from:
https://github.com/CppTrainingHub/mini-llama-cpp/blob/main/scripts/download_demo_model.py
"""

import argparse
from http.client import HTTPException
import os
from pathlib import Path
import sys
from urllib.parse import quote
import urllib.request


DEFAULT_REPO = "qwen/Qwen2-0.5B-Instruct-GGUF"
DEFAULT_REMOTE_FILE = "qwen2-0_5b-instruct-q8_0.gguf"
DEFAULT_FILE = "Qwen2-0.5B-Instruct-Q8_0.gguf"
DEFAULT_DIR = "models/chat"


def is_valid_gguf(path):
    """Check the file signature; this is not a full model integrity check."""
    try:
        with Path(path).open("rb") as stream:
            return stream.read(4) == b"GGUF"
    except OSError:
        return False


def is_git_lfs_pointer(path):
    try:
        with Path(path).open("rb") as stream:
            return stream.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def get_modelscope_download_url(repo, filename):
    return (f"https://modelscope.cn/models/{quote(repo, safe='/')}/resolve/master/"
            f"{quote(filename, safe='/')}")


def download_file(url, dest):
    print(f"Downloading {url}\n  -> {dest}", flush=True)
    downloaded = 0
    last_progress = -1
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length", "0"))
            with Path(dest).open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    megabytes = downloaded / (1024 * 1024)
                    if total > 0:
                        percent = min(100, downloaded * 100 // total)
                        if percent != last_progress:
                            print(f"\r  {megabytes:.1f} / {total / (1024 * 1024):.1f} MB"
                                  f" ({percent}%)", end="", flush=True)
                            last_progress = percent
                    else:
                        print(f"\r  {megabytes:.1f} MB", end="", flush=True)
            if total > 0 and downloaded != total:
                raise ValueError(f"Incomplete download: expected {total} bytes, got {downloaded}")
    finally:
        print(flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download demo GGUF chat model from ModelScope")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="ModelScope repository")
    parser.add_argument("--remote-file", default=DEFAULT_REMOTE_FILE,
                        help="GGUF filename in the remote repository")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Local GGUF filename")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Download directory, relative to the current working directory")
    args = parser.parse_args(argv)
    dest = Path(args.dir) / args.file
    partial = dest.with_name(dest.name + ".part")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            if is_valid_gguf(dest):
                print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
                return 0
            reason = "Git LFS pointer" if is_git_lfs_pointer(dest) else "invalid GGUF file"
            print(f"Replacing {reason} after successful download: {dest}")

        url = get_modelscope_download_url(args.repo, args.remote_file)
        download_file(url, partial)
        if not is_valid_gguf(partial):
            raise ValueError("Downloaded file does not have the GGUF signature")
        os.replace(partial, dest)
        print(f"Downloaded: {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
        return 0
    except (OSError, ValueError, HTTPException) as error:
        print(f"Error downloading model: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download interrupted", file=sys.stderr)
        return 130
    finally:
        # Never leave a failed transfer at the final model path.
        if partial.is_file():
            partial.unlink()


if __name__ == "__main__":
    sys.exit(main())
