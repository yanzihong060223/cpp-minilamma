#!/usr/bin/env python3
"""Check exact one/four-thread logits for the actual threaded forward modes."""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from compare_logits import compare_directories
from test_golden import (GQA_OVERRIDES, binary_argument, check_cpp_contract,
                         cpp_command, expected_contract, export_fixture,
                         run_command)


def main():
    binary = binary_argument(argparse.ArgumentParser(description=__doc__))
    work = Path(tempfile.mkdtemp(prefix="yan-golden-threaded-"))
    cases = 0
    try:
        for fixture, overrides in (("tiny", None), ("gqa", GQA_OVERRIDES)):
            model = work / fixture / "model"
            manifest = export_fixture(model, overrides)
            expected = expected_contract(manifest, "hello", 16)
            for mode in ("threaded", "threaded-simd"):
                directories = []
                for threads in (1, 4):
                    directory = work / fixture / f"{mode}-{threads}"
                    run_command(cpp_command(
                        binary, model, "hello", 16, directory, mode, threads))
                    check_cpp_contract(directory, expected, mode, threads)
                    directories.append(directory)
                compare_directories(
                    directories[1], directories[0],
                    vocab_size=expected["vocab_size"],
                    expected_steps=expected["step_count"], reference_kind="cpp",
                    exact=True, verbose=False)
                cases += 1
                print(f"PASS {fixture}/{mode}: 1/4 threads; "
                      f"{expected['step_count']} identical logits steps")
    except Exception as error:
        print(f"threaded artifacts retained: {work}", file=sys.stderr)
        print(f"RESULT: FAILED: {error}", file=sys.stderr)
        return 1
    shutil.rmtree(work)
    print(f"threaded cases: {cases}")
    print("RESULT: ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
