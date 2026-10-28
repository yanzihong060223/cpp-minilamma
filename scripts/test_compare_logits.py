#!/usr/bin/env python3
"""Fault injection for the golden-logits file, number, and sampling checks."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from compare_logits import compare_directories


class CompareLogitsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="yan-compare-logits-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cpp = self.root / "cpp"
        self.python = self.root / "python"
        self.cpp.mkdir()
        self.python.mkdir()
        self.metadata = {
            "format_version": 1,
            "vocab_size": 4,
            "bos_id": 1,
            "prompt_tokens": [1, 2],
            "requested_generation": 2,
            "max_seq_len": 16,
            "generated_count": 2,
            "step_count": 3,
        }
        for side in ("cpp", "python"):
            self.write_metadata(side, self.metadata)
            self.write_tokens(side, [1, 1])
            for step in range(3):
                self.write_logits(side, step,
                                  [0.25 + step * 0.01, 1.0 + step * 0.1, -0.5, 0.0])

    def paths(self, side):
        if side == "cpp":
            return self.cpp, "logits_step", "generation_tokens.txt", "cpp_run.json"
        return self.python, "python_logits_step", "python_generation_tokens.txt", "python_run.json"

    def step_path(self, side, step):
        directory, prefix, _, _ = self.paths(side)
        return directory / f"{prefix}{step}.bin"

    def write_logits(self, side, step, values):
        self.step_path(side, step).write_bytes(np.asarray(values, dtype="<f4").tobytes())

    def read_logits(self, side, step):
        return np.frombuffer(self.step_path(side, step).read_bytes(), dtype="<f4").copy()

    def write_metadata(self, side, metadata):
        directory, _, _, name = self.paths(side)
        (directory / name).write_text(json.dumps(metadata), encoding="utf-8")

    def write_tokens(self, side, tokens):
        directory, _, name, _ = self.paths(side)
        (directory / name).write_text("\n".join(str(token) for token in tokens),
                                      encoding="utf-8")

    def compare(self, **kwargs):
        return compare_directories(self.cpp, self.python, verbose=False, **kwargs)

    def test_complete_runs_pass_with_only_final_success_marker(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = compare_directories(self.cpp, self.python,
                                         vocab_size=4, expected_steps=3)
        self.assertEqual(result, {
            "max_abs_error": 0.0, "max_rel_error": 0.0,
            "step_count": 3, "generated_count": 2,
        })
        self.assertEqual(output.getvalue().count("RESULT: ALL PASSED"), 1)
        self.assertIn("FILE SET: PASSED", output.getvalue())
        self.assertIn("LOGITS: PASSED", output.getvalue())
        self.assertIn("SAMPLING: PASSED", output.getvalue())

    def test_opposite_producer_files_can_share_a_directory(self):
        for path in self.python.iterdir():
            (self.cpp / path.name).write_bytes(path.read_bytes())
        (self.cpp / "notes.txt").write_text("not a logits file")
        compare_directories(self.cpp, self.cpp, expected_steps=3, verbose=False)

    def test_missing_one_side_or_both_last_steps_fail(self):
        self.step_path("cpp", 2).unlink()
        with self.assertRaisesRegex(ValueError, "FILE SET.*missing=.*2"):
            self.compare()
        self.step_path("python", 2).unlink()
        with self.assertRaisesRegex(ValueError, "FILE SET.*missing=.*2"):
            self.compare()

    def test_hole_and_extra_file_fail_even_with_correct_file_count(self):
        self.step_path("cpp", 1).rename(self.step_path("cpp", 3))
        with self.assertRaisesRegex(ValueError, "missing=.*1.*extra=.*3"):
            self.compare()

    def test_extra_step_fails(self):
        self.write_logits("python", 3, [0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "FILE SET.*extra=.*3"):
            self.compare()

    def test_malformed_or_non_file_steps_fail(self):
        for name in ("logits_step00.bin", "logits_step-1.bin",
                     "logits_stepx.bin", "logits_step3.txt"):
            with self.subTest(name=name):
                path = self.cpp / name
                path.write_bytes(bytes(16))
                with self.assertRaisesRegex(ValueError, "invalid step filename"):
                    self.compare()
                path.unlink()
        path = self.cpp / "logits_step3.bin"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "invalid step filename or file type"):
            self.compare()

    def test_empty_collections_fail(self):
        for side in ("cpp", "python"):
            for step in range(3):
                self.step_path(side, step).unlink()
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.compare()

    def test_missing_metadata_fails(self):
        (self.python / "python_run.json").unlink()
        with self.assertRaisesRegex(ValueError, "cannot read valid metadata"):
            self.compare()

    def test_invalid_metadata_types_counts_and_context_fail(self):
        variants = {
            "format_version": [0, 2, True],
            "vocab_size": [0, -1, 4.0, True],
            "bos_id": [4, -1, True],
            "prompt_tokens": [[], [2, 1], [1, -1], [1, 4], [1, True], "1 2"],
            "requested_generation": [-1, 2.0, True],
            "max_seq_len": [0, 1, True],
            "generated_count": [0, 1, 3, True],
            "step_count": [0, 2, 4, True],
        }
        for field, values in variants.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    metadata = copy.deepcopy(self.metadata)
                    metadata[field] = value
                    self.write_metadata("cpp", metadata)
                    with self.assertRaises(ValueError):
                        self.compare()
        self.write_metadata("cpp", self.metadata)
        metadata = copy.deepcopy(self.metadata)
        del metadata["requested_generation"]
        self.write_metadata("cpp", metadata)
        with self.assertRaisesRegex(ValueError, "missing metadata fields"):
            self.compare()

    def test_metadata_mismatch_and_authoritative_expected_values_fail(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["prompt_tokens"] = [1, 3]
        self.write_metadata("python", metadata)
        with self.assertRaisesRegex(ValueError, "metadata mismatch for prompt_tokens"):
            self.compare()
        self.write_metadata("python", self.metadata)
        with self.assertRaisesRegex(ValueError, "expected vocab_size=5"):
            self.compare(vocab_size=5)
        with self.assertRaisesRegex(ValueError, "expected_steps=4"):
            self.compare(expected_steps=4)

    def test_both_metadata_cannot_claim_truncated_step_count(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["step_count"] = 2
        for side in ("cpp", "python"):
            self.write_metadata(side, metadata)
            self.step_path(side, 2).unlink()
        with self.assertRaisesRegex(ValueError, "metadata step_count must be 3"):
            self.compare()

    def test_duplicate_metadata_fields_fail(self):
        path = self.cpp / "cpp_run.json"
        path.write_text('{"format_version":1,"format_version":1}')
        with self.assertRaisesRegex(ValueError, "Duplicate metadata field"):
            self.compare()

    def test_wrong_byte_lengths_fail(self):
        for size in (0, 15, 17, 32):
            with self.subTest(size=size):
                self.step_path("cpp", 0).write_bytes(bytes(size))
                with self.assertRaisesRegex(ValueError, "expected 16 bytes"):
                    self.compare()

    def test_nan_inf_fail_even_when_both_sides_are_equal(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                for side in ("cpp", "python"):
                    self.write_logits(side, 0, [bad, 1, 0, 0])
                with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                    self.compare()

    def test_absolute_error_is_independent_of_relative_error(self):
        self.write_logits("python", 0, [100, 1, 0, 0])
        self.write_logits("cpp", 0, [100.001, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "tolerance exceeded.*step=0.*max_abs_error"):
            self.compare()

    def test_relative_error_is_independent_of_absolute_error(self):
        self.write_logits("python", 0, [0.002, 1, 0, 0])
        self.write_logits("cpp", 0, [0.0020003, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "tolerance exceeded.*max_rel_error"):
            self.compare()

    def test_relative_mask_boundary_and_empty_mask(self):
        self.write_logits("python", 0, [0.0009999, 1, 0, 0])
        self.write_logits("cpp", 0, [0.0010499, 1, 0, 0])
        self.compare()  # This token is excluded despite a large relative error.
        self.write_logits("python", 0, [0.001, 1, 0, 0])
        self.write_logits("cpp", 0, [0.0010002, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
            self.compare()
        for step in range(3):
            self.write_logits("python", step, [0, 0.0003, -0.0002, 0.0001])
            self.write_logits("cpp", step, [0.00001, 0.0003, -0.0002, 0.0001])
        result = self.compare()
        self.assertEqual(result["max_rel_error"], 0.0)

    def test_below_and_above_absolute_tolerance(self):
        self.write_logits("python", 0, [1, 2, 0, 0])
        self.write_logits("cpp", 0, [1.0000999, 2, 0, 0])
        self.compare()
        self.write_logits("cpp", 0, [1.0001001, 2, 0, 0])
        with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
            self.compare()

    def test_near_tie_top1_mismatch_fails_within_numerical_tolerance(self):
        self.write_logits("python", 0, [1, 1.00001, 0, 0])
        self.write_logits("cpp", 0, [1.00002, 1.00001, 0, 0])
        with self.assertRaisesRegex(ValueError, "top-1 mismatch.*cpp_top1=0.*reference_top1=1"):
            self.compare()

    def test_equal_ties_choose_lowest_token_id(self):
        for side in ("cpp", "python"):
            for step in range(3):
                self.write_logits(side, step, [1, 1, 0, 0])
            self.write_tokens(side, [0, 0])
        self.compare()

    def test_missing_illegal_and_wrong_length_token_files_fail(self):
        path = self.cpp / "generation_tokens.txt"
        for text in ("", "1", "1 1 1", "-1 1", "4 1", "1x 1"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "SAMPLING"):
                    self.compare()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "cannot read token file"):
            self.compare()

    def test_one_or_both_identically_wrong_generation_files_fail(self):
        self.write_tokens("cpp", [1, 0])
        with self.assertRaisesRegex(ValueError, "generation index=1, step=2.*greedy argmax=1"):
            self.compare()
        self.write_tokens("python", [1, 0])
        with self.assertRaisesRegex(ValueError, "generation index=1, step=2.*greedy argmax=1"):
            self.compare()

    def test_zero_one_and_context_clipped_generation_step_counts(self):
        for requested, capacity, generated, steps in ((0, 16, 0, 2),
                                                       (1, 16, 1, 2),
                                                       (100, 3, 1, 2),
                                                       (100, 2, 0, 2)):
            with self.subTest(requested=requested, capacity=capacity):
                metadata = copy.deepcopy(self.metadata)
                metadata.update(requested_generation=requested,
                                max_seq_len=capacity,
                                generated_count=generated, step_count=steps)
                for side in ("cpp", "python"):
                    self.write_metadata(side, metadata)
                    self.write_tokens(side, [1] * generated)
                    self.step_path(side, 2).unlink(missing_ok=True)
                result = self.compare(expected_steps=steps)
                self.assertEqual(result["generated_count"], generated)

    def test_exact_cpp_thread_comparison_and_signed_zero(self):
        other = self.root / "cpp4"
        other.mkdir()
        for path in self.cpp.iterdir():
            (other / path.name).write_bytes(path.read_bytes())
        compare_directories(self.cpp, other, reference_kind="cpp", exact=True,
                            verbose=False)
        values = self.read_logits("cpp", 0)
        values[3] = -0.0
        (other / "logits_step0.bin").write_bytes(values.astype("<f4").tobytes())
        with self.assertRaisesRegex(ValueError, "exact thread comparison differs"):
            compare_directories(self.cpp, other, reference_kind="cpp", exact=True,
                                verbose=False)

    def test_cli_returns_nonzero_and_never_prints_success_on_failure(self):
        self.step_path("cpp", 2).unlink()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("compare_logits.py")),
             "--cpp-dir", str(self.cpp), "--py-dir", str(self.python)],
            capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RESULT: FAILED", result.stderr)
        self.assertNotIn("RESULT: ALL PASSED", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
