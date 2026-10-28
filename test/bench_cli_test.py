"""Exercise the chapter benchmark CLI using the existing tiny model exporter."""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = str(Path(sys.argv.pop(1)).resolve())
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "tiny_exporter", ROOT / "model/tinymodel/export_model.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class BenchCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="yan-bench-cli-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.ascii = cls.root / "ascii"
        cls.json_model = cls.root / "json"
        with contextlib.redirect_stdout(io.StringIO()):
            EXPORTER.export_model(cls.ascii, 42)
            EXPORTER.export_model(cls.json_model, 42, json_vocab=True)

    def run_cli(self, *args, cwd=None, success=True):
        result = subprocess.run(
            [RUNNER, *(str(arg) for arg in args)],
            cwd=cwd or self.root, capture_output=True, text=True, timeout=15)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(result.stderr.strip(), "Failure should explain the error")
        return result

    def field(self, output, name):
        match = re.search(r"^" + re.escape(name) + r":[ \t]*(.+)$", output, re.M)
        self.assertIsNotNone(match, f"Missing {name!r} in output:\n{output}")
        return match.group(1).strip()

    def check_counts(self, output, prompt, generated):
        self.assertEqual(int(self.field(output, "n_prompt_tokens")), prompt)
        self.assertEqual(int(self.field(output, "n_generate_tokens")), generated)
        self.assertEqual(int(self.field(output, "n_decode_tokens")),
                         max(0, generated - 1))
        for name in ("prefill_ms", "decode_ms", "tokens/s(total)", "tokens/s(Decode)"):
            value = float(self.field(output, name))
            self.assertTrue(math.isfinite(value), name)
            self.assertGreaterEqual(value, 0.0, name)

    def copied_model(self):
        directory = tempfile.TemporaryDirectory(prefix="yan-bench-invalid-")
        self.addCleanup(directory.cleanup)
        model = Path(directory.name) / "model"
        shutil.copytree(self.json_model, model)
        return model

    def test_exporter_preserves_weights_and_default_ascii_manifest(self):
        ascii_manifest = json.loads((self.ascii / "model.json").read_text())
        json_manifest = json.loads((self.json_model / "model.json").read_text())
        self.assertEqual(ascii_manifest["tokenizer"], {
            "type": "ASCII", "pathname": "", "bos_id": 128,
            "eos_id": 129, "unk_id": 130,
        })
        self.assertFalse((self.ascii / "vocab.json").exists())
        self.assertEqual(json_manifest["tokenizer"], {
            "type": "JsonVocab", "pathname": "vocab.json", "bos_id": 128,
            "eos_id": 129, "unk_id": 130,
        })
        self.assertEqual(ascii_manifest["config"], json_manifest["config"])
        self.assertEqual(ascii_manifest["tensors"], json_manifest["tensors"])
        self.assertEqual((self.ascii / "model.bin").read_bytes(),
                         (self.json_model / "model.bin").read_bytes())

        vocab = json.loads((self.json_model / "vocab.json").read_text())
        self.assertEqual(len(vocab), 131)
        self.assertEqual(vocab[:128], [
            {"id": i, "context": chr(i), "special": False} for i in range(128)
        ])
        self.assertEqual(vocab[128:], [
            {"id": 128, "context": "<BOS>", "special": True},
            {"id": 129, "context": "<EOS>", "special": True},
            {"id": 130, "context": "<UNK>", "special": True},
        ])

    def test_help_does_not_require_a_model(self):
        for args in (("--help",), ("bench", "--help")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertIn("mini-llama bench", result.stdout)

    def test_chapter_command_reports_threads_rates_and_native_mode(self):
        result = self.run_cli(
            "bench", "json", "--tokenizer", "json/vocab.json",
            "--prompt", "hello", "--n-predict", "4", "--threads", "4",
            "--verbose")
        self.assertEqual(self.field(result.stdout, "threads"), "4")
        self.assertEqual(self.field(result.stdout, "linear_mode"), "Native")
        self.assertIn(self.field(result.stdout, "compiled_simd"),
                      ("NEON", "AVX2/FMA", "scalar fallback"))
        self.check_counts(result.stdout, 6, 4)
        self.assertIn("Tensor: logits", result.stdout)
        self.assertIn("KV cache:", result.stdout)

    def test_thread_alias_and_metadata_vocab_are_resolved_from_model_directory(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        result = self.run_cli(
            "bench", self.json_model, "-threads", "4", "--n-predict", "1",
            cwd=elsewhere)
        self.assertEqual(self.field(result.stdout, "threads"), "4")
        self.check_counts(result.stdout, 6, 1)

    def test_original_ascii_model_needs_no_external_vocab(self):
        result = self.run_cli("bench", self.ascii, "--n-predict", "0")
        self.check_counts(result.stdout, 6, 0)

    def test_zero_one_and_empty_prompt_generation(self):
        for count in (0, 1):
            with self.subTest(count=count):
                result = self.run_cli(
                    "bench", self.json_model, "--n-predict", count,
                    "--threads", "1")
                self.check_counts(result.stdout, 6, count)
                self.assertEqual(float(self.field(result.stdout, "tokens/s(Decode)")), 0.0)
        result = self.run_cli(
            "bench", self.json_model, "--prompt", "", "--n-predict", "0")
        self.check_counts(result.stdout, 1, 0)

    def test_generation_is_clipped_to_remaining_context(self):
        result = self.run_cli(
            "bench", self.json_model, "--prompt", "a" * 125,
            "--n-predict", "4", "--threads", "1")
        self.check_counts(result.stdout, 126, 2)
        self.run_cli(
            "bench", self.json_model, "--prompt", "a" * 128,
            "--n-predict", "0", success=False)

    def test_zero_threads_reports_resolved_positive_count(self):
        result = self.run_cli(
            "bench", self.json_model, "--threads", "0", "--n-predict", "0")
        self.assertGreater(int(self.field(result.stdout, "threads")), 0)

    def test_invalid_commands_and_options_fail(self):
        for args in ((), ("unknown",), ("bench",),
                     ("bench", self.json_model, "--unknown", "1")):
            with self.subTest(args=args):
                self.run_cli(*args, success=False)
        for option in ("--tokenizer", "--prompt", "--n-predict", "--threads", "-threads"):
            with self.subTest(missing_value=option):
                self.run_cli("bench", self.json_model, option, success=False)
        for option in ("--threads", "-threads", "--n-predict"):
            for value in ("-1", "1x", "1.5", "", "2147483648"):
                with self.subTest(option=option, value=value):
                    self.run_cli("bench", self.json_model, option, value, success=False)

    def test_missing_models_weights_and_vocab_fail(self):
        self.run_cli("bench", self.root / "missing", success=False)
        model = self.copied_model()
        (model / "model.bin").unlink()
        self.run_cli("bench", model, success=False)
        self.run_cli("bench", self.json_model, "--tokenizer", "missing.json",
                     success=False)
        self.run_cli("bench", self.json_model, "--tokenizer", "", success=False)

    def test_vocab_size_and_special_id_mismatches_fail(self):
        model = self.copied_model()
        path = model / "vocab.json"
        original = json.loads(path.read_text())
        for mismatch in ("size", "special_ids"):
            with self.subTest(mismatch=mismatch):
                vocab = [dict(entry) for entry in original]
                if mismatch == "size":
                    vocab.append({"id": 131, "context": "<EXTRA>", "special": True})
                else:
                    vocab[128]["id"], vocab[129]["id"] = 129, 128
                path.write_text(json.dumps(vocab), encoding="utf-8")
                self.run_cli("bench", model, success=False)


if __name__ == "__main__":
    unittest.main()
