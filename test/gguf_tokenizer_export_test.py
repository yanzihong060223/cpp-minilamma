"""Exporter tests, including C++ public-API comparisons for both loading paths."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exporter", ROOT / "tools/export_gguf_tokenizer.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)
COMPARE = sys.argv.pop(1) if len(sys.argv) > 1 else None


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def value(kind, data):
    if kind == 8:
        return string(data)
    if kind == 9:
        element, items = data
        return struct.pack("<IQ", element, len(items)) + b"".join(value(element, item) for item in items)
    return struct.pack("<" + EXPORTER.FORMATS[kind], data)


def fields():
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [b for b in range(256) if b not in kept]
    symbols = {b: chr(b) for b in kept}
    symbols.update({b: chr(256 + i) for i, b in enumerate(missing)})
    tokens = ["<bos>", "<eos>", "<unk>"] + [symbols[b] for b in range(256)]
    tokens += ["he", "hel", "hell", "hello", "ab", "bc", "aa", "<ctrl>", "<ctrl>long", "<控制>", "<|literal|>"]
    types = [3 if t in ("<ctrl>", "<ctrl>long", "<控制>") else 1 for t in tokens]
    prefix = EXPORTER.PREFIX
    return {
        "general.name": (8, "fixture"),  # exercise skipping unrelated metadata
        "general.flags": (9, (7, [True, False])),
        prefix + "model": (8, "gpt2"),
        prefix + "tokens": (9, (8, tokens)),
        prefix + "token_type": (9, (5, types)),
        prefix + "scores": (9, (6, [0.25] * len(tokens))),
        prefix + "merges": (9, (8, ["b c", "a b", "h e", "he l", "hel l", "hell o", "a a"])),
        prefix + "bos_token_id": (4, 0),
        prefix + "eos_token_id": (4, 1),
        prefix + "unknown_token_id": (4, 2),
        prefix + "add_bos_token": (7, False),
        prefix + "add_eos_token": (7, False),
        "tokenizer.chat_template": (8, "用户\n{{messages}}\t"),
    }


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="yan-gguf-export-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "model.gguf"
        self.output = self.root / "export"
        self.data = fields()

    def write(self, version=3):
        metadata = b"".join(string(key) + struct.pack("<I", kind) + value(kind, data)
                            for key, (kind, data) in self.data.items())
        self.source.write_bytes(b"GGUF" + struct.pack("<IQQ", version, 0, len(self.data)) + metadata)

    def export(self, version=3):
        self.write(version)
        EXPORTER.export_tokenizer(self.source, self.output)

    def compare(self, *args):
        if COMPARE is None:
            self.skipTest("pass the gguf_tokenizer_compare executable to enable C++ comparisons")
        result = subprocess.run([COMPARE, str(self.source), str(self.output), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_no_bpe(self):
        self.assertTrue((self.output / "tokenizer_metadata.json").exists())
        self.assertTrue((self.output / "tokens.tsv").exists())
        for name in ("vocab.json", "merges.txt", "special_tokens.json"):
            self.assertFalse((self.output / name).exists())

    def test_exports_exact_ids_rules_and_reports(self):
        self.export()
        vocab = json.loads((self.output / "vocab.json").read_text())
        self.assertEqual(vocab, dict(zip(self.data[EXPORTER.PREFIX + "tokens"][1][1], range(len(vocab)))))
        self.assertEqual((self.output / "merges.txt").read_text().splitlines()[1:],
                         self.data[EXPORTER.PREFIX + "merges"][1][1])
        report = json.loads((self.output / "tokenizer_metadata.json").read_text())
        self.assertNotIn("general.name", report)
        self.assertEqual(report["tokenizer.chat_template"]["value"], "用户\n{{messages}}\t")
        self.assertEqual(report[EXPORTER.PREFIX + "scores"]["element_type"], "FLOAT32")
        rows = (self.output / "tokens.tsv").read_text().splitlines()
        self.assertEqual(len(rows), len(vocab) + 1)
        self.assertTrue(rows[1].endswith('"<bos>"'))
        self.compare()

    def test_gguf_v2(self):
        self.export(version=2)
        self.compare()

    def test_missing_special_ids(self):
        for key in ("bos_token_id", "eos_token_id", "unknown_token_id"):
            del self.data[EXPORTER.PREFIX + key]
        self.export()
        config = json.loads((self.output / "special_tokens.json").read_text())
        for key in ("bos_id", "eos_id", "unk_id", "pad_id"):
            self.assertEqual(config[key], -1)
        self.compare()

    def test_pad_role_without_control_type(self):
        tokens = self.data[EXPORTER.PREFIX + "tokens"][1][1]
        tokens.append("<填充>")
        self.data[EXPORTER.PREFIX + "token_type"][1][1].append(1)
        self.data[EXPORTER.PREFIX + "scores"][1][1].append(0.25)
        self.data[EXPORTER.PREFIX + "padding_token_id"] = (4, len(tokens) - 1)
        self.export()
        config = json.loads((self.output / "special_tokens.json").read_text())
        self.assertEqual(config["pad_id"], len(tokens) - 1)
        self.assertIn(config["pad_id"], config["special_ids"])
        self.compare("--pad")

    def test_pad_can_share_bos_and_eos(self):
        self.data[EXPORTER.PREFIX + "padding_token_id"] = (4, 0)
        self.data[EXPORTER.PREFIX + "eos_token_id"] = (4, 0)
        self.export()
        config = json.loads((self.output / "special_tokens.json").read_text())
        self.assertEqual(config["pad_id"], 0)
        self.assertEqual(config["special_ids"].count(0), 1)
        self.compare()

    def test_invalid_pad_is_rejected(self):
        for index, payload in enumerate(((5, -1), (5, -2), (4, 99999), (7, True))):
            with self.subTest(payload=payload):
                self.data[EXPORTER.PREFIX + "padding_token_id"] = payload
                self.output = self.root / f"bad-pad-{index}"
                self.write()
                with self.assertRaisesRegex(ValueError, "Invalid vocabulary ID"):
                    EXPORTER.export_tokenizer(self.source, self.output)
                self.assert_no_bpe()
                if COMPARE:
                    result = subprocess.run([COMPARE, str(self.source), str(self.output)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Invalid GGUF special token ID", result.stderr)

    def test_template_export_preserves_exact_utf8(self):
        for index, template in enumerate(("", "用户\r\n{{ messages }}\t\n😀")):
            with self.subTest(template=template):
                self.output = self.root / f"template-{index}"
                self.data["tokenizer.chat_template"] = (8, template)
                self.export()
                self.assertEqual((self.output / "chat_template.txt").read_bytes(), template.encode("utf-8"))

    def test_absent_template_does_not_create_file(self):
        del self.data["tokenizer.chat_template"]
        self.export()
        self.assertFalse((self.output / "chat_template.txt").exists())

    def test_invalid_template_type(self):
        self.data["tokenizer.chat_template"] = (5, 123)
        self.write()
        with self.assertRaisesRegex(ValueError, "chat_template must be a string"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()

    def test_shared_bos_eos_and_missing_unk(self):
        self.data[EXPORTER.PREFIX + "eos_token_id"] = (4, 0)
        del self.data[EXPORTER.PREFIX + "unknown_token_id"]
        self.export()
        self.compare()

    def test_empty_merges(self):
        self.data[EXPORTER.PREFIX + "merges"] = (9, (8, []))
        self.export()
        self.compare()

    def test_missing_unk_with_unencodable_byte(self):
        del self.data[EXPORTER.PREFIX + "unknown_token_id"]
        # Replace the FF byte symbol with an ordinary, unmerged BPE token.
        self.data[EXPORTER.PREFIX + "tokens"][1][1][258] = "zz"
        self.export()
        self.compare("--missing-byte")

    def test_control_character_token_is_escaped_in_tsv(self):
        self.data[EXPORTER.PREFIX + "tokens"][1][1][-2] = "<ctrl\n\t\0>"
        self.export()
        rows = (self.output / "tokens.tsv").read_text().splitlines()
        self.assertEqual(len(rows), len(self.data[EXPORTER.PREFIX + "tokens"][1][1]) + 1)
        self.assertIn(r'"<ctrl\n\t\u0000>"', rows[-2])

    def test_output_not_overwritten(self):
        self.export()
        before = (self.output / "vocab.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assertEqual((self.output / "vocab.json").read_bytes(), before)

    def test_existing_template_not_overwritten(self):
        self.write()
        self.output.mkdir()
        template = self.output / "chat_template.txt"
        template.write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "already exist"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assertEqual(template.read_bytes(), b"keep me")
        self.assertFalse((self.output / "vocab.json").exists())

    def test_unsupported_model_keeps_reports(self):
        self.data[EXPORTER.PREFIX + "model"] = (8, "llama")
        self.write()
        with self.assertRaisesRegex(ValueError, "not supported"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()
        self.assertEqual((self.output / "chat_template.txt").read_bytes(),
                         self.data["tokenizer.chat_template"][1].encode("utf-8"))

    def test_invalid_vocabulary_keeps_reports(self):
        self.data[EXPORTER.PREFIX + "tokens"][1][1][-1] = "a"
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()

    def test_invalid_special_id_keeps_reports(self):
        self.data[EXPORTER.PREFIX + "bos_token_id"] = (5, -1)
        self.write()
        with self.assertRaisesRegex(ValueError, "Invalid vocabulary ID"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()

    def test_invalid_merge_keeps_reports(self):
        self.data[EXPORTER.PREFIX + "merges"] = (9, (8, ["a missing"]))
        self.write()
        with self.assertRaisesRegex(ValueError, "missing or special"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()

    def test_wrong_array_type(self):
        self.data[EXPORTER.PREFIX + "tokens"] = (9, (5, [1, 2]))
        self.write()
        with self.assertRaisesRegex(ValueError, "Invalid metadata type"):
            EXPORTER.export_tokenizer(self.source, self.output)
        self.assert_no_bpe()

    def test_optional_fields_with_invalid_types_are_not_silently_ignored(self):
        for index, (name, payload) in enumerate((
                ("bos_token_id", (7, True)), ("model", (5, 123)),
                ("token_type", (9, (8, ["wrong"]))),
                ("token_type", (9, (5, []))))):
            with self.subTest(name=name, payload=payload):
                self.data = fields()
                self.data[EXPORTER.PREFIX + name] = payload
                self.output = self.root / f"invalid-{index}"
                self.write()
                with self.assertRaises(ValueError):
                    EXPORTER.export_tokenizer(self.source, self.output)
                self.assert_no_bpe()
                if COMPARE:
                    result = subprocess.run([COMPARE, str(self.source), str(self.output)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("GGUF", result.stderr)

    def test_bad_magic_version_and_truncated_data(self):
        self.write()
        original = self.source.read_bytes()
        for broken in (b"", b"NOPE" + original[4:], original[:4] + struct.pack("<I", 1) + original[8:],
                       original[:20], original[:-1]):
            with self.subTest(length=len(broken)):
                self.source.write_bytes(broken)
                with self.assertRaises(ValueError):
                    EXPORTER.read_metadata(self.source)

    def test_unknown_metadata_type(self):
        self.source.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("tokenizer.x") + struct.pack("<I", 99))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            EXPORTER.read_metadata(self.source)

    def test_cli_success_and_error_status(self):
        self.write()
        command = [sys.executable, str(ROOT / "tools/export_gguf_tokenizer.py"), str(self.source),
                   "--output-dir", str(self.output)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_cli_out_dir_alias_and_default(self):
        self.write()
        command = [sys.executable, str(ROOT / "tools/export_gguf_tokenizer.py"), str(self.source)]
        alias = subprocess.run(command + ["--out-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(alias.returncode, 0, alias.stderr)
        default = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        for name in EXPORTER.OUTPUTS:
            self.assertEqual((self.output / name).read_bytes(),
                             (self.root / "models/chat" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
