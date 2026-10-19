"""Exercise the actual chat executable with tiny deterministic GGUF models."""

from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

RUNNER = str(Path(sys.argv.pop(1)).resolve())
GENERATE = str(Path(sys.argv.pop(1)).resolve())
TEMPLATE = ("{% for message in messages %}"
            "{{ message.role + ':' + message.content + ';' }}"
            "{% endfor %}{{ 'assistant:' }}")


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def model_file(path, context=160, template=TEMPLATE, missing_byte=None,
               nan_logits=False, q8=False, eos=False):
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    symbols = {b: chr(b) for b in kept}
    symbols.update({b: chr(256 + i) for i, b in enumerate(b for b in range(256) if b not in kept)})
    tokens = [symbols[b] for b in range(256) if b != missing_byte] + ["<eos>"]
    dim = 32 if q8 else 4
    metadata = []

    def field(key, kind, data):
        metadata.append(string(key) + struct.pack("<I", kind) + data)

    field("general.architecture", 8, string("qwen2"))
    for key, value in {
        "block_count": 1, "embedding_length": dim, "feed_forward_length": dim * 2,
        "attention.head_count": 2, "attention.head_count_kv": 1,
        "context_length": context,
    }.items():
        field("qwen2." + key, 4, struct.pack("<I", value))
    field("qwen2.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-5))
    field("tokenizer.ggml.model", 8, string("gpt2"))
    field("tokenizer.ggml.tokens", 9,
          struct.pack("<IQ", 8, len(tokens)) + b"".join(string(t) for t in tokens))
    field("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 0))
    field("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", len(tokens) - 1))
    field("tokenizer.chat_template", 8, string(template))

    weights = []

    def weight(name, shape, fill=0.0, values=None):
        count = 1
        for dimension in shape:
            count *= dimension
        values = values if values is not None else [fill] * count
        if q8 and len(shape) == 2:
            # Exact Q8_0 fixture blocks: scale=1, values restricted to 0 or 1.
            payload = b"".join(struct.pack("<e", 1.0) +
                               struct.pack("32b", *(int(v) for v in values[i:i + 32]))
                               for i in range(0, count, 32))
            kind = 8
        else:
            payload = struct.pack("<" + "f" * count, *values)
            kind = 0
        weights.append((name, shape, kind, payload))

    weight("token_embd.weight", (dim, len(tokens)), 1.0)
    weight("output_norm.weight", (dim,), float("nan") if nan_logits else 1.0)
    output = [0.0] * (len(tokens) * dim)
    chosen = len(tokens) - 1 if eos else tokens.index("x")
    output[chosen * dim:(chosen + 1) * dim] = [1.0] * dim
    weight("output.weight", (dim, len(tokens)), values=output)
    weight("blk.0.attn_norm.weight", (dim,), 1.0)
    weight("blk.0.ffn_norm.weight", (dim,), 1.0)
    for name, shape in {
        "attn_q": (dim, dim), "attn_k": (dim, dim // 2),
        "attn_v": (dim, dim // 2), "attn_output": (dim, dim),
        "ffn_gate": (dim, dim * 2), "ffn_up": (dim, dim * 2),
        "ffn_down": (dim * 2, dim),
    }.items():
        weight("blk.0." + name + ".weight", shape)

    descriptors, payloads, offset = [], [], 0
    for name, shape, kind, payload in weights:
        descriptors.append(string(name) + struct.pack("<I", len(shape)) +
                           struct.pack("<" + "Q" * len(shape), *shape) +
                           struct.pack("<IQ", kind, offset))
        padded = payload + bytes((-len(payload)) % 32)
        payloads.append(padded)
        offset += len(padded)
    header = (b"GGUF" + struct.pack("<IQQ", 3, len(weights), len(metadata)) +
              b"".join(metadata) + b"".join(descriptors))
    path.write_bytes(header + bytes((-len(header)) % 32) + b"".join(payloads))


def prompt(messages):
    return "".join(role + ":" + text + ";" for role, text in messages) + "assistant:"


class InteractiveTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="yan-gguf-chat-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tiny.gguf"
        model_file(self.path)

    def run_chat(self, text, limit="2"):
        args = [RUNNER, str(self.path)]
        if limit is not None:
            args.append(limit)
        result = subprocess.run(args, input=text, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def counts(self, output, label):
        return [int(n) for n in re.findall(re.escape(label) + r"\s*:?\s*(\d+)", output)]

    def test_multiple_turns_and_clear_use_full_native_template_history(self):
        result = self.run_chat("hello\nb\n/stats\n/clear\n/stats\nc\n/stats\n/exit\n")
        first = len(prompt([("user", "hello")]))
        second = len(prompt([("user", "hello"), ("assistant", "xx"), ("user", "b")]))
        third = len(prompt([("user", "c")]))
        self.assertEqual(self.counts(result.stderr, "Prompt tokens"), [first, second, third])
        self.assertEqual(self.counts(result.stdout, "messages"), [4, 0, 2])
        self.assertEqual(self.counts(result.stdout, "context tokens"), [second + 2, 0, third + 2])
        self.assertEqual(self.counts(result.stdout, "total prompt tokens"), [first + second, 0, third])
        self.assertEqual(self.counts(result.stdout, "generate tokens"), [4, 0, 2])
        self.assertEqual(result.stdout.count("assistant  > xx"), 3)
        self.assertEqual(result.stderr.count("Loading model:"), 1)
        self.assertNotIn("Preparing F32 weights", result.stderr)
        self.assertIn(f"Prefill {second}/{second}", result.stderr)
        self.assertNotIn("FAIL", result.stderr)

    def test_commands_and_whitespace_do_not_create_messages(self):
        result = self.run_chat("\n \t\n/help\n/params\n/parmas\n/unknown\n/stats\n/exit\n")
        self.assertIn("/params", result.stdout)
        self.assertEqual(result.stdout.count("temperature:"), 2)
        self.assertIn("Unknown command", result.stdout)
        self.assertEqual(self.counts(result.stdout, "messages"), [0])
        self.assertNotIn("Prefill", result.stderr)

    def test_eof_and_last_line_without_newline(self):
        self.assertNotIn("Prefill", self.run_chat("").stderr)
        result = self.run_chat("hello")
        self.assertEqual(result.stdout.count("assistant  > xx"), 1)

    def test_default_limit_and_preserved_message_spaces(self):
        result = self.run_chat(" hello \n/exit\n", limit=None)
        self.assertIn("generated_tokens=16; stop=token limit", result.stderr)
        self.assertEqual(self.counts(result.stderr, "Prompt tokens"),
                         [len(prompt([("user", " hello ")]))])

    def test_overflow_rejects_only_current_turn(self):
        model_file(self.path, context=80)
        result = self.run_chat("hello\n" + "z" * 100 + "\n/stats\nb\n/stats\n/exit\n")
        self.assertIn("exceeds model context length", result.stderr)
        self.assertEqual(self.counts(result.stdout, "messages"), [2, 4])
        expected = len(prompt([("user", "hello"), ("assistant", "xx"), ("user", "b")]))
        self.assertEqual(self.counts(result.stderr, "Prompt tokens")[-1], expected)

    def test_template_encoding_and_prefill_failures_do_not_commit(self):
        cases = [
            ({"template": "{% if True %}"}, "hello", "prompt build"),
            ({"missing_byte": ord("Z")}, "Z", "tokenization"),
            ({"nan_logits": True}, "hello", "prefill"),
        ]
        for options, text, stage in cases:
            with self.subTest(stage=stage):
                model_file(self.path, **options)
                result = self.run_chat(text + "\n/stats\n/exit\n")
                self.assertIn("FAIL [" + stage + "]", result.stderr)
                self.assertEqual(self.counts(result.stdout, "messages"), [0])
                self.assertEqual(self.counts(result.stdout, "context tokens"), [0])
                self.assertEqual(self.counts(result.stdout, "generate tokens"), [0])

    def test_eos_stops_without_outputting_special_token(self):
        model_file(self.path, eos=True)
        result = self.run_chat("hello\n/stats\n/exit\n")
        self.assertIn("generated_tokens=0; stop=EOS", result.stderr)
        self.assertNotIn("<eos>", result.stdout)
        self.assertEqual(self.counts(result.stdout, "messages"), [2])

    def test_q8_forward_and_single_turn_regression(self):
        model_file(self.path, q8=True)
        result = self.run_chat("hello\nb\n/exit\n")
        self.assertEqual(result.stdout.count("assistant  > xx"), 2)
        self.assertNotIn("FAIL", result.stderr)
        single = subprocess.run([GENERATE, str(self.path), "hello", "2"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
        self.assertEqual(single.stdout, "xx\n")
        self.assertIn("PASS:", single.stderr)
        self.assertNotIn("Preparing F32 weights", single.stderr)

    def test_help_and_invalid_startup_arguments(self):
        help_result = subprocess.run([RUNNER, "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("usage: gguf_run", help_result.stdout)
        for limit in ("0", "-1", "x", "1x", "99999999999999999999"):
            result = subprocess.run([RUNNER, str(self.path), limit],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL [arguments]", result.stderr)
            self.assertNotIn("Loading model:", result.stderr)
        missing = subprocess.run([RUNNER, str(self.path) + ".missing"],
                                 capture_output=True, text=True, timeout=10)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("FAIL [model load]", missing.stderr)


if __name__ == "__main__":
    unittest.main()
