#!/usr/bin/env python3
"""Inspect GGUF tokenizer metadata and export files for this project's BpeTokenizer."""

import argparse
import json
import struct
import sys
from pathlib import Path

# GGUF metadata value types, in wire order.
TYPES = ("UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
         "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64")
FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
OUTPUTS = ("tokenizer_metadata.json", "tokens.tsv", "vocab.json",
           "merges.txt", "special_tokens.json", "chat_template.txt")
PREFIX = "tokenizer.ggml."


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, size):
        if size > self.size - self.stream.tell():
            raise ValueError(f"Truncated GGUF at offset {self.stream.tell()}")
        return self.stream.read(size)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        length = self.number("Q")
        if keep:
            return self.read(length).decode("utf-8")
        self.skip(length)

    def skip(self, size):
        if size > self.size - self.stream.tell():
            raise ValueError("Truncated GGUF metadata")
        self.stream.seek(size, 1)

    def value(self, kind, keep=True, depth=0):
        if kind not in range(len(TYPES)) or depth > 8:
            raise ValueError(f"Unsupported metadata type/nesting: {kind}")
        if kind == 8:
            return self.string(keep), None
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if element not in range(len(TYPES)) or count > self.size - self.stream.tell():
                raise ValueError("Invalid metadata array type or length")
            values = [] if keep else None
            for _ in range(count):
                value, _ = self.value(element, keep, depth + 1)
                if keep:
                    values.append(value)
            return values, TYPES[element]
        if keep:
            return self.number(FORMATS[kind]), None
        self.skip(struct.calcsize(FORMATS[kind]))
        return None, None


def read_metadata(path):
    with path.open("rb") as stream:
        reader = Reader(stream)
        if reader.read(4) != b"GGUF":
            raise ValueError("Expected little-endian GGUF magic")
        version = reader.number("I")
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        reader.number("Q")  # tensor count; tensor descriptors and weights are not read
        count = reader.number("Q")
        if count > reader.size // 12:
            raise ValueError("Invalid metadata count")
        result, seen = {}, set()
        for _ in range(count):
            key, kind = reader.string(), reader.number("I")
            if key in seen:
                raise ValueError(f"Duplicate metadata key: {key}")
            seen.add(key)
            keep = key.startswith("tokenizer.")
            value, element = reader.value(kind, keep)
            if keep:
                result[key] = {"type": TYPES[kind], "value": value}
                if element is not None:
                    result[key]["element_type"] = element
        return result


def byte_symbols():
    kept = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    return {chr(b) for b in kept} | {chr(256 + n) for n in range(256 - len(kept))}


def bpe_files(metadata):
    def field(name, kind, element=None, default=None):
        entry = metadata.get(PREFIX + name)
        if entry is None:
            return default
        if entry["type"] != kind or (element and entry.get("element_type") != element):
            raise ValueError(f"Invalid metadata type: {PREFIX + name}")
        return entry["value"]

    model = field("model", "STRING")
    if model not in (None, "gpt2"):
        raise ValueError(f"Tokenizer model {model!r} is not supported by the byte BPE loader")
    tokens = field("tokens", "ARRAY", "STRING")
    merges = field("merges", "ARRAY", "STRING")
    if not tokens or len(tokens) >= 2**31 or merges is None:
        raise ValueError("A nonempty token array and a merges array are required")
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Empty or duplicate vocabulary token")
    vocab = {token: index for index, token in enumerate(tokens)}
    types = field("token_type", "ARRAY", "INT32")
    if types is not None and len(types) != len(tokens):
        raise ValueError("token_type length must equal vocabulary size")
    specials = {i for i, kind in enumerate(types or []) if kind == 3}
    config = dict.fromkeys(("bos_id", "eos_id", "unk_id", "pad_id"), -1)
    for source, target in (("bos_token_id", "bos_id"), ("eos_token_id", "eos_id"),
                           ("unknown_token_id", "unk_id"), ("padding_token_id", "pad_id")):
        entry = metadata.get(PREFIX + source)
        if entry is None:
            continue
        value = entry["value"]
        if entry["type"] not in ("UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "UINT64", "INT64") or not 0 <= value < len(tokens):
            raise ValueError(f"Invalid vocabulary ID: {source}")
        config[target] = value
        specials.add(value)
    config["special_ids"] = sorted(specials)
    symbols = byte_symbols()
    for index, token in enumerate(tokens):
        if index not in specials and any(char not in symbols for char in token):
            raise ValueError(f"Token {index} contains a non-BPE byte symbol")
    seen = set()
    for rank, merge in enumerate(merges):
        pair = merge.split(" ")
        if len(pair) != 2 or not all(pair) or any(c.isspace() for part in pair for c in part):
            raise ValueError(f"Invalid merge at rank {rank}")
        if merge in seen:
            raise ValueError(f"Duplicate merge at rank {rank}")
        seen.add(merge)
        for token in (pair[0], pair[1], "".join(pair)):
            if token not in vocab or vocab[token] in specials:
                raise ValueError(f"Merge {rank} references a missing or special token")
    return {"vocab.json": json_text(vocab),
            "merges.txt": "#version: 0.2\n" + "".join(m + "\n" for m in merges),
            "special_tokens.json": json_text(config)}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def export_tokenizer(source, output):
    if any((output / name).exists() for name in OUTPUTS):
        raise ValueError("Output files already exist; choose another output directory")
    metadata = read_metadata(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / OUTPUTS[0]).write_text(json_text(metadata), encoding="utf-8")
    tokens = metadata.get(PREFIX + "tokens", {}).get("value", [])
    types = metadata.get(PREFIX + "token_type", {}).get("value", [])
    scores = metadata.get(PREFIX + "scores", {}).get("value", [])
    with (output / "tokens.tsv").open("w", encoding="utf-8") as stream:
        stream.write("id\ttoken_type\tscore\ttoken_json\n")
        if isinstance(tokens, list):
            for index, token in enumerate(tokens):
                kind = types[index] if isinstance(types, list) and index < len(types) else ""
                score = scores[index] if isinstance(scores, list) and index < len(scores) else ""
                stream.write(f"{index}\t{kind}\t{score}\t{json.dumps(token, ensure_ascii=False)}\n")
    template = metadata.get("tokenizer.chat_template")
    if template is not None:
        if template["type"] != "STRING":
            raise ValueError("tokenizer.chat_template must be a string")
        with (output / "chat_template.txt").open("xb") as stream:
            stream.write(template["value"].encode("utf-8"))
    # Inspection files remain available even when BPE conversion is unsupported.
    files = bpe_files(metadata)
    for name, text in files.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=Path("models/chat"),
                        help="Output directory (default: models/chat)")
    args = parser.parse_args()
    try:
        export_tokenizer(args.gguf, args.output_dir)
    except (OSError, ValueError, struct.error) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported tokenizer inspection reports and BPE files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
