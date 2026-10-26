#!/usr/bin/env python3
"""Export deterministic random F32 weights for yan-minillama's JSON/BIN loader."""

import argparse
import json
from pathlib import Path

import numpy as np


CONFIG = {
    "vocab_size": 131,
    "dim": 32,
    "hidden_dim": 86,
    "layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "head_dim": 8,
    "max_seq_len": 128,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
}


def export_model(output_dir: Path, seed: int, json_vocab: bool = False,
                 config_overrides=None) -> None:
    rng = np.random.RandomState(seed)
    config = dict(CONFIG)
    if config_overrides is not None:
        unknown = set(config_overrides) - config.keys()
        if unknown:
            raise ValueError(f"Unknown tiny config fields: {sorted(unknown)}")
        config.update(config_overrides)
    for name in ("vocab_size", "dim", "hidden_dim", "layers", "n_heads",
                 "n_kv_heads", "head_dim", "max_seq_len"):
        if type(config[name]) is not int or not 0 < config[name] <= 2147483647:
            raise ValueError(f"{name} must be a positive integer")
    if config["vocab_size"] != 131:
        raise ValueError("The tiny ASCII/JSON tokenizers require vocab_size=131")
    if (config["dim"] != config["n_heads"] * config["head_dim"] or
            config["n_heads"] % config["n_kv_heads"] or config["head_dim"] % 2):
        raise ValueError("Invalid tiny attention dimensions")
    for name in ("rope_theta", "rms_norm_eps"):
        value = config[name]
        if (type(value) not in (int, float) or not np.isfinite(value) or
                value <= 0 or value > np.finfo(np.float32).max or
                np.float32(value) <= 0):
            raise ValueError(f"{name} must be finite and positive in F32")
    dim = config["dim"]
    hidden = config["hidden_dim"]
    q_dim = config["n_heads"] * config["head_dim"]
    kv_dim = config["n_kv_heads"] * config["head_dim"]
    tensors = []

    def add(name, shape, norm=False):
        if norm:
            data = np.ones(shape, dtype=np.float32)
        else:
            data = rng.randn(*shape).astype(np.float32) * np.float32(0.02)
        tensors.append((name, np.ascontiguousarray(data, dtype="<f4")))

    add("token_embedding", (config["vocab_size"], dim))
    for layer in range(config["layers"]):
        prefix = f"layers.{layer}."
        add(prefix + "attention_norm", (dim,), norm=True)
        add(prefix + "wq", (q_dim, dim))
        add(prefix + "wk", (kv_dim, dim))
        add(prefix + "wv", (kv_dim, dim))
        add(prefix + "wo", (dim, q_dim))
        add(prefix + "ffn_norm", (dim,), norm=True)
        add(prefix + "w_gate", (hidden, dim))
        add(prefix + "w_up", (hidden, dim))
        add(prefix + "w_down", (dim, hidden))
    add("final_norm", (dim,), norm=True)
    add("lm_head", (config["vocab_size"], dim))

    metadata = []
    offset = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "model.bin"
    with bin_path.open("wb") as weights:
        for name, data in tensors:
            payload = data.tobytes(order="C")
            weights.write(payload)
            metadata.append({
                "name": name,
                "shape": list(data.shape),
                "dtype": "float32",
                "offset": offset,
                "byte_size": len(payload),
            })
            offset += len(payload)

    manifest = {
        "config": config,
        "tokenizer": {
            "type": "ASCII",
            "pathname": "",
            "bos_id": 128,
            "eos_id": 129,
            "unk_id": 130,
        },
        "tensors": metadata,
    }
    if json_vocab:
        # Match the existing ASCII tokenizer IDs without changing the weights.
        vocab = [
            {"id": i, "context": chr(i), "special": False}
            for i in range(128)
        ]
        vocab.extend([
            {"id": 128, "context": "<BOS>", "special": True},
            {"id": 129, "context": "<EOS>", "special": True},
            {"id": 130, "context": "<UNK>", "special": True},
        ])
        vocab_path = output_dir / "vocab.json"
        vocab_path.write_text(
            json.dumps(vocab, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["tokenizer"]["type"] = "JsonVocab"
        manifest["tokenizer"]["pathname"] = "vocab.json"
        print(f"Wrote {vocab_path}")

    json_path = output_dir / "model.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {bin_path}")
    print(f"Seed: {seed}; tensors: {len(tensors)}; parameters: {offset // 4}; bytes: {offset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--json-vocab", action="store_true",
                        help="Export vocab.json with ASCII-compatible token IDs")
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1")
    export_model(args.output_dir, args.seed, args.json_vocab)


if __name__ == "__main__":
    main()
