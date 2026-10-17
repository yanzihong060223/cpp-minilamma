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


def export_model(output_dir: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)
    config = dict(CONFIG)
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
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1")
    export_model(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
