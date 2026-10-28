"""Independent NumPy F32 reference for the project's JSON/BIN tiny model.

Weights are row-major [out, in]. A ReferenceModel owns one persistent K/V
cache; callers supply the token's absolute position to forward_token.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


_DIMENSIONS = (
    "vocab_size", "dim", "hidden_dim", "layers", "n_heads",
    "n_kv_heads", "head_dim", "max_seq_len",
)


@dataclass
class ReferenceModel:
    config: dict
    weights: dict
    bos_id: int
    key_cache: np.ndarray
    value_cache: np.ndarray

    def reset_cache(self):
        self.key_cache.fill(0)
        self.value_cache.fill(0)


def _integer(value, name, minimum=0, maximum=2147483647):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value, name):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _expected_shapes(config):
    dim = config["dim"]
    hidden = config["hidden_dim"]
    q_dim = config["n_heads"] * config["head_dim"]
    kv_dim = config["n_kv_heads"] * config["head_dim"]
    shapes = {
        "token_embedding": (config["vocab_size"], dim),
        "final_norm": (dim,),
        "lm_head": (config["vocab_size"], dim),
    }
    for layer in range(config["layers"]):
        prefix = f"layers.{layer}."
        shapes.update({
            prefix + "attention_norm": (dim,),
            prefix + "wq": (q_dim, dim),
            prefix + "wk": (kv_dim, dim),
            prefix + "wv": (kv_dim, dim),
            prefix + "wo": (dim, q_dim),
            prefix + "ffn_norm": (dim,),
            prefix + "w_gate": (hidden, dim),
            prefix + "w_up": (hidden, dim),
            prefix + "w_down": (dim, hidden),
        })
    return shapes


def load_model(config_path, weights_path):
    """Read configuration and F32 tensors without using any C++ output."""
    manifest = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Model JSON must be an object")
    fields = manifest.get("config", manifest)
    if not isinstance(fields, dict):
        raise ValueError("Model config must be an object")
    config = {name: _integer(fields[name], name, 1) for name in _DIMENSIONS}
    for name in ("rope_theta", "rms_norm_eps"):
        value = fields[name]
        if (type(value) not in (int, float) or not math.isfinite(value) or
                value <= 0 or value > np.finfo(np.float32).max):
            raise ValueError(f"{name} must be finite, positive and representable as F32")
        config[name] = np.float32(value)
        if config[name] <= 0:
            raise ValueError(f"{name} underflows to zero in F32")
    if (config["head_dim"] % 2 or
            config["dim"] != config["n_heads"] * config["head_dim"] or
            config["n_heads"] % config["n_kv_heads"]):
        raise ValueError("Invalid attention dimensions")

    tokenizer = manifest.get("tokenizer", {})
    if not isinstance(tokenizer, dict):
        raise ValueError("Tokenizer metadata must be an object")
    tokenizer_type = tokenizer.get("type", "ASCII")
    if tokenizer_type not in ("ASCII", "JsonVocab"):
        raise ValueError("Golden JSON models support ASCII or JsonVocab metadata")
    default_bos = 1 if tokenizer_type == "JsonVocab" else 128
    bos_id = _integer(tokenizer.get("bos_id", default_bos), "bos_id")
    if bos_id >= config["vocab_size"]:
        raise ValueError("BOS token is outside the vocabulary")

    entries = manifest.get("tensors")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Model must contain a nonempty tensor manifest")
    # Check the manifest count before constructing the per-layer shape map.
    if len(entries) < 3 + 9 * config["layers"]:
        raise ValueError("Missing required tensors for configured layers")
    expected = _expected_shapes(config)
    payload = Path(weights_path).read_bytes()
    weights = {}
    ranges = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Tensor metadata must be an object")
        name = entry.get("name")
        shape = entry.get("shape")
        if not isinstance(name, str) or not name or name in weights:
            raise ValueError(f"Invalid or duplicate tensor name: {name!r}")
        if entry.get("dtype") != "float32":
            raise ValueError(f"{name}: golden supports only float32 tensors")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"{name}: shape must be a nonempty array")
        shape = tuple(_integer(size, f"{name} dimension", 1) for size in shape)
        offset = _integer(entry.get("offset"), f"{name} offset", maximum=len(payload))
        byte_size = _integer(entry.get("byte_size"), f"{name} byte_size",
                             maximum=len(payload))
        count = math.prod(shape)
        if byte_size != count * 4 or offset + byte_size > len(payload):
            raise ValueError(f"{name}: invalid tensor byte range or size")
        if name in expected and shape != expected[name]:
            raise ValueError(f"{name}: expected shape {expected[name]}, got {shape}")
        values = np.frombuffer(payload, dtype="<f4", count=count, offset=offset)
        weights[name] = _finite(values.astype(np.float32).reshape(shape), name)
        ranges.append((offset, offset + byte_size))
    missing = set(expected) - weights.keys()
    if missing:
        raise ValueError(f"Missing required tensors: {sorted(missing)}")
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping tensor byte ranges")
    if ranges[-1][1] != len(payload):
        raise ValueError("Weights file contains trailing bytes")

    cache_shape = (config["layers"], config["max_seq_len"],
                   config["n_kv_heads"], config["head_dim"])
    return ReferenceModel(config, weights, bos_id,
                          np.zeros(cache_shape, dtype=np.float32),
                          np.zeros(cache_shape, dtype=np.float32))


def encode_prompt(model, prompt):
    """Tiny golden tokenization: BOS followed by literal ASCII IDs, no EOS."""
    if not isinstance(prompt, str) or not prompt.isascii():
        raise ValueError("Golden prompt must contain only ASCII characters")
    tokens = [model.bos_id] + [ord(character) for character in prompt]
    if any(token >= model.config["vocab_size"] for token in tokens):
        raise ValueError("Prompt token is outside the vocabulary")
    if len(tokens) > model.config["max_seq_len"]:
        raise ValueError("Prompt exceeds model context length")
    return tokens


def rms_norm(x, weight, eps):
    # The CPU implementation intentionally uses double for this reduction.
    promoted = x.astype(np.float64)
    denominator = np.sqrt(np.mean(promoted * promoted) + float(eps))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid RMSNorm denominator")
    return (promoted * weight.astype(np.float64) / denominator).astype(np.float32)


def rope(x, pos, theta):
    """Normal RoPE pairs adjacent coordinates independently in every head."""
    result = x.copy()
    head_dim = x.shape[1]
    for pair in range(head_dim // 2):
        angle = pos * math.pow(float(theta), -2.0 * pair / head_dim)
        cosine, sine = math.cos(angle), math.sin(angle)
        first = x[:, 2 * pair].astype(np.float64)
        second = x[:, 2 * pair + 1].astype(np.float64)
        result[:, 2 * pair] = first * cosine - second * sine
        result[:, 2 * pair + 1] = first * sine + second * cosine
    return result


def attention(model, layer, pos, q, k, v):
    model.key_cache[layer, pos] = k
    model.value_cache[layer, pos] = v
    config = model.config
    group_size = config["n_heads"] // config["n_kv_heads"]
    scale = np.float32(1) / np.sqrt(np.float32(config["head_dim"]))
    out = np.empty_like(q)
    for head in range(config["n_heads"]):
        kv_head = head // group_size
        keys = model.key_cache[layer, :pos + 1, kv_head]
        values = model.value_cache[layer, :pos + 1, kv_head]
        scores = (keys @ q[head]) * scale
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= np.sum(probabilities, dtype=np.float32)
        out[head] = probabilities @ values
    return out.reshape(-1)


def forward_token(model, pos, token_id):
    """Compute one token and write its K/V at the explicit absolute position."""
    config, weights = model.config, model.weights
    pos = _integer(pos, "position")
    token_id = _integer(token_id, "token_id")
    if pos >= config["max_seq_len"] or token_id >= config["vocab_size"]:
        raise ValueError("Token or position is outside the configured range")

    x = weights["token_embedding"][token_id].copy()
    for layer in range(config["layers"]):
        prefix = f"layers.{layer}."
        normed = rms_norm(x, weights[prefix + "attention_norm"],
                          config["rms_norm_eps"])
        q = (weights[prefix + "wq"] @ normed).reshape(
            config["n_heads"], config["head_dim"])
        k = (weights[prefix + "wk"] @ normed).reshape(
            config["n_kv_heads"], config["head_dim"])
        v = (weights[prefix + "wv"] @ normed).reshape(
            config["n_kv_heads"], config["head_dim"])
        q = rope(q, pos, config["rope_theta"])
        k = rope(k, pos, config["rope_theta"])
        attended = attention(model, layer, pos, q, k, v)
        x = x + weights[prefix + "wo"] @ attended

        normed = rms_norm(x, weights[prefix + "ffn_norm"],
                          config["rms_norm_eps"])
        gate = weights[prefix + "w_gate"] @ normed
        up = weights[prefix + "w_up"] @ normed
        with np.errstate(over="ignore", under="ignore"):
            activated = gate * (np.float32(1) /
                                (np.float32(1) + np.exp(-gate)))
        x = _finite(x + weights[prefix + "w_down"] @ (activated * up),
                    f"layer {layer} hidden state")

    normed = rms_norm(x, weights["final_norm"], config["rms_norm_eps"])
    return _finite((weights["lm_head"] @ normed).astype(np.float32), "logits")


def reference_run(model, prompt="hello", n_predict=16):
    """Run independent greedy generation, preserving the existing decode count."""
    requested = _integer(n_predict, "n_predict")
    prompt_tokens = encode_prompt(model, prompt)
    generated_count = min(requested, model.config["max_seq_len"] - len(prompt_tokens))
    model.reset_cache()
    steps = [forward_token(model, pos, token)
             for pos, token in enumerate(prompt_tokens)]
    generated = []
    logits = steps[-1]
    for index in range(generated_count):
        # NumPy argmax returns the first maximum, matching greedy tie-breaking.
        token = int(np.argmax(logits))
        generated.append(token)
        if index + 1 < generated_count:
            logits = forward_token(model, len(prompt_tokens) + index, token)
            steps.append(logits)
    expected_steps = len(prompt_tokens) + max(generated_count - 1, 0)
    if len(steps) != expected_steps:
        raise ValueError("Internal reference step count mismatch")
    metadata = {
        "format_version": 1,
        "vocab_size": model.config["vocab_size"],
        "bos_id": model.bos_id,
        "prompt_tokens": prompt_tokens,
        "requested_generation": requested,
        "max_seq_len": model.config["max_seq_len"],
        "generated_count": generated_count,
        "step_count": expected_steps,
    }
    return steps, generated, metadata
