#!/usr/bin/env python3
"""Strictly compare complete C++ and NumPy F32 golden-logits runs."""

import argparse
from itertools import islice
import json
from pathlib import Path
import re
import sys

import numpy as np


ABS_TOLERANCE = 1e-4
REL_TOLERANCE = 1e-4
REL_MINIMUM = 1e-3
COMMON_FIELDS = (
    "format_version", "vocab_size", "bos_id", "prompt_tokens",
    "requested_generation", "max_seq_len", "generated_count", "step_count",
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata field: {key}")
        result[key] = value
    return result


def _integer(metadata, name, minimum=0):
    value = metadata.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"FILE SET: metadata {name} must be an integer >= {minimum}")
    return value


def _metadata(path):
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(f"FILE SET: cannot read valid metadata {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"FILE SET: metadata must be an object: {path}")
    missing = set(COMMON_FIELDS) - metadata.keys()
    if missing:
        raise ValueError(f"FILE SET: missing metadata fields in {path}: {sorted(missing)}")

    if _integer(metadata, "format_version", 1) != 1:
        raise ValueError(f"FILE SET: unsupported metadata format version in {path}")
    vocab_size = _integer(metadata, "vocab_size", 1)
    bos_id = _integer(metadata, "bos_id")
    if bos_id >= vocab_size:
        raise ValueError(f"FILE SET: metadata BOS is outside the vocabulary: {path}")
    prompt = metadata["prompt_tokens"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"FILE SET: metadata prompt_tokens must include BOS: {path}")
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in prompt):
        raise ValueError(f"FILE SET: metadata prompt token is outside the vocabulary: {path}")
    if prompt[0] != bos_id or any(token > 127 for token in prompt[1:]):
        raise ValueError(f"FILE SET: metadata prompt must be BOS followed by ASCII IDs: {path}")

    requested = _integer(metadata, "requested_generation")
    capacity = _integer(metadata, "max_seq_len", 1)
    if len(prompt) > capacity:
        raise ValueError(f"FILE SET: metadata prompt exceeds context capacity: {path}")
    generated = min(requested, capacity - len(prompt))
    steps = len(prompt) + max(generated - 1, 0)
    if _integer(metadata, "generated_count") != generated:
        raise ValueError(f"FILE SET: metadata generated_count must be {generated}: {path}")
    if _integer(metadata, "step_count", 1) != steps:
        raise ValueError(f"FILE SET: metadata step_count must be {steps}: {path}")
    return metadata


def _step_files(directory, prefix, expected_steps):
    pattern = re.compile(re.escape(prefix) + r"(0|[1-9][0-9]*)\.bin")
    files = {}
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise ValueError(f"FILE SET: cannot list {directory}: {error}") from error
    for path in entries:
        if not path.name.startswith(prefix):
            continue
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            raise ValueError(f"FILE SET: invalid step filename or file type: {path}")
        files[int(match.group(1))] = path

    extra = sorted(index for index in files if index >= expected_steps)
    if len(files) != expected_steps or extra:
        # Do not construct an enormous set from untrusted metadata.
        missing = list(islice((index for index in range(expected_steps)
                               if index not in files), 8))
        raise ValueError(
            f"FILE SET: {directory}/{prefix} expected {expected_steps} steps, "
            f"found {len(files)}; missing={missing}, extra={extra[:8]}")
    return files


def _tokens(path, metadata):
    try:
        words = path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"SAMPLING: cannot read token file {path}: {error}") from error
    if any(re.fullmatch(r"[0-9]+", word) is None for word in words):
        raise ValueError(f"SAMPLING: token file requires nonnegative decimal IDs: {path}")
    try:
        tokens = [int(word) for word in words]
    except ValueError as error:
        raise ValueError(f"SAMPLING: invalid token integer in {path}: {error}") from error
    if len(tokens) != metadata["generated_count"]:
        raise ValueError(
            f"SAMPLING: {path} expected {metadata['generated_count']} tokens, "
            f"found {len(tokens)}")
    if any(token >= metadata["vocab_size"] for token in tokens):
        raise ValueError(f"SAMPLING: generated token is outside the vocabulary: {path}")
    return tokens


def _logits(path, vocab_size):
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"FILE SET: cannot read logits {path}: {error}") from error
    if len(payload) != 4 * vocab_size:
        raise ValueError(
            f"FILE SET: {path} expected {4 * vocab_size} bytes, found {len(payload)}")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"LOGITS: NaN/Inf in {path}")
    return values, payload


def _location(metadata, step, generated):
    prompt = metadata["prompt_tokens"]
    phase = "prefill" if step < len(prompt) else "decode"
    token = prompt[step] if step < len(prompt) else generated[step - len(prompt)]
    return f"step={step} phase={phase} pos={step} input_token={token}"


def compare_directories(cpp_dir, py_dir, vocab_size=None, expected_steps=None,
                        *, reference_kind="python", exact=False, verbose=True):
    """Validate run completeness, numerical logits, and actual greedy sampling.

    ``reference_kind='cpp'`` reads a second C++ dump for thread comparisons.
    Runtime metadata such as mode and thread count can differ; the common run
    semantics must match. ``exact=True`` additionally requires identical bytes.
    All failures raise ValueError. No files are written by this function.
    """
    if reference_kind not in ("python", "cpp"):
        raise ValueError("FILE SET: reference_kind must be 'python' or 'cpp'")
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= 0):
        raise ValueError("FILE SET: vocab_size must be a positive integer")
    if expected_steps is not None and (type(expected_steps) is not int or expected_steps <= 0):
        raise ValueError("FILE SET: expected_steps must be a positive integer")

    cpp_dir, py_dir = Path(cpp_dir), Path(py_dir)
    ref_prefix = "python_logits_step" if reference_kind == "python" else "logits_step"
    ref_tokens = "python_generation_tokens.txt" if reference_kind == "python" else "generation_tokens.txt"
    ref_metadata = "python_run.json" if reference_kind == "python" else "cpp_run.json"
    cpp_meta = _metadata(cpp_dir / "cpp_run.json")
    py_meta = _metadata(py_dir / ref_metadata)
    for field in COMMON_FIELDS:
        if cpp_meta[field] != py_meta[field]:
            raise ValueError(
                f"FILE SET: metadata mismatch for {field}: "
                f"cpp={cpp_meta[field]!r}, reference={py_meta[field]!r}")
    if vocab_size is not None and cpp_meta["vocab_size"] != vocab_size:
        raise ValueError(
            f"FILE SET: expected vocab_size={vocab_size}, "
            f"metadata has {cpp_meta['vocab_size']}")
    if expected_steps is not None and cpp_meta["step_count"] != expected_steps:
        raise ValueError(
            f"FILE SET: expected_steps={expected_steps}, "
            f"metadata has {cpp_meta['step_count']}")
    vocab_size = cpp_meta["vocab_size"]
    steps = cpp_meta["step_count"]
    cpp_files = _step_files(cpp_dir, "logits_step", steps)
    py_files = _step_files(py_dir, ref_prefix, steps)
    cpp_generated = _tokens(cpp_dir / "generation_tokens.txt", cpp_meta)
    py_generated = _tokens(py_dir / ref_tokens, py_meta)
    max_abs_error = 0.0
    max_rel_error = 0.0
    cpp_top1, py_top1 = [], []
    for step in range(steps):
        cpp, cpp_bytes = _logits(cpp_files[step], vocab_size)
        reference, reference_bytes = _logits(py_files[step], vocab_size)
        cpp64 = cpp.astype(np.float64)
        py64 = reference.astype(np.float64)
        differences = np.abs(cpp64 - py64)
        abs_token = int(np.argmax(differences))
        abs_error = float(differences[abs_token])
        mask = np.abs(py64) >= REL_MINIMUM
        relative = np.zeros(vocab_size, dtype=np.float64)
        relative[mask] = differences[mask] / np.abs(py64[mask])
        rel_token = int(np.argmax(relative))
        rel_error = float(relative[rel_token])
        cpp_id, py_id = int(np.argmax(cpp)), int(np.argmax(reference))
        cpp_top1.append(cpp_id)
        py_top1.append(py_id)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        location = _location(cpp_meta, step, cpp_generated)
        details = (
            f"{location} max_abs_error={abs_error:.9g} token={abs_token} "
            f"cpp={cpp64[abs_token]:.9g} py={py64[abs_token]:.9g}; "
            f"max_rel_error={rel_error:.9g} token={rel_token} "
            f"cpp={cpp64[rel_token]:.9g} py={py64[rel_token]:.9g}; "
            f"cpp_top1={cpp_id} reference_top1={py_id}")
        if abs_error > ABS_TOLERANCE or rel_error > REL_TOLERANCE:
            raise ValueError("LOGITS: tolerance exceeded: " + details)
        if cpp_id != py_id:
            raise ValueError("LOGITS: top-1 mismatch: " + details)
        if exact and cpp_bytes != reference_bytes:
            raise ValueError("LOGITS: exact thread comparison differs: " + details)
    if verbose:
        print(f"FILE SET: PASSED ({steps} complete steps, vocab_size={vocab_size})")
        print(f"LOGITS: PASSED (max_abs_error={max_abs_error:.9g}, "
              f"max_rel_error={max_rel_error:.9g}, every top-1 matches)")

    first_sample = len(cpp_meta["prompt_tokens"]) - 1
    for name, generated, top1 in (("cpp", cpp_generated, cpp_top1),
                                   ("reference", py_generated, py_top1)):
        for index, token in enumerate(generated):
            step = first_sample + index
            if token != top1[step]:
                raise ValueError(
                    f"SAMPLING: {name} generation index={index}, step={step}: "
                    f"file token={token}, greedy argmax={top1[step]}")
    if cpp_generated != py_generated:
        index = next(index for index, pair in enumerate(zip(cpp_generated, py_generated))
                     if pair[0] != pair[1])
        raise ValueError(
            f"SAMPLING: first generation mismatch at index={index}: "
            f"cpp={cpp_generated[index]}, reference={py_generated[index]}")
    if verbose:
        print(f"SAMPLING: PASSED ({len(cpp_generated)} generated tokens)")
        print("RESULT: ALL PASSED")
    return {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "step_count": steps,
        "generated_count": len(cpp_generated),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-dir", required=True, type=Path)
    parser.add_argument("--py-dir", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()
    try:
        compare_directories(args.cpp_dir, args.py_dir, args.vocab_size,
                            args.expected_steps)
    except (OSError, ValueError) as error:
        print(f"RESULT: FAILED\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
