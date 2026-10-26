#!/usr/bin/env python3
"""Generate independent NumPy logits and greedy tokens for F32 alignment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from reference_forward import load_model, reference_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("-n", "--n-predict", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output_dir
        if output.exists() and (list(output.glob("python_logits_step*.bin")) or
                (output / "python_generation_tokens.txt").exists() or
                (output / "python_run.json").exists()):
            raise ValueError("Output directory already contains Python golden results")
        model = load_model(args.config, args.weights)
        steps, generated, metadata = reference_run(model, args.prompt, args.n_predict)
        output.mkdir(parents=True, exist_ok=True)
        for step, logits in enumerate(steps):
            with (output / f"python_logits_step{step}.bin").open("xb") as stream:
                stream.write(np.asarray(logits, dtype="<f4").tobytes(order="C"))
        with (output / "python_generation_tokens.txt").open("x", encoding="utf-8") as stream:
            stream.write(" ".join(map(str, generated)) + ("\n" if generated else ""))
        # A complete metadata file is the final marker of a successful export.
        with (output / "python_run.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"Wrote {len(steps)} Python logits steps and {len(generated)} generated tokens to {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"make_golden: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
