from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.successor_mlp_features import run_successor_mlp_features


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose and causally patch Layer 3-4 GPT-2 MLP intermediate features."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fit-examples-per-k", type=int, default=2)
    parser.add_argument("--eval-examples-per-k", type=int, default=2)
    parser.add_argument(
        "--support-sizes",
        type=_int_tuple,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    )
    parser.add_argument("--random-replicates", type=int, default=4)
    parser.add_argument("--patch-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_successor_mlp_features(
        args.run_dir,
        fit_examples_per_k=args.fit_examples_per_k,
        eval_examples_per_k=args.eval_examples_per_k,
        support_sizes=args.support_sizes,
        random_replicates=args.random_replicates,
        patch_batch_size=args.patch_batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")
    print(f"SUCCESSOR_MLP_FEATURE_DIR={args.run_dir / 'analysis' / 'successor_mlp_features'}")


if __name__ == "__main__":
    main()

