from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.successor_conversion import run_successor_conversion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose v10 successor attention evidence through MLP and residual paths."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples-per-k", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_successor_conversion(
        args.run_dir,
        examples_per_k=args.examples_per_k,
        device=args.device,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")
    print(f"SUCCESSOR_CONVERSION_DIR={args.run_dir / 'analysis' / 'successor_conversion'}")


if __name__ == "__main__":
    main()
