from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.hidden_state_patching import run_hidden_state_patching


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict v10 hidden-state patching experiments")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--examples-per-pair", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_hidden_state_patching(
        args.run_dir,
        device=args.device,
        examples_per_pair=args.examples_per_pair,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
