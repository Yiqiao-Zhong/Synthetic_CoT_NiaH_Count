from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_counting_v10.successor_patching import run_successor_patching


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v10 M_k successor/close head patching")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--examples-per-k", type=int, default=2)
    parser.add_argument("--random-replicates", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_successor_patching(
        args.run_dir,
        examples_per_k=args.examples_per_k,
        random_replicates=args.random_replicates,
        device=args.device,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()

