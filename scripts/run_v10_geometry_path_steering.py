from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.geometry_path_steering import run_geometry_path_steering


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run v10 count-centroid transplant, chord, and curved-path steering."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples-per-count", type=int, default=4)
    parser.add_argument("--alphas", type=_floats, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument(
        "--nonadjacent-offsets",
        type=_ints,
        default=(-10, -5, -3, -2, 2, 3, 5, 10),
    )
    parser.add_argument("--patch-batch-size", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_geometry_path_steering(
        args.run_dir,
        examples_per_count=args.examples_per_count,
        alphas=args.alphas,
        nonadjacent_offsets=args.nonadjacent_offsets,
        patch_batch_size=args.patch_batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows", flush=True)
    print(
        f"GEOMETRY_PATH_STEERING_DIR={args.run_dir / 'analysis' / 'geometry_path_steering'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
