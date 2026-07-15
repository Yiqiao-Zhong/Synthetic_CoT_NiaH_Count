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
    parser.add_argument("--rollout-examples", type=int, default=None)
    parser.add_argument("--rollout-centroid-examples", type=int, default=None)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=32)
    parser.add_argument("--rollout-receiver-count", type=int, default=5)
    parser.add_argument("--rollout-receiver-progress", type=int, default=4)
    parser.add_argument("--rollout-donor-count", type=int, default=10)
    parser.add_argument("--rollout-donor-progress", type=int, default=7)
    parser.add_argument(
        "--rollout-layers",
        default=None,
        help="Comma-separated 1-based layers; default is the final layer only",
    )
    parser.add_argument(
        "--rollout-patch-policies",
        default="one_shot",
        help="Comma-separated one_shot/persistent policies",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rollout_layers = (
        [int(value) for value in args.rollout_layers.split(",") if value.strip()]
        if args.rollout_layers
        else None
    )
    rollout_patch_policies = [
        value.strip()
        for value in args.rollout_patch_policies.split(",")
        if value.strip()
    ]
    outputs = run_hidden_state_patching(
        args.run_dir,
        device=args.device,
        examples_per_pair=args.examples_per_pair,
        rollout_examples=args.rollout_examples,
        rollout_centroid_examples=args.rollout_centroid_examples,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        rollout_receiver_count=args.rollout_receiver_count,
        rollout_receiver_progress=args.rollout_receiver_progress,
        rollout_donor_count=args.rollout_donor_count,
        rollout_donor_progress=args.rollout_donor_progress,
        rollout_layers=rollout_layers,
        rollout_patch_policies=rollout_patch_policies,
        overwrite=args.overwrite,
    )
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
