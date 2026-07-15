from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.head_state_bidirectional import run_bidirectional_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v10 bidirectional attention-head / hidden-state causal analysis"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--centroid-examples-per-count", type=int, default=4)
    parser.add_argument("--eval-examples-per-count", type=int, default=2)
    parser.add_argument("--state-to-head-examples-per-bin", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    outputs = run_bidirectional_analysis(
        args.run_dir.resolve(),
        centroid_examples_per_count=args.centroid_examples_per_count,
        eval_examples_per_count=args.eval_examples_per_count,
        state_to_head_examples_per_bin=args.state_to_head_examples_per_bin,
        device=args.device,
    )
    output_dir = args.run_dir.resolve() / "analysis" / "head_state_bidirectional"
    print(f"V10_BIDIRECTIONAL_DIR={output_dir}")
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
