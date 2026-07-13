from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


# The local workstation can contain optional Pandas accelerators compiled for a
# different NumPy ABI. They are not needed for these experiments.
if "pyarrow" not in sys.modules:
    pyarrow_stub = types.ModuleType("pyarrow")
    pyarrow_stub.__version__ = "0.0.0"
    pyarrow_stub.Array = type("Array", (), {})
    pyarrow_stub.ChunkedArray = type("ChunkedArray", (), {})
    sys.modules["pyarrow"] = pyarrow_stub
for optional_module in ("numexpr", "bottleneck"):
    sys.modules.setdefault(optional_module, None)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_counting_v10.report_followups import run_report_followups


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict local causal follow-ups used in the v10 report")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = run_report_followups(args.run_dir, device=args.device, overwrite=args.overwrite)
    print("V10_REPORT_FOLLOWUPS=", args.run_dir / "analysis" / "report_followups")
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
