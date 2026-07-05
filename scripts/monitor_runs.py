from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def read_last_jsonl(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last = ""
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and last:
                break
            last = char.decode("utf-8", errors="ignore") + last
    last = last.strip()
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"raw": last}


def checkpoint_summary(run_dir: Path) -> str:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return "-"
    names = sorted(path.name for path in ckpt_dir.iterdir() if path.is_dir())
    if not names:
        return "-"
    if "final" in names:
        return "final"
    return names[-1]


def gpu_summary() -> list[str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        return [f"GPU: unavailable ({exc.__class__.__name__})"]
    if not out:
        return ["GPU: no output"]
    rows = []
    for idx, line in enumerate(out.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(f"GPU {idx}: {parts[0]} | util={parts[1]}% | mem={parts[2]}/{parts[3]} MiB")
        else:
            rows.append(f"GPU {idx}: {line}")
    return rows


def format_value(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def collect_rows(runs_dir: Path, max_steps: int | None) -> list[dict]:
    rows = []
    for run_dir in sorted(path for path in runs_dir.rglob("*_seed*") if path.is_dir()):
        log_path = run_dir / "train_log.jsonl"
        last = read_last_jsonl(log_path)
        mtime = log_path.stat().st_mtime if log_path.exists() else None
        age = time.time() - mtime if mtime is not None else None
        step = last.get("step") if last else None
        progress = None
        if step is not None and max_steps:
            progress = 100.0 * float(step) / float(max_steps)
        rows.append(
            {
                "run": run_dir.relative_to(runs_dir).as_posix(),
                "step": step,
                "progress": progress,
                "loss": last.get("total_weighted_loss") if last else None,
                "val": last.get("val_total_weighted_loss") if last else None,
                "tf": last.get("val_tf_count_acc") if last else None,
                "age": age,
                "ckpt": checkpoint_summary(run_dir),
            }
        )
    return rows


def print_report(runs_dir: Path, max_steps: int | None, show_gpu: bool) -> None:
    print("=" * 120)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"runs_dir={runs_dir}")
    if show_gpu:
        for row in gpu_summary():
            print(row)
    rows = collect_rows(runs_dir, max_steps)
    if not rows:
        print("No run directories found yet.")
        return
    header = f"{'run':58} {'step':>8} {'%':>7} {'loss':>10} {'val':>10} {'tf':>8} {'age':>8} {'ckpt':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        step = "-" if row["step"] is None else str(row["step"])
        pct = "-" if row["progress"] is None else f"{row['progress']:.1f}"
        age = "-" if row["age"] is None else f"{row['age']:.0f}s"
        print(
            f"{row['run'][:58]:58} "
            f"{step:>8} "
            f"{pct:>7} "
            f"{format_value(row['loss']):>10} "
            f"{format_value(row['val']):>10} "
            f"{format_value(row['tf'], 3):>8} "
            f"{age:>8} "
            f"{row['ckpt'][:16]:>16}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor trace-counting training logs and checkpoints.")
    parser.add_argument("--runs_dir", default="runs/trace_count_v0_seed0")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    while True:
        print_report(runs_dir, args.max_steps, show_gpu=not args.no_gpu)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
