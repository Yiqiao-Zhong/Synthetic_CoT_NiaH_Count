from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Trace_Count_v16_2_Colab.ipynb"


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str, cell_id: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"tags": tags} if tags else {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build() -> Path:
    cells = [
        markdown(
            """
            # Trace Count v16_2: three-character sets in Tiny Shakespeare

            This isolated revision prepares a reproducible pool of distinct three-character
            sets, mixes raw-language and counting-formatted training examples through
            `TASK_OCCURRENCE_RATIO`, and evaluates fixed raw/task/mixture suites on train,
            held-out validation, and final-only test regions.
            """,
            "title",
        ),
        markdown("## 1. Mount Google Drive first", "drive-heading"),
        code(
            """
            from pathlib import Path

            DRIVE_RESULTS_ROOT = Path(
                "/content/drive/MyDrive/Colab Notebooks/NIAH_synthetic/"
                "Synthetic_CoT_NiaH_Count-main/colab_results"
            )
            DRIVE_READY = False
            if Path("/content").exists():
                from google.colab import drive
                if not Path("/content/drive/MyDrive").exists():
                    drive.mount("/content/drive")
                DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
                DRIVE_READY = True
            """,
            "drive-login",
            ["google-drive-login"],
        ),
        markdown("## 2. Repository and environment", "setup-heading"),
        code(
            """
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            # Change this to the exact Drive folder containing pyproject.toml.
            DRIVE_REPO_ROOT = Path(
                "/content/drive/MyDrive/Colab Notebooks/NIAH_synthetic/"
                "Synthetic_CoT_NiaH_Count-main"
            )

            assert DRIVE_REPO_ROOT.exists(), DRIVE_REPO_ROOT
            assert (DRIVE_REPO_ROOT / "pyproject.toml").exists(), (
                f"pyproject.toml not found under {DRIVE_REPO_ROOT}"
            )

            # Copy source to the Colab VM: training against Drive is substantially slower.
            LOCAL_REPO_ROOT = Path("/content/Synthetic_CoT_NiaH_Count-main")
            shutil.copytree(
                DRIVE_REPO_ROOT,
                LOCAL_REPO_ROOT,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".venv*",
                    "__pycache__",
                    "runs",
                    "artifacts",
                    "colab_results",
                ),
            )

            repo = LOCAL_REPO_ROOT
            os.chdir(repo)
            # Preserve Colab's binary-compatible NumPy/pandas/scientific stack.
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."],
                check=True,
            )

            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import Image, Markdown, display

            corpus_path = (
                repo
                / "src"
                / "synthetic_counting_v11"
                / "resources"
                / "tiny_shakespeare"
                / "input.txt"
            )
            assert corpus_path.exists(), f"Tiny Shakespeare is missing: {corpus_path}"

            print({
                "drive_repo": str(DRIVE_REPO_ROOT),
                "working_repo": str(repo),
                "corpus": str(corpus_path),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
            """,
            "environment-setup",
        ),
        code(
            """
            RUN_TESTS = True
            if RUN_TESTS:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v16_2.py"],
                    check=True,
                )
            """,
            "tests",
        ),
        markdown("## 3. Easy-to-edit settings", "settings-heading"),
        code(
            """
            VERSION = "v16_2"
            PRESET = "main"                 # use "debug" for an end-to-end check
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            TASK_OCCURRENCE_RATIO = 0.05     # 0 = raw Shakespeare; 1 = all counting tasks
            COUNT_MAX_THRESHOLD = 10
            NEEDLE_POOL_SIZE = 100
            NEEDLE_POOL_FREQUENCY_THRESHOLD = 0.04
            OUT_ROOT = "runs/synthetic_counting_v16_2"
            RUN_NAME = None                  # default name records all important settings
            SKIP_COMPLETED = True
            CHECKPOINT_SYNC_ROOT = (
                DRIVE_RESULTS_ROOT / "v16_2_live_checkpoints" if DRIVE_READY else None
            )

            from synthetic_counting_v16_2.config import preset_config
            PLANNED_CONFIG = preset_config(
                PRESET,
                seed=SEED,
                device=DEVICE,
                task_occurrence_ratio=TASK_OCCURRENCE_RATIO,
                count_max_threshold=COUNT_MAX_THRESHOLD,
                needle_pool_size=NEEDLE_POOL_SIZE,
                needle_pool_frequency_threshold=NEEDLE_POOL_FREQUENCY_THRESHOLD,
            )
            print(PLANNED_CONFIG.to_dict())
            """,
            "runtime-settings",
        ),
        markdown("## 4. Prepare split, pool, and fixed evaluation suites", "prepare-heading"),
        code(
            """
            base_cmd = [
                sys.executable, "-u", "-m", "synthetic_counting_v16_2.run_v16_2",
                "--preset", PRESET,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO),
                "--count-max-threshold", str(COUNT_MAX_THRESHOLD),
                "--needle-pool-size", str(NEEDLE_POOL_SIZE),
                "--needle-pool-frequency-threshold", str(NEEDLE_POOL_FREQUENCY_THRESHOLD),
                "--out-root", OUT_ROOT,
            ]
            if RUN_NAME is not None:
                base_cmd += ["--run-name", RUN_NAME]
            if CHECKPOINT_SYNC_ROOT is not None:
                base_cmd += ["--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT)]
            if SKIP_COMPLETED:
                base_cmd.append("--skip-completed")
            subprocess.run([*base_cmd, "--stage", "prepare"], check=True)

            from synthetic_counting_v16_2.config import default_run_name
            RUN_DIR = Path(OUT_ROOT) / (RUN_NAME or default_run_name(PLANNED_CONFIG))
            display(pd.read_csv(RUN_DIR / "tables" / "needle_pool.csv"))
            display(Image(filename=str(RUN_DIR / "figures" / "needle_pool_frequency_distribution.png")))
            """,
            "prepare-data",
        ),
        markdown("## 5. Train, analyze, and plot", "run-heading"),
        code(
            """
            subprocess.run([*base_cmd, "--stage", "train,attention,state,plots"], check=True)
            print("RUN_DIR =", RUN_DIR.resolve())
            """,
            "run-pipeline",
        ),
        markdown("## 6. Inspect train-versus-held-out loss curves", "inspect-heading"),
        code(
            """
            losses = pd.read_csv(RUN_DIR / "tables" / "eval_loss_curves.csv")
            display(losses)
            display(Image(filename=str(RUN_DIR / "figures" / "learning_loss_suites_train_vs_heldout.png")))
            test_summary = RUN_DIR / "tables" / "test_loss_summary.csv"
            if test_summary.exists():
                display(Markdown("**Final-only untouched test results**"))
                display(pd.read_csv(test_summary))
            """,
            "inspect-losses",
        ),
        markdown("## 7. Save the complete result bundle", "save-heading"),
        code(
            """
            import shutil
            from datetime import datetime

            if DRIVE_READY:
                destination = DRIVE_RESULTS_ROOT / f"{RUN_DIR.name}_{datetime.now():%Y%m%d_%H%M%S}"
                shutil.copytree(RUN_DIR, destination, dirs_exist_ok=True)
                print("Saved:", destination)
            else:
                print("Drive unavailable; results remain at", RUN_DIR.resolve())
            """,
            "save-drive",
            ["google-drive-save"],
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
