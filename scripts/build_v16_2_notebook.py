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

            # Keep the uploaded source and result paths derived from one editable root.
            DRIVE_REPO_ROOT = Path(
                "/content/drive/MyDrive/Colab Notebooks/NIAH_synthetic/"
                "Synthetic_CoT_NiaH_Count-main"
            )
            DRIVE_RESULTS_ROOT = DRIVE_REPO_ROOT / "colab_results"
            DRIVE_READY = False
            if Path("/content").exists():
                from google.colab import drive
                if not Path("/content/drive/MyDrive").exists():
                    drive.mount("/content/drive")
                DRIVE_READY = True
                print("Drive mounted; source expected at", DRIVE_REPO_ROOT)
            else:
                print("Local runtime: Drive mount skipped")
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

            assert DRIVE_READY, "Run the Google Drive mount cell before environment setup"
            assert DRIVE_REPO_ROOT.exists(), DRIVE_REPO_ROOT
            assert (DRIVE_REPO_ROOT / "pyproject.toml").exists(), (
                f"pyproject.toml not found under {DRIVE_REPO_ROOT}"
            )
            assert DRIVE_RESULTS_ROOT == DRIVE_REPO_ROOT / "colab_results"
            DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

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
                    ".pytest_cache",
                    ".ruff_cache",
                    ".mypy_cache",
                    "*.egg-info",
                    "runs",
                    "artifacts",
                    "colab_results",
                ),
            )

            repo = LOCAL_REPO_ROOT
            os.chdir(repo)

            # Validate Colab's preinstalled binary stack without modifying it. If this
            # fails, discard the runtime rather than mixing binary package generations.
            scientific_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import matplotlib,numpy,pandas,seaborn,torch,tqdm",
                ],
                capture_output=True,
                text=True,
            )
            if scientific_probe.returncode:
                print(scientific_probe.stdout)
                print(scientific_probe.stderr)
                raise RuntimeError(
                    "Colab scientific-package imports are inconsistent. Use Runtime > "
                    "Disconnect and delete runtime, reconnect, and rerun from the first cell."
                )

            # Preserve Colab's binary-compatible NumPy/pandas/scientific stack.
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."],
                check=True,
            )

            # Editable-install .pth files are processed only when Python starts. Make the
            # copied src-layout package importable in this already-running notebook kernel.
            src_root = str(repo / "src")
            if src_root not in sys.path:
                sys.path.insert(0, src_root)

            import synthetic_counting_v16_2
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import Image, Markdown, display

            package_path = Path(synthetic_counting_v16_2.__file__).resolve()
            assert package_path.is_relative_to(Path(src_root)), (
                f"Notebook kernel imported stale package from {package_path}"
            )
            subprocess_package_path = Path(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import synthetic_counting_v16_2 as p; print(p.__file__)",
                    ],
                    text=True,
                ).strip()
            ).resolve()
            assert subprocess_package_path.is_relative_to(Path(src_root)), (
                f"Subprocess imported stale package from {subprocess_package_path}"
            )

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
                "src_root": src_root,
                "kernel_package": str(package_path),
                "subprocess_package": str(subprocess_package_path),
                "corpus": str(corpus_path),
                "python": sys.version.split()[0],
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
            RUN_TESTS = False  # optional developer preflight; not required for training
            if RUN_TESTS:
                test_process = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v16_2.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print(test_process.stdout, end="")
                test_process.check_returncode()
            else:
                print("Optional repository tests skipped; pipeline validation remains enabled.")
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
            WEIGHT_DECAY = 0.01              # AdamW decay on all trainable parameters; 0.0 disables it
            FINAL_COUNT_LOSS_WEIGHT = 1.0    # >1 upweights the final numeric answer target
            COT_TRACE_LOSS_WEIGHT = 1.0      # >1 upweights CoT trace indices and marker characters
            RUN_ROPE_NONTHINKING = True
            RUN_ROPE_THINKING = True
            RUN_RPE_NONTHINKING = True
            RUN_RPE_THINKING = True
            MAX_TRAIN_STEPS = 10_000         # optimizer steps for each enabled model
            EVAL_EXAMPLES_PER_COUNT = 100    # examples for each count; suite size = this value x COUNT_MAX_THRESHOLD
            NEEDLE_POOL_SIZE = 100
            NEEDLE_POOL_FREQUENCY_THRESHOLD = 0.04
            OUT_ROOT = "runs/synthetic_counting_v16_2"
            RUN_NAME = None                  # default name records all important settings
            SKIP_COMPLETED = True
            CHECKPOINT_SYNC_ROOT = (
                DRIVE_RESULTS_ROOT / "v16_2_live_checkpoints" if DRIVE_READY else None
            )

            ENABLED_MODEL_VARIANTS = tuple(
                variant
                for enabled, variant in (
                    (RUN_ROPE_NONTHINKING, "rope/nonthinking"),
                    (RUN_ROPE_THINKING, "rope/thinking"),
                    (RUN_RPE_NONTHINKING, "rpe/nonthinking"),
                    (RUN_RPE_THINKING, "rpe/thinking"),
                )
                if enabled
            )
            if not ENABLED_MODEL_VARIANTS:
                raise ValueError("Enable at least one of the four model variants")

            from synthetic_counting_v16_2.config import preset_config
            PLANNED_CONFIG = preset_config(
                PRESET,
                seed=SEED,
                device=DEVICE,
                task_occurrence_ratio=TASK_OCCURRENCE_RATIO,
                count_max_threshold=COUNT_MAX_THRESHOLD,
                weight_decay=WEIGHT_DECAY,
                final_count_loss_weight=FINAL_COUNT_LOSS_WEIGHT,
                cot_trace_loss_weight=COT_TRACE_LOSS_WEIGHT,
                enabled_model_variants=ENABLED_MODEL_VARIANTS,
                train_steps=MAX_TRAIN_STEPS,
                eval_examples_per_count=EVAL_EXAMPLES_PER_COUNT,
                needle_pool_size=NEEDLE_POOL_SIZE,
                needle_pool_frequency_threshold=NEEDLE_POOL_FREQUENCY_THRESHOLD,
            )
            EVAL_EXAMPLES_PER_SUITE = EVAL_EXAMPLES_PER_COUNT * COUNT_MAX_THRESHOLD
            PERIODIC_TF_EXAMPLES_PER_MODEL = 7 * EVAL_EXAMPLES_PER_SUITE
            print({
                "config": PLANNED_CONFIG.to_dict(),
                "enabled_model_variants": ENABLED_MODEL_VARIANTS,
                "number_of_models": len(ENABLED_MODEL_VARIANTS),
                "weight_decay": WEIGHT_DECAY,
                "max_steps_per_model": MAX_TRAIN_STEPS,
                "total_planned_optimizer_steps": len(ENABLED_MODEL_VARIANTS) * MAX_TRAIN_STEPS,
                "eval_examples_per_suite": EVAL_EXAMPLES_PER_SUITE,
                "periodic_teacher_forced_examples_per_model": PERIODIC_TF_EXAMPLES_PER_MODEL,
            })
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
                "--weight-decay", str(WEIGHT_DECAY),
                "--final-count-loss-weight", str(FINAL_COUNT_LOSS_WEIGHT),
                "--cot-trace-loss-weight", str(COT_TRACE_LOSS_WEIGHT),
                "--train-steps", str(MAX_TRAIN_STEPS),
                "--eval-examples-per-count", str(EVAL_EXAMPLES_PER_COUNT),
                "--needle-pool-size", str(NEEDLE_POOL_SIZE),
                "--needle-pool-frequency-threshold", str(NEEDLE_POOL_FREQUENCY_THRESHOLD),
                "--out-root", OUT_ROOT,
            ]
            for variant in ENABLED_MODEL_VARIANTS:
                base_cmd += ["--model-variant", variant]
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
