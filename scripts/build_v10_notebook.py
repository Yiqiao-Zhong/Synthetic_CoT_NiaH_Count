from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Trace_Count_v10_Colab.ipynb"


def markdown(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str, cell_id: str, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata,
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # Trace Count v10: Count-30 Learning Dynamics and Causal Mechanisms

        v10 使用两个**独立训练**的 v2-style Transformer：一个 non-thinking，一个 thinking。
        它取消 v5 的 mode switch，固定 prompt 长度 256，并把 needle 数量扩展为 1–30。

        ```text
        non-thinking: <BOS> prompt <Ans> <n> <EOS>
        thinking:     <BOS> prompt <Think> <1> M1 ... <n> Mn </Think> <Ans> <n> <EOS>
        ```

        Trace index 与最终答案共享 `<1>...<30>`。模型均为随机初始化 GPT-2：4 Layers × 4 heads，
        `d_model=256`，MLP=1024，learned absolute-position embedding。

        Notebook 分成四个可恢复阶段：`train`、`attention`、`state`、`plots`。`all` 会依次运行全部阶段。
        完整协议见 `docs/pipelines/pipeline_v10_two_model_count30_causal.md`。
        """,
        "v10-title",
    ),
    markdown("## 1. Google Drive login", "drive-heading"),
    code(
        r"""
        from pathlib import Path
        import sys

        DRIVE_RESULTS_ROOT = Path(
            "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
            "Synthetic_CoT_NiaH_Count/colab_results"
        )
        DRIVE_READY = False

        def ensure_google_drive() -> bool:
            global DRIVE_READY
            if "google.colab" not in sys.modules and not Path("/content").exists():
                print("Not running in Colab; Drive mount skipped.")
                return False
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            DRIVE_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            DRIVE_READY = True
            print("Google Drive ready:", DRIVE_RESULTS_ROOT)
            return True

        ensure_google_drive()
        """,
        "drive-login",
        ["google-drive-login"],
    ),
    markdown("## 2. Environment and repository", "setup-heading"),
    code(
        r"""
        from pathlib import Path
        import os
        import subprocess
        import sys

        REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
        preferred = Path("/content/Synthetic_CoT_NiaH_Count")
        candidates = [Path.cwd(), preferred]
        repo = next(
            (path.resolve() for path in candidates if (path / "pyproject.toml").exists()),
            None,
        )
        if repo is None:
            subprocess.run(["git", "clone", REPO_URL, str(preferred)], check=True)
            repo = preferred
        elif (repo / ".git").exists() and Path("/content").exists():
            subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=False)

        os.chdir(repo)
        print("Repo:", repo)

        # Install one mutually compatible scientific stack before importing it.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numpy<2",
                "pandas>=2.1,<3",
                "scipy>=1.11,<2",
                "matplotlib>=3.8",
                "seaborn>=0.13",
                "transformers>=4.41,<5",
                "tqdm>=4.66",
            ],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)

        import numpy as np
        import pandas as pd
        import torch
        import transformers

        print({
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        """,
        "environment-setup",
    ),
    code(
        r"""
        RUN_TESTS = True
        if RUN_TESTS:
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v10.py"],
                check=True,
            )
        """,
        "v10-tests",
    ),
    markdown(
        r"""
        ## 3. Runtime settings

        - `PRESET="debug"`：极小模型与 8 steps，只验证整条 pipeline。
        - `PRESET="main"`：正式 4-Layer × 4-head 模型，两个模型各 10,000 steps。
        - `STAGE="all"`：训练完成后继续 attention、hidden-state 和绘图。
        - 长实验更稳妥的做法是依次设为 `train`、`attention`、`state`、`plots`。断线后保持同一
          `RUN_NAME`，`SKIP_COMPLETED=True`，会先从 Drive 恢复。

        默认**不早停**。原因是完整 learning dynamics 本身是实验结果，而且两个模型在不同 step 停止会
        破坏比较。pipeline 仍会保存 `best`、每 1000-step 和 `final` checkpoint。
        """,
        "runtime-explanation",
    ),
    code(
        r"""
        import torch

        PRESET = "main"             # "debug" or "main"
        STAGE = "all"               # "all", "train", "attention", "state", "plots"
        RUN_NAME = f"v10_{PRESET}_seed1234"
        OUT_ROOT = "runs/synthetic_counting_v10"
        SKIP_COMPLETED = True
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        DRIVE_SYNC_ROOT = (
            DRIVE_RESULTS_ROOT / "v10_live_checkpoints"
            if DRIVE_READY else None
        )
        print({
            "PRESET": PRESET,
            "STAGE": STAGE,
            "RUN_NAME": RUN_NAME,
            "DEVICE": DEVICE,
            "DRIVE_SYNC_ROOT": str(DRIVE_SYNC_ROOT) if DRIVE_SYNC_ROOT else None,
        })
        """,
        "runtime-settings",
    ),
    markdown("## 4. Run v10 pipeline", "run-heading"),
    code(
        r"""
        import os
        import subprocess
        import sys
        from pathlib import Path

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "synthetic_counting_v10.run_v10",
            "--preset", PRESET,
            "--stage", STAGE,
            "--device", DEVICE,
            "--out-root", OUT_ROOT,
            "--run-name", RUN_NAME,
        ]
        if SKIP_COMPLETED:
            cmd.append("--skip-completed")
        if DRIVE_SYNC_ROOT is not None:
            cmd += ["--checkpoint-sync-root", str(DRIVE_SYNC_ROOT)]

        print(" ".join(cmd), flush=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        captured = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            captured.append(line.rstrip())
        returncode = proc.wait()
        if returncode:
            print("---- Last 160 log lines ----")
            print("\n".join(captured[-160:]))
            raise subprocess.CalledProcessError(returncode, cmd)

        RUN_DIR = Path(OUT_ROOT) / RUN_NAME
        print("RUN_DIR =", RUN_DIR.resolve())
        """,
        "run-pipeline",
    ),
    markdown(
        r"""
        ## 5. Training dynamics

        **Teacher-forced final-count accuracy**：输入完整 gold sequence，并读取 `<Ans>` hidden state 的下一
        token logits；只在 `<1>...<30>` 中取 argmax。它测的是给定正确前缀后的 final-count readout。

        **Autoregressive final-count accuracy**：只输入 prompt 与生成起点，让模型自己生成 trace/answer，再解析
        最终数字。它同时包含 trace generation error 与 final readout error。

        三个难度桶 `1-10`、`11-20`、`21-30` 都包含等量 exact-count classes，因此图中每个桶是十个 count
        accuracy 的宏平均，而不是被某个常见 count 主导。
        """,
        "training-definitions",
    ),
    code(
        r"""
        from IPython.display import Image, Markdown, display
        import pandas as pd

        def display_png(path: Path, width: int = 1100):
            if path.exists():
                display(Image(filename=str(path), width=width))
            else:
                print("Missing figure:", path)

        config = pd.read_json(RUN_DIR / "config.json", typ="series")
        display(config.to_frame("value"))
        for name in ["eval_dynamics_by_bin.csv", "eval_dynamics_by_count.csv", "eval_dynamics_losses.csv"]:
            path = RUN_DIR / "tables" / name
            if path.exists():
                frame = pd.read_csv(path)
                display(Markdown(f"**{name}**"))
                display(frame.tail(24))

        for name in [
            "training_loss_components.png",
            "accuracy_dynamics_by_count_bin.png",
            "final_accuracy_by_exact_count.png",
        ]:
            display_png(RUN_DIR / "figures" / "training" / name)
        """,
        "show-training-results",
    ),
    markdown(
        r"""
        ## 6. Attention heads: definitions and causal tests

        设 Layer `l`、head `h` 在 query `q` 对第 `j` 个 prompt needle 位置 `p_j` 的 attention 为
        $A_{lh}(q,p_j)$。

        **Broad attention score**：先计算 needle 总质量
        $m_N=\sum_j A_{lh}(q,p_j)$，再把 needle 子集内权重归一化为 $r_j$，计算
        $H_N=-\sum_j r_j\log r_j/\log n$。最终
        $B_{lh}=m_N H_N$。它同时要求“看向 needles”与“在多个 needles 间广泛覆盖”。

        **Targeted retrieval score**：thinking trace 中 `<k>` 正在预测 `M_k`，定义
        $T_{lh}=\mathbb{E}_{x,k}[A_{lh}(<k>,p_k)]$。另外保存 correct-top1 和
        diagonal dominance。

        **Top-n ablation**：按上述 score 排序后，从 top-1 一直累积 mask 到 top-16，并与相同 n 的固定随机
        排序比较。纵轴 margin drop 是未干预正确 token logit margin 减去 mask 后 margin。

        **Patching**：marker identity 实验保持 count/positions 不变，只改第 k 个 marker，再把 clean head output
        patch 回 corrupt run。count-offset 实验覆盖 ±1、±2、±3、±5、±10，并保存 donor/receiver absolute
        position，防止把 learned-position effect 错当成抽象加法。
        """,
        "attention-definitions",
    ),
    code(
        r"""
        attention_root = RUN_DIR / "analysis" / "attention_causal"
        for name in [
            "attention_candidate_signatures.png",
            "nonthinking_final_query_attention_categories.png",
            "thinking_final_query_attention_categories.png",
            "topn_head_ablation.png",
            "thinking_topn_trace_ablation.png",
            "retrieval_patching_topn.png",
            "nonthinking_count_offset_head_patching.png",
            "thinking_count_offset_head_patching.png",
        ]:
            display_png(attention_root / "figures" / name)

        for name in [
            "attention_head_summary.csv",
            "topn_ablation_summary.csv",
            "retrieval_patching_summary.csv",
            "count_offset_head_patching_summary.csv",
        ]:
            path = attention_root / "tables" / name
            if path.exists():
                display(Markdown(f"**{name}**"))
                display(pd.read_csv(path).head(40))
        """,
        "show-attention-results",
    ),
    markdown(
        r"""
        ## 7. Hidden-state manifold and causal count state

        每个 Layer 的 residual state 都在四类 anchor 提取：non-thinking `<Ans>`、thinking `<Ans>`、
        thinking `<k>`、thinking `M_k`。前两类 label 是 total count，后两类 label 是 trace progress `k`。

        独立 train/eval 样本用于拟合 adjacent-centroid direction 与 ridge direction。PCA 图对每个 Layer 分开
        拟合，并记录前 2、3、6 个 PC 的累计解释方差。高 probe/PCA 可读性只说明 representation 可读，
        不等于模型因果使用它。

        因果部分包括：

        1. 沿 count direction 做加性 steering；
        2. final `<Ans>` residual 从 donor count `m` 替换到 receiver count `n`；
        3. thinking trace 内把 `M_m` residual 替换到 `M_n`；
        4. final donor patch 到早期位置，测试是否诱导 premature `</Think>`；
        5. earlier donor patch 到 final 位置，测试是否诱导继续输出 `<m+1>`。

        所有 patch 表都保留绝对位置差。若 effect 与 offset 同时也与 position delta 完全共线，不能直接宣称
        模型实现了抽象的 `+n/-n` 运算。
        """,
        "state-definitions",
    ),
    code(
        r"""
        state_root = RUN_DIR / "analysis" / "state_causal"
        for name in [
            "direction_geometry_by_layer.png",
            "pca_2_3_6_component_variance.png",
            "nonthinking_final_answer_pca2d_by_layer.png",
            "thinking_final_answer_pca2d_by_layer.png",
            "thinking_fixed_trace_answer_pca2d_by_layer.png",
            "thinking_trace_index_pca2d_by_layer.png",
            "thinking_trace_marker_pca2d_by_layer.png",
            "nonthinking_final_answer_pca3d_by_layer.png",
            "thinking_final_answer_pca3d_by_layer.png",
            "thinking_fixed_trace_answer_pca3d_by_layer.png",
            "thinking_trace_index_pca3d_by_layer.png",
            "thinking_trace_marker_pca3d_by_layer.png",
            "geometry_steering_by_layer.png",
            "final_state_m_to_n_transplant.png",
            "trace_progress_m_to_n_transplant.png",
        ]:
            display_png(state_root / "figures" / name)

        for name in [
            "direction_geometry.csv",
            "manifold_geometry.csv",
            "steering_summary.csv",
            "final_state_transplant_summary.csv",
            "trace_progress_transplant_summary.csv",
        ]:
            path = state_root / "tables" / name
            if path.exists():
                display(Markdown(f"**{name}**"))
                display(pd.read_csv(path).head(60))
        """,
        "show-state-results",
    ),
    markdown("## 8. Save a timestamped result bundle to Google Drive", "save-heading"),
    code(
        r"""
        import json
        import shutil
        from datetime import datetime

        DRIVE_SAVE_COMPLETED = False
        if DRIVE_READY:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            RESULT_BUNDLE = DRIVE_RESULTS_ROOT / f"{RUN_NAME}_{stamp}"
            if RESULT_BUNDLE.exists():
                shutil.rmtree(RESULT_BUNDLE)
            shutil.copytree(RUN_DIR, RESULT_BUNDLE)
            manifest = {
                "experiment": "v10",
                "run_name": RUN_NAME,
                "preset": PRESET,
                "stage": STAGE,
                "source_run_dir": str(RUN_DIR.resolve()),
                "saved_at": stamp,
            }
            (RESULT_BUNDLE / "bundle_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            DRIVE_SAVE_COMPLETED = True
            print("Saved:", RESULT_BUNDLE)
        else:
            print("Drive is unavailable; local results remain at", RUN_DIR)
        """,
        "save-drive",
        ["google-drive-save"],
    ),
    markdown("## 9. Optional runtime disconnect", "disconnect-heading"),
    code(
        r"""
        AUTO_DISCONNECT = False

        if AUTO_DISCONNECT and DRIVE_SAVE_COMPLETED and "google.colab" in sys.modules:
            from google.colab import runtime
            runtime.unassign()
        else:
            print("Runtime kept alive. Set AUTO_DISCONNECT=True after checking the bundle.")
        """,
        "optional-disconnect",
        ["auto-disconnect"],
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Trace_Count_v10_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
