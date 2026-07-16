from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


VERSIONS = {
    "v11": {
        "title": "Position-Encoding Comparison at Fixed Small Capacity",
        "summary": (
            "固定 v10/v2 counting task 与 4-layer x 4-head GPT-2-style 计算骨架，"
            "把 hidden size 缩到 64，并在严格配对初始化下比较 learned APE、RoPE 与 learned RPE。"
        ),
        "data": "prompt length 256; needle count 1-30; streaming uniform synthetic haystack",
        "positions": "APE, RoPE, RPE",
        "runs": "3 positional encodings x 2 output modes = 6 independently trained models",
    },
    "v12": {
        "title": "Longer Context and Count-50 at Small Capacity",
        "summary": (
            "复用 v10-style 数据/render/training 路径与 v11 的小模型，将 prompt 扩到 512，"
            "needle count 扩到 1-50；只使用 learned APE。"
        ),
        "data": "prompt length 512; needle count 1-50; streaming uniform synthetic haystack",
        "positions": "learned APE only",
        "runs": "1 positional encoding x 2 output modes = 2 independently trained models",
    },
    "v13": {
        "title": "Fixed-Dataset Training at Small Capacity",
        "summary": (
            "保持 v11 的 APE、长度 256、count 1-30 与 64 维模型，但训练 prompt 不再在线重采样；"
            "先持久化一个平衡 finite pool，再从同一 pool 有放回采样。"
        ),
        "data": "prompt length 256; needle count 1-30; fixed balanced training pool",
        "positions": "learned APE only",
        "runs": "1 positional encoding x 2 output modes = 2 independently trained models",
    },
    "v14": {
        "title": "Shakespeare Character Haystack at Small Capacity",
        "summary": (
            "保持 v11 的 APE、小模型与 count 设置，将 uniform noise 换成 Shakespeare 公版文本的"
            "连续 character-level token window，再随机覆盖位置插入 marker。"
        ),
        "data": "prompt length 256; needle count 1-30; contiguous Shakespeare character haystack",
        "positions": "learned APE only",
        "runs": "1 positional encoding x 2 output modes = 2 independently trained models",
    },
}


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


def build_cells(version: str, spec: dict[str, str]) -> list[dict]:
    module = f"synthetic_counting_{version}.run_{version}"
    position_default = "ape"
    return [
        markdown(
            f"""
            # Trace Count {version}: {spec['title']}

            {spec['summary']}

            **受控协议**

            | 项目 | 设置 |
            | --- | --- |
            | Transformer | random-init v10-style pre-LN causal Transformer; **4 layers, 4 heads, d_model=64, MLP=256** |
            | 数据 | {spec['data']} |
            | 位置编码 | {spec['positions']} |
            | 输出模式 | 两个独立模型：non-thinking 与 thinking；不共享训练参数 |
            | run 数 | {spec['runs']} |
            | 数字 token | trace index 与最终答案共享同一套 `<1>...<N>` token |
            | 分析范围 | Section 4 learning dynamics；Section 5 descriptive attention；Section 6 hidden-state geometry |

            ```text
            non-thinking: <BOS> prompt <Ans> <n> <EOS>
            thinking:     <BOS> prompt <Think> <1> M1 ... <n> Mn </Think> <Ans> <n> <EOS>
            ```

            v12-v14 可以复用 v10-style 实现，但**不复用 v10 的 d_model=256 参数规模**；配置会硬性拒绝任何非 64 维模型。
            """,
            f"{version}-title",
        ),
        markdown("## 1. 在训练前挂载 Google Drive", "drive-heading"),
        code(
            """
            from pathlib import Path
            import sys

            DRIVE_RESULTS_ROOT = Path(
                "/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/"
                "Synthetic_CoT_NiaH_Count/colab_results"
            )
            DRIVE_READY = False

            def ensure_google_drive() -> bool:
                global DRIVE_READY
                if not Path("/content").exists():
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
        markdown("## 2. 环境与仓库", "setup-heading"),
        code(
            """
            from pathlib import Path
            import json
            import os
            import signal
            import subprocess
            import sys

            REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
            preferred = Path("/content/Synthetic_CoT_NiaH_Count")
            search = [Path.cwd(), *Path.cwd().parents, preferred]
            repo = next((p.resolve() for p in search if (p / "pyproject.toml").exists()), None)
            if repo is None:
                subprocess.run(["git", "clone", REPO_URL, str(preferred)], check=True)
                repo = preferred
            elif Path("/content").exists() and (repo / ".git").exists():
                subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=False)
            os.chdir(repo)
            print("Repo:", repo)

            # Test the compiled scientific stack in a fresh process. Repair the
            # whole ABI-compatible group together only when the probe fails.
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json,numpy,pandas,scipy,matplotlib,seaborn; "
                    "print(json.dumps({'numpy':numpy.__version__,'pandas':pandas.__version__}))",
                ],
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                print(probe.stderr[-2000:])
                subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                        "--force-reinstall", "numpy==1.26.4", "pandas==2.2.3",
                        "scipy==1.13.1", "matplotlib==3.8.4", "seaborn==0.13.2",
                    ],
                    check=True,
                )
                print("Scientific-stack ABI repaired. Restarting the Colab runtime once.")
                sys.stdout.flush()
                if Path("/content").exists():
                    os.kill(os.getpid(), signal.SIGKILL)
                raise RuntimeError("Restart the kernel and rerun all cells.")

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
                check=True,
            )
            try:
                import plotly
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "plotly>=5"], check=True)

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import Image, Markdown, display

            print({
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
            RUN_TESTS = True
            if RUN_TESTS:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "tests/test_synthetic_counting_v11_v14.py"],
                    check=True,
                )
            """,
            "tests",
        ),
        markdown(
            f"""
            ## 3. Runtime settings

            `debug` 用极小数据和 6 steps 检查整条 pipeline；`main` 使用正式 10,000 steps。
            `STAGE="all"` 依次运行 train、attention、state、plots。训练 checkpoint 每 1,000 steps
            同步到 Drive；断线后保持同一个 `RUN_NAME` 与 `SKIP_COMPLETED=True` 即可恢复。

            本版本正式训练：**{spec['runs']}**。
            """,
            "runtime-heading",
        ),
        code(
            f"""
            VERSION = "{version}"
            PRESET = "main"                 # "debug" or "main"
            STAGE = "all"                   # train, attention, state, plots, or all
            SEED = 1234
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            OUT_ROOT = f"runs/synthetic_counting_{{VERSION}}"
            RUN_NAME = f"{{VERSION}}_{{PRESET}}_seed{{SEED}}"
            SKIP_COMPLETED = True
            CHECKPOINT_SYNC_ROOT = (
                DRIVE_RESULTS_ROOT / f"{{VERSION}}_live_checkpoints" if DRIVE_READY else None
            )

            # Resolve and validate the exact preset before starting an expensive run.
            # v12-v14 may reuse the v10-style computation path, but their hidden
            # width must remain the v11-scale d_model=64.
            from synthetic_counting_v11.config import preset_config

            PLANNED_CONFIG = preset_config(VERSION, PRESET, seed=SEED, device=DEVICE)
            assert (
                PLANNED_CONFIG.n_layer,
                PLANNED_CONFIG.n_head,
                PLANNED_CONFIG.n_embd,
                PLANNED_CONFIG.n_inner,
            ) == (4, 4, 64, 256)

            print({{
                "VERSION": VERSION,
                "PRESET": PRESET,
                "STAGE": STAGE,
                "DEVICE": DEVICE,
                "OUT_ROOT": OUT_ROOT,
                "RUN_NAME": RUN_NAME,
                "CHECKPOINT_SYNC_ROOT": str(CHECKPOINT_SYNC_ROOT) if CHECKPOINT_SYNC_ROOT else None,
                "seq_len": PLANNED_CONFIG.seq_len,
                "count_range": f"{{PLANNED_CONFIG.count_min}}-{{PLANNED_CONFIG.count_max}}",
                "position_encodings": PLANNED_CONFIG.position_encodings,
                "architecture": "4 layers x 4 heads x d_model 64; MLP 256",
            }})
            """,
            "runtime-settings",
        ),
        markdown(f"## 4. Run {version} pipeline", "run-heading"),
        code(
            f"""
            import subprocess
            import sys
            from pathlib import Path

            cmd = [
                sys.executable, "-u", "-m", "{module}",
                "--preset", PRESET,
                "--stage", STAGE,
                "--device", DEVICE,
                "--seed", str(SEED),
                "--out-root", OUT_ROOT,
                "--run-name", RUN_NAME,
            ]
            if SKIP_COMPLETED:
                cmd.append("--skip-completed")
            if CHECKPOINT_SYNC_ROOT is not None:
                cmd += ["--checkpoint-sync-root", str(CHECKPOINT_SYNC_ROOT)]

            print(" ".join(cmd), flush=True)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured = []
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                captured.append(line.rstrip())
            returncode = process.wait()
            if returncode:
                print("---- Last 160 log lines ----")
                print("\\n".join(captured[-160:]))
                raise subprocess.CalledProcessError(returncode, cmd)

            RUN_DIR = Path(OUT_ROOT) / RUN_NAME
            assert (RUN_DIR / "config.json").exists(), RUN_DIR
            print("RUN_DIR =", RUN_DIR.resolve())
            """,
            "run-pipeline",
        ),
        markdown(
            r"""
            ## 4. Learning dynamics

            **Teacher-forced final-count accuracy**：给定完整 gold prefix，在 `<Ans>` 后预测最终数字；只在
            count-number token 集合上取 argmax。它测试“给定正确中间上下文后的答案读取”，不等同于自由生成。

            **Autoregressive final-count accuracy**：模型从 prompt 后自由生成完整 completion，再解析最终数字是否
            等于 gold count。thinking 模式中的上游 trace 错误会传播到此指标。

            **Trace-index / trace-marker accuracy**：仅适用于 thinking。前者逐位置检查 `<k>`，后者逐位置检查
            第 k 个 marker identity；均在 gold prefix 下计算。曲线按每十个 count 划分难度区间。

            **Loss**：next-token cross entropy 只覆盖 completion。non-thinking 覆盖最终 count 与 EOS；thinking
            还覆盖完整 index-marker trace、`</Think>`、`<Ans>`、最终 count 与 EOS。
            """,
            "learning-definitions",
        ),
        code(
            """
            cfg = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
            display(pd.DataFrame([{
                "version": cfg["version"],
                "seq_len": cfg["seq_len"],
                "count_range": f"{cfg['count_min']}-{cfg['count_max']}",
                "position_encodings": ", ".join(cfg["position_encodings"]),
                "layers": cfg["n_layer"],
                "heads": cfg["n_head"],
                "d_model": cfg["n_embd"],
                "mlp": cfg["n_inner"],
                "training_data_mode": cfg["training_data_mode"],
                "noise_source": cfg["noise_source"],
            }]))

            for name in ["learning_loss.png", "learning_accuracy_by_bin.png", "final_accuracy_by_count.png"]:
                display(Image(filename=str(RUN_DIR / "figures" / name)))
            display(pd.read_csv(RUN_DIR / "tables" / "time_to_99.csv"))
            """,
            "learning-results",
        ),
        markdown(
            r"""
            ## 5. Descriptive attention

            对 layer `l`、head `h`，令 `A[l,h,q,j]` 是 query 位置 `q` 对 key 位置 `j` 的 softmax attention。

            - **Prompt-needle mass**：对全部 prompt needle 位置求和。它只说明 head 看了多少 needle。
            - **Normalized needle entropy**：先把 needle 子集内的 attention 重新归一化，再计算 entropy / log(n)。
              1 表示在 n 个 needles 间均匀覆盖，0 表示集中到一个 needle。
            - **Broad-attention score**：`prompt_needle_mass * normalized_needle_entropy`；同时奖励 needle 总 mass
              和跨 needle 的广覆盖。
            - **Raw k-to-k mass**：thinking 的第 k 个 trace index query `<k>` 对 prompt 中按位置排序的第 k 个
              needle 的原始 attention 权重。
            - **Correct top-1**：只在 prompt-needle 子集内比较，最大 attention 是否落在第 k 个 needle。
            - **Diagonal dominance**：raw k-to-k mass / prompt-needle mass。它是 needle 子集内的相对占比；即使
              接近 1，raw mass 仍可能很低，因此必须与 raw mass 一起读。
            - **Trace-marker readout mass**：最终 `<Ans>` query 投向所有 trace marker 位置的 attention 总和。

            这些都是**描述性 routing 指标**，不是因果证据；它们用于比较 PE 是否更容易形成 broad aggregation
            或 targeted retrieval 候选 head。
            """,
            "attention-definitions",
        ),
        code(
            """
            attention_summary = pd.read_csv(RUN_DIR / "tables" / "attention_summary.csv")
            positions = cfg["position_encodings"]
            for position_encoding in positions:
                display(Markdown(f"### {position_encoding.upper()} attention signatures"))
                display(Image(filename=str(RUN_DIR / "figures" / f"attention_signatures_{position_encoding}.png")))
                display(Image(filename=str(RUN_DIR / "figures" / f"targeted_retrieval_by_bin_{position_encoding}.png")))

            display(
                attention_summary[
                    attention_summary["count_bin"].eq("all")
                ].sort_values(
                    ["position_encoding", "mode", "query_kind", "correct_prompt_needle_mass"],
                    ascending=[True, True, True, False],
                ).groupby(
                    ["position_encoding", "mode", "query_kind"], as_index=False
                ).head(4)
            )
            """,
            "attention-results",
        ),
        markdown(
            r"""
            ## 6. Descriptive hidden-state geometry

            每个 token 在 embedding 后和四个 Transformer layers 后各有一个 64 维 residual-stream state
            (`layer=0..4`)。本节提取：两种模式的最终 `<Ans>` query，以及 thinking trace 中每个 `<k>` index
            和随后的 marker state。

            - **Nearest-centroid accuracy**：训练集按 exact count/progress 求 64 维 centroid；测试 state 分给标准化
              欧氏距离最近的 centroid。
            - **Position-only baseline**：只用 absolute token position 做同样分类，排除“probe 只读出长度/位置”。
            - **Ridge R2 / MAE**：把 count/progress 当连续标量，检验 state 是否沿近似线性 count direction 排列。
            - **PCA**：只对各 exact-count centroid 做 PCA，报告 PC1-PC6 explained-variance ratio 与累计覆盖；图中
              每个点是一个 count/progress 类均值，不是单个样本。

            这些仍是表征可读性证据，不单独证明某个方向被模型因果使用。
            """,
            "state-definitions",
        ),
        code(
            """
            probes = pd.read_csv(RUN_DIR / "tables" / "state_probe_summary.csv")
            centroids = pd.read_csv(RUN_DIR / "tables" / "state_centroids_pca.csv")
            variance = pd.read_csv(RUN_DIR / "tables" / "state_pca_variance.csv")

            for position_encoding in cfg["position_encodings"]:
                display(Markdown(f"### {position_encoding.upper()} hidden-state summary"))
                display(Image(filename=str(RUN_DIR / "figures" / f"state_probe_{position_encoding}.png")))
                display(Image(filename=str(RUN_DIR / "figures" / f"state_pca_variance_{position_encoding}.png")))
                for path in sorted((RUN_DIR / "figures").glob(f"state_centroids_{position_encoding}_*.png")):
                    display(Image(filename=str(path)))

            display(probes.sort_values(["position_encoding", "mode", "site", "layer"]))
            """,
            "state-results",
        ),
        markdown(
            "### 可交互 3D centroid manifold\n\n修改下方选择项后重跑。PC 可从 PC1-PC6 任取三轴；如果某类不足 6 个非零 PC，图会自动使用现有列。",
            "state-interactive-heading",
        ),
        code(
            f"""
            import plotly.express as px

            POSITION_ENCODING = "{position_default}"
            MODE = "thinking"               # "nonthinking" or "thinking"
            SITE = "final_answer"            # final_answer, trace_index, trace_marker
            LAYER = 4                         # 0=embedding, 1..4=after each Transformer layer
            PC_X, PC_Y, PC_Z = "pc1", "pc2", "pc3"

            selected = centroids[
                centroids["position_encoding"].eq(POSITION_ENCODING)
                & centroids["mode"].eq(MODE)
                & centroids["site"].eq(SITE)
                & centroids["layer"].eq(LAYER)
            ].copy()
            missing = [column for column in (PC_X, PC_Y, PC_Z) if column not in selected]
            if selected.empty:
                raise ValueError("No centroid rows for the selected PE/mode/site/layer combination.")
            if missing:
                raise ValueError(f"Unavailable PCA columns: {{missing}}")
            figure = px.scatter_3d(
                selected,
                x=PC_X,
                y=PC_Y,
                z=PC_Z,
                color="state_label",
                text="state_label",
                color_continuous_scale="Viridis",
                title=(
                    f"{{POSITION_ENCODING.upper()}} | {{MODE}} | {{SITE}} | "
                    f"layer {{LAYER}} | {{PC_X.upper()}}, {{PC_Y.upper()}}, {{PC_Z.upper()}}"
                ),
            )
            figure.update_traces(marker={{"size": 6}}, textposition="top center")
            figure.show()
            """,
            "interactive-manifold",
        ),
        markdown("## 7. 保存完整结果到 Google Drive", "save-heading"),
        code(
            """
            from datetime import datetime
            import shutil

            DRIVE_SAVE_COMPLETED = False
            SAVED_RESULT_DIR = None
            if DRIVE_READY:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                SAVED_RESULT_DIR = DRIVE_RESULTS_ROOT / f"{RUN_NAME}_{timestamp}"
                shutil.copytree(RUN_DIR, SAVED_RESULT_DIR, dirs_exist_ok=True)
                DRIVE_SAVE_COMPLETED = True
                print("Saved complete result bundle:", SAVED_RESULT_DIR)
            else:
                print("Drive is unavailable; local run remains at", RUN_DIR.resolve())
            """,
            "save-drive",
            ["google-drive-save"],
        ),
        markdown("## 8. 保存成功后自动断开 Colab", "disconnect-heading"),
        code(
            """
            AUTO_DISCONNECT = True
            if AUTO_DISCONNECT and DRIVE_SAVE_COMPLETED and Path("/content").exists():
                from google.colab import runtime
                print("Drive save verified; disconnecting this Colab runtime.")
                runtime.unassign()
            else:
                print("Auto-disconnect skipped.")
            """,
            "auto-disconnect",
            ["auto-disconnect"],
        ),
    ]


def build_notebook(version: str, spec: dict[str, str]) -> Path:
    notebook = {
        "cells": build_cells(version, spec),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"Trace_Count_{version}_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / f"Trace_Count_{version}_Colab.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    for version, spec in VERSIONS.items():
        print(build_notebook(version, spec))
