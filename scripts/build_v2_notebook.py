from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Trace_Count_v2_Colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        r'''
# Trace Count v2: fixed count-token type, quantity OOD

这个 notebook 跑一个更干净的 NiaH-like counting 实验：**count 的 token 种类固定，ID/OOD 只改变数量**。

核心设计：

- **ID:** 序列长度 `50, 100, 200`，needle 数量 `0-5`。
- **OOD:** 同样的序列长度，但 needle 数量 `6-10`。
- **答案格式:** 不再用 `<C0> ... <C10>` 这种不同 count token，而是用同一个 `<CNT>` 重复 `k` 次。例如 count=7 的答案是 `<ANS> <CNT> ... <CNT> <EOS>`。
- **think 模型:** source 后生成 `<Think> <TICK>, marker ... <Think> <ANS> <CNT>*k <EOS>`。
- **answer-only 模型:** source 后直接生成 `<ANS> <CNT>*k <EOS>`。
- **accuracy 定义:** 无论是否带 think，主指标都只看最终答案 span 里 `<CNT>` 的数量是否等于真实 count。

要回答的问题：

1. 带 think token 的模型和不带 think token 的模型，在 **ID / OOD** final count 上有什么差异？
2. hidden states 里有没有线性的 counter direction？哪种提取方法更好？
3. 用不同 count direction 做 steering，能不能改善 OOD counting？
4. think token 是否改变 attention 到 source needles 的分布，从而影响 targeted retrieval / counting？
        '''
    ),
    code(
        r'''
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime

REPO_URL = "https://github.com/Twist-Shan/Synthetic_CoT_NiaH_Count.git"
INSTALL_PACKAGE = True

IN_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if IN_COLAB:
    repo_dir = Path("/content/Synthetic_CoT_NiaH_Count")
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    os.chdir(repo_dir)

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if INSTALL_PACKAGE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

print("cwd:", ROOT)
print("python:", sys.executable)
print("platform:", platform.platform())
try:
    import torch

    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))
        '''
    ),
    md(
        r'''
## Runtime Settings

默认跑完整 v2 pipeline：生成两份可比数据、训练两个模型、评估 ID/OOD、fit probe/direction、做 OOD projection、做 generation steering、做 attention analysis。

如果只想补某一阶段，可以把 `PIPELINE_STAGE` 改成：

- `data`
- `train`
- `eval`
- `probe`
- `projection`
- `steering`
- `attention`
- `all`

`SKIP_COMPLETED = True` 时，已有且非空的结果会跳过，适合断点续跑。
        '''
    ),
    code(
        r'''
RUN_TESTS = True
EXPERIMENT_NAME = "trace_count_v2_seed0_full_colab"

MODEL_CONFIG = "configs/model/small_main.yaml"
MODEL_NAME = "small_main"
SEED = 0

DATA_ROOT = "data/trace_count_v2_seed0"
OUT_ROOT = f"runs/{EXPERIMENT_NAME}"

LENGTHS = "50,100,200"
ID_COUNTS = "0:5"
OOD_COUNTS = "6:10"
MAX_COUNT = 10
NOISE_VOCAB_SIZE = 64

EXAMPLES_PER_PAIR_TRAIN = 512
EXAMPLES_PER_PAIR_VAL = 128

MAX_STEPS = 10000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
EVAL_EVERY = 1000
SAVE_EVERY = 0
PROGRESS_EVERY = 100

EVAL_LIMIT = 2048
PROBE_LIMIT = 2048
PROJECTION_LIMIT = 2048
STEERING_LIMIT = 512
ATTENTION_LIMIT = 512

PIPELINE_STAGE = "all"
SKIP_COMPLETED = True
VARIANTS = "think_trace_repeat_count,answer_only_repeat_count"
STEERING_ALPHAS = "-6,-4,-2,-1,0,1,2,4,6"

DIRECTION_LAYERS = "all"
PROBE_LAYERS = "all"
PROBE_ANCHORS = "ans,think_close,source_marker,trace_index,trace_marker"
DIRECTION_ANCHORS = "ans,think_close,source_marker,trace_index,trace_marker"
DIRECTION_SPECS = (
    "layer_2:ans:total_count,layer_4:ans:total_count,"
    "layer_2:source_marker:running_count,layer_4:source_marker:running_count,"
    "layer_2:think_close:total_count,layer_4:think_close:total_count,"
    "layer_2:trace_marker:k,layer_4:trace_marker:k"
)
ATTENTION_SPLITS = "val_id,val_count_ood"
ATTENTION_QUERY_ANCHORS = "ans,think_close,count"

RESULT_DIR_OVERRIDE = None

settings = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "DATA_ROOT": DATA_ROOT,
    "OUT_ROOT": OUT_ROOT,
    "MODEL_CONFIG": MODEL_CONFIG,
    "MODEL_NAME": MODEL_NAME,
    "SEED": SEED,
    "LENGTHS": LENGTHS,
    "ID_COUNTS": ID_COUNTS,
    "OOD_COUNTS": OOD_COUNTS,
    "MAX_COUNT": MAX_COUNT,
    "MAX_STEPS": MAX_STEPS,
    "BATCH_SIZE": BATCH_SIZE,
    "PIPELINE_STAGE": PIPELINE_STAGE,
    "VARIANTS": VARIANTS,
    "EVAL_LIMIT": EVAL_LIMIT,
    "PROBE_LIMIT": PROBE_LIMIT,
    "PROJECTION_LIMIT": PROJECTION_LIMIT,
    "STEERING_LIMIT": STEERING_LIMIT,
    "ATTENTION_LIMIT": ATTENTION_LIMIT,
}
print(json.dumps(settings, indent=2, ensure_ascii=False))
        '''
    ),
    code(
        r'''
if RUN_TESTS:
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], check=True)
        '''
    ),
    md(
        r'''
## Run v2 Pipeline

这一格会顺序跑两个模型：

- `think_trace_repeat_count`: 带 think trace，trace 里每个 needle 用 `<TICK>, marker` 表示。
- `answer_only_repeat_count`: 不带 think trace，source 后直接输出最终 count。

两者 source 数据随机种子相同，所以 ID/OOD 比较更干净。训练目标都是 all-token next-token prediction (`full_sequence`)，不加 final-count 权重。
        '''
    ),
    code(
        r'''
cmd = [
    sys.executable,
    "-u",
    "scripts/run_v2_repeat_count.py",
    "--data_root",
    DATA_ROOT,
    "--out_root",
    OUT_ROOT,
    "--model_config",
    MODEL_CONFIG,
    "--model_name",
    MODEL_NAME,
    "--seed",
    str(SEED),
    "--lengths",
    LENGTHS,
    "--id_counts",
    ID_COUNTS,
    "--ood_counts",
    OOD_COUNTS,
    "--max_count",
    str(MAX_COUNT),
    "--noise_vocab_size",
    str(NOISE_VOCAB_SIZE),
    "--examples_per_pair_train",
    str(EXAMPLES_PER_PAIR_TRAIN),
    "--examples_per_pair_val",
    str(EXAMPLES_PER_PAIR_VAL),
    "--max_steps",
    str(MAX_STEPS),
    "--batch_size",
    str(BATCH_SIZE),
    "--learning_rate",
    str(LEARNING_RATE),
    "--warmup_steps",
    str(WARMUP_STEPS),
    "--eval_every",
    str(EVAL_EVERY),
    "--eval_limit",
    str(EVAL_LIMIT),
    "--probe_limit",
    str(PROBE_LIMIT),
    "--projection_limit",
    str(PROJECTION_LIMIT),
    "--steering_limit",
    str(STEERING_LIMIT),
    "--attention_limit",
    str(ATTENTION_LIMIT),
    "--save_every",
    str(SAVE_EVERY),
    "--progress_every",
    str(PROGRESS_EVERY),
    "--variants",
    VARIANTS,
    "--stage",
    PIPELINE_STAGE,
    "--probe_layers",
    PROBE_LAYERS,
    "--direction_layers",
    DIRECTION_LAYERS,
    "--probe_anchors",
    PROBE_ANCHORS,
    "--direction_anchors",
    DIRECTION_ANCHORS,
    "--projection_specs",
    DIRECTION_SPECS,
    "--steering_direction_specs",
    DIRECTION_SPECS,
    f"--steering_alphas={STEERING_ALPHAS}",
    "--attention_splits",
    ATTENTION_SPLITS,
    "--attention_query_anchors",
    ATTENTION_QUERY_ANCHORS,
]
if SKIP_COMPLETED:
    cmd.append("--skip_completed")

Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
pipeline_log = Path(OUT_ROOT) / "v2_pipeline.log"
print(" ".join(cmd), flush=True)
with pipeline_log.open("a", encoding="utf-8") as log:
    log.write("\n\n" + "=" * 100 + "\n")
    log.write(datetime.now().isoformat() + "\n")
    log.write(" ".join(cmd) + "\n")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    returncode = proc.wait()
if returncode != 0:
    raise subprocess.CalledProcessError(returncode, cmd)
print("Pipeline log:", pipeline_log)
        '''
    ),
    md(
        r'''
## Load Results

如果你是在 Colab 跑完后把结果下载回本地，只需要设置 `RESULT_DIR_OVERRIDE` 指向结果 bundle。否则这里默认读取当前 notebook 刚生成的 `data/...` 和 `runs/...`。
        '''
    ),
    code(
        r'''
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 130

VARIANT_INFO = {
    "think_trace_repeat_count": {
        "label": "think trace",
        "run_name": "think_trace_repeat_count_full_sequence_seed0",
        "description": "source + <Think> trace + <Think> + repeated <CNT> answer",
    },
    "answer_only_repeat_count": {
        "label": "answer only",
        "run_name": "answer_only_repeat_count_full_sequence_seed0",
        "description": "source + repeated <CNT> answer",
    },
}
REQUESTED_VARIANTS = [v.strip() for v in VARIANTS.split(",") if v.strip()]

if RESULT_DIR_OVERRIDE:
    RESULT_DIR = Path(RESULT_DIR_OVERRIDE)
    BASE_DATA = RESULT_DIR / "data"
    BASE_RUNS = RESULT_DIR / "runs"
else:
    RESULT_DIR = None
    BASE_DATA = Path(DATA_ROOT)
    BASE_RUNS = Path(OUT_ROOT)

DATA_DIRS = {v: BASE_DATA / v for v in REQUESTED_VARIANTS}
RUNS = {v: BASE_RUNS / MODEL_NAME / VARIANT_INFO[v]["run_name"] for v in REQUESTED_VARIANTS}

def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path, limit=None):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def safe_read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

missing = []
for variant in REQUESTED_VARIANTS:
    if not DATA_DIRS[variant].exists():
        missing.append(str(DATA_DIRS[variant]))
    if not RUNS[variant].exists():
        missing.append(str(RUNS[variant]))
if missing:
    display(Markdown("**Missing paths.** Run the pipeline cell first, or set `RESULT_DIR_OVERRIDE`.\n\n" + "\n".join(f"- `{p}`" for p in missing)))
else:
    display(Markdown("**Loaded v2 paths**"))
    display(pd.DataFrame(
        [
            {
                "variant": VARIANT_INFO[v]["label"],
                "data_dir": str(DATA_DIRS[v]),
                "run_dir": str(RUNS[v]),
                "format": VARIANT_INFO[v]["description"],
            }
            for v in REQUESTED_VARIANTS
        ]
    ))
        '''
    ),
    md(
        r'''
## Experiment Design Check

下面这张表确认数据设置。重点看 `count_range`：

- `val_id` 是训练分布内 count：`0-5`。
- `val_count_ood` 是数量外推 count：`6-10`。
- 两个模型使用相同 source marker 词表和相同 answer token 类型 `<CNT>`，所以 OOD 主要考察“数量/停止位置是否能外推”。
        '''
    ),
    code(
        r'''
dataset_rows = []
for variant, data_dir in DATA_DIRS.items():
    meta_path = data_dir / "dataset_metadata.json"
    if not meta_path.exists():
        continue
    meta = read_json(meta_path)
    for split, spec in meta["split_specs"].items():
        counts = list(spec["counts"])
        dataset_rows.append(
            {
                "variant": VARIANT_INFO[variant]["label"],
                "task_format": meta.get("task_format"),
                "split": split,
                "n_total": meta["split_counts"].get(split),
                "lengths": ", ".join(map(str, spec["lengths"])),
                "count_range": f"{min(counts)}-{max(counts)}",
                "count_values": ", ".join(map(str, counts)),
                "examples_per_pair": spec["examples_per_pair"],
            }
        )
dataset_df = pd.DataFrame(dataset_rows)
display(dataset_df)

display(Markdown(
    """
**坐标/组别说明。** 后面的 ID/OOD 图里，`ID` 指 `val_id`，也就是训练见过的 count 范围 `0-5`；`OOD` 指 `val_count_ood`，也就是没见过的数量范围 `6-10`。`think trace` 和 `answer only` 是两套独立训练的模型。
"""
))
        '''
    ),
    md(
        r'''
## Final-count Accuracy: Two Models, ID vs OOD

图里只评价最终 count：

- **横轴:** 模型类型，`think trace` 或 `answer only`。
- **纵轴:** final-count accuracy，预测的 `<CNT>` 重复次数是否等于真实 needle 数。
- **颜色:** split，蓝色是 ID (`0-5`)，红色是 OOD (`6-10`)。
- **teacher-forced:** 给模型真实前缀，只看下一步/answer span 的 count 读出能力。
- **autoregressive:** 只给 source，让模型自己生成 think/answer，然后统计最终 `<CNT>` 数量。
        '''
    ),
    code(
        r'''
eval_rows = []
for variant, run_dir in RUNS.items():
    summary_path = run_dir / "eval" / "summary_metrics.json"
    if not summary_path.exists():
        continue
    summary = read_json(summary_path)
    for split, split_metrics in summary.items():
        for mode in ["teacher_forced", "autoregressive"]:
            metrics = split_metrics.get(mode)
            if not metrics:
                continue
            eval_rows.append(
                {
                    "variant": VARIANT_INFO[variant]["label"],
                    "variant_key": variant,
                    "split": split,
                    "split_label": "ID: count 0-5" if split == "val_id" else "OOD: count 6-10",
                    "mode": mode,
                    "count_accuracy": metrics.get("count_accuracy"),
                    "mae": metrics.get("mean_absolute_error"),
                    "invalid_answer_rate": metrics.get("invalid_answer_rate"),
                    "undercount_rate": metrics.get("undercount_rate"),
                    "overcount_rate": metrics.get("overcount_rate"),
                    "trace_exact_match": metrics.get("trace_exact_match"),
                    "format_validity": metrics.get("format_validity"),
                }
            )
eval_df = pd.DataFrame(eval_rows)
display(eval_df.sort_values(["mode", "variant", "split"]))

if not eval_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    palette = {"ID: count 0-5": "#2f6fed", "OOD: count 6-10": "#e03131"}
    for ax, mode in zip(axes, ["teacher_forced", "autoregressive"]):
        plot_df = eval_df[eval_df["mode"] == mode]
        sns.barplot(
            data=plot_df,
            x="variant",
            y="count_accuracy",
            hue="split_label",
            palette=palette,
            ax=ax,
        )
        ax.set_title("Teacher-forced final count" if mode == "teacher_forced" else "Autoregressive final count")
        ax.set_xlabel("model")
        ax.set_ylabel("final-count accuracy" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.legend(title="split")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    fig.suptitle("V2 main comparison: fixed count-token type, ID vs OOD", y=1.03, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
        '''
    ),
    md(
        r'''
## Accuracy by Count

这张图看模型是不是只在训练范围内会数，还是能外推到更大的数量：

- **横轴:** 真实 needle 数量。
- **纵轴:** final-count accuracy。
- **线条颜色:** 两个模型。
- **线型:** teacher-forced / autoregressive。
- 竖虚线右侧是 OOD 数量区间 `6-10`。
        '''
    ),
    code(
        r'''
pred_rows = []
for variant, run_dir in RUNS.items():
    for split in ["val_id", "val_count_ood"]:
        pred_path = run_dir / "eval" / f"predictions_{split}.jsonl"
        for row in read_jsonl(pred_path):
            base = {
                "variant": VARIANT_INFO[variant]["label"],
                "variant_key": variant,
                "split": split,
                "split_label": "ID: count 0-5" if split == "val_id" else "OOD: count 6-10",
                "true_count": int(row["true_count"]),
                "seq_len": int(row["seq_len"]),
            }
            if "tf_pred_count" in row:
                pred_rows.append(
                    {
                        **base,
                        "mode": "teacher_forced",
                        "pred_count": row.get("tf_pred_count"),
                        "correct": bool(row.get("tf_correct", False)),
                        "format_valid": True,
                    }
                )
            if "ar_pred_count" in row:
                pred = row.get("ar_pred_count")
                pred_rows.append(
                    {
                        **base,
                        "mode": "autoregressive",
                        "pred_count": pred,
                        "correct": pred == row["true_count"],
                        "format_valid": bool(row.get("ar_format_valid", False)),
                    }
                )

pred_df = pd.DataFrame(pred_rows)
if pred_df.empty:
    display(Markdown("No prediction rows found yet."))
else:
    by_count = (
        pred_df.groupby(["variant", "mode", "split", "true_count"], as_index=False)
        .agg(accuracy=("correct", "mean"), format_validity=("format_valid", "mean"), n=("correct", "size"))
        .sort_values(["variant", "mode", "true_count"])
    )
    display(by_count)
    plt.figure(figsize=(10, 4.6))
    sns.lineplot(
        data=by_count,
        x="true_count",
        y="accuracy",
        hue="variant",
        style="mode",
        markers=True,
        dashes=True,
    )
    plt.axvline(5.5, color="black", linestyle="--", linewidth=1)
    plt.text(2.5, 1.03, "ID counts 0-5", ha="center", va="bottom", fontsize=9)
    plt.text(8.0, 1.03, "OOD counts 6-10", ha="center", va="bottom", fontsize=9)
    plt.ylim(-0.03, 1.08)
    plt.xlabel("true number of needles")
    plt.ylabel("final-count accuracy")
    plt.title("Final-count accuracy by true count")
    plt.legend(title="model / decoding")
    plt.tight_layout()
    plt.show()
        '''
    ),
    md(
        r'''
## Training Curves

下面展示训练过程：

- **横轴:** training step。
- **纵轴:** loss。
- **颜色:** loss segment 或模型。
- `total_weighted_loss` 是训练目标的总体 loss；本实验不加 final-count 权重，所以它基本就是 all-token next-token prediction loss。
- `count_loss` 是 `<CNT>` answer span 的 loss。
- `trace_index_loss` 是 think trace 里的 `<TICK>` loss。
- `trace_marker_loss` 是 think trace 里需要复述 source marker 的 loss。
        '''
    ),
    code(
        r'''
train_rows = []
for variant, run_dir in RUNS.items():
    for row in read_jsonl(run_dir / "train_log.jsonl"):
        row = dict(row)
        row["variant"] = VARIANT_INFO[variant]["label"]
        row["variant_key"] = variant
        train_rows.append(row)
train_df = pd.DataFrame(train_rows)

if train_df.empty:
    display(Markdown("No training log rows found yet."))
else:
    display(train_df.tail(10))
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=train_df, x="step", y="total_weighted_loss", hue="variant")
    plt.xlabel("training step")
    plt.ylabel("total weighted loss")
    plt.title("Training loss by model")
    plt.tight_layout()
    plt.show()

    segment_cols = [c for c in ["count_loss", "trace_index_loss", "trace_marker_loss", "answer_prefix_loss", "eos_loss"] if c in train_df.columns]
    if segment_cols:
        seg_df = train_df.melt(id_vars=["step", "variant"], value_vars=segment_cols, var_name="segment", value_name="loss")
        plt.figure(figsize=(11, 4.6))
        sns.lineplot(data=seg_df, x="step", y="loss", hue="segment", style="variant")
        plt.xlabel("training step")
        plt.ylabel("loss")
        plt.title("Segment losses")
        plt.tight_layout()
        plt.show()
        '''
    ),
    md(
        r'''
## Probe and Count Direction Extraction

这里比较几种 count direction 的提取方法。它们都用 `val_id` hidden states 训练线性 probe / ridge direction：

- `ans:total_count`: 在 `<ANS>` 位置读出总 count。
- `source_marker:running_count`: 在每个 source needle 位置读出 running count，也就是第几个 needle。
- `think_close:total_count`: 在第二个 `<Think>` 位置读出总 count，只适用于 think 模型。
- `trace_marker:k`: 在 trace 中每个 marker 位置读出第几个 needle，只适用于 think 模型。

图里：

- **横轴:** direction/probe 方法。
- **纵轴:** ridge `R²`，越高代表这个 hidden-state 位置越线性地编码目标 count。
- **颜色:** 模型类型。
        '''
    ),
    code(
        r'''
probe_rows = []
direction_rows = []
for variant, run_dir in RUNS.items():
    p = safe_read_csv(run_dir / "probes" / "probe_summary.csv")
    if not p.empty:
        p["variant"] = VARIANT_INFO[variant]["label"]
        p["variant_key"] = variant
        probe_rows.append(p)
    d = safe_read_csv(run_dir / "directions" / "direction_summary.csv")
    if not d.empty:
        d["variant"] = VARIANT_INFO[variant]["label"]
        d["variant_key"] = variant
        direction_rows.append(d)

probe_df = pd.concat(probe_rows, ignore_index=True) if probe_rows else pd.DataFrame()
direction_df = pd.concat(direction_rows, ignore_index=True) if direction_rows else pd.DataFrame()

if probe_df.empty:
    display(Markdown("No probe summary found yet."))
else:
    display(Markdown("**Top ridge probes**"))
    top_probe = (
        probe_df[probe_df["probe_type"] == "ridge"]
        .sort_values("r2", ascending=False)
        .head(20)
    )
    display(top_probe)

if direction_df.empty:
    display(Markdown("No direction summary found yet."))
else:
    direction_df["ok_bool"] = direction_df["ok"].astype(str).str.lower().eq("true")
    direction_df["method"] = direction_df["layer"] + ":" + direction_df["anchor"] + ":" + direction_df["target"]
    rel = direction_df[direction_df["ok_bool"]].copy()
    rel = rel[rel["method"].isin([m.strip() for m in DIRECTION_SPECS.split(",") if m.strip()])]
    display(Markdown("**Direction extraction summary used for steering/projection**"))
    display(rel.sort_values(["variant", "target", "anchor", "layer"]))

    plt.figure(figsize=(12, 4.8))
    sns.barplot(data=rel, x="method", y="r2", hue="variant")
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("direction method = layer:anchor:target")
    plt.ylabel("ridge R² on ID")
    plt.title("Which extracted count directions are most linear on ID?")
    plt.tight_layout()
    plt.show()
        '''
    ),
    md(
        r'''
## Direction Projection on OOD Counts

这一步不是改模型，而是把 OOD hidden states 投影到 ID 学到的 count direction 上，检查 direction 是否外推：

- **横轴:** OOD true count (`6-10`)。
- **纵轴:** ridge direction 预测出的 count / running count。
- **颜色:** direction 方法。
- **列:** 模型类型。

如果线大致跟 `y=x` 同步上升，说明这个 direction 至少在表征空间里对 OOD 数量有外推趋势。
        '''
    ),
    code(
        r'''
projection_rows = []
for variant, run_dir in RUNS.items():
    ex = safe_read_csv(run_dir / "direction_projection" / "direction_projection_examples.csv")
    if not ex.empty:
        ex["variant"] = VARIANT_INFO[variant]["label"]
        ex["variant_key"] = variant
        ex["method"] = ex["layer"] + ":" + ex["anchor"] + ":" + ex["target"]
        projection_rows.append(ex)
projection_df = pd.concat(projection_rows, ignore_index=True) if projection_rows else pd.DataFrame()

if projection_df.empty:
    display(Markdown("No direction projection examples found yet."))
else:
    display(projection_df.head())
    plot_methods = [m.strip() for m in DIRECTION_SPECS.split(",") if m.strip()]
    proj_plot = projection_df[projection_df["method"].isin(plot_methods)].copy()
    proj_summary = (
        proj_plot.groupby(["variant", "method", "true_count"], as_index=False)
        .agg(mean_pred=("pred_value", "mean"), mae=("pred_value", lambda x: np.mean(np.abs(x - proj_plot.loc[x.index, "target_value"]))))
    )
    g = sns.FacetGrid(proj_summary, col="variant", height=4, aspect=1.25, sharey=False)
    g.map_dataframe(sns.lineplot, x="true_count", y="mean_pred", hue="method", marker="o")
    for ax in g.axes.flat:
        lo, hi = int(proj_summary["true_count"].min()), int(proj_summary["true_count"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y=x")
        ax.set_xlabel("OOD true count")
        ax.set_ylabel("mean projected count value")
    g.add_legend(title="direction method")
    g.fig.suptitle("OOD projection onto ID-fitted count directions", y=1.05, fontsize=13, fontweight="bold")
    plt.show()
        '''
    ),
    md(
        r'''
## Generation Steering

这一步把 ID 上学到的 count direction 加到 autoregressive generation 的 hidden state 中，只在模型已经生成 `<ANS>` 之后 steering。

图里：

- **横轴:** steering strength `alpha`。
- **纵轴:** OOD final-count accuracy 或平均预测 count。
- **颜色:** direction 方法。
- **列:** 模型类型。

如果某条线在 `alpha != 0` 时比 `alpha = 0` 更高，说明该 direction 对 OOD counting 有可干预的因果效果。
        '''
    ),
    code(
        r'''
steer_rows = []
for variant, run_dir in RUNS.items():
    s = safe_read_csv(run_dir / "generation_steering" / "generation_steering_summary.csv")
    if not s.empty:
        s["variant"] = VARIANT_INFO[variant]["label"]
        s["variant_key"] = variant
        steer_rows.append(s)
steer_df = pd.concat(steer_rows, ignore_index=True) if steer_rows else pd.DataFrame()

if steer_df.empty:
    display(Markdown("No steering summary found yet."))
else:
    display(steer_df.sort_values(["variant", "method", "alpha"]))
    best = (
        steer_df.sort_values(["variant", "accuracy", "format_validity"], ascending=[True, False, False])
        .groupby("variant", as_index=False)
        .head(5)
    )
    display(Markdown("**Best steering settings by OOD accuracy**"))
    display(best[["variant", "method", "alpha", "accuracy", "mae", "mean_pred_count", "format_validity"]])

    g = sns.FacetGrid(steer_df, col="variant", height=4.2, aspect=1.25, sharey=True)
    g.map_dataframe(sns.lineplot, x="alpha", y="accuracy", hue="method", marker="o")
    for ax in g.axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("steering alpha")
        ax.set_ylabel("OOD final-count accuracy")
    g.add_legend(title="direction method")
    g.fig.suptitle("Generation steering on OOD counts", y=1.05, fontsize=13, fontweight="bold")
    plt.show()

    g = sns.FacetGrid(steer_df, col="variant", height=4.2, aspect=1.25, sharey=True)
    g.map_dataframe(sns.lineplot, x="alpha", y="mean_pred_count", hue="method", marker="o")
    for ax in g.axes.flat:
        ax.axhspan(6, 10, color="#e03131", alpha=0.08, label="OOD target range")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("steering alpha")
        ax.set_ylabel("mean predicted count")
    g.add_legend(title="direction method")
    g.fig.suptitle("Does steering move the generated count upward/downward?", y=1.05, fontsize=13, fontweight="bold")
    plt.show()
        '''
    ),
    md(
        r'''
## Attention Analysis

这里看模型在 answer/think/count 位置是否把 attention 集中到 source needles 上：

- **横轴:** transformer layer。
- **纵轴 1:** `marker_enrichment`，source needle 单 token 平均 attention / source noise 单 token 平均 attention。大于 1 表示更偏向 needle。
- **纵轴 2:** `top_source_marker_rate`，source 里 attention 最大的位置是不是 needle。
- **颜色:** split，ID 或 OOD。
- **列:** 模型类型和 query anchor。

这不是最终性能指标，而是解释 targeted retrieval 的辅助证据：think token 如果有用，可能表现为 answer 前后更稳定地关注 source needles。
        '''
    ),
    code(
        r'''
attention_rows = []
missing_attention = []
for variant, run_dir in RUNS.items():
    a = safe_read_csv(run_dir / "attention" / "attention_summary.csv")
    if a.empty:
        missing_attention.append({"variant": VARIANT_INFO[variant]["label"], "path": str(run_dir / "attention" / "attention_summary.csv")})
    else:
        a["variant"] = VARIANT_INFO[variant]["label"]
        a["variant_key"] = variant
        a["split_label"] = a["split"].map({"val_id": "ID: count 0-5", "val_count_ood": "OOD: count 6-10"}).fillna(a["split"])
        attention_rows.append(a)

attention_df = pd.concat(attention_rows, ignore_index=True) if attention_rows else pd.DataFrame()
if missing_attention:
    display(Markdown("**Some attention summaries are missing or empty.**"))
    display(pd.DataFrame(missing_attention))

if attention_df.empty:
    display(Markdown("No attention rows found. Rerun with `PIPELINE_STAGE = 'attention'` and `SKIP_COMPLETED = True`."))
else:
    attn_agg = (
        attention_df.groupby(["variant", "split_label", "query_anchor", "layer"], as_index=False)
        .agg(
            marker_enrichment=("marker_enrichment", "mean"),
            top_source_marker_rate=("top_source_marker_rate", "mean"),
            source_mass=("source_mass", "mean"),
            marker_mass=("marker_mass", "mean"),
            noise_mass=("noise_mass", "mean"),
        )
    )
    display(attn_agg.head())
    for metric, ylabel in [
        ("marker_enrichment", "needle attention enrichment"),
        ("top_source_marker_rate", "top source token is needle rate"),
    ]:
        g = sns.FacetGrid(attn_agg, row="query_anchor", col="variant", height=3.2, aspect=1.25, sharey=False)
        g.map_dataframe(sns.lineplot, x="layer", y=metric, hue="split_label", marker="o")
        for ax in g.axes.flat:
            ax.set_xlabel("transformer layer")
            ax.set_ylabel(ylabel)
        g.add_legend(title="split")
        g.fig.suptitle(f"Attention analysis: {metric}", y=1.02, fontsize=13, fontweight="bold")
        plt.show()
        '''
    ),
    md(
        r'''
## Chinese Analysis Summary

下面这个 cell 会根据当前结果自动生成中文总结。它会先写明模型和数据设置，再分别回答：

1. ID/OOD final count 是否成功；
2. think token 是否有帮助；
3. probe/direction 是否显示 counter；
4. steering 是否能改善 OOD；
5. attention 是否支持 targeted retrieval 的解释。
        '''
    ),
    code(
        r'''
def _metric(df, variant, split, mode, column):
    if df.empty:
        return None
    sub = df[(df["variant"] == variant) & (df["split"] == split) & (df["mode"] == mode)]
    if sub.empty:
        return None
    value = sub.iloc[0].get(column)
    return None if pd.isna(value) else float(value)

def _fmt(value):
    return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.3f}"

analysis_lines = []
analysis_lines.append("### 实验设置")
analysis_lines.append(f"- 模型：`{MODEL_NAME}`，配置 `{MODEL_CONFIG}`，seed `{SEED}`，训练 `{MAX_STEPS}` steps，batch size `{BATCH_SIZE}`。")
analysis_lines.append("- 训练目标：all-token next-token prediction (`full_sequence`)，没有 final-count 加权。")
analysis_lines.append(f"- 数据：source 长度 `{LENGTHS}`；ID count `{ID_COUNTS}`；OOD count `{OOD_COUNTS}`；positive marker 词表固定为 `X/Y/Z`。")
analysis_lines.append("- v2 的关键控制：答案统一用 `<CNT>` 重复 k 次表示 count，think trace 统一用 `<TICK>` 表示每一次计数，因此 ID/OOD 的主要差异是数量，不是新 token 类别。")

if not eval_df.empty:
    analysis_lines.append("\n### 1. 两个模型的 ID/OOD final-count 表现")
    for variant_label in [VARIANT_INFO[v]["label"] for v in REQUESTED_VARIANTS]:
        id_ar = _metric(eval_df, variant_label, "val_id", "autoregressive", "count_accuracy")
        ood_ar = _metric(eval_df, variant_label, "val_count_ood", "autoregressive", "count_accuracy")
        id_tf = _metric(eval_df, variant_label, "val_id", "teacher_forced", "count_accuracy")
        ood_tf = _metric(eval_df, variant_label, "val_count_ood", "teacher_forced", "count_accuracy")
        analysis_lines.append(
            f"- `{variant_label}`: autoregressive ID={_fmt(id_ar)}, OOD={_fmt(ood_ar)}；teacher-forced ID={_fmt(id_tf)}, OOD={_fmt(ood_tf)}。"
        )
    analysis_lines.append("- 这里的 accuracy 只看最终 `<CNT>` 数量，不要求 think trace 完全正确。若 teacher-forced 高但 autoregressive 低，说明内部读出 count 可以，但自由生成格式/停止机制失败；若两者 OOD 都低，说明 count 表征本身也没有外推。")

if not direction_df.empty:
    analysis_lines.append("\n### 2. Probe / count direction")
    ok_dir = direction_df[direction_df["ok"].astype(str).str.lower().eq("true")].copy()
    if not ok_dir.empty:
        ok_dir["method"] = ok_dir["layer"] + ":" + ok_dir["anchor"] + ":" + ok_dir["target"]
        top = ok_dir.sort_values("r2", ascending=False).head(5)
        for _, row in top.iterrows():
            analysis_lines.append(f"- `{row['variant']}` 的 `{row['method']}`: R²={_fmt(float(row['r2']))}, MAE={_fmt(float(row['mae']))}。")
        analysis_lines.append("- 如果 `source_marker:running_count` 或 `trace_marker:k` R² 高，说明模型在每个 needle/trace 位置形成了局部递增 counter；如果 `ans:total_count` 或 `think_close:total_count` R² 高，说明最终读出处聚合了总数。")

if not steer_df.empty:
    analysis_lines.append("\n### 3. Steering")
    zero = steer_df[np.isclose(steer_df["alpha"], 0.0)]
    best = steer_df.sort_values(["variant", "accuracy", "format_validity"], ascending=[True, False, False]).groupby("variant").head(1)
    for _, row in best.iterrows():
        z = zero[zero["variant"] == row["variant"]]
        baseline = None if z.empty else float(z["accuracy"].max())
        delta = None if baseline is None else float(row["accuracy"]) - baseline
        analysis_lines.append(
            f"- `{row['variant']}` 最好 steering 是 `{row['method']}` alpha={row['alpha']}: OOD acc={_fmt(float(row['accuracy']))}, baseline(alpha=0)={_fmt(baseline)}, delta={_fmt(delta)}。"
        )
    analysis_lines.append("- 如果 steering 只改变 mean predicted count 但不提高 accuracy，说明 direction 能推动数量偏置，但还没有精确控制停止位置。")

if not attention_df.empty:
    analysis_lines.append("\n### 4. Attention / targeted retrieval")
    attn_best = attention_df.sort_values("marker_enrichment", ascending=False).head(5)
    for _, row in attn_best.iterrows():
        analysis_lines.append(
            f"- `{row['variant']}` split `{row['split']}` query `{row['query_anchor']}` layer {row['layer']} head {row['head']}: marker_enrichment={_fmt(float(row['marker_enrichment']))}, top_marker_rate={_fmt(float(row['top_source_marker_rate']))}。"
        )
    analysis_lines.append("- `marker_enrichment > 1` 表示该 query 位置更偏向 source needles 而不是 noise。若 think 模型在 OOD 仍保持较高 enrichment，但 OOD accuracy 低，瓶颈更可能是 counting/termination；若 enrichment 本身掉了，瓶颈更像 retrieval。")

analysis_lines.append("\n### 下一步建议")
analysis_lines.append("- 如果 v2 OOD 仍差，可以把 OOD 进一步拆成 interpolation (`0-5` 内不同长度) 和 extrapolation (`6-10`)；同时增加 curriculum：先训练 `0-3`，再扩到 `0-5`，看 counter 是否更可外推。")
analysis_lines.append("- 如果 probe R² 高但 steering 效果弱，建议尝试 activation patching 或者在 answer span 的每一步用不同 alpha schedule，而不是固定 alpha。")
analysis_lines.append("- 如果 answer-only 和 think trace 的差距小，说明当前 think trace 可能只是额外监督而没有形成可用算法；可以把 trace 改成显式 running total，例如 `<TICK> <CNT>` 前缀累积。")

display(Markdown("\n".join(analysis_lines)))
        '''
    ),
    md(
        r'''
## Save Result Bundle to Google Drive / Local `colab_results`

这个 cell 会把 data、runs、当前 notebook 和关键配置打包到结果目录。

Colab 默认保存到：

`/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results/`

本地默认保存到：

`colab_results/`
        '''
    ),
    code(
        r'''
SAVE_RESULTS = True
DRIVE_RESULTS_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/CoT_Counting/Synthetic_CoT_NiaH_Count/colab_results")
LOCAL_RESULTS_ROOT = Path("colab_results")

if SAVE_RESULTS:
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")
        results_root = DRIVE_RESULTS_ROOT
    else:
        results_root = LOCAL_RESULTS_ROOT

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = results_root / f"{EXPERIMENT_NAME}_{MODEL_NAME}_steps{MAX_STEPS}_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    def copy_dir(src, dst):
        src = Path(src)
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    copy_dir(DATA_ROOT, bundle_dir / "data")
    copy_dir(OUT_ROOT, bundle_dir / "runs")
    (bundle_dir / "notebooks").mkdir(exist_ok=True)
    notebook_src = Path("notebooks/Trace_Count_v2_Colab.ipynb")
    if notebook_src.exists():
        shutil.copy2(notebook_src, bundle_dir / "notebooks" / notebook_src.name)
    shutil.copy2("scripts/run_v2_repeat_count.py", bundle_dir / "run_v2_repeat_count.py")
    with (bundle_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(settings | {"saved_at": timestamp, "repo": REPO_URL}, f, indent=2, ensure_ascii=False)
    print("Saved result bundle:", bundle_dir)
        '''
    ),
    md(
        r'''
## Commit / Push Notebook and Code to GitHub

这个 cell 默认不会执行。确认结果和 notebook 没问题后，把 `PUSH_TO_GITHUB = True`。

默认只提交代码、notebook 和测试，不提交大型 `runs/` / `data/` 结果目录。结果建议放 Google Drive。
        '''
    ),
    code(
        r'''
PUSH_TO_GITHUB = False
ADD_RESULTS_TO_GIT = False
COMMIT_MESSAGE = "Add Trace Count v2 repeated-count notebook"

if PUSH_TO_GITHUB:
    paths = [
        "notebooks/Trace_Count_v2_Colab.ipynb",
        "scripts/run_v2_repeat_count.py",
        "src/trace_counting",
        "tests",
        "README.md",
        "pyproject.toml",
    ]
    if ADD_RESULTS_TO_GIT:
        paths += [DATA_ROOT, OUT_ROOT]
    subprocess.run(["git", "status", "--short"], check=False)
    subprocess.run(["git", "add", *paths], check=True)
    subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=False)
    subprocess.run(["git", "push"], check=True)
else:
    print("PUSH_TO_GITHUB is False; no git command was run.")
        '''
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
