from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable


DEFAULT_RUN_DIR = Path("colab_results/v6_separator_trace_main_seed1234_20260708_175919")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def mean(values: Iterable[object]) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return fmean(vals) if vals else math.nan


def weighted_mean(rows: list[dict[str, str]], metric: str, weight_col: str = "n_examples") -> float:
    total = 0.0
    weight = 0.0
    for row in rows:
        val = as_float(row.get(metric))
        if math.isnan(val):
            continue
        w = as_float(row.get(weight_col))
        if math.isnan(w) or w <= 0:
            w = 1.0
        total += val * w
        weight += w
    return total / weight if weight else math.nan


def group_weighted(
    rows: list[dict[str, str]],
    keys: list[str],
    metrics: list[str],
    weight_col: str = "n_examples",
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, object]] = []
    for key, vals in grouped.items():
        item: dict[str, object] = dict(zip(keys, key))
        item["n_examples"] = sum(
            as_float(v.get(weight_col)) if not math.isnan(as_float(v.get(weight_col))) else 1.0
            for v in vals
        )
        for metric in metrics:
            item[metric] = weighted_mean(vals, metric, weight_col)
        out.append(item)
    return out


def fmt(value: object, digits: int = 3) -> str:
    val = as_float(value)
    if math.isnan(val):
        return html_lib.escape(str(value))
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    val = as_float(value)
    if math.isnan(val):
        return "n/a"
    return f"{100 * val:.{digits}f}%"


def code(text: object) -> str:
    return f"<code>{html_lib.escape(str(text))}</code>"


def table_html(rows: list[dict[str, object]], columns: list[tuple[str, str, str]]) -> str:
    header = "".join(f"<th>{html_lib.escape(label)}</th>" for _key, label, _kind in columns)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _label, kind in columns:
            value = row.get(key, "")
            if kind == "pct":
                text = pct(value)
            elif kind == "num":
                text = fmt(value)
            elif kind == "sci":
                text = fmt(value, 2)
            elif kind == "int":
                val = as_float(value)
                text = "n/a" if math.isnan(val) else str(int(round(val)))
            else:
                text = html_lib.escape(str(value))
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(f"<tr><td colspan=\"{len(columns)}\">No data</td></tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(run_dir: Path, rel_path: str, title: str, caption: str) -> str:
    path = run_dir / rel_path
    if not path.exists():
        return ""
    return f"""
    <figure class="figure">
      <h4>{html_lib.escape(title)}</h4>
      <img src="{image_uri(path)}" alt="{html_lib.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def figure_grid(items: list[str], columns: int = 2) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    cls = "grid three" if columns == 3 else "grid two"
    return f"<div class=\"{cls}\">{''.join(items)}</div>"


def jsonl_stats(path: Path) -> dict[str, object]:
    counts: Counter[int] = Counter()
    repeated: Counter[int] = Counter()
    total = 0
    repeated_total = 0
    if not path.exists():
        return {"total": 0, "repeated_rate": math.nan, "counts": counts, "repeated_by_count": {}}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            count = int(obj["count"])
            total += 1
            counts[count] += 1
            is_repeated = len(set(obj["needle_markers"])) < len(obj["needle_markers"])
            repeated_total += int(is_repeated)
            repeated[count] += int(is_repeated)
    repeated_by_count = {
        count: repeated[count] / counts[count]
        for count in sorted(counts)
        if counts[count]
    }
    return {
        "total": total,
        "repeated_rate": repeated_total / total if total else math.nan,
        "counts": counts,
        "repeated_by_count": repeated_by_count,
    }


def final_step(rows: list[dict[str, str]]) -> int:
    steps = [int(as_float(r.get("step"))) for r in rows if not math.isnan(as_float(r.get("step")))]
    return max(steps) if steps else 0


def first_solved_step(eval_rows: list[dict[str, str]], model_type: str, eval_mode: str, require_trace: bool = False) -> int | None:
    steps = sorted({int(as_float(r["step"])) for r in eval_rows if not math.isnan(as_float(r.get("step")))})
    for step in steps:
        sub = [
            r
            for r in eval_rows
            if int(as_float(r.get("step"))) == step
            and r.get("model_type") == model_type
            and r.get("eval_mode") == eval_mode
        ]
        if not sub:
            continue
        acc = weighted_mean(sub, "accuracy")
        trace = weighted_mean(sub, "trace_exact_match_rate")
        if acc >= 0.999 and (not require_trace or trace >= 0.999):
            return step
    return None


def aggregate_attention(path: Path) -> dict[tuple[str, ...], dict[str, float]]:
    metrics = [
        "correct_top1_rate",
        "diagonal_dominance",
        "needle_attention_mass",
        "top_n_retrieval_recall",
        "ans_to_all_needles_mass",
        "attention_entropy_over_prompt_body",
        "entropy_over_prompt_body",
    ]
    sums: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("model_type", ""),
                row.get("query_anchor", ""),
                row.get("subset", ""),
                row.get("layer", ""),
                row.get("head", ""),
            )
            for metric in metrics:
                val = as_float(row.get(metric))
                if math.isnan(val):
                    continue
                sums[key][metric] += val
                counts[key][metric] += 1
    out: dict[tuple[str, ...], dict[str, float]] = {}
    for key, metric_sums in sums.items():
        item: dict[str, float] = {}
        item["n_queries"] = max(counts[key].values(), default=0)
        for metric, total in metric_sums.items():
            item[metric] = total / counts[key][metric]
        out[key] = item
    return out


def selected_attention_rows(attn: dict[tuple[str, ...], dict[str, float]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sep_rows: list[dict[str, object]] = []
    for subset in ["all_examples", "repeated_marker_examples", "unique_marker_examples"]:
        key = ("thinking_sep_trace", "sep_token_k", subset, "3", "2")
        if key in attn:
            sep_rows.append({"model": "thinking_sep_trace", "query": "sep_token_k", "subset": subset, "layer": 3, "head": 2, **attn[key]})
    non_rows: list[dict[str, object]] = []
    for head in ["0", "1", "2", "3"]:
        key = ("non_thinking", "ans_token", "all_examples", "1", head)
        if key in attn:
            non_rows.append({"model": "non_thinking", "query": "ans_token", "subset": "all_examples", "layer": 1, "head": int(head), **attn[key]})
    non_rows = sorted(non_rows, key=lambda r: (as_float(r.get("top_n_retrieval_recall")), as_float(r.get("ans_to_all_needles_mass"))), reverse=True)
    return sep_rows, non_rows


def probe_summary_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    final_focus = [
        r
        for r in rows
        if r.get("label_type") == "final_count"
        and r.get("anchor_type") in {"ans_token", "last_prompt_token", "think_start", "think_end", "pre_ans_token"}
        and r.get("layer") in {"0", "3"}
    ]
    final_focus = sorted(final_focus, key=lambda r: (r.get("model_type", ""), r.get("anchor_type", ""), as_float(r.get("layer"))))
    final_out = []
    for r in final_focus:
        final_out.append(
            {
                "model_type": r.get("model_type", ""),
                "anchor_type": r.get("anchor_type", ""),
                "layer": r.get("layer", ""),
                "probe_accuracy": as_float(r.get("probe_accuracy")),
                "position_only_accuracy": as_float(r.get("position_only_accuracy")),
                "trace_length_only_accuracy": as_float(r.get("trace_length_only_accuracy")),
                "probe_minus_position": as_float(r.get("probe_accuracy")) - as_float(r.get("position_only_accuracy")),
                "probe_r2": as_float(r.get("probe_r2")),
            }
        )

    prefix_focus = [
        r
        for r in rows
        if r.get("model_type") == "thinking_sep_trace"
        and r.get("label_type") == "prefix_count"
        and r.get("anchor_type") in {"sep_token_k", "marker_token_k", "pre_sep_k", "post_marker_k"}
        and r.get("layer") in {"0", "3"}
    ]
    prefix_focus = sorted(prefix_focus, key=lambda r: (r.get("anchor_type", ""), as_float(r.get("layer"))))
    prefix_out = []
    for r in prefix_focus:
        prefix_out.append(
            {
                "anchor_type": r.get("anchor_type", ""),
                "layer": r.get("layer", ""),
                "probe_accuracy": as_float(r.get("probe_accuracy")),
                "position_only_accuracy": as_float(r.get("position_only_accuracy")),
                "trace_length_only_accuracy": as_float(r.get("trace_length_only_accuracy")),
                "token_id_only_accuracy": as_float(r.get("token_id_only_accuracy")),
                "probe_minus_position": as_float(r.get("probe_accuracy")) - as_float(r.get("position_only_accuracy")),
            }
        )
    return final_out, prefix_out


def build_report(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    vocab_size = manifest.get("vocab_size", 91)

    final_bin = read_rows(run_dir / "metrics_final_test_by_bin.csv") or read_rows(run_dir / "metrics" / "metrics_final_test_by_bin.csv")
    final_count = read_rows(run_dir / "metrics_final_test_by_count.csv") or read_rows(run_dir / "metrics" / "metrics_final_test_by_count.csv")
    eval_bin = read_rows(run_dir / "metrics_eval_by_bin.csv") or read_rows(run_dir / "metrics" / "metrics_eval_by_bin.csv")
    train_rows = read_rows(run_dir / "metrics_train.csv") or read_rows(run_dir / "metrics" / "metrics_train.csv")
    probe_rows = read_rows(run_dir / "probes" / "probe_metrics.csv")
    attn = aggregate_attention(run_dir / "attention" / "attention_metrics.csv")
    sep_attn_rows, non_attn_rows = selected_attention_rows(attn)
    final_probe_rows, prefix_probe_rows = probe_summary_rows(probe_rows)

    final_metrics = [
        "accuracy",
        "eval_completion_loss",
        "eval_trace_loss",
        "eval_final_answer_loss",
        "mae",
        "trace_exact_match_rate",
        "trace_marker_precision",
        "trace_marker_recall",
        "trace_delimiter_count_accuracy",
        "premature_close_rate",
        "missing_close_rate",
        "ans_generated_rate",
        "think_close_generated_rate",
    ]
    final_overall = sorted(
        group_weighted(final_bin, ["model_type", "eval_mode"], final_metrics),
        key=lambda r: (str(r["model_type"]), str(r["eval_mode"])),
    )
    final_by_bin = sorted(
        group_weighted(final_bin, ["model_type", "eval_mode", "count_bin"], final_metrics),
        key=lambda r: (
            str(r["model_type"]),
            str(r["eval_mode"]),
            {"low": 0, "mid": 1, "high": 2}.get(str(r["count_bin"]), 99),
        ),
    )

    train_step = final_step(train_rows)
    non_solved = first_solved_step(eval_bin, "non_thinking", "direct")
    sep_solved = first_solved_step(eval_bin, "thinking_sep_trace", "generated_trace", require_trace=True)
    oracle_solved = first_solved_step(eval_bin, "thinking_sep_trace", "oracle_trace_final_readout")

    test_stats = jsonl_stats(run_dir / "data" / "test_pool.jsonl")
    val_stats = jsonl_stats(run_dir / "data" / "val_pool.jsonl")
    count_rows = [
        {
            "count": count,
            "test_examples": test_stats["counts"].get(count, 0),  # type: ignore[index,union-attr]
            "test_repeated_rate": test_stats["repeated_by_count"].get(count, math.nan),  # type: ignore[index,union-attr]
        }
        for count in range(int(cfg["min_count"]), int(cfg["max_count"]) + 1)
    ]

    headline = {
        (r["model_type"], r["eval_mode"]): r
        for r in final_overall
    }
    non = headline.get(("non_thinking", "direct"), {})
    sep_gen = headline.get(("thinking_sep_trace", "generated_trace"), {})
    sep_oracle = headline.get(("thinking_sep_trace", "oracle_trace_final_readout"), {})

    curve_figs = figure_grid(
        [
            figure(
                run_dir,
                "plots/train_loss_vs_step.png",
                "训练损失随 step 变化",
                "横轴是 training step，纵轴是 masked completion loss。蓝线/橙线分别是 non-thinking 与 separator-thinking。注意 thinking 模型被监督的 completion 更长，所以训练损失主要用于看是否收敛，不宜直接解读为两个模型难度完全等价。",
            ),
            figure(
                run_dir,
                "plots/eval_final_answer_loss_vs_step.png",
                "最终答案 token 的验证损失",
                "横轴是 training step，纵轴是 final-answer cross-entropy。这里比较的是最终 count token 的读出难度；separator-thinking 的 oracle-trace readout 使用 gold separator trace 前缀后预测最终 count。",
            ),
        ],
        columns=2,
    )

    behavior_figs = figure_grid(
        [
            figure(
                run_dir,
                "plots/eval_accuracy_by_bin_vs_step.png",
                "按 count bin 的准确率学习曲线",
                "横轴是 training step，纵轴是 exact final-count accuracy。线型/颜色区分 model_type、eval_mode 与 count_bin。count bin 定义为 low=1-3，mid=4-6，high=7-10。",
            ),
            figure(
                run_dir,
                "plots/final_accuracy_by_count.png",
                "最终 checkpoint 按精确 count 的准确率",
                "横轴是 gold count 1-10，纵轴是 exact final-count accuracy。该图回答模型是否在某些具体 count 上掉点，而不是只看平均值。",
            ),
        ],
        columns=2,
    )

    heatmap_figs = figure_grid(
        [
            figure(
                run_dir,
                "plots/accuracy_heatmap_by_count_and_step_non_thinking.png",
                "Non-thinking 准确率热图",
                "横轴是 training step，纵轴是 gold count，颜色是 exact final-count accuracy。这个图展示直接从 prompt 后的 <Ans> 预测 count 的学习过程。",
            ),
            figure(
                run_dir,
                "plots/accuracy_heatmap_by_count_and_step_thinking_generated_trace.png",
                "Separator-thinking 自由生成 trace 的准确率热图",
                "横轴是 training step，纵轴是 gold count，颜色是 generated-trace 模式下最终 count 是否正确。这里 trace 不是 gold，而是模型自己生成。",
            ),
            figure(
                run_dir,
                "plots/accuracy_heatmap_by_count_and_step_thinking_oracle_trace.png",
                "Separator-thinking oracle trace 读出热图",
                "横轴是 training step，纵轴是 gold count，颜色是给定 gold <Sep>, marker trace 后最终 count 的准确率。它隔离的是 final-answer readout，而不是 trace generation。",
            ),
        ],
        columns=3,
    )

    trace_figs = figure_grid(
        [
            figure(
                run_dir,
                "plots/trace_exact_by_count.png",
                "Generated separator trace 的 exact match",
                "横轴是 gold count，纵轴是 trace_exact_match_rate。该指标要求生成的 trace tokens 与 gold <Sep>, marker_1, ..., <Sep>, marker_n 完全一致。",
            ),
            figure(
                run_dir,
                "plots/trace_delimiter_count_accuracy_by_count.png",
                "<Sep> 数量是否等于 gold count",
                "横轴是 gold count，纵轴是 trace_delimiter_count_accuracy。这个指标只检查生成 trace 里的 <Sep> 个数是否正确，是 trace exact 的一个更弱版本。",
            ),
            figure(
                run_dir,
                "plots/trace_marker_precision_recall_by_count.png",
                "Trace marker precision / recall",
                "横轴是 gold count，纵轴是 marker precision/recall。precision/recall 基于生成 marker 序列与 gold marker 序列的 LCS，因此保留顺序信息，不是简单集合匹配。",
            ),
            figure(
                run_dir,
                "plots/premature_close_missing_close_by_count.png",
                "Trace 结构错误率",
                "横轴是 gold count，纵轴是错误率。premature close 表示 </Think> 过早出现，missing close 表示生成中没有正确闭合 thinking 段。",
            ),
        ],
        columns=2,
    )

    attention_figs = figure_grid(
        [
            figure(
                run_dir,
                "attention/attention_thinking_sep_correct_top1_by_layer_head.png",
                "Sep-token targeted retrieval: top-1 是否命中第 k 个 needle",
                "横轴是 attention head，纵轴是 layer，颜色是 correct_top1_rate。query 是第 k 个 <Sep> token；如果 top-attended prompt needle 正好是第 k 个 needle，则记为 1。",
            ),
            figure(
                run_dir,
                "attention/attention_thinking_sep_diagonal_dominance_by_layer_head.png",
                "Sep-token attention 的 diagonal dominance",
                "横轴是 head，纵轴是 layer，颜色是 diagonal_dominance。它比较第 k 个 <Sep> 对第 k 个 prompt needle 的 attention 与对其他 prompt needles 的平均 attention。",
            ),
            figure(
                run_dir,
                "attention/attention_thinking_sep_needle_mass_by_layer_head.png",
                "Sep-token 分配到 prompt needles 的总 attention mass",
                "横轴是 head，纵轴是 layer，颜色是 needle_attention_mass。数值越高，说明该 head 越集中看 prompt 中的 marker/needle token，而不是 noise token。",
            ),
        ],
        columns=3,
    )

    attention_matrix_figs = figure_grid(
        [
            figure(
                run_dir,
                "attention/attention_matrix_thinking_sep_best_head_low.png",
                "Best sep head 的 k-to-j attention matrix: low count",
                "横轴是 prompt needle j，纵轴是 trace item k，颜色是 attention weight。理想 targeted retrieval 是对角线亮，即第 k 个 sep/trace step 看第 k 个 prompt needle。",
            ),
            figure(
                run_dir,
                "attention/attention_matrix_thinking_sep_best_head_mid.png",
                "Best sep head 的 k-to-j attention matrix: mid count",
                "横轴是 prompt needle j，纵轴是 trace item k，颜色是 attention weight。该矩阵用于判断模型是在 sequential retrieval，而不是只看某一个固定位置。",
            ),
            figure(
                run_dir,
                "attention/attention_matrix_thinking_sep_best_head_high.png",
                "Best sep head 的 k-to-j attention matrix: high count",
                "横轴是 prompt needle j，纵轴是 trace item k，颜色是 attention weight。high bin 对重复 marker 更敏感，因为 marker identity 更容易重复。",
            ),
        ],
        columns=3,
    )

    non_attention_figs = figure_grid(
        [
            figure(
                run_dir,
                "attention/attention_nonthinking_topn_recall_by_layer_head.png",
                "Non-thinking answer-token top-n retrieval recall",
                "横轴是 head，纵轴是 layer，颜色是 top_n_retrieval_recall。query 是 <Ans> token；取 prompt body 中 attention 最高的 n 个位置，看其中有多少是 gold needles。",
            ),
            figure(
                run_dir,
                "attention/attention_nonthinking_ans_needle_mass_by_layer_head.png",
                "Non-thinking answer-token needle mass",
                "横轴是 head，纵轴是 layer，颜色是 <Ans> 对所有 prompt needles 的 attention mass。它和 top-n recall 一起看：top-n 可以命中 needles，但总质量可能仍然分散。",
            ),
            figure(
                run_dir,
                "attention/attention_unique_vs_repeated_marker_diagnostic.png",
                "Unique vs repeated marker diagnostic",
                "横轴/组别比较 unique-marker 与 repeated-marker 子集，纵轴是 retrieval 指标。repeated marker 子集更能排除只按 marker identity 检索的解释。",
            ),
        ],
        columns=3,
    )

    probe_figs = figure_grid(
        [
            figure(
                run_dir,
                "probes/probe_final_count_accuracy_heatmap_non_thinking.png",
                "Non-thinking final-count probe",
                "横轴是 anchor/layer，纵轴或颜色是 probe accuracy。probe 标签是最终 count；用于看 hidden state 中是否线性可读出 count。",
            ),
            figure(
                run_dir,
                "probes/probe_final_count_accuracy_heatmap_thinking_sep_trace.png",
                "Separator-thinking final-count probe",
                "横轴是 anchor/layer，颜色是 final-count probe accuracy。注意 think trace 的长度本身与 count 绑定，所以需要和 position/trace-length baseline 一起解读。",
            ),
            figure(
                run_dir,
                "probes/probe_prefix_count_accuracy_heatmap_thinking_sep_trace.png",
                "Separator trace prefix-count probe",
                "横轴是 anchor/layer，颜色是 prefix-count probe accuracy。prefix-count 是第 k 个 trace item 的 k 值；在 v6 中它很容易被 token position 解释。",
            ),
            figure(
                run_dir,
                "probes/probe_sep_token_prefix_probe_minus_position_baseline.png",
                "Sep-token prefix probe 减 position-only baseline",
                "横轴是 layer/anchor，纵轴或颜色是 probe_accuracy - position_only_accuracy。接近 0 说明线性 probe 没有超过位置基线，不能当作独立 counter 证据。",
            ),
        ],
        columns=2,
    )

    final_count_table_rows = [
        {
            "count": r.get("count", ""),
            "model_type": r.get("model_type", ""),
            "eval_mode": r.get("eval_mode", ""),
            "accuracy": as_float(r.get("accuracy")),
            "trace_exact_match_rate": as_float(r.get("trace_exact_match_rate")),
            "trace_delimiter_count_accuracy": as_float(r.get("trace_delimiter_count_accuracy")),
            "n_examples": as_float(r.get("n_examples")),
        }
        for r in final_count
        if r.get("eval_mode") != "oracle_trace_final_readout"
    ]
    final_count_table_rows = sorted(
        final_count_table_rows,
        key=lambda r: (str(r["model_type"]), str(r["eval_mode"]), int(as_float(r["count"]))),
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Trace Count v6 Separator-Trace Report</title>
<style>
:root {{
  --ink: #172033;
  --muted: #5f6b7a;
  --line: #d9e2ec;
  --card: #ffffff;
  --bg: #f6f8fb;
  --accent: #2563eb;
  --green: #15803d;
}}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.58;
}}
main {{
  width: min(1180px, calc(100vw - 48px));
  margin: 34px auto 64px;
}}
h1 {{ font-size: 34px; line-height: 1.18; margin: 0 0 8px; }}
h2 {{ font-size: 24px; margin: 34px 0 14px; border-top: 1px solid var(--line); padding-top: 24px; }}
h3 {{ font-size: 19px; margin: 22px 0 8px; }}
h4 {{ font-size: 16px; margin: 0 0 10px; }}
p {{ margin: 9px 0; }}
.hero, .card, .figure {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.hero {{ padding: 24px 28px; }}
.card {{ padding: 18px 20px; margin: 16px 0; }}
.lead {{ color: var(--muted); font-size: 16px; }}
.takeaway {{
  border-left: 4px solid var(--green);
  background: #f0fdf4;
  padding: 14px 16px;
  margin: 18px 0;
}}
code {{
  background: #eef2f7;
  border-radius: 4px;
  padding: 1px 5px;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
}}
pre {{
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 8px;
  overflow-x: auto;
  padding: 14px 16px;
  font-size: 13px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  font-size: 14px;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{ background: #f1f5f9; font-weight: 650; }}
.grid {{
  display: grid;
  gap: 16px;
  align-items: start;
  margin: 16px 0;
}}
.grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.figure {{
  margin: 0;
  padding: 14px;
}}
.figure img {{
  display: block;
  width: 100%;
  height: auto;
  max-height: 430px;
  object-fit: contain;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: #fff;
}}
.grid.three .figure img {{ max-height: 320px; }}
figcaption {{
  color: var(--muted);
  font-size: 13.5px;
  margin-top: 10px;
}}
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}}
.metric {{
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 9px;
  padding: 14px 16px;
}}
.metric-label {{ color: var(--muted); font-size: 13px; }}
.metric-value {{ font-size: 27px; font-weight: 750; margin-top: 4px; }}
.note {{ color: var(--muted); font-size: 14px; }}
@media (max-width: 900px) {{
  main {{ width: min(100vw - 24px, 1180px); }}
  .grid.two, .grid.three, .metric-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Trace Count v6: separator trace 能否替代数字 index trace?</h1>
    <p class="lead">结果目录：{code(run_dir)}</p>
    <p>v6 与 v2 的核心任务保持一致：在长度 {cfg["seq_len"]} 的 synthetic NIAH prompt 中数出 marker/needle 的数量。唯一关键变化是 thinking trace 不再写数字 index，而是用重复的 {code("<Sep>")} 分隔每个被检索到的 marker。</p>
    <pre>non-thinking:      &lt;BOS&gt; prompt_tokens &lt;Ans&gt; &lt;count&gt; &lt;EOS&gt;
separator-thinking:&lt;BOS&gt; prompt_tokens &lt;Think/&gt; &lt;Sep&gt; marker_1 ... &lt;Sep&gt; marker_n &lt;/Think&gt; &lt;Ans&gt; &lt;count&gt; &lt;EOS&gt;</pre>
    <div class="takeaway"><b>一句话结论。</b> 在这个 ID setting 里，数字 index 不是行为成功的必要条件：separator-thinking 自由生成 trace 与最终答案都达到 100%，并且出现了非常强的 {code("sep_token_k -> kth prompt needle")} targeted retrieval head。与此同时，prefix-count probe 很大程度被位置/trace-length 基线解释，不能单独当作“内部 counter direction”的证据。</div>
  </section>

  <h2>1. 实验设定</h2>
  <div class="card">
    <h3>数据生成</h3>
    <p>每条样本先生成 {code(str(cfg["seq_len"]))} 个 prompt token。Noise vocabulary 为 {code(cfg["noise_vocab_size"])} 个 {code("<N0> ... <N63>")}；marker vocabulary 为 {code(cfg["marker_vocab_size"])} 个 {code("<A> ... <J>")}。gold count 范围是 {code(f'{cfg["min_count"]}-{cfg["max_count"]}')}。每个 count 在 validation/test 中均衡采样。</p>
    <p>count bin 定义：{code("low=1-3")}，{code("mid=4-6")}，{code("high=7-10")}。test pool 每个 count 有 {eval_cfg["test_examples_per_count"]} 条样本，总计 {test_stats["total"]} 条；整体重复 marker 比例为 {pct(test_stats["repeated_rate"])}。重复 marker 重要，因为这时模型不能只靠 marker token identity，而必须按顺序取第 k 个 needle。</p>
    {table_html(count_rows, [("count", "gold count", "int"), ("test_examples", "test examples", "int"), ("test_repeated_rate", "repeated-marker rate", "pct")])}
  </div>
  <div class="card">
    <h3>模型与训练</h3>
    <p>本次训练两个模型：{code("non_thinking")} 与 {code("thinking_sep_trace")}。结构为 GPT-2 LM head，小模型配置：{model_cfg["n_layer"]} layers、{model_cfg["n_head"]} heads/layer、hidden size {model_cfg["n_embd"]}、vocab size {vocab_size}、learned absolute positional embedding、{model_cfg["n_positions"]} max positions、dropout 0。这里不是 RoPE。</p>
    <p>训练目标是 masked next-token prediction：prompt prefix 不计 loss；non-thinking 只监督 {code("<count>, <EOS>")}；separator-thinking 监督 trace tokens、{code("</Think>")}、{code("<Ans>")}、最终 count 和 {code("<EOS>")}。没有 loss-mask ablation，也没有 final-answer reweighting。</p>
    <p>训练参数：seed {cfg["seed"]}，{train_cfg["train_steps"]} steps，batch size {train_cfg["batch_size"]}，learning rate {train_cfg["learning_rate"]}，warmup {train_cfg["warmup_steps"]}，weight decay {train_cfg["weight_decay"]}，eval every {train_cfg["eval_every"]} steps。</p>
  </div>

  <h2>2. 总体行为结果</h2>
  <div class="metric-grid">
    <div class="metric"><div class="metric-label">non-thinking direct final accuracy</div><div class="metric-value">{pct(non.get("accuracy"))}</div></div>
    <div class="metric"><div class="metric-label">separator-thinking generated-trace final accuracy</div><div class="metric-value">{pct(sep_gen.get("accuracy"))}</div></div>
    <div class="metric"><div class="metric-label">separator-thinking trace exact match</div><div class="metric-value">{pct(sep_gen.get("trace_exact_match_rate"))}</div></div>
  </div>
  <p>final test 是每个 count 1000 条、总计 10000 条样本。结果饱和：non-thinking、separator-thinking generated trace、separator-thinking oracle trace 三种模式的最终 count accuracy 都是 100%。separator-thinking 的 trace exact、marker precision/recall、delimiter count accuracy 也都是 100%，premature close 和 missing close 均为 0。</p>
  <p>学习速度上，按 validation bin 聚合，non-thinking direct 首次达到全 bin 100% 的 step 是 {code(non_solved if non_solved is not None else "n/a")}；separator-thinking generated trace 且 trace exact 首次达到全 bin 100% 的 step 是 {code(sep_solved if sep_solved is not None else "n/a")}；oracle trace final readout 首次全解是 {code(oracle_solved if oracle_solved is not None else "n/a")}。</p>
  {table_html(final_overall, [("model_type", "model", "text"), ("eval_mode", "eval mode", "text"), ("n_examples", "n", "int"), ("accuracy", "final acc", "pct"), ("mae", "MAE", "num"), ("trace_exact_match_rate", "trace exact", "pct"), ("trace_marker_precision", "marker precision", "pct"), ("trace_marker_recall", "marker recall", "pct"), ("trace_delimiter_count_accuracy", "Sep-count acc", "pct")])}
  {curve_figs}
  {behavior_figs}
  {heatmap_figs}

  <h2>3. Separator trace 是否真的生成对了?</h2>
  <p>这一部分专门看 thinking 模型生成的 trace，而不是最终答案。generated-trace evaluation 从 {code("<Think/>")} 后自由 greedy generation，解析 {code("</Think>")}、{code("<Ans>")} 和最终 count。trace_exact_match_rate 要求完整 trace token 序列完全等于 gold；marker precision/recall 基于生成 marker 序列和 gold marker 序列的 LCS；delimiter count accuracy 只看 {code("<Sep>")} 个数是否等于 count。</p>
  {trace_figs}
  <div class="card">
    <h3>按精确 count 的 final/trace 表</h3>
    <p class="note">oracle-trace rows 在这里省略，因为它们只是给定 gold trace 后读最终答案；关键对比是 non-thinking direct 与 separator-thinking generated trace。</p>
    {table_html(final_count_table_rows, [("model_type", "model", "text"), ("eval_mode", "eval mode", "text"), ("count", "gold count", "int"), ("n_examples", "n", "int"), ("accuracy", "final acc", "pct"), ("trace_exact_match_rate", "trace exact", "pct"), ("trace_delimiter_count_accuracy", "Sep-count acc", "pct")])}
  </div>

  <h2>4. Attention: separator-thinking 是否形成 targeted retrieval?</h2>
  <p>attention 分析使用 balanced attention pool：每个 count {eval_cfg["attention_examples_per_count"]} 条样本。对 separator-thinking，核心 query 是第 k 个 {code("<Sep>")} token；我们检查这个 query 是否主要看 prompt 中第 k 个 needle。对 non-thinking，query 是 {code("<Ans>")} token；取 prompt body 中 attention 最高的 n 个位置，检查其中多少是 gold needles。</p>
  <div class="takeaway"><b>关键发现。</b> separator-thinking 的 layer 3 head 2 是一个非常清楚的 targeted retrieval head：在 all examples 上 correct-top1={pct(sep_attn_rows[0].get("correct_top1_rate") if sep_attn_rows else math.nan)}，diagonal dominance={fmt(sep_attn_rows[0].get("diagonal_dominance") if sep_attn_rows else math.nan)}，needle attention mass={fmt(sep_attn_rows[0].get("needle_attention_mass") if sep_attn_rows else math.nan)}。在 repeated-marker subset 上仍然 correct-top1={pct(sep_attn_rows[1].get("correct_top1_rate") if len(sep_attn_rows) > 1 else math.nan)}，这说明它不是单纯按 marker identity 匹配。</div>
  {table_html(sep_attn_rows, [("query", "query anchor", "text"), ("subset", "subset", "text"), ("layer", "layer", "int"), ("head", "head", "int"), ("n_queries", "queries", "int"), ("correct_top1_rate", "correct top-1", "pct"), ("diagonal_dominance", "diagonal dominance", "num"), ("needle_attention_mass", "needle mass", "num"), ("entropy_over_prompt_body", "prompt entropy", "num")])}
  {attention_figs}
  {attention_matrix_figs}
  <div class="card">
    <h3>Non-thinking 的注意力不是同一种显式 trace retrieval</h3>
    <p>non-thinking 的 layer 1 heads 在 top-n retrieval recall 上也能到 100%，说明它也能把 top attention positions 排到 needles 上。但总 needle mass 只有约 0.13 左右，prompt entropy 约 5.39，明显比 separator-thinking 的 sep head 更分散。因此它可能学到了一种足够解题的 prompt scanning/readout 机制，但不像 sep-thinking 那样在每个 trace step 上形成清晰的 k-to-k diagonal retrieval。</p>
    {table_html(non_attn_rows, [("query", "query anchor", "text"), ("layer", "layer", "int"), ("head", "head", "int"), ("n_queries", "queries", "int"), ("top_n_retrieval_recall", "top-n recall", "pct"), ("ans_to_all_needles_mass", "needle mass", "num"), ("attention_entropy_over_prompt_body", "prompt entropy", "num")])}
  </div>
  {non_attention_figs}

  <h2>5. Probe: 能不能从 hidden state 线性读出 count?</h2>
  <p>probe 使用 ridge/logistic-style readout 从不同 anchor/layer 的 hidden state 预测 count 或 prefix-count。这里必须和 baseline 一起看：{code("position_only")} 只用 token 位置，{code("trace_length_only")} 只用 trace 长度，{code("token_id_only")} 只用 token identity。v6 的 separator trace 长度与 count 强绑定，所以很多看似完美的 prefix-count probe 其实被位置或 trace length 完全解释。</p>
  <div class="takeaway"><b>谨慎解读。</b> separator trace 的 prefix-count probe 在 {code("sep_token_k")} 等 anchor 上接近 100%，但 position-only baseline 也是 100%，所以这不是一个干净的 internal counter direction 证据。相比之下，non-thinking 在 prompt/answer anchor 上的 final-count probe 超过位置基线，说明 final count 信息确实进入 hidden state；但这仍然不是 causal proof。</div>
  {table_html(final_probe_rows, [("model_type", "model", "text"), ("anchor_type", "anchor", "text"), ("layer", "layer", "text"), ("probe_accuracy", "probe acc", "pct"), ("position_only_accuracy", "position-only acc", "pct"), ("trace_length_only_accuracy", "trace-length acc", "pct"), ("probe_minus_position", "probe - position", "num"), ("probe_r2", "R2", "num")])}
  {table_html(prefix_probe_rows, [("anchor_type", "prefix anchor", "text"), ("layer", "layer", "text"), ("probe_accuracy", "probe acc", "pct"), ("position_only_accuracy", "position-only acc", "pct"), ("trace_length_only_accuracy", "trace-length acc", "pct"), ("token_id_only_accuracy", "token-id acc", "pct"), ("probe_minus_position", "probe - position", "num")])}
  {probe_figs}

  <h2>6. 结论与下一步</h2>
  <div class="card">
    <p><b>结论 1。</b> 在当前 v6 ID setting 中，数字 trace index 不是必要脚手架。用固定 {code("<Sep>")} delimiter 替换数字 index 后，thinking 模型仍然可以完整生成 trace，并达到 100% final accuracy。</p>
    <p><b>结论 2。</b> separator-thinking 仍然形成了类似 v2 的 targeted retrieval attention。最强证据是 layer 3 head 2 在第 k 个 {code("<Sep>")} token 上对第 k 个 prompt needle 的对角 attention；这个 head 在 repeated-marker subset 中同样成立。</p>
    <p><b>结论 3。</b> 目前 probe 结果不能单独证明“显式 counter direction”。v6 去掉了数字 index token identity，但引入了更直接的 trace-position/trace-length 信息：第 k 个 sep token 的位置本身就等于 prefix count。因此 prefix-count probe 必须做 causal tests 或位置控制后再解释。</p>
    <p><b>局限。</b> 这次没有 ID/OOD split，也没有 causal ablation/patching；所有 count 1-10 都在训练/验证/测试分布内。因此它回答的是“去掉 numeric index 后，ID synthetic task 是否仍可解、是否仍有 targeted retrieval head”，而不是 OOD 泛化或 head 必要性。</p>
    <p><b>建议下一步。</b> 对 layer 3 head 2 做 v3.2 风格 causal patching/ablation：在 generated trace 的关键 {code("<Sep>")} query 上 ablate 该 head，看 trace exact 和 final answer 是否下降；同时加入 position-jitter 或 variable filler between trace items，打断 prefix-count 与 absolute position 的绑定。</p>
  </div>
</main>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone HTML report for Trace Count v6 results.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir
    html = build_report(run_dir)
    out_root = run_dir / "report.html"
    out_nested = run_dir / "report" / "report.html"
    out_nested.parent.mkdir(parents=True, exist_ok=True)
    out_root.write_text(html, encoding="utf-8")
    out_nested.write_text(html, encoding="utf-8")
    print(out_root)
    print(out_nested)


if __name__ == "__main__":
    main()
