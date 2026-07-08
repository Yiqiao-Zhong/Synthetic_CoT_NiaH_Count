from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fnum(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt(value: object, digits: int = 3) -> str:
    x = fnum(value)
    if x is None:
        return html.escape(str(value)) if value not in (None, "") else "-"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def pct(value: object) -> str:
    x = fnum(value)
    return "-" if x is None else f"{100 * x:.1f}%"


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def group_mean(rows: list[dict[str, str]], keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    out: list[dict[str, object]] = []
    for key, rs in sorted(grouped.items()):
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = len(rs)
        for metric in metrics:
            values = [fnum(r.get(metric)) for r in rs]
            values = [v for v in values if v is not None]
            item[metric] = mean(values) if values else ""
        out.append(item)
    return out


def table_html(rows: list[dict[str, object]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{esc(label)}</th>" for _, label in columns)
    body_parts = []
    for row in shown:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            cells.append(f"<td>{esc(value)}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    more = ""
    if limit is not None and len(rows) > limit:
        more = f"<p class='muted'>Showing {limit} of {len(rows)} rows.</p>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table></div>{more}"


def img_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str) -> str:
    uri = img_data_uri(path)
    if not uri:
        return f"<div class='figure missing'><h4>{esc(title)}</h4><p>{esc(caption)}</p><p class='muted'>Missing: {esc(path.name)}</p></div>"
    return (
        "<figure class='figure'>"
        f"<h4>{esc(title)}</h4>"
        f"<img src='{uri}' alt='{esc(title)}'>"
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def top_rows(
    rows: list[dict[str, str]],
    score_key: str,
    n: int = 10,
    reverse: bool = True,
    pred=None,
) -> list[dict[str, object]]:
    pred = pred or (lambda row: True)
    scored = []
    for row in rows:
        score = fnum(row.get(score_key))
        if score is None or not pred(row):
            continue
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return [dict(row) for _, row in scored[:n]]


def summarize_patch(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("anchor_name", ""), row.get("hook_name", ""), row.get("layer", ""))].append(row)
    out = []
    for (anchor, hook, layer), rs in grouped.items():
        move_rate = mean(1.0 if r.get("patched_moves_toward_donor") == "True" else 0.0 for r in rs)
        effects = [abs(fnum(r.get("causal_effect_size")) or 0.0) for r in rs]
        recoveries = [fnum(r.get("logit_recovery_toward_donor_count")) for r in rs]
        recoveries = [x for x in recoveries if x is not None]
        out.append(
            {
                "anchor_name": anchor,
                "hook_name": hook,
                "layer": layer,
                "n": len(rs),
                "move_rate": move_rate,
                "mean_abs_pred_shift": mean(effects) if effects else "",
                "mean_logit_recovery": mean(recoveries) if recoveries else "",
            }
        )
    out.sort(key=lambda r: (fnum(r["move_rate"]) or 0.0, fnum(r["mean_abs_pred_shift"]) or 0.0), reverse=True)
    return out


def summarize_steering(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("control_type", ""), row.get("direction_type", ""))].append(row)
    controls = []
    for (control, direction), rs in grouped.items():
        shifts = [abs(fnum(r.get("mean_count_shift")) or 0.0) for r in rs]
        accs = [fnum(r.get("steered_accuracy")) for r in rs]
        accs = [x for x in accs if x is not None]
        kls = [fnum(r.get("kl_count_distribution_vs_base")) for r in rs]
        kls = [x for x in kls if x is not None]
        controls.append(
            {
                "control_type": control,
                "direction_type": direction,
                "n": len(rs),
                "mean_abs_count_shift": mean(shifts) if shifts else "",
                "max_abs_count_shift": max(shifts) if shifts else "",
                "mean_steered_accuracy": mean(accs) if accs else "",
                "mean_kl_vs_base": mean(kls) if kls else "",
            }
        )
    controls.sort(key=lambda r: fnum(r["max_abs_count_shift"]) or 0.0, reverse=True)

    real = [
        dict(r)
        for r in rows
        if r.get("control_type") == "none"
        and abs(fnum(r.get("alpha")) or 0.0) > 0
        and abs(fnum(r.get("mean_count_shift")) or 0.0) > 0
    ]
    real.sort(key=lambda r: abs(fnum(r.get("mean_count_shift")) or 0.0), reverse=True)
    return controls, real[:12]


def metric_bullets(summary: dict[str, object]) -> str:
    items = []
    for label, value in summary.items():
        items.append(f"<li><strong>{esc(label)}</strong>: {esc(value)}</li>")
    return "<ul class='compact'>" + "".join(items) + "</ul>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("colab_results/v4_main_seed1234"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_path = args.out or (run_dir / "report.html")
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"

    config = read_json(run_dir / "config.json")
    vocab = read_json(run_dir / "vocab.json")
    vocab_size = len(vocab.get("token_to_id", {})) if isinstance(vocab, dict) else ""

    behavior = read_csv(tables_dir / "behavior_eval.csv")
    behavior_by_count = read_csv(tables_dir / "behavior_accuracy_by_count.csv")
    probes = read_csv(tables_dir / "probe_results.csv")
    residualized = read_csv(tables_dir / "probe_residualized.csv")
    directions = read_csv(tables_dir / "direction_metrics.csv")
    input_geom = read_csv(tables_dir / "input_geometry_results.csv")
    patching = read_csv(tables_dir / "interchange_patching_results.csv")
    steering = read_csv(tables_dir / "steering_results.csv")

    behavior_summary = group_mean(
        behavior,
        ["model_type", "eval_mode"],
        ["final_accuracy", "trace_exact_rate", "invalid_rate", "final_mae", "final_answer_ce", "count_token_rate"],
    )
    behavior_bin_summary = group_mean(
        behavior,
        ["model_type", "eval_mode", "count_bin"],
        ["final_accuracy", "final_mae", "invalid_rate"],
    )

    top_probe_acc = top_rows(
        probes,
        "accuracy",
        n=12,
        pred=lambda r: r.get("probe_type") == "multiclass_logistic" and r.get("raw_or_residualized") == "raw",
    )
    top_probe_r2 = top_rows(
        probes,
        "r2",
        n=12,
        pred=lambda r: r.get("probe_type") == "ridge_scalar" and r.get("raw_or_residualized") == "raw",
    )
    top_resid_acc = top_rows(
        residualized,
        "accuracy",
        n=10,
        pred=lambda r: r.get("probe_type") == "multiclass_logistic",
    )
    top_direction = top_rows(directions, "projection_r2", n=12)
    patch_summary = summarize_patch(patching)
    steering_controls, steering_top = summarize_steering(steering)

    count_min = config.get("count_min", "")
    count_max = config.get("count_max", "")
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    probe_cfg = config.get("probe", {})
    steering_cfg = config.get("steering", {})

    all_acc = [fnum(r.get("final_accuracy")) for r in behavior]
    all_acc = [x for x in all_acc if x is not None]
    min_acc = min(all_acc) if all_acc else ""
    trace_rows = [r for r in behavior if r.get("trace_exact_rate") not in ("", None)]
    trace_mean = mean(fnum(r.get("trace_exact_rate")) or 0.0 for r in trace_rows) if trace_rows else ""

    executive = {
        "主任务最终计数": f"所有 eval mode 平均/逐 count 都为 {pct(min_acc)}" if min_acc != "" else "missing",
        "thinking trace": f"generated/oracle trace exact rate = {pct(trace_mean)}" if trace_mean != "" else "not applicable",
        "最强 patch": (
            f"{patch_summary[0]['anchor_name']} / {patch_summary[0]['hook_name']}，move-to-donor={pct(patch_summary[0]['move_rate'])}"
            if patch_summary
            else "missing"
        ),
        "最强 steering": (
            f"{steering_top[0]['anchor_name']} / layer {steering_top[0]['layer']} / alpha={steering_top[0]['alpha']}，mean count shift={fmt(steering_top[0]['mean_count_shift'])}"
            if steering_top
            else "actual count-token prediction almost saturated; no nonzero mean shift recorded",
        ),
    }

    behavior_cols = [
        ("model_type", "model"),
        ("eval_mode", "eval mode"),
        ("n", "n examples"),
        ("final_accuracy", "final accuracy"),
        ("trace_exact_rate", "trace exact"),
        ("invalid_rate", "invalid"),
        ("final_mae", "MAE"),
        ("count_token_rate", "count-token rate"),
    ]
    for row in behavior_summary:
        for k in ["final_accuracy", "trace_exact_rate", "invalid_rate", "count_token_rate"]:
            if row.get(k) != "":
                row[k] = pct(row[k])
        if row.get("final_mae") != "":
            row["final_mae"] = fmt(row["final_mae"])

    for row in behavior_bin_summary:
        for k in ["final_accuracy", "invalid_rate"]:
            if row.get(k) != "":
                row[k] = pct(row[k])
        if row.get("final_mae") != "":
            row["final_mae"] = fmt(row["final_mae"])

    probe_cols = [
        ("model_type", "model"),
        ("target", "target"),
        ("anchor_name", "anchor"),
        ("hook_name", "hook"),
        ("layer", "layer"),
        ("accuracy", "accuracy"),
        ("r2", "R2"),
        ("position_baseline_acc", "position baseline"),
        ("token_baseline_acc", "token baseline"),
        ("leakage_prone", "leakage-prone"),
    ]
    for row in top_probe_acc + top_probe_r2 + top_resid_acc:
        for key in ["accuracy", "r2", "position_baseline_acc", "token_baseline_acc"]:
            if row.get(key, "") != "":
                row[key] = fmt(row[key])

    direction_cols = [
        ("target", "target"),
        ("direction_type", "direction"),
        ("model_type", "model"),
        ("anchor_name", "anchor"),
        ("hook_name", "hook"),
        ("layer", "layer"),
        ("projection_slope", "projection slope"),
        ("projection_r2", "projection R2"),
        ("cosine_with_unembedding", "cosine vs unembed"),
    ]
    for row in top_direction:
        for key in ["projection_slope", "projection_r2", "cosine_with_unembedding"]:
            if row.get(key, "") != "":
                row[key] = fmt(row[key])

    patch_cols = [
        ("anchor_name", "anchor"),
        ("hook_name", "hook"),
        ("layer", "layer"),
        ("n", "pairs"),
        ("move_rate", "move-to-donor rate"),
        ("mean_abs_pred_shift", "mean |pred shift|"),
        ("mean_logit_recovery", "mean logit recovery"),
    ]
    for row in patch_summary:
        row["move_rate"] = pct(row["move_rate"])
        row["mean_abs_pred_shift"] = fmt(row["mean_abs_pred_shift"])
        row["mean_logit_recovery"] = fmt(row["mean_logit_recovery"])

    steering_control_cols = [
        ("control_type", "control"),
        ("direction_type", "direction"),
        ("n", "rows"),
        ("mean_abs_count_shift", "mean |count shift|"),
        ("max_abs_count_shift", "max |count shift|"),
        ("mean_steered_accuracy", "mean steered acc"),
        ("mean_kl_vs_base", "mean KL vs base"),
    ]
    for row in steering_controls:
        for key in ["mean_abs_count_shift", "max_abs_count_shift", "mean_kl_vs_base"]:
            row[key] = fmt(row[key])
        row["mean_steered_accuracy"] = pct(row["mean_steered_accuracy"])

    steering_top_cols = [
        ("anchor_name", "anchor"),
        ("hook_name", "hook"),
        ("layer", "layer"),
        ("direction_type", "direction"),
        ("alpha", "alpha"),
        ("mean_count_shift", "mean count shift"),
        ("steered_accuracy", "steered acc"),
        ("mean_gold_logit_change", "gold logit delta"),
        ("mean_target_plus_one_logprob_change", "+1 logprob delta"),
    ]
    for row in steering_top:
        for key in ["alpha", "mean_count_shift", "mean_gold_logit_change", "mean_target_plus_one_logprob_change"]:
            row[key] = fmt(row.get(key))
        row["steered_accuracy"] = pct(row.get("steered_accuracy"))

    input_geom_cols = [
        ("anchor_name", "anchor"),
        ("layer", "layer"),
        ("direction_type", "direction"),
        ("perturbation", "perturbation"),
        ("projection_slope", "projection slope"),
        ("projection_r2", "projection R2"),
    ]
    for row in input_geom:
        for key in ["projection_slope", "projection_r2"]:
            row[key] = fmt(row.get(key))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Count v4 Report - {esc(run_dir.name)}</title>
<style>
:root {{
  --ink: #142033;
  --muted: #5e6b7e;
  --line: #dbe4f0;
  --panel: #ffffff;
  --soft: #f5f8fc;
  --blue: #2563eb;
  --green: #12805c;
  --orange: #b45309;
  --red: #b42318;
}}
body {{
  margin: 0;
  background: #eef3f9;
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  line-height: 1.6;
}}
.page {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}}
h1 {{ font-size: 34px; margin: 0 0 8px; }}
h2 {{ font-size: 24px; margin: 36px 0 14px; border-top: 1px solid var(--line); padding-top: 28px; }}
h3 {{ font-size: 18px; margin: 22px 0 10px; }}
p {{ margin: 8px 0 14px; }}
code {{
  background: #e9eef6;
  border-radius: 5px;
  padding: 1px 5px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}}
.muted {{ color: var(--muted); }}
.hero, .card, .figure {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}
.hero {{ padding: 24px; }}
.card {{ padding: 18px 20px; margin: 16px 0; }}
.grid {{ display: grid; gap: 16px; }}
.grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
@media (max-width: 900px) {{ .grid.two, .grid.three {{ grid-template-columns: 1fr; }} }}
.kpi {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
.kpi div {{ background: var(--soft); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }}
.kpi strong {{ display: block; font-size: 22px; }}
.compact {{ margin-top: 8px; }}
.compact li {{ margin: 4px 0; }}
.callout {{
  border-left: 5px solid var(--green);
  background: #ecfdf5;
  padding: 14px 16px;
  border-radius: 8px;
  margin: 16px 0;
}}
.warning {{
  border-left: 5px solid var(--orange);
  background: #fff7ed;
  padding: 14px 16px;
  border-radius: 8px;
  margin: 16px 0;
}}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
th {{ background: #eef4fb; font-weight: 700; }}
tr:nth-child(even) td {{ background: #fafcff; }}
.figure {{ margin: 16px 0; padding: 14px; }}
.figure h4 {{ margin: 2px 0 10px; font-size: 16px; }}
.figure img {{
  display: block;
  width: 100%;
  max-height: 430px;
  object-fit: contain;
  background: white;
  border-radius: 8px;
}}
.figure.compact img {{ max-height: 340px; }}
figcaption {{ color: var(--muted); font-size: 14px; margin-top: 10px; }}
.tag {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #e0ecff; color: #1d4ed8; font-size: 12px; margin-right: 4px; }}
</style>
</head>
<body>
<main class="page">
<section class="hero">
  <h1>Trace Count v4 结果报告</h1>
  <p class="muted">Run directory: <code>{esc(str(run_dir))}</code><br>Generated: {esc(generated)}</p>
  <p>
    v4 是一个 <strong>v2-style steering</strong> 实验：训练一个不输出 trace 的模型和一个输出 indexed thinking trace 的模型，
    然后在 trained checkpoint 上做 probe、direction extraction、interchange patching 和 activation steering。
    这份报告重点回答：模型是否已经学会计数、hidden state 中是否有线性 count direction、这些 direction 是否可以因果操控最终答案。
  </p>
  {metric_bullets(executive)}
  <div class="kpi">
    <div><span class="muted">vocab size</span><strong>{esc(vocab_size)}</strong></div>
    <div><span class="muted">count range</span><strong>{esc(count_min)}-{esc(count_max)}</strong></div>
    <div><span class="muted">sequence length</span><strong>{esc(config.get('seq_len', ''))}</strong></div>
    <div><span class="muted">seed</span><strong>{esc(', '.join(map(str, config.get('seeds', []))))}</strong></div>
  </div>
</section>

<h2>1. 实验设定</h2>
<div class="card">
  <h3>Task and model</h3>
  <p>
    输入是长度 <code>{esc(config.get('seq_len', ''))}</code> 的 synthetic NIAH-style 序列，里面出现 <code>{esc(count_min)}-{esc(count_max)}</code> 个 marker/needle。
    非 thinking 模型看到 prompt 后直接在 <code>&lt;Ans&gt;</code> 后预测最终 count token；
    thinking 模型先生成或被 teacher-forced 一个 indexed trace：<code>&lt;Think/&gt; &lt;1&gt; marker &lt;2&gt; marker ... &lt;/Think&gt; &lt;Ans&gt; count</code>。
  </p>
  <p>
    模型是随机初始化的 HuggingFace <code>GPT2LMHeadModel</code>，使用 learned absolute positional embeddings，
    层数 <code>{esc(model_cfg.get('n_layer'))}</code>，heads <code>{esc(model_cfg.get('n_head'))}</code>，hidden size <code>{esc(model_cfg.get('n_embd'))}</code>，
    max positions <code>{esc(model_cfg.get('n_positions'))}</code>。训练 <code>{esc(train_cfg.get('steps'))}</code> steps，
    batch size <code>{esc(train_cfg.get('batch_size'))}</code>，LR <code>{esc(train_cfg.get('lr'))}</code>，warmup <code>{esc(train_cfg.get('warmup_steps'))}</code>。
  </p>
  <p>
    Probe 每个 count 训练/测试各约 <code>{esc(probe_cfg.get('examples_per_count_train'))}</code>/<code>{esc(probe_cfg.get('examples_per_count_test'))}</code> 个 hidden-state samples；
    steering 在每个 count 上用 <code>{esc(steering_cfg.get('examples_per_count'))}</code> 个 examples，alpha grid =
    <code>{esc(steering_cfg.get('alpha_grid'))}</code>。
  </p>
</div>

<div class="warning">
  <strong>读图时要注意。</strong>
  v4 的主任务在 main setting 下已经完全学会了，所以 final answer logits 很饱和。
  因此 probe 很容易达到 1.0，不代表这个方向一定可控；更强的 causal evidence 来自 patching/steering 是否能改变最终 prediction。
</div>

<h2>2. Behavioral evaluation：模型是否会数</h2>
<p>
  这里的横向比较是不同模型和 eval mode。<code>non_thinking/direct</code> 直接读答案；
  <code>thinking/oracle_trace</code> 使用 gold trace 后预测答案；
  训练过程中也保存了 generated-trace behavior，若存在则表示模型自己生成 trace 再读答案。
  <code>final accuracy</code> 只看最后 count token 是否等于 gold count；<code>trace exact</code> 只对 thinking trace 有意义。
</p>
{table_html(behavior_summary, behavior_cols)}
<p>
  按 low/mid/high count bins 分开看，准确率也没有暴露明显难度差异：
</p>
{table_html(behavior_bin_summary, [('model_type','model'),('eval_mode','eval mode'),('count_bin','count bin'),('n','n'),('final_accuracy','accuracy'),('final_mae','MAE'),('invalid_rate','invalid')])}
<div class="callout">
  <strong>结果。</strong>
  v4 main 的行为结果是一个 ceiling result：non-thinking 和 thinking 都达到 100% final-count accuracy，
  thinking 的 trace exact 也达到 100%。所以后续 mechanistic 分析不能再问“是否会做题”，而应问“答案信息在哪里、能否被干预”。
</div>

<h2>3. Probe：count 信息在哪里可读</h2>
<p>
  Probe 部分有两类指标。Logistic probe 的纵轴是 count 分类准确率；ridge probe 的纵轴是用 hidden state 线性预测 count 的 <code>R2</code>。
  横轴是取 hidden state 的 anchor：例如 <code>ans_token</code> 是答案位置，<code>index_k_pos</code> 是 trace 中第 k 个 index token，
  <code>marker_k_pos</code>/<code>post_marker_k</code> 是 trace marker 附近位置。颜色表示 layer 或 hook。
</p>
<div class="grid two">
{figure(figures_dir / "probe_acc_by_layer_anchor.png", "Figure 1a. Probe accuracy by layer and anchor", "横轴是 anchor_name，纵轴是 multiclass logistic probe 的 count accuracy，颜色是 layer。高值说明该位置 hidden state 线性可分出 count；但若 position/token baseline 也很高，要小心它只是 index token 泄漏。")}
{figure(figures_dir / "probe_r2_by_layer_anchor.png", "Figure 1b. Ridge R2 by layer and anchor", "横轴是 anchor_name，纵轴是 ridge regression 对 count 的 R2，颜色是 layer。R2 接近 1 表示隐藏状态沿某个线性方向几乎连续编码 count。")}
</div>
{figure(figures_dir / "probe_minus_baseline_heatmap.png", "Figure 1c. Probe accuracy minus position baseline", "横轴是 layer，纵轴是 anchor_name，颜色是 probe accuracy 减去 position baseline。接近 0 说明 probe 可能主要利用位置/格式；大于 0 才更像非平凡的 hidden-state count information。")}

<h3>Top probe rows</h3>
<p>下面只列最强的若干 rows，方便定位。<code>leakage-prone=True</code> 通常表示 anchor 与 index token 或固定位置强绑定，不能直接解释为模型内部真的“数出来”。</p>
{table_html(top_probe_acc, probe_cols, limit=12)}
<h3>Top ridge rows</h3>
{table_html(top_probe_r2, probe_cols, limit=12)}
<h3>Residualized probes</h3>
<p>Residualized probe 是在去掉 position/token/trace-length baseline 后再试图读 count。若 residualized 后仍高，说明 hidden state 中有超出简单格式变量的 count signal。</p>
{table_html(top_resid_acc, probe_cols, limit=10)}

<h2>4. Direction extraction：哪些 count direction 看起来像输出方向</h2>
<p>
  Direction 部分比较多种方向：<code>ridge</code> 是回归 count 得到的方向，
  <code>dom</code>/<code>matched_delta</code> 是按 count 差构造的均值差方向，
  <code>logistic_adjacent</code> 是相邻 count 分类方向，
  <code>unembedding_adjacent</code> 是输出层里相邻 count token 的读出方向，
  <code>shuffled_label_probe</code> 是负控。
</p>
<div class="grid two">
{figure(figures_dir / "direction_cosine_heatmap.png", "Figure 2a. Direction cosine with unembedding adjacent direction", "横轴是 anchor_name，纵轴是 direction_type，颜色是该 direction 与 count-token unembedding adjacent direction 的 cosine。颜色越红，越接近输出层把 count+1/相邻 count 区分开的方向；这说明 readout alignment，但不等于因果可控。")}
{figure(figures_dir / "projection_by_count.png", "Figure 2b. Projection slope by anchor and direction", "横轴是 anchor_name，纵轴是 projection_slope：hidden state 投影到该 direction 后随 gold count 增长的斜率；颜色是 direction_type。斜率越大，说明 count 在该方向上的几何变化越明显。")}
</div>
{figure(figures_dir / "input_geometry_projection_trajectories.png", "Figure 2c. Input geometry projection summary", "横轴是 anchor_name，纵轴是 projection_slope，颜色是 direction_type。这个图检查输入/trace token geometry 自身是否已经携带 count 结构；如果 embedding 或 resid_pre_layer_0 就接近完美，说明一部分 signal 来自显式 index token/格式。")}
<h3>Top projection directions</h3>
{table_html(top_direction, direction_cols, limit=12)}
<h3>Input geometry rows</h3>
{table_html(input_geom, input_geom_cols, limit=8)}

<div class="warning">
  <strong>解释。</strong>
  v4 里很多 prefix/trace anchor 的 R2 接近 1，尤其是 indexed trace 的 <code>&lt;k&gt;</code> token 附近。
  这很可能是因为 trace 显式包含 index token，count/prefix-count 已经被 token identity 或固定结构编码。
  因而 direction/probe 需要和 causal steering 一起看，不能单独作为机制结论。
</div>

<h2>5. Interchange patching：替换 hidden state 会不会把答案搬过去</h2>
<p>
  Interchange patching 使用 receiver prompt 和 donor prompt，二者 count 不同。
  实验把 donor run 在某个 anchor/layer 的 hidden state 替换到 receiver run 的同一位置，
  然后看 patched prediction 是否朝 donor count 移动。<code>move-to-donor rate</code> 是 patched prediction 更接近 donor count 的比例；
  <code>mean |pred shift|</code> 是 patched prediction 相对 receiver base prediction 的平均绝对位移。
</p>
{figure(figures_dir / "interchange_patch_matrix.png", "Figure 3. Interchange patch matrix", "横轴是 donor_count，纵轴是 receiver_count，颜色/数字是平均 causal_effect_size，即 patch 后 prediction 相对 receiver base prediction 的位移。若某个 component 真携带可读答案，patch 应该把 receiver 往 donor count 方向推。")}
<h3>Top patching summaries</h3>
{table_html(patch_summary, patch_cols, limit=12)}
<div class="callout">
  <strong>结果。</strong>
  最强 patch 是直接在 <code>ans_token</code> 的 residual post layer 上替换 hidden state，move-to-donor rate 为 100%。
  这说明答案位置 residual 已经足以决定最终 count token。相反，大部分 trace/prefix anchor 的 direct patch 对最终答案几乎没有影响，
  因而当前 v4 patching 不能证明“中间 trace 位置本身足以驱动最终答案”。
</div>

<h2>6. Steering：沿 count direction 加激活是否能改答案</h2>
<p>
  Steering 在固定 anchor/layer 上加入 <code>alpha * direction</code>。
  横轴 alpha 表示干预强度；纵轴常见有两种：<code>mean_pred_steered</code> 是平均预测 count，
  <code>mean_count_shift</code> 是相对 base prediction 的平均变化。Controls 包括 random unit direction 和 zero intervention。
</p>
<div class="grid two">
{figure(figures_dir / "steering_controls.png", "Figure 4a. Steering controls", "横轴是 control_type，纵轴是 mean_count_shift，颜色是 direction_type。真实 ridge direction 若显著超过 random/zero，才说明不是任意扰动导致。这里 random 和 zero 基本为 0。")}
{figure(figures_dir / "steering_dose_response_top_configs.png", "Figure 4b. Top steering dose response", "横轴是 alpha，纵轴是 mean_pred_steered，颜色/线型是最强 steering config。若方向有效，应出现随 alpha 单调变化的 dose-response。")}
</div>
{figure(figures_dir / "steering_heatmap_anchor_layer.png", "Figure 4c. Steering heatmap by anchor and layer", "横轴是 layer，纵轴是 anchor_name，颜色是平均 mean_count_shift。颜色接近 0 说明该 anchor/layer 的平均 steering 对最终 prediction 几乎不动；深色表示更强的 count shift。")}
<h3>Control summary</h3>
{table_html(steering_controls, steering_control_cols, limit=12)}
<h3>Top nonzero steering rows</h3>
{table_html(steering_top, steering_top_cols, limit=12)}
<div class="warning">
  <strong>结果。</strong>
  Steering 的实质效果很弱。最强可见 row 是 <code>ans_token / resid_post_layer_0 / ridge / alpha=-6</code>，
  平均 count shift 约 -0.114，steered accuracy 降到约 88.6%；其他多数 rows 的最终 count prediction 不变。
  这说明 v4 找到了一个能轻微改变答案 logits/prediction 的方向，但在 main setting 下还不是强、稳定、可泛化的 count-control knob。
</div>

<h2>7. 总结与下一步</h2>
<div class="card">
  <h3>结论</h3>
  <ul>
    <li><strong>行为层面：</strong>v4 main 已经彻底解决 synthetic count task；两个模型都达到 100% final accuracy。</li>
    <li><strong>表征层面：</strong>final answer position 和 indexed trace positions 都有高度可 probe 的 count signal；但 trace index token 带来很强 leakage/format baseline。</li>
    <li><strong>方向层面：</strong>ridge direction 对 count projection 的 R2 很高，但与 unembedding adjacent direction 的对齐程度随 anchor 变化；高 R2 不自动等价于 steering 成功。</li>
    <li><strong>因果层面：</strong>patch answer-position residual 可以直接搬运答案；trace/prefix anchor 的 direct patch 贡献弱。Steering 存在弱效应，但整体被饱和的 final-answer readout 抑制。</li>
  </ul>
  <h3>建议</h3>
  <ul>
    <li>如果要证明“可控 count direction”，下一版应在非饱和 regime 或更困难 OOD 上做 steering，让 prediction 不是 100% ceiling。</li>
    <li>对 indexed trace，probe 必须显式报告 token/position baseline；最好用 v6 那种 separator trace 或去掉显式 index token 来减少泄漏。</li>
    <li>Patch 需要更局部的路径测试：从 trace retrieval head 到 answer readout 的具体链路，而不是只 patch 单个 residual anchor。</li>
  </ul>
</div>

<p class="muted">Source tables: <code>tables/*.csv</code>. Source figures: <code>figures/*.png</code>.</p>
</main>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
