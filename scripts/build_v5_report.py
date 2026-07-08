from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    try:
        if value == "":
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
        w = as_float(row.get(weight_col))
        if math.isnan(val):
            continue
        if math.isnan(w) or w <= 0:
            w = 1.0
        total += val * w
        weight += w
    return total / weight if weight else math.nan


def fmt(value: object, digits: int = 3) -> str:
    val = as_float(value)
    if math.isnan(val):
        return html.escape(str(value))
    if abs(val) < 0.001 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    val = as_float(value)
    if math.isnan(val):
        return html.escape(str(value))
    return f"{100 * val:.{digits}f}%"


def code(text: object) -> str:
    return f"<code>{html.escape(str(text))}</code>"


def group_weighted(
    rows: list[dict[str, str]],
    keys: list[str],
    metrics: list[str],
    weight_col: str = "n_examples",
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    out: list[dict[str, object]] = []
    for key, vals in grouped.items():
        item: dict[str, object] = dict(zip(keys, key))
        item["n_rows"] = len(vals)
        item["n_examples"] = sum(as_float(v.get(weight_col)) if not math.isnan(as_float(v.get(weight_col))) else 1.0 for v in vals)
        for metric in metrics:
            item[metric] = weighted_mean(vals, metric, weight_col)
        out.append(item)
    return out


def group_mean(rows: list[dict[str, str]], keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        for metric in metrics:
            grouped[key][metric].append(row.get(metric, ""))
    out: list[dict[str, object]] = []
    for key, vals in grouped.items():
        item: dict[str, object] = dict(zip(keys, key))
        item["n"] = max((len(v) for v in vals.values()), default=0)
        for metric in metrics:
            item[metric] = mean(vals[metric])
        out.append(item)
    return out


def table_html(rows: list[dict[str, object]], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    header = "".join(f"<th>{html.escape(label)}</th>" for _key, label in columns)
    body = []
    for row in shown:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                text = fmt(value)
            else:
                text = html.escape(str(value))
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(f"<tr><td colspan=\"{len(columns)}\">No data</td></tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, title: str, caption: str, *, wide: bool = False) -> str:
    if not path.exists():
        return ""
    cls = "figure wide" if wide else "figure"
    return f"""
    <figure class="{cls}">
      <h3>{html.escape(title)}</h3>
      <img src="{image_uri(path)}" alt="{html.escape(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def setting_box(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"""
    <div class="setting">
      <div class="setting-title">{html.escape(title)}</div>
      <table class="setting-table"><tbody>{body}</tbody></table>
    </div>
    """


def final_step(rows: list[dict[str, str]]) -> int:
    steps = [int(as_float(r.get("step"))) for r in rows if not math.isnan(as_float(r.get("step")))]
    return max(steps) if steps else 0


def first_perfect_step(eval_rows: list[dict[str, str]], mode: str) -> int | None:
    steps = sorted({int(as_float(r["step"])) for r in eval_rows if not math.isnan(as_float(r.get("step")))})
    for step in steps:
        sub = [r for r in eval_rows if int(as_float(r.get("step"))) == step and r.get("mode") == mode]
        if not sub:
            continue
        acc = weighted_mean(sub, "final_accuracy")
        trace = weighted_mean(sub, "trace_exact")
        if acc >= 1.0 and (mode == "nonthinking" or trace >= 1.0):
            return step
    return None


def top_rows(rows: list[dict[str, str]], metric: str, n: int, *, mode: str | None = None) -> list[dict[str, object]]:
    filtered = [r for r in rows if mode is None or r.get("mode") == mode]
    filtered = sorted(filtered, key=lambda r: as_float(r.get(metric)), reverse=True)
    return [dict(r) for r in filtered[:n]]


def best_probe_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    anchors = {"think_open_pos", "think_close_pos", "pre_count_pos", "count_pos", "trace_marker_1", "trace_marker_5", "post_trace_marker_5"}
    filtered = [r for r in rows if r.get("anchor_name") in anchors]
    filtered = sorted(filtered, key=lambda r: (r.get("mode", ""), r.get("target", ""), r.get("anchor_name", ""), -as_float(r.get("accuracy"))))
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in filtered:
        key = (row.get("mode", ""), row.get("target", ""), row.get("anchor_name", ""))
        if key not in best:
            best[key] = row
    return [dict(v) for v in best.values()]


def matrix_diag_summary(rows: list[dict[str, str]], layer: int, head: int) -> dict[str, float]:
    sub = [r for r in rows if int(as_float(r.get("layer"))) == layer and int(as_float(r.get("head"))) == head]
    diag = [as_float(r.get("attention")) for r in sub if int(as_float(r.get("trace_index"))) == int(as_float(r.get("needle_index")))]
    off = [as_float(r.get("attention")) for r in sub if int(as_float(r.get("trace_index"))) != int(as_float(r.get("needle_index")))]
    vals = [as_float(r.get("attention")) for r in sub]
    return {
        "diag_mean": mean(diag),
        "off_diag_mean": mean(off),
        "max_attention": max([v for v in vals if not math.isnan(v)], default=math.nan),
    }


def build_report(run_dir: Path) -> str:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
    tables = run_dir / "tables"
    figures = run_dir / "figures"

    train = read_rows(tables / "train_log.csv")
    eval_rows = read_rows(tables / "eval_by_step.csv")
    probe = read_rows(tables / "probe_results.csv")
    attention = read_rows(tables / "attention_metrics.csv")
    attention_matrix = read_rows(tables / "attention_trace_matrix.csv")
    ambiguous = read_rows(tables / "ambiguous_prefix.csv")
    similarity = read_rows(tables / "mode_hidden_similarity.csv")

    step = final_step(eval_rows)
    final = [r for r in eval_rows if int(as_float(r.get("step"))) == step]
    final_by_mode = {
        r["mode"]: r
        for r in group_weighted(
            final,
            ["mode"],
            [
                "final_accuracy",
                "final_mae",
                "undercount_rate",
                "overcount_rate",
                "trace_exact",
                "trace_marker_precision",
                "trace_marker_recall",
                "premature_close_rate",
                "missing_close_rate",
                "invalid_count_rate",
            ],
        )
    }
    final_by_bin = sorted(
        group_weighted(final, ["mode", "count_bin"], ["final_accuracy", "trace_exact", "trace_marker_recall", "premature_close_rate"]),
        key=lambda r: (str(r["mode"]), {"low": 0, "mid": 1, "high": 2}.get(str(r["count_bin"]), 99)),
    )
    final_by_count = sorted(
        group_weighted(final, ["mode", "count"], ["final_accuracy", "final_mae", "trace_exact", "trace_marker_recall"]),
        key=lambda r: (str(r["mode"]), int(as_float(r["count"]))),
    )
    final_amb = [r for r in ambiguous if int(as_float(r.get("step"))) == step]
    amb_summary = group_weighted(
        final_amb,
        [],
        [
            "p_close_after_think",
            "p_any_marker_after_think",
            "p_gold_first_marker_after_think",
            "argmax_is_close",
            "argmax_is_gold_first_marker",
        ],
    )[0]
    amb_by_bin = sorted(
        group_weighted(
            final_amb,
            ["count_bin"],
            [
                "p_close_after_think",
                "p_any_marker_after_think",
                "p_gold_first_marker_after_think",
                "argmax_is_close",
                "argmax_is_gold_first_marker",
            ],
        ),
        key=lambda r: {"low": 0, "mid": 1, "high": 2}.get(str(r["count_bin"]), 99),
    )
    attention_top = top_rows(attention, "correct_top1", 8, mode="thinking")
    best_attn = attention_top[0] if attention_top else {}
    best_layer = int(as_float(best_attn.get("layer"))) if best_attn else -1
    best_head = int(as_float(best_attn.get("head"))) if best_attn else -1
    diag_summary = matrix_diag_summary(attention_matrix, best_layer, best_head) if best_attn else {}
    nonthinking_attn = top_rows(attention, "needle_mass", 8, mode="nonthinking")
    probe_rows = best_probe_rows(probe)
    sim_rows = sorted(group_mean(similarity, ["layer"], ["cosine_similarity"]), key=lambda r: int(as_float(r["layer"])))

    train_last = train[-1] if train else {}
    train_first = train[0] if train else {}
    thinking = final_by_mode.get("thinking", {})
    nonthinking = final_by_mode.get("nonthinking", {})
    first_non = first_perfect_step(eval_rows, "nonthinking")
    first_think = first_perfect_step(eval_rows, "thinking")

    model = config["model"]
    tr = config["train"]
    vocab_size = len(vocab.get("id_to_token", []))
    setting = setting_box(
        "实验设定",
        [
            ("模型", f"单个随机初始化 GPT-2 LM，learned absolute positional embeddings；{code(model['n_layer'])} layers, {code(model['n_head'])} heads/layer, hidden size {code(model['n_embd'])}, MLP size {code(model['n_inner'])}, context {code(model['n_positions'])}。"),
            ("词表", f"{code(vocab_size)} tokens = 4 个 special tokens + 64 个 noise tokens + 10 个 marker tokens + 10 个 final count tokens。"),
            ("数据", f"prompt 长度 {code(tr['seq_len'])}；每个 prompt 内有 count={code(tr['count_min'])}..{code(tr['count_max'])} 个 needle/marker，needle 位置随机，marker 从 10 类中采样。"),
            ("mixed-format 训练", f"一个模型同时训练两种渲染；batch 中 thinking fraction = {code(tr['thinking_fraction'])}。训练步数 {code(tr['train_steps'])}，batch size {code(tr['batch_size'])}，lr={code(tr['lr'])}，warmup={code(tr['warmup_steps'])}，seed={code(tr['seed'])}。"),
            ("thinking 渲染", f"{code('<BOS> prompt <Think/> marker_1 ... marker_n </Think> <Cn> <EOS>')}。本 run 中 {code('trace_indices=false')}，trace 里没有数字 index，只生成 marker 序列。"),
            ("non-thinking 渲染", f"{code('<BOS> prompt <Think/> </Think> <Cn> <EOS>')}。注意：no-thinking 训练只监督 final count 和 EOS，不监督 {code('</Think>')}；eval 时把 {code('</Think>')} 作为前缀给定。"),
            ("no-conflict mask", f"{code('ablate_no_conflict_mask=false')}，也就是不在 ambiguous prefix {code('<Think/>')} 上同时要求模型有时输出 marker、有时输出 {code('</Think>')}。"),
            ("评估", f"final-count accuracy 只看最后数字 token 是否等于 gold count；thinking 还报告生成的 marker trace 是否与 gold marker 序列完全一致。每个 count eval {code(tr['eval_examples_per_count'])} 个样本。"),
        ],
    )
    behavior_setting = setting_box(
        "行为实验怎么读",
        [
            ("final_accuracy", "restricted count-token argmax 是否等于 gold count；non-thinking 和 thinking 的主结果都看这个。"),
            ("trace_exact", "thinking free-run 生成的 marker trace 是否逐 token 等于 prompt 中 needle marker 序列。"),
            ("trace_marker_recall/precision", "生成 trace 与 gold marker 序列的 LCS recall/precision；用于区分局部漏 marker 和完全失败。"),
            ("ambiguous prefix", f"只给 {code('<BOS> prompt <Think/>')}，检查下一 token 概率：若 no-conflict mask 有效，模型应输出第一个 marker，而不是提前输出 {code('</Think>')}。"),
        ],
    )
    probe_setting = setting_box(
        "probe 和 attention 设置",
        [
            ("hidden cache", f"final checkpoint 上缓存 thinking/nonthinking 两种渲染的 hidden states；probe 每个 count {code(tr['probe_examples_per_count'])} 个样本。"),
            ("probe target", f"{code('final_count')} 是最终 count；thinking trace 位置还测试 {code('prefix_count')}。报告使用 centroid classifier + ridge regression 的 accuracy/R2/MAE。"),
            ("probe 注意事项", f"{code('count_pos')} 已经在 count token 本身，天然泄漏答案；很多 prefix_count anchor 的 position/trace_len baseline 也接近 1。因此 probe 是表示诊断，不是因果证据。"),
            ("attention", f"thinking 模式在每个 trace marker k 上看它是否 attend 到第 k 个 prompt needle；attention 每个 count {code(tr['attention_examples_per_count'])} 个样本。non-thinking 只看 {code('think_close_pos/pre_count_pos')} 对 prompt needles 的总 mass。"),
        ],
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic NIAH Counting v5 Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5b6475;
      --line: #dce2ec;
      --band: #f7f9fc;
      --accent: #2454d6;
      --accent-soft: #eaf0ff;
      --good: #117a39;
      --good-soft: #eaf8ef;
      --warn: #8a4b05;
      --warn-soft: #fff5df;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.65;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 72px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin-top: 42px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 24px; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; }}
    p, li {{ font-size: 16px; }}
    code {{ padding: 1px 5px; border-radius: 4px; background: #eef1f7; font-family: "SFMono-Regular", Consolas, monospace; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
    th {{ background: var(--band); text-align: left; }}
    .subtitle {{ color: var(--muted); margin-bottom: 22px; font-size: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0 28px; }}
    .card {{ padding: 14px 16px; border: 1px solid var(--line); border-radius: 10px; background: var(--band); }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    .value {{ font-weight: 800; font-size: 20px; }}
    .setting {{ margin: 18px 0 22px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 10px; background: #fbfcff; }}
    .setting-title {{ font-weight: 800; margin-bottom: 8px; font-size: 16px; }}
    .setting-table {{ margin: 0; font-size: 13.5px; }}
    .setting-table th {{ width: 190px; background: #f0f3fa; }}
    .callout {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--accent); border-radius: 8px; background: var(--accent-soft); }}
    .positive {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--good); border-radius: 8px; background: var(--good-soft); }}
    .warning {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--warn); border-radius: 8px; background: var(--warn-soft); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
    .figure {{ margin: 22px 0; padding: 16px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
    .figure img {{ width: 100%; max-height: 500px; object-fit: contain; display: block; margin: 0 auto; }}
    .figure.wide img {{ max-height: 560px; }}
    figcaption {{ color: var(--muted); margin-top: 12px; font-size: 14px; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{
      main {{ padding: 24px 16px 56px; }}
      .cards, .grid2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Synthetic NIAH Counting v5: Mixed Thinking Toggle Report</h1>
  <div class="subtitle">
    本报告整合 <code>{html.escape(str(run_dir))}</code> 中的 v5 结果：一个 GPT-2-style transformer 同时支持 thinking trace 和 no-thinking 直接答题两种格式。
  </div>

  <div class="cards">
    <div class="card"><div class="label">final non-thinking acc</div><div class="value">{pct(nonthinking.get("final_accuracy"))}</div></div>
    <div class="card"><div class="label">final thinking acc</div><div class="value">{pct(thinking.get("final_accuracy"))}</div></div>
    <div class="card"><div class="label">thinking trace exact</div><div class="value">{pct(thinking.get("trace_exact"))}</div></div>
    <div class="card"><div class="label">P(close after &lt;Think/&gt;)</div><div class="value">{fmt(amb_summary.get("p_close_after_think"))}</div></div>
  </div>

  {setting}

  <div class="positive">
    <b>一句话结论。</b>
    这次 v5 main run 是一个很干净的成功 toggle：同一个模型在 step {step} 时 non-thinking final accuracy =
    <b>{pct(nonthinking.get("final_accuracy"))}</b>，thinking final accuracy =
    <b>{pct(thinking.get("final_accuracy"))}</b>，thinking trace exact =
    <b>{pct(thinking.get("trace_exact"))}</b>。两个模式第一次同时达到完美 final accuracy / trace exact 的 checkpoint 约为
    non-thinking step <b>{first_non}</b>、thinking step <b>{first_think}</b>。
  </div>
  <div class="warning">
    <b>但 claim 边界要写清楚。</b>
    v5 证明的是“一个模型能在给定模式 token/prefix 时执行两种格式”。non-thinking eval 是把 <code>&lt;/Think&gt;</code>
    直接放进前缀后再预测 count；它不是在测试模型是否能自主决定要不要 think。
    <code>ambiguous_prefix</code> 反而说明：如果只给 <code>&lt;Think/&gt;</code>，模型几乎必然继续输出第一个 marker，而不是关闭 thinking。
  </div>

  <h2>1. 训练与行为结果</h2>
  {behavior_setting}
  <div class="grid2">
    {figure(
        figures / "train_loss_by_step_and_mode.png",
        "Figure 1. Training loss by step and supervised component",
        "横轴是 training step，纵轴是 masked next-token-prediction cross entropy。不同颜色表示 total loss、thinking trace loss、thinking final-count loss、non-thinking final-count loss。thinking trace loss 覆盖 marker trace + close/count/EOS；non-thinking final-count loss 只覆盖 count/EOS，因此不同 loss 分量的 token 数不完全可比。",
    )}
    {figure(
        figures / "final_accuracy_by_step_mode.png",
        "Figure 2. Final-count accuracy over training",
        "横轴是 training step，纵轴是 final-count exact accuracy。蓝/橙两组分别是 non-thinking 与 thinking。这里的 accuracy 都只看最终数字 token 是否等于 gold count；thinking 的 trace 是否正确由后续图单独报告。",
    )}
  </div>
  <p>
    训练在 10k steps 后几乎完全收敛：最后一条 training total loss 为
    <b>{fmt(train_last.get("loss_total"))}</b>，thinking trace loss 为
    <b>{fmt(train_last.get("loss_thinking_trace"))}</b>，thinking final-count loss 为
    <b>{fmt(train_last.get("loss_thinking_final_count"))}</b>，non-thinking final-count loss 为
    <b>{fmt(train_last.get("loss_nonthinking_final_count"))}</b>。
    早期 step 500 时，non-thinking 平均 final accuracy 约 82.2%，thinking 约 64.1%，说明两个模式不是一开始就靠简单先验答对，而是在训练中逐步学会。
  </p>
  {table_html(
      [
          {
              "mode": mode,
              "final_accuracy": pct(row.get("final_accuracy")),
              "final_mae": fmt(row.get("final_mae")),
              "trace_exact": pct(row.get("trace_exact")) if mode == "thinking" else "n/a",
              "trace_recall": pct(row.get("trace_marker_recall")) if mode == "thinking" else "n/a",
              "premature_close": pct(row.get("premature_close_rate")) if mode == "thinking" else "n/a",
              "n_examples": int(as_float(row.get("n_examples"))),
          }
          for mode, row in final_by_mode.items()
      ],
      [
          ("mode", "mode"),
          ("n_examples", "eval examples"),
          ("final_accuracy", "final acc"),
          ("final_mae", "final MAE"),
          ("trace_exact", "trace exact"),
          ("trace_recall", "trace recall"),
          ("premature_close", "premature close"),
      ],
  )}

  <div class="grid2">
    {figure(
        figures / "final_accuracy_by_count_mode.png",
        "Figure 3. Final checkpoint accuracy by exact count",
        "横轴是 gold count 1..10，纵轴是 final-count exact accuracy，颜色表示 mode。两条线都在 1.0，说明 v5 在训练分布内的所有 count 上都学会了直接读出最终计数。",
    )}
    {figure(
        figures / "trace_metrics_by_count.png",
        "Figure 4. Thinking trace quality by count",
        "横轴是 gold count，纵轴是 trace metric rate。trace_exact 表示完整 marker trace 完全匹配；precision/recall 基于生成 trace 与 gold marker 序列的 LCS；premature/missing close 是格式错误率。最终 checkpoint 全部为 trace exact=1、premature/missing close=0。",
    )}
  </div>
  {table_html(
      [
          {
              "mode": r["mode"],
              "count_bin": r["count_bin"],
              "n_examples": int(as_float(r["n_examples"])),
              "final_accuracy": pct(r["final_accuracy"]),
              "trace_exact": pct(r["trace_exact"]) if r["mode"] == "thinking" else "n/a",
              "trace_recall": pct(r["trace_marker_recall"]) if r["mode"] == "thinking" else "n/a",
              "premature_close": pct(r["premature_close_rate"]) if r["mode"] == "thinking" else "n/a",
          }
          for r in final_by_bin
      ],
      [
          ("mode", "mode"),
          ("count_bin", "count bin"),
          ("n_examples", "examples"),
          ("final_accuracy", "final acc"),
          ("trace_exact", "trace exact"),
          ("trace_recall", "trace recall"),
          ("premature_close", "premature close"),
      ],
  )}

  <h2>2. No-conflict mask 与 ambiguous prefix</h2>
  {figure(
        figures / "ambiguous_prefix_probs_by_step.png",
        "Figure 5. Ambiguous-prefix next-token probabilities",
        "横轴是 training step，纵轴是在只给出 `<BOS> prompt <Think/>` 后的下一 token 概率。三条线分别是 P(`</Think>`)、P(any marker)、P(gold first marker)。最终 P(close) 约为 1.7e-8，而 P(gold first marker) 接近 1，说明 no-conflict mask 成功避免了“同一个前缀既要关 think 又要开始 trace”的监督冲突。",
        wide=True,
  )}
  <p>
    这张图很关键，因为它解释 v5 为什么能同时支持两种格式：non-thinking 的 <code>&lt;/Think&gt;</code>
    不在 ambiguous prefix 上作为训练目标出现，因此模型在 bare <code>&lt;Think/&gt;</code> 后几乎总是继续生成 marker trace。
    如果想走 no-thinking 模式，外部必须把 <code>&lt;/Think&gt;</code> 放进前缀，模型再从这个已关闭的状态预测 count。
  </p>
  {table_html(
      [
          {
              "count_bin": r["count_bin"],
              "n_examples": int(as_float(r["n_examples"])),
              "p_close": fmt(r["p_close_after_think"]),
              "p_any_marker": pct(r["p_any_marker_after_think"], 3),
              "p_gold_first": pct(r["p_gold_first_marker_after_think"], 3),
              "argmax_close": pct(r["argmax_is_close"]),
              "argmax_gold": pct(r["argmax_is_gold_first_marker"]),
          }
          for r in amb_by_bin
      ],
      [
          ("count_bin", "count bin"),
          ("n_examples", "examples"),
          ("p_close", "P(close)"),
          ("p_any_marker", "P(any marker)"),
          ("p_gold_first", "P(gold first marker)"),
          ("argmax_close", "argmax close"),
          ("argmax_gold", "argmax gold first"),
      ],
  )}

  <h2>3. Attention: thinking 有弱到中等的 diagonal retrieval，non-thinking 更像总量聚合</h2>
  {probe_setting}
  {figure(
        figures / "attention_trace_to_prompt_best_head.png",
        "Figure 6. Thinking trace-to-prompt attention diagnostic",
        "热图横轴是 head，纵轴是 layer，颜色是 diagonal_dominance：在 thinking trace 的第 k 个 marker query 上，该 head 对第 k 个 prompt needle 的 attention mass 占所有 prompt needles attention mass 的比例。颜色越高，越像 targeted retrieval。",
        wide=True,
  )}
  <p>
    最强 thinking attention head 是 <b>L{best_layer}H{best_head}</b>：
    correct_top1 = <b>{fmt(best_attn.get("correct_top1"))}</b>，
    diagonal_dominance = <b>{fmt(best_attn.get("diagonal_dominance"))}</b>，
    needle_mass = <b>{fmt(best_attn.get("needle_mass"))}</b>。
    在 trace-to-prompt matrix 里，这个 head 的 diagonal mean =
    <b>{fmt(diag_summary.get("diag_mean"))}</b>，off-diagonal mean =
    <b>{fmt(diag_summary.get("off_diag_mean"))}</b>，最大单格 attention =
    <b>{fmt(diag_summary.get("max_attention"))}</b>。
    这说明 v5 确实形成了“第 k 个 trace marker 倾向看第 k 个 prompt needle”的结构，但它比 v2 中 index-token head 那种接近完美的 diagonal retrieval 弱很多。
  </p>
  {table_html(
      [
          {
              "head": f"L{r.get('layer')}H{r.get('head')}",
              "query_anchor": r.get("query_anchor"),
              "correct_top1": fmt(r.get("correct_top1")),
              "diagonal_dominance": fmt(r.get("diagonal_dominance")),
              "needle_mass": fmt(r.get("needle_mass")),
              "needle_vs_noise_ratio": fmt(r.get("needle_vs_noise_ratio")),
              "entropy": fmt(r.get("entropy")),
          }
          for r in attention_top
      ],
      [
          ("head", "thinking head"),
          ("query_anchor", "query"),
          ("correct_top1", "correct top1"),
          ("diagonal_dominance", "diag dominance"),
          ("needle_mass", "needle mass"),
          ("needle_vs_noise_ratio", "needle/noise"),
          ("entropy", "entropy"),
      ],
  )}
  <p>
    non-thinking 没有 trace step k，因此不能定义“第 k 个 trace token 是否看第 k 个 needle”。这里只看
    <code>think_close_pos/pre_count_pos</code> 对所有 prompt needles 的总 attention mass。最强 non-thinking heads 的 needle mass
    可到 0.73 左右，说明直接答题路径也强烈利用 prompt needle 集合，但不是 v2/v5 thinking 那种逐步 diagonal retrieval。
  </p>
  {table_html(
      [
          {
              "head": f"L{r.get('layer')}H{r.get('head')}",
              "query_anchor": r.get("query_anchor"),
              "needle_mass": fmt(r.get("needle_mass")),
              "needle_vs_noise_ratio": fmt(r.get("needle_vs_noise_ratio")),
              "entropy": fmt(r.get("entropy")),
          }
          for r in nonthinking_attn[:6]
      ],
      [
          ("head", "non-thinking head"),
          ("query_anchor", "query"),
          ("needle_mass", "needle mass"),
          ("needle_vs_noise_ratio", "needle/noise"),
          ("entropy", "entropy"),
      ],
  )}

  <h2>4. Hidden-state probes 与 mode similarity</h2>
  <div class="grid2">
    {figure(
        figures / "mode_hidden_similarity.png",
        "Figure 7. Hidden similarity between modes",
        "横轴是 layer，纵轴是 cosine similarity。对同一个 base example，分别渲染 thinking prefix 和 non-thinking prefix，到 think_close_pos 附近取 hidden state 后计算 cosine。中层相似度较低，最后层升高，说明两个模式在中间层走了不同 computation，但最终 readout 前会重新靠近。",
    )}
    {figure(
        figures / "confusion_matrix_thinking.png",
        "Figure 8. Thinking final-count confusion matrix",
        "横轴是 predicted count，纵轴是 gold count，颜色是每个 gold count 行内的预测比例。最终 checkpoint 对角线为 1，说明 thinking 模式没有系统性 under/over-count。",
    )}
  </div>
  <div class="grid2">
    {figure(
        figures / "confusion_matrix_nonthinking.png",
        "Figure 9. Non-thinking final-count confusion matrix",
        "横轴是 predicted count，纵轴是 gold count，颜色是每个 gold count 行内的预测比例。最终 checkpoint 对角线为 1，说明 no-thinking 直接答题也没有系统性 under/over-count。",
    )}
    <div>
      {table_html(
          [
              {"layer": r["layer"], "cosine_similarity": fmt(r["cosine_similarity"]), "n": int(as_float(r["n"]))}
              for r in sim_rows
          ],
          [("layer", "layer"), ("cosine_similarity", "mean cosine"), ("n", "n vectors")],
      )}
    </div>
  </div>
  <p>
    probe 结果整体很高，但需要谨慎解释。尤其 <code>count_pos</code> anchor 已经包含答案 token，
    <code>trace_len_baseline_acc</code> 对 final_count 也是 1.0，因为 trace_len 本身等于 count。
    因此这些 probe 只能说明 hidden state 中存在可读出的 count 信息，不能说明存在可 steering 的 causal count direction。
  </p>
  {table_html(
      [
          {
              "mode": r.get("mode"),
              "target": r.get("target"),
              "anchor": r.get("anchor_name"),
              "layer": r.get("layer"),
              "accuracy": fmt(r.get("accuracy")),
              "r2": fmt(r.get("r2")),
              "mae": fmt(r.get("mae")),
              "position_baseline": fmt(r.get("position_baseline_acc")),
              "trace_len_baseline": fmt(r.get("trace_len_baseline_acc")),
              "leakage_prone": r.get("leakage_prone"),
          }
          for r in probe_rows
      ],
      [
          ("mode", "mode"),
          ("target", "target"),
          ("anchor", "anchor"),
          ("layer", "best layer"),
          ("accuracy", "probe acc"),
          ("r2", "ridge R2"),
          ("mae", "MAE"),
          ("position_baseline", "position baseline"),
          ("trace_len_baseline", "trace-len baseline"),
          ("leakage_prone", "leakage-prone"),
      ],
      max_rows=28,
  )}

  <h2>5. 综合分析与下一步</h2>
  <div class="callout">
    <p><b>v5 支持的结论。</b></p>
    <p>
      在 ID 分布内，一个小型 GPT-2-style transformer 可以通过 no-conflict label mask 学会 mixed thinking toggle：
      给 <code>&lt;Think/&gt;</code> 后让它继续生成时，它会生成完整 marker trace 并给出正确 count；
      给 <code>&lt;Think/&gt; &lt;/Think&gt;</code> 后，它也能直接输出正确 count。
      因此“thinking trace”和“direct answer”并不一定需要两个模型。
    </p>
    <p>
      机制上，thinking 路径出现了 diagonal retrieval，但强度只有中等：
      best head correct_top1 约 <b>{fmt(best_attn.get("correct_top1"))}</b>，不是 v2 那种接近 1 的 targeted head。
      一个合理解释是：v5 的 trace 没有数字 index，trace token 只是 marker 本身，因此模型更可能混合使用局部生成、marker identity 和 prompt retrieval，而不是形成一个非常干净的 index-token retrieval head。
    </p>
  </div>
  <div class="warning">
    <p><b>不应过度 claim 的部分。</b></p>
    <ul>
      <li>这不是 OOD generalization 结果；当前只测 count 1..10、长度 256 的训练分布。</li>
      <li>这不是自主“是否思考”的决策实验；no-thinking 模式由外部提供 <code>&lt;/Think&gt;</code>。</li>
      <li>probe 不是因果证据；存在 count_pos、trace_len、position 等强泄漏基线。</li>
      <li>attention 是机制线索，不是充分/必要性证明。若要证明 v5 的 causal circuit，需要补 v5-specific ablation/path patching。</li>
    </ul>
  </div>
  <p>
    建议下一步做两个补充实验：第一，训练一个 <code>ablate_no_conflict_mask=true</code> 的对照，看 ambiguous prefix 是否真的崩；
    第二，对 v5 的 best thinking retrieval head 做类似 v3.2 的 local causal patching，测试 clean head output 是否能恢复 trace marker 或 final count。
  </p>

  <h2>6. 文件与复现信息</h2>
  <table>
    <tbody>
      <tr><th>run_dir</th><td>{code(run_dir)}</td></tr>
      <tr><th>config</th><td>{code(run_dir / "config.json")}</td></tr>
      <tr><th>tables</th><td>{code(run_dir / "tables")}</td></tr>
      <tr><th>figures</th><td>{code(run_dir / "figures")}</td></tr>
      <tr><th>checkpoint</th><td>{code(run_dir / "checkpoints" / "final.pt")}</td></tr>
      <tr><th>generated by</th><td>{code("scripts/build_v5_report.py")}</td></tr>
    </tbody>
  </table>
  <p class="small">
    This report was generated from v5 CSV tables and PNG figures. All images are embedded as base64 for portability.
  </p>
</main>
</body>
</html>
"""
    return html_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a detailed HTML report for Synthetic NIAH Counting v5.")
    parser.add_argument("run_dir", type=Path, help="Path to the v5 run directory containing config.json, tables, and figures.")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path. Defaults to run_dir/report.html.")
    args = parser.parse_args()
    out = args.out or args.run_dir / "report.html"
    out.write_text(build_report(args.run_dir), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
