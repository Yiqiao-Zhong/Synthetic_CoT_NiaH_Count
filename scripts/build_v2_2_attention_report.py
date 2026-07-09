from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    r"colab_results\v2_2_attention_diagnostics_seed1234_20260709_212435"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 3) -> str:
    x = number(value)
    if math.isnan(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    x = number(value)
    if math.isnan(x):
        return "NA"
    return f"{100 * x:.{digits}f}%"


def esc(value: object) -> str:
    return html.escape(str(value))


def head_label(row: dict[str, str]) -> str:
    return f"L{int(number(row.get('layer')))}H{int(number(row.get('head')))}"


def find_head(rows: Iterable[dict[str, str]], layer: int, head: int) -> dict[str, str]:
    for row in rows:
        if int(number(row.get("layer"))) == layer and int(number(row.get("head"))) == head:
            return row
    return {}


def top(rows: list[dict[str, str]], key: str, n: int = 1) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: number(row.get(key), -math.inf), reverse=True)[:n]


def table_html(
    rows: list[dict[str, object]],
    columns: list[tuple[str, str, str]],
    *,
    css_class: str = "",
) -> str:
    if not rows:
        return "<p class='muted'>该表没有可用记录。</p>"
    header = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    body: list[str] = []
    for row in rows:
        cells: list[str] = []
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "num":
                text = fmt(value)
            elif kind == "pct":
                text = pct(value)
            elif kind == "int":
                x = number(value)
                text = "NA" if math.isnan(x) else str(int(x))
            elif kind == "code":
                text = f"<code>{esc(value)}</code>"
            else:
                text = esc(value)
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<div class='table-wrap {esc(css_class)}'><table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def image_data(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def figure(
    path: Path,
    number_label: str,
    title: str,
    details: list[tuple[str, str]],
    *,
    wide: bool = False,
) -> str:
    if not path.exists():
        return (
            "<figure class='figure missing'>"
            f"<h4>{esc(number_label)}. {esc(title)}</h4>"
            f"<p>缺少图片：<code>{esc(path.name)}</code></p></figure>"
        )
    detail_html = "".join(
        f"<div><dt>{esc(label)}</dt><dd>{value}</dd></div>" for label, value in details
    )
    return f"""
    <figure class="figure{' wide' if wide else ''}">
      <h4><span>{esc(number_label)}</span>{esc(title)}</h4>
      <img src="{image_data(path)}" alt="{esc(title)}">
      <figcaption><dl>{detail_html}</dl></figcaption>
    </figure>
    """


def evidence_row(
    component: str,
    observation: str,
    evidence_type: str,
    verdict: str,
    missing: str,
    level: str,
) -> dict[str, object]:
    return {
        "component": component,
        "observation": observation,
        "evidence_type": evidence_type,
        "verdict": verdict,
        "missing": missing,
        "level": level,
    }


def mechanism_diagrams() -> str:
    return """
    <div class="mechanism-grid">
      <section class="mechanism-panel">
        <div class="mechanism-kicker">候选机制 A</div>
        <h3>Non-thinking：宽分布 prompt scan → 直接聚合 → count</h3>
        <div class="flow">
          <div class="flow-node prompt-node">
            <b>Prompt body</b>
            <div class="tokens"><span>N</span><span class="needle">M</span><span>N</span><span class="needle">M</span><span>…</span><span class="needle">M</span></div>
            <small>256 tokens；needle 稀疏分布于 noise 中</small>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-node">
            <b>分布式聚合</b>
            <small>L1 heads 对 marker 有 per-token enrichment，但大部分总质量仍覆盖整个 prompt</small>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-node answer-node"><b>&lt;Ans&gt; → n</b><small>一次 next-token readout</small></div>
        </div>
        <div class="support support-medium"><b>当前支持：</b>描述性 attention 很一致；尚无 non-thinking 专属因果 ablation。</div>
      </section>
      <section class="mechanism-panel">
        <div class="mechanism-kicker">候选机制 B</div>
        <h3>CoT：indexed targeted retrieval → trace → 分布式 readout</h3>
        <div class="flow">
          <div class="flow-node prompt-node">
            <b>Prompt needles</b>
            <div class="tokens"><span class="needle">M₁</span><span class="needle">M₂</span><span>…</span><span class="needle">Mₖ</span></div>
          </div>
          <div class="flow-arrow green">→</div>
          <div class="flow-node trace-node"><b>&lt;k&gt; ↔ Mₖ</b><small>L3H3/L3H1 对第 k 个 prompt needle 做对角检索</small></div>
          <div class="flow-arrow orange">→</div>
          <div class="flow-node answer-node"><b>Trace / prompt readout</b><small>多个 heads 聚合；最终 count 对 trace 干预高度敏感</small></div>
          <div class="flow-arrow">→</div>
          <div class="flow-node output-node"><b>n</b></div>
        </div>
        <div class="support support-strong"><b>当前支持：</b>retrieval 与 trace mediation 较强；“如何产生下一个 index”及局部 aggregation 路径仍未被因果定位。</div>
      </section>
    </div>
    """


def build_report(root: Path, output: Path) -> None:
    analysis_tables = root / "analysis" / "tables"
    analysis_figures = root / "analysis" / "figures"
    follow_tables = root / "followup_mechanism" / "tables"
    follow_figures = root / "followup_mechanism" / "figures"

    result_manifest = read_json(root / "manifest.json")
    source_root = root.parent / "v2_marker_trace_main_seed1234_20260706_215757"
    source_manifest = read_json(source_root / "manifest.json")
    source_config = source_manifest.get("config", {})

    non = read_csv(analysis_tables / "nonthinking_head_summary.csv")
    index_heads = read_csv(analysis_tables / "thinking_index_head_summary.csv")
    l3 = [row for row in index_heads if int(number(row.get("layer"))) == 3]
    answer_heads = read_csv(analysis_tables / "thinking_answer_head_summary.csv")
    conflict = read_csv(analysis_tables / "thinking_answer_prompt_trace_conflict_summary.csv")
    valid_ablation = read_csv(analysis_tables / "thinking_head_output_multi_ablation.csv")

    next_index = read_csv(follow_tables / "next_index_retrieval_head_summary.csv")
    successor = read_csv(follow_tables / "successor_transition_head_summary.csv")
    successor_ablation = read_csv(follow_tables / "successor_head_ablation_head_summary.csv")
    follow_mask = read_csv(follow_tables / "answer_multihead_mask_summary.csv")
    override_rows = read_csv(follow_tables / "trace_length_override.csv")

    non_best = top(non, "top_n_retrieval_recall", 4)
    non_max_bos = top(non, "bos_mass", 1)[0]
    non_best_needle = top(non, "prompt_needles_mass", 1)[0]
    for row in non_best:
        row["head_label"] = head_label(row)
        row["needle_enrichment"] = number(row.get("needle_per_token_mass")) / max(
            number(row.get("noise_per_token_mass")), 1e-12
        )

    l3h0 = find_head(index_heads, 3, 0)
    l3h1 = find_head(index_heads, 3, 1)
    l3h2 = find_head(index_heads, 3, 2)
    l3h3 = find_head(index_heads, 3, 3)
    l3_table: list[dict[str, object]] = []
    for row in sorted(l3, key=lambda item: int(number(item.get("head")))):
        l3_table.append(
            {
                "head": head_label(row),
                "correct_top1_rate": row.get("correct_top1_rate"),
                "diag_share_of_needle_mass": row.get("diag_share_of_needle_mass"),
                "correct_prompt_needle_mass": row.get("correct_prompt_needle_mass"),
                "all_prompt_needles_mass": row.get("all_prompt_needles_mass"),
                "prompt_noise_mass": row.get("prompt_noise_mass"),
                "bos_mass": row.get("bos_mass"),
                "plus_one_score": row.get("plus_one_score"),
            }
        )

    next_top = top(next_index, "correct_top1", 6)
    next_table: list[dict[str, object]] = []
    for row in next_top:
        next_table.append(
            {
                "head": head_label(row),
                "correct_top1": row.get("correct_top1"),
                "correct_prompt_needle_mass": row.get("correct_prompt_needle_mass"),
                "all_prompt_needles_mass": row.get("all_prompt_needles_mass"),
                "prompt_noise_mass": row.get("prompt_noise_mass"),
                "previous_trace_marker_mass": row.get("previous_trace_marker_mass"),
            }
        )
    successor_top = top(successor, "next_prompt_needle_mass", 6)
    successor_table: list[dict[str, object]] = []
    for row in successor_top:
        successor_table.append(
            {
                "head": head_label(row),
                "next_prompt_needle_mass": row.get("next_prompt_needle_mass"),
                "all_prompt_needles_mass": row.get("all_prompt_needles_mass"),
                "current_marker_self_mass": row.get("current_marker_self_mass"),
                "previous_marker_mass": row.get("previous_marker_mass"),
                "prompt_noise_mass": row.get("prompt_noise_mass"),
            }
        )

    broad_top = top(answer_heads, "broad_prompt_aggregate_score", 5)
    trace_top = top(answer_heads, "trace_readout_score", 5)
    answer_candidate_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for role, rows in (("broad prompt", broad_top), ("trace readout", trace_top)):
        for row in rows:
            label = head_label(row)
            key = f"{role}:{label}"
            if key in seen:
                continue
            seen.add(key)
            answer_candidate_rows.append(
                {
                    "role": role,
                    "head": label,
                    "prompt_mass": row.get("prompt_mass"),
                    "trace_mass": row.get("trace_mass"),
                    "needle_enrichment": row.get("needle_enrichment"),
                    "broad_prompt_aggregate_score": row.get("broad_prompt_aggregate_score"),
                    "trace_readout_score": row.get("trace_readout_score"),
                }
            )

    baseline = next((row for row in valid_ablation if row.get("condition") == "none"), {})
    selected_conditions = {
        "mask_L3H3_main_retrieval",
        "mask_L3H1_H3_retrieval",
        "mask_layer1_all",
        "mask_layer2_all",
        "mask_L3_all",
        "mask_layer4_all",
        "mask_top2_answer_broad_prompt",
        "mask_top4_answer_trace_readout",
        "mask_index_top4_plus_answer_broad_top4",
        "mask_all_16_heads_sanity",
    }
    selected_ablation: list[dict[str, object]] = []
    for row in valid_ablation:
        if row.get("condition") not in selected_conditions:
            continue
        selected_ablation.append(
            {
                "condition": row.get("condition"),
                "mask_heads": row.get("mask_heads"),
                "answer_acc": row.get("answer_acc"),
                "answer_margin_drop": number(baseline.get("answer_count_margin"))
                - number(row.get("answer_count_margin")),
                "marker_acc": row.get("trace_marker_acc"),
                "marker_margin_drop": number(baseline.get("trace_marker_margin"))
                - number(row.get("trace_marker_margin")),
            }
        )

    relation_stats: dict[str, dict[str, float]] = {}
    for relation in ("trace shorter", "equal", "trace longer"):
        relation_stats[relation] = {
            "n": 0.0,
            "follow_trace": 0.0,
            "follow_prompt": 0.0,
            "abs_error": 0.0,
        }
    for row in override_rows:
        prompt_count = int(number(row.get("prompt_count")))
        trace_count = int(number(row.get("trace_count")))
        relation = (
            "trace shorter"
            if trace_count < prompt_count
            else "trace longer"
            if trace_count > prompt_count
            else "equal"
        )
        stats = relation_stats[relation]
        stats["n"] += 1
        stats["follow_trace"] += number(row.get("follows_trace"), 0.0)
        stats["follow_prompt"] += number(row.get("follows_prompt"), 0.0)
        stats["abs_error"] += abs(number(row.get("pred_count")) - trace_count)
    override_table: list[dict[str, object]] = []
    for relation, stats in relation_stats.items():
        n = max(stats["n"], 1.0)
        override_table.append(
            {
                "relation": relation,
                "n": stats["n"],
                "follow_trace": stats["follow_trace"] / n,
                "follow_prompt": stats["follow_prompt"] / n,
                "mean_abs_pred_minus_trace": stats["abs_error"] / n,
            }
        )

    successor_mask_degenerate = bool(successor_ablation) and all(
        abs(number(row.get("margin_drop"), 0.0)) < 1e-12 for row in successor_ablation
    )
    answer_mask_degenerate = bool(follow_mask) and all(
        abs(number(row.get("margin_drop"), 0.0)) < 1e-12
        and abs(number(row.get("accuracy_drop"), 0.0)) < 1e-12
        for row in follow_mask
    )

    evidence = [
        evidence_row(
            "Non-thinking / prompt scan",
            f"L1 四个 heads 的 top-n recall 均为 1.0；最佳 head 的 prompt entropy={fmt(non_best[0].get('prompt_entropy_normalized'))}，noise mass={fmt(non_best[0].get('prompt_noise_mass'))}。",
            "attention，描述性",
            "支持宽分布扫描，并非稀疏 pinpoint retrieval。",
            "需要在 non-thinking 模型上做 head/MLP mask、marker deletion 与 activation patching。",
            "中",
        ),
        evidence_row(
            "Non-thinking / marker selection",
            f"{head_label(non_best_needle)} 的 needle mass={fmt(non_best_needle.get('prompt_needles_mass'))}；L1H2 的每-token needle/noise enrichment={fmt(number(find_head(non, 1, 2).get('needle_per_token_mass')) / max(number(find_head(non, 1, 2).get('noise_per_token_mass')), 1e-12))}。",
            "attention，描述性",
            "模型对 marker identity 有选择性，不是简单均匀平均 noise。",
            "还没有证明这些 enriched heads 对 count logit 必要。",
            "中",
        ),
        evidence_row(
            "CoT / k→k targeted retrieval",
            f"L3H3 correct top-1={pct(l3h3.get('correct_top1_rate'))}，correct needle mass={fmt(l3h3.get('correct_prompt_needle_mass'))}；plus-one score={fmt(l3h3.get('plus_one_score'))}。",
            "aligned attention + teacher-forced mask",
            "强支持第 k 个 index token 检索第 k 个 prompt needle，而非只看前一个数字做 +1。",
            "单独 mask L3H3 只降 margin、不降 accuracy；路径有冗余。",
            "强",
        ),
        evidence_row(
            "CoT / successor transition",
            f"下一个 index token 处 L3H3 top-1={pct(next_top[0].get('correct_top1'))}；前一个 marker query 处 {head_label(successor_top[0])} 给 next needle mass={fmt(successor_top[0].get('next_prompt_needle_mass'))}。",
            "attention，描述性",
            "说明下一轮检索在 index token 出现后重新对齐到下一个 needle。",
            "当前 successor 单-head causal mask 为退化全零结果，尚未定位“谁产生下一个 index”。",
            "中偏弱",
        ),
        evidence_row(
            "CoT / trace mediation",
            "±1 prompt/trace conflict 的 900 个样本均跟随 trace；trace-longer 的广域 override 跟随率为 98.9%。",
            "行为干预",
            "强支持最终答案读取 trace，而非把 trace 当装饰。",
            "trace-shorter 跟随率仅 35.4%，说明 final answer 不是 trace length 的纯函数。",
            "强但有边界",
        ),
        evidence_row(
            "CoT / final aggregation",
            f"最终 &lt;Ans&gt; 同时存在 broad head {head_label(broad_top[0])} 与 trace-readout head {head_label(trace_top[0])}；mask L1 all 后 answer accuracy={pct(next(row for row in selected_ablation if row['condition']=='mask_layer1_all')['answer_acc'])}。",
            "attention + 全序列 head mask",
            "支持分布式聚合；L1/L2 比单一 readout head 更关键。",
            "mask 未局部化到 &lt;Ans&gt;，无法把效应唯一归因于 final aggregation。",
            "中强",
        ),
    ]

    config_rows = [
        {"item": "任务", "value": "固定长度 symbolic NIAH counting；同一 base example 渲染为 non-thinking 与 thinking 两种序列"},
        {"item": "Prompt", "value": f"长度 {source_config.get('seq_len', 256)}；64 种 noise token；10 种 marker token；count 1–10 均匀采样"},
        {"item": "词表", "value": "90 tokens = 6 special + 64 noise + 10 markers + 10 single-token numbers"},
        {"item": "模型", "value": f"两套独立随机初始化 GPT-2 decoder；{source_config.get('n_layer', 4)} layers × {source_config.get('n_head', 4)} heads；d_model={source_config.get('n_embd', 256)}"},
        {"item": "位置编码", "value": f"GPT-2 learned absolute position embeddings；n_positions={source_config.get('n_positions', 320)}；不是 RoPE"},
        {"item": "训练", "value": f"{source_config.get('train_steps', 10000)} steps；batch={source_config.get('batch_size', 128)}；AdamW lr={source_config.get('learning_rate', 3e-4)}；seed={source_config.get('seed', 1234)}"},
        {"item": "Loss mask", "value": "随机 prompt 不计 loss；non-thinking 监督 <count>, <EOS>；thinking 监督完整 indexed trace、边界、最终 count 与 <EOS>"},
        {"item": "v2.2 样本", "value": f"主 attention：每 count {result_manifest.get('examples_per_count', 100)}；follow-up：每 count {result_manifest.get('followup_examples_per_count', 50)}；causal subset：每 count {result_manifest.get('followup_causal_examples_per_count', 10)}"},
    ]

    metric_rows = [
        {"metric": "category attention mass", "definition": "对 query q 与某 token 类别 C，先算 Σ_{j∈C} A(q,j)，再在样本/对齐的 trace step 上取均值。不同类别的 mass 可相加近似为 1。"},
        {"metric": "top-n retrieval recall", "definition": "仅用于 non-thinking <Ans>：在 256 个 prompt 位置中取 attention 最大的 n=gold count 个位置；与真实 needles 的交集大小除以 n。它是排名指标，不是 raw mass。"},
        {"metric": "correct top-1", "definition": "用于 thinking index token k：只在 prompt needle 位置中比较 attention；若 argmax 指向左到右第 k 个 needle，则记 1，最后平均。"},
        {"metric": "correct needle mass", "definition": "A(index_k, prompt_needle_k) 的 raw attention weight。比 top-1 更严格：top-1 可在总 needle mass 很小时仍很高。"},
        {"metric": "diagonal share", "definition": "correct_needle_mass / all_prompt_needles_mass。衡量 needle 内部是否对角对齐，但不保证该 head 把大量总 attention 放到 needles。"},
        {"metric": "plus-one score", "definition": "index_token_k 对前一个 trace index 与前一个 trace marker 的 attention mass 之和；用于检查是否主要靠局部 trace 做 +1。"},
        {"metric": "needle enrichment", "definition": "每个 needle token 的平均 mass / 每个 noise token 的平均 mass；纠正 10 个以内 needle 对 246+ noise 的基数差。"},
        {"metric": "broad prompt score", "definition": "prompt_mass × normalized_prompt_entropy × |corr(count, prompt_needles_mass)|。这是候选-head 排序用启发式，不是因果效应。"},
        {"metric": "trace readout score", "definition": "trace_mass × |corr(count, trace_mass)|。同样是启发式候选分数。"},
        {"metric": "logit margin", "definition": "gold count logit − 其余 9 个 count logits 的最大值。margin>0 表示 gold 在数字词表中排名第一；mask 后 margin 下降表示信心被削弱。"},
    ]

    css = """
    :root { color-scheme: light; --ink:#172033; --muted:#5f6b7c; --line:#dce3ec; --soft:#f5f8fc; --blue:#2f6feb; --green:#168a4b; --orange:#c76217; --red:#b42318; }
    * { box-sizing: border-box; }
    body { margin:0; background:#f3f6fa; color:var(--ink); font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.68; }
    main { max-width:1320px; margin:0 auto; background:#fff; padding:52px 64px 80px; }
    h1 { margin:0 0 8px; font-size:34px; letter-spacing:0; }
    h2 { margin:54px 0 20px; padding-top:20px; border-top:1px solid var(--line); font-size:26px; letter-spacing:0; }
    h3 { margin:26px 0 12px; font-size:19px; letter-spacing:0; }
    h4 { font-size:17px; margin:0 0 12px; letter-spacing:0; }
    p { margin:10px 0 16px; }
    code { padding:2px 6px; border-radius:4px; background:#eef2f7; color:#26364d; }
    .subtitle { color:var(--muted); margin:0; }
    .meta { margin-top:18px; color:var(--muted); font-size:14px; }
    .lead { font-size:18px; max-width:1050px; }
    .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:24px 0 10px; }
    .summary-item { border-top:3px solid var(--blue); background:var(--soft); padding:16px; min-height:120px; }
    .summary-item b { display:block; margin-bottom:6px; }
    .summary-item span { color:var(--muted); font-size:14px; }
    .callout { margin:18px 0; padding:16px 18px; background:#eef6ff; border-left:5px solid var(--blue); }
    .callout.warning { background:#fff7e8; border-color:var(--orange); }
    .callout.danger { background:#fff1f0; border-color:var(--red); }
    .callout.success { background:#edf9f1; border-color:var(--green); }
    .mechanism-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:24px 0; }
    .mechanism-panel { border:1px solid var(--line); border-radius:8px; padding:20px; }
    .mechanism-kicker { color:var(--blue); font-size:13px; font-weight:700; text-transform:uppercase; }
    .flow { display:grid; grid-template-columns:minmax(0,1.3fr) 32px minmax(0,1fr) 32px minmax(0,.75fr); align-items:stretch; gap:6px; margin:18px 0; }
    .mechanism-panel:nth-child(2) .flow { grid-template-columns:minmax(0,1.05fr) 28px minmax(0,1fr) 28px minmax(0,1fr) 28px 54px; }
    .flow-node { border:1px solid #b9c7db; background:#f8fafc; border-radius:7px; padding:12px; min-width:0; }
    .flow-node b { display:block; margin-bottom:7px; }
    .flow-node small { display:block; color:var(--muted); margin-top:7px; }
    .prompt-node { background:#eef5ff; }.trace-node { background:#eefaf2; }.answer-node { background:#fff7e8; }.output-node { display:flex; align-items:center; justify-content:center; font-size:22px; }
    .flow-arrow { align-self:center; text-align:center; font-size:28px; color:var(--blue); }.flow-arrow.green{color:var(--green)}.flow-arrow.orange{color:var(--orange)}
    .tokens { display:flex; flex-wrap:wrap; gap:4px; }.tokens span { border:1px solid #b9c7db; border-radius:4px; padding:1px 6px; background:#fff; font-size:13px; }.tokens .needle { background:#e3f6e9; border-color:#6ac28c; }
    .support { margin-top:12px; padding:10px 12px; font-size:14px; }.support-medium{background:#fff7e8}.support-strong{background:#edf9f1}
    .figure-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; align-items:start; margin:18px 0 26px; }
    .figure-grid.three { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .figure { margin:0; border:1px solid var(--line); border-radius:8px; padding:16px; background:#fff; min-width:0; }
    .figure.wide { grid-column:1/-1; }
    .figure h4 span { color:var(--blue); margin-right:7px; }
    .figure img { display:block; width:100%; height:auto; max-height:430px; object-fit:contain; margin:0 auto 12px; }
    .figure.wide img { max-height:560px; }
    figcaption { color:#344258; font-size:14px; }
    figcaption dl { margin:0; }
    figcaption dl>div { display:grid; grid-template-columns:92px 1fr; gap:8px; padding:5px 0; border-top:1px solid #edf1f6; }
    figcaption dt { font-weight:700; } figcaption dd { margin:0; }
    .table-wrap { overflow-x:auto; margin:14px 0 24px; border:1px solid var(--line); border-radius:6px; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { background:#f1f5fa; white-space:nowrap; } tbody tr:last-child td { border-bottom:0; }
    .evidence-table td:first-child { font-weight:700; min-width:180px; }
    .level { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2f7; white-space:nowrap; }
    .status-line { display:grid; grid-template-columns:190px 1fr; gap:10px; padding:8px 0; border-bottom:1px solid var(--line); }
    .status-line b { color:#26364d; }
    .audit { background:#fff8ed; padding:16px 18px; border-left:5px solid var(--orange); margin:18px 0; }
    .muted { color:var(--muted); }
    .formula { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; background:#f4f7fb; padding:10px 12px; border-radius:5px; overflow-wrap:anywhere; }
    ul,ol { padding-left:23px; } li { margin:7px 0; }
    footer { margin-top:54px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }
    @media (max-width:980px) { main{padding:36px 28px 64px}.summary{grid-template-columns:1fr 1fr}.mechanism-grid{grid-template-columns:1fr}.figure-grid,.figure-grid.three{grid-template-columns:1fr 1fr}.mechanism-panel:nth-child(2) .flow,.flow{grid-template-columns:1fr}.flow-arrow{transform:rotate(90deg)} }
    @media (max-width:680px) { main{padding:28px 16px 52px}h1{font-size:28px}.summary,.figure-grid,.figure-grid.three{grid-template-columns:1fr}.figure.wide{grid-column:auto}.status-line{grid-template-columns:1fr}.figure img,.figure.wide img{max-height:none}figcaption dl>div{grid-template-columns:1fr;gap:0} }
    @media print { body{background:#fff}main{max-width:none;padding:24px}.figure,.mechanism-panel{break-inside:avoid}.figure img{max-height:420px} }
    """

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Counting v2.2：两种 counting mechanism 的证据审计</title>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>Synthetic Counting v2.2：两种 counting mechanism 的证据审计</h1>
    <p class="subtitle">Non-thinking 的直接分布式计数，和 CoT 的 indexed retrieval + trace-mediated readout</p>
    <p class="meta">结果目录：<code>{esc(root.name)}</code> · 单 seed 1234 · 报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  </header>

  <div class="summary">
    <div class="summary-item"><b>Non-thinking</b><span>不是 BOS sink。L1 heads 对 needles 有高排名和 4.7–5.9× per-token enrichment，但总 attention 仍宽分布在 prompt。</span></div>
    <div class="summary-item"><b>CoT retrieval</b><span>L3H3/L3H1 在 index token k 处稳定检索第 k 个 prompt needle；最强 head 的 raw correct mass 达 0.831。</span></div>
    <div class="summary-item"><b>CoT readout</b><span>±1 conflict 中最终答案 100% 跟 trace；但强行缩短很多时只有 35.4% 跟 trace，说明存在边界与混合信号。</span></div>
    <div class="summary-item"><b>还没证明的部分</b><span>谁产生下一个 index、retrieval 如何局部传入 final count、non-thinking 的因果 circuit，都还缺有效的局部 patching。</span></div>
  </div>

  <h2>1. 我们希望区分的两种 counting mechanism</h2>
  <p class="lead">两种模型最终都能在 1–10 个 needles 上达到 100% 准确率，但“答对”并不说明内部算法相同。v2.2 的目标是区分：non-thinking 是否直接在 prompt 上做分布式聚合；CoT 是否先按序检索每个 needle、写入 trace，再从 trace 得到最终 count。</p>
  {mechanism_diagrams()}
  <div class="callout"><b>报告的证据标准。</b> attention 只能定位候选信息流；prompt/trace 冲突属于行为干预；真正的机制因果证据需要有效的 head/residual/path patching。本报告把三者严格分开，不把高 attention 自动写成因果。</div>

  <h2>2. 实验设定</h2>
  {table_html(config_rows, [("item","项目","text"),("value","精确定义","text")])}
  <p>两个模型使用完全相同的 base-example generator、词表和 GPT-2 架构，但参数独立训练。non-thinking 序列为 <code>&lt;BOS&gt; prompt &lt;Ans&gt; n &lt;EOS&gt;</code>；thinking 序列为 <code>&lt;BOS&gt; prompt &lt;Think/&gt; 1 M₁ 2 M₂ … n Mₙ &lt;/Think&gt; &lt;Ans&gt; n &lt;EOS&gt;</code>。最终准确率始终只看 <code>&lt;Ans&gt;</code> 后的数字。</p>

  <h3>指标如何计算</h3>
  {table_html(metric_rows, [("metric","指标","code"),("definition","计算与解释","text")])}

  <h2>3. 行为基线：两个模型都把训练分布做满了</h2>
  <p>v2 main 在 step 10,000 时，non-thinking 与 thinking 对 count 1–10 的最终答案准确率均为 1.0；thinking 的 free-run trace exact match 也为 1.0。因此这里的机制差异不是由一个模型“没学会任务”造成的。与此同时，这是同分布、单 seed、固定长度实验，不能据此推断 OOD 泛化。</p>

  <h2>4. Mechanism A：Non-thinking 更像宽分布 prompt scan，而不是 BOS sink</h2>
  <div class="callout success"><b>主结论。</b> L1 四个 heads 的 top-n recall 均为 1.0，但 attention entropy 约 0.98、noise mass 约 0.88–0.90。它们能把 needles 排到前面，却没有把全部质量稀疏集中到 needles；同时最大 BOS mass 只有 {fmt(non_max_bos.get('bos_mass'))}。这更符合“marker-sensitive 的宽分布扫描/聚合”，不符合“把计数存在 BOS”或“逐 needle pinpoint retrieval”。</div>
  <div class="figure-grid">
    {figure(analysis_figures/'nonthinking_16head_category_signature.png','Figure 1','Non-thinking：16 个 heads 的 attention 类别签名',[
        ('Query','每个样本的 <code>&lt;Ans&gt;</code> token；该位置直接预测最终数字。'),
        ('横轴','互斥 token 类别：BOS、<code>&lt;Ans&gt;</code> self、last prompt token、所有 prompt needles、所有 prompt noise、其他上下文。'),
        ('纵轴','16 个 heads，记为 L1H0–L4H3。'),
        ('颜色','该 head 分给该类别的平均 raw attention mass；同一 head 各类别近似求和为 1。'),
        ('结果','大多数 heads 的主体质量在 prompt noise；L1 heads 同时对 needle token 有显著 per-token enrichment。')
    ],wide=True)}
  </div>
  <div class="figure-grid">
    {figure(analysis_figures/'nonthinking_topn_recall_by_head.png','Figure 2','Non-thinking：top-n needle retrieval recall',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','在 prompt attention 最大的 n=gold count 个位置里，真实 needles 所占比例。'),('解读','L1 四头均为 1.0：排序能识别 needles；该指标不说明 needles 获得了大部分总质量。')
    ])}
    {figure(analysis_figures/'nonthinking_prompt_needles_mass_by_head.png','Figure 3','Non-thinking：所有 prompt needles 的 raw mass',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','<code>Σ attention(&lt;Ans&gt;, needle_j)</code>，再跨样本平均。'),('结果',f"最高为 {head_label(non_best_needle)} 的 {fmt(non_best_needle.get('prompt_needles_mass'))}；信号存在但并不占主导。")
    ])}
    {figure(analysis_figures/'nonthinking_bos_mass_by_head.png','Figure 4','Non-thinking：BOS attention mass',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','<code>attention(&lt;Ans&gt;, BOS)</code> 的样本均值。'),('结果',f"最大值 {fmt(non_max_bos.get('bos_mass'))}，不足以支持 BOS 是主要计数存储位。")
    ])}
  </div>
  {table_html(non_best,[("head_label","head","text"),("top_n_retrieval_recall","top-n recall","pct"),("prompt_needles_mass","needle mass","num"),("prompt_noise_mass","noise mass","num"),("prompt_entropy_normalized","prompt entropy","num"),("needle_enrichment","per-token enrichment","num")])}
  <p><b>当前能说：</b>non-thinking 已学会 marker-sensitive 排名，并以高熵方式覆盖 prompt。<b>当前不能说：</b>这些 L1 heads 就是 count accumulator。因为本轮没有对 non-thinking 单独做有效的 head/MLP ablation 或 prompt-marker causal patching。</p>

  <h2>5. Mechanism B 第一段：CoT 在 index token 上做 k→k targeted retrieval</h2>
  <div class="callout success"><b>最清楚的机制证据。</b>在 trace 的第 k 个 index token 上，L3H3 对第 k 个 prompt needle 的 correct top-1 为 {pct(l3h3.get('correct_top1_rate'))}，raw correct mass={fmt(l3h3.get('correct_prompt_needle_mass'))}；L3H1 分别为 {pct(l3h1.get('correct_top1_rate'))} 与 {fmt(l3h1.get('correct_prompt_needle_mass'))}。两头的 plus-one score 只有 {fmt(l3h3.get('plus_one_score'))}/{fmt(l3h1.get('plus_one_score'))}，因此它们不是主要盯着前一 trace 数字做局部 +1。</div>
  <div class="figure-grid three">
    {figure(analysis_figures/'thinking_index_correct_top1_by_head.png','Figure 5','Thinking：第 k 个 index token 的 correct top-1',[
        ('Query','teacher-forced trace 中的 numeric index token k。'),('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','只在 prompt needle 位置中比较，argmax 恰为第 k 个 needle 的比例。'),('结果','L3H3≈1.00，L3H1≈0.98；显示清晰的顺序对齐。')
    ])}
    {figure(analysis_figures/'thinking_index_correct_needle_mass_by_head.png','Figure 6','Thinking：第 k 个正确 needle 的 raw mass',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','<code>attention(index_k, prompt_needle_k)</code> 的均值。'),('解读','这比 top-1 更严格；L3H0/L3H2 虽有一定 top-1，但 raw mass 小且大部分 attention 在 noise。')
    ])}
    {figure(analysis_figures/'thinking_index_plus_one_score_by_head.png','Figure 7','Thinking：局部 plus-one score',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','index_k 对 previous index token 与 previous trace marker 的 mass 之和。'),('结果','最强 targeted heads 的值接近 0；不支持“仅看上一数字然后 +1”作为它们的主要算法。')
    ])}
  </div>
  {table_html(l3_table,[("head","L3 head","text"),("correct_top1_rate","correct top-1","pct"),("diag_share_of_needle_mass","diagonal share","num"),("correct_prompt_needle_mass","correct mass","num"),("all_prompt_needles_mass","all needle mass","num"),("prompt_noise_mass","noise mass","num"),("bos_mass","BOS mass","num"),("plus_one_score","plus-one","num")])}
  <div class="callout warning"><b>为什么 L3H2 不是同等级 retrieval head？</b>它的 correct top-1={pct(l3h2.get('correct_top1_rate'))}，看起来不低；但 correct mass 只有 {fmt(l3h2.get('correct_prompt_needle_mass'))}，noise mass 达 {fmt(l3h2.get('prompt_noise_mass'))}，BOS mass 仅 {fmt(l3h2.get('bos_mass'))}。所以问题不是 BOS，而是“在大量 noise 背景上有一个弱的正确排序”。L3H0 也类似：top-1={pct(l3h0.get('correct_top1_rate'))}，但 correct mass 只有 {fmt(l3h0.get('correct_prompt_needle_mass'))}。</div>

  <h2>6. Mechanism B 第二段：如何从第 k 项走到第 k+1 项？</h2>
  <p>follow-up 把两个时刻分开：一是 <b>marker_k 位置</b>（它要预测 next index 或 <code>&lt;/Think&gt;</code>）；二是 <b>index_{'{'}k+1{'}'} 位置</b>（它要检索下一枚 prompt needle）。结果说明后半段很清楚，前半段仍未被因果定位。</p>
  <div class="figure-grid">
    {figure(follow_figures/'next_index_correct_top1.png','Figure 8','下一轮 index token：correct next-needle top-1',[
        ('Query','teacher-forced 的 <code>index_{k+1}</code> token。'),('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','在 prompt needles 中，top-1 是否为第 k+1 个 needle。'),('结果',f"{head_label(next_top[0])}={pct(next_top[0].get('correct_top1'))}，{head_label(next_top[1])}={pct(next_top[1].get('correct_top1'))}：同一 retrieval 子电路在下一轮重新对齐。")
    ])}
    {figure(follow_figures/'next_index_correct_prompt_needle_mass.png','Figure 9','下一轮 index token：correct next-needle raw mass',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','<code>attention(index_{k+1}, prompt_needle_{k+1})</code>。'),('结果',f"{head_label(next_top[0])}={fmt(next_top[0].get('correct_prompt_needle_mass'))}，说明不只是排名优势，而是高质量 targeted retrieval。")
    ])}
    {figure(follow_figures/'successor_next_prompt_needle_mass.png','Figure 10','marker_k query 对下一枚 prompt needle 的 mass',[
        ('Query','trace 中的 marker_k；其 hidden state 预测 next index 或关闭 trace。'),('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','对 prompt_needle_{k+1} 的 raw attention mass；最后一个 marker 没有 next needle 时排除。'),('结果',f"最高为 {head_label(successor_top[0])}={fmt(successor_top[0].get('next_prompt_needle_mass'))}，但它同时给所有 needles {fmt(successor_top[0].get('all_prompt_needles_mass'))}，更像候选 successor/broad-needle head。")
    ])}
  </div>
  <div class="figure-grid">
    <div>{table_html(next_table,[("head","head","text"),("correct_top1","next top-1","pct"),("correct_prompt_needle_mass","next needle mass","num"),("all_prompt_needles_mass","all needle mass","num"),("prompt_noise_mass","noise mass","num"),("previous_trace_marker_mass","previous marker mass","num")])}</div>
    <div>{table_html(successor_table,[("head","head","text"),("next_prompt_needle_mass","next needle mass","num"),("all_prompt_needles_mass","all needle mass","num"),("current_marker_self_mass","self mass","num"),("previous_marker_mass","previous marker mass","num"),("prompt_noise_mass","noise mass","num")])}</div>
  </div>
  <div class="audit"><b>因果审计：这一段仍然缺关键证据。</b>新 follow-up 的 <code>successor_head_ablation</code> 对 16 个 heads 都返回完全相同的 clean/masked margin，所有 margin drop 精确为 0（退化={str(successor_mask_degenerate).lower()}）。这意味着该 mask 没有真正改变 forward，不能据此声称 successor 是“分布式冗余”或“没有 head 负责”。此外旧图中的 next-token margin 对所有 heads 恒为同一个 model-level 数值 17.64，也不是 head-specific 指标。当前可靠结论只有：<i>index_{'{'}k+1{'}'} 出现后，L3H3/L3H1 会检索下一枚 needle</i>；谁促成 index_{'{'}k+1{'}'} 的生成仍未知。</div>

  <h2>7. Mechanism B 第三段：最终 aggregation/readout</h2>
  <p>在最终 <code>&lt;Ans&gt;</code> query，attention 中同时出现两类候选 heads：L1 的高熵 broad-prompt heads，以及 L2/L4 的 trace-marker readout heads。它们说明模型保留了 prompt 与 trace 两路信息，但仅凭 attention 不能判断哪一路决定最终数字。</p>
  <div class="figure-grid">
    {figure(analysis_figures/'thinking_answer_16head_category_signature.png','Figure 11','Thinking 最终 &lt;Ans&gt;：16 heads 的类别签名',[
        ('Query','teacher-forced 的最终 <code>&lt;Ans&gt;</code> token。'),('横轴','prompt needles/noise、trace index/marker、BOS、think boundary、answer self 等互斥类别。'),('纵轴','L1H0–L4H3。'),('颜色','每个 head 分给该类别的平均 raw mass。'),('结果','L1 以 prompt-wide attention 为主；L2/L4 出现明显 trace-marker mass。')
    ],wide=True)}
  </div>
  <div class="figure-grid three">
    {figure(analysis_figures/'thinking_answer_broad_prompt_score_by_head.png','Figure 12','Broad prompt aggregate 候选分数',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','prompt_mass × prompt_entropy × |corr(count, prompt needle mass)|。'),('注意','这是候选排序启发式，不是因果量。L1H1 最高。')
    ])}
    {figure(analysis_figures/'thinking_answer_trace_readout_score_by_head.png','Figure 13','Trace readout 候选分数',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','trace_mass × |corr(count, trace_mass)|。'),('注意','也是启发式；L4H2 与 L2H3 最突出。')
    ])}
    {figure(analysis_figures/'thinking_answer_needle_enrichment_by_head.png','Figure 14','最终答案处 needle/noise per-token enrichment',[
        ('横轴','head index 0–3。'),('纵轴','layer 1–4。'),('颜色','mean attention per needle token / mean attention per noise token。'),('解读','纠正 prompt 中 noise 数量远多于 needle 的基数差。')
    ])}
  </div>
  {table_html(answer_candidate_rows,[("role","候选角色","text"),("head","head","text"),("prompt_mass","prompt mass","num"),("trace_mass","trace mass","num"),("needle_enrichment","needle enrichment","num"),("broad_prompt_aggregate_score","broad score","num"),("trace_readout_score","trace score","num")])}

  <h3>行为干预：最终答案是否真的使用 trace？</h3>
  <div class="figure-grid">
    {figure(analysis_figures/'thinking_answer_prompt_trace_conflict.png','Figure 15','±1 prompt/trace conflict',[
        ('干预','保持一侧 clean，把 prompt count 或 teacher-forced trace count 减 1；两侧数字冲突。'),('横轴','两种 conflict condition。'),('纵轴','follow rate 与 trace−prompt logit preference。'),('结果','两种条件各 450 例；最终预测 100% 跟随 trace，trace count logit 分别高出 prompt count 14.81 / 16.12。'),('含义','这是 trace mediation 的强行为证据，但干预幅度只有 1。')
    ])}
    {figure(follow_figures/'trace_length_override_follows_trace.png','Figure 16','广域 trace-length override',[
        ('干预','固定 prompt，强制 teacher-forced trace 长度为 1–10，再读取最终 count。'),('横轴','forced trace count。'),('纵轴','原始 prompt count。'),('颜色','最终预测是否等于 forced trace count 的比例。'),('结果','trace 比 prompt 更长时跟随率 98.9%；相等时 100%；trace 更短时仅 35.4%。'),('含义','最终答案强烈使用 trace，但短 trace 会触发训练分布外/不完整轨迹行为，不能说它是 trace length 的纯函数。')
    ])}
  </div>
  {table_html(conflict,[("condition","±1 condition","code"),("n","n","int"),("trace_follow_rate","trace follow","pct"),("prompt_follow_rate","prompt follow","pct"),("mean_trace_minus_prompt_logit","trace−prompt logit","num")])}
  {table_html(override_table,[("relation","广域 override 关系","text"),("n","n","int"),("follow_trace","trace follow","pct"),("follow_prompt","prompt follow","pct"),("mean_abs_pred_minus_trace","mean |pred−trace|","num")])}

  <h3>有效的 teacher-forced head-output mask</h3>
  <div class="callout warning"><b>如何读 causal mask。</b>这里把指定 heads 的 attention output 在完整 teacher-forced forward 中置零，再看最终 count 与 trace-marker 的 accuracy/margin。它是有效干预，但作用于所有 token positions，因此能证明某组 heads 对任务必要，不能精确证明它只在 <code>&lt;Ans&gt;</code> 做 aggregation。</div>
  <div class="figure-grid">
    {figure(analysis_figures/'thinking_head_output_multi_ablation.png','Figure 17','Thinking head-output multi-ablation',[
        ('横轴','被 mask 的单 head、head group 或整层。'),('纵轴','panel 分别给最终答案/trace-marker accuracy 与相对 baseline 的 logit-margin 变化。'),('Baseline',f"answer accuracy={pct(baseline.get('answer_acc'))}，answer margin={fmt(baseline.get('answer_count_margin'))}；marker accuracy={pct(baseline.get('trace_marker_acc'))}，marker margin={fmt(baseline.get('trace_marker_margin'))}。"),('结果','L3H3 单独主要降低 retrieval margin；L3 全层显著破坏 trace marker 但最终答案仍正确；L1/L2 全层会显著打掉最终答案和 trace。')
    ],wide=True)}
  </div>
  {table_html(selected_ablation,[("condition","mask condition","code"),("mask_heads","masked heads","code"),("answer_acc","answer acc","pct"),("answer_margin_drop","answer margin drop","num"),("marker_acc","marker acc","pct"),("marker_margin_drop","marker margin drop","num")])}
  <div class="audit"><b>不要使用新 follow-up 的第二套 answer multi-head mask 作为证据。</b>其中从 1 head 到 all 16 heads 的 margin/accuracy drop 全部精确为 0（退化={str(answer_mask_degenerate).lower()}），说明 head mask 没接入模型 forward。Figure 17 来自同一结果包的主 v2.2 ablation，能产生非零、分组特异的效应，因此报告只采用后者。</div>

  <h2>8. 证据账本：现在到底支持了什么？</h2>
  {table_html(evidence,[("component","机制环节","text"),("observation","观测","text"),("evidence_type","证据类型","text"),("verdict","当前结论","text"),("missing","仍缺什么","text"),("level","支持力度","text")],css_class='evidence-table')}

  <h2>9. 最合理的当前机制图景</h2>
  <div class="status-line"><b>Non-thinking</b><span>最符合 <b>direct distributed counting</b>：早层 heads 在高熵 prompt scan 中对 marker token 做富集/排序，后续 residual/MLP 将稀疏 marker 信号聚合为 count。BOS sink 与稀疏单-head retrieval 均不受支持。但“聚合发生在哪一层、是不是线性累加”还没被因果证明。</span></div>
  <div class="status-line"><b>CoT trace generation</b><span>最符合 <b>indexed targeted retrieval</b>：index token k 为 query，L3H3/L3H1 定位 prompt needle k；marker token 被复制进 trace。该 retrieval 不是主要靠看上一数字做 +1。</span></div>
  <div class="status-line"><b>CoT successor</b><span>index_{'{'}k+1{'}'} 出现后，L3 retrieval heads 会重新定位 needle_{'{'}k+1{'}'}。marker_k → index_{'{'}k+1{'}'} 的转换目前只看到 L2H2 等候选 attention 模式，没有有效 causal ablation，因此不能声称已找到 successor head。</span></div>
  <div class="status-line"><b>CoT final answer</b><span>最终 readout 同时可见 prompt-wide 与 trace-marker 信号；冲突干预表明 trace 对最终答案有强控制力。有效全序列 mask 又显示 L1/L2 的 distributed heads 很关键。因此最稳妥的说法是：<b>trace-mediated、但并非单一 trace-readout head 独立完成的分布式 aggregation</b>。</span></div>

  <h2>10. 缺失证据与下一轮最关键实验</h2>
  <ol>
    <li><b>修复并验证 localized head mask。</b>先做 sanity check：mask all heads 必须显著改变 logits；再在 marker_k、index_{'{'}k+1{'}'}、<code>&lt;Ans&gt;</code> 三个 query position 分别 mask。这样才能回答 successor 与 final aggregation。</li>
    <li><b>做 clean/corrupt path patching。</b>改变第 k+1 个 needle 的位置或 identity，patch marker_k residual、L2H2 output、index_{'{'}k+1{'}'} residual、L3H3/L3H1 output，测 next-index/next-marker logit recovery。</li>
    <li><b>把 non-thinking 也纳入因果分析。</b>按 L1 marker-enriched heads、全 L1、MLP1/2 分组 ablation，并做 marker deletion / noise-preserving replacement；否则 direct prompt scan 仍只是 attention-level 解释。</li>
    <li><b>把 final answer 干预局部化。</b>只在 <code>&lt;Ans&gt;</code> query mask broad heads 或 trace-readout heads，同时 patch clean trace residual 到 corrupt trace；区分“生成 trace 所必需”与“读取 trace 所必需”。</li>
    <li><b>避免把极短 forced trace 当作同分布因果干预。</b>广域 override 的不对称说明长度 1–4 对高-count prompt 是强分布外输入。后续可使用长度相同但最后 index/marker identity 被替换的 conflict，以保持格式和长度不变。</li>
    <li><b>重复 seeds。</b>目前所有机制结论都来自 seed 1234。论文级结论至少需要 3 seeds，并报告 head role 是否跨 seed 对齐或仅功能等价。</li>
  </ol>

  <h2>11. 结论</h2>
  <p class="lead"><b>目前 synthetic 结果已经清楚区分出两种内部策略：</b>non-thinking 表现为 marker-sensitive 的宽分布 prompt scan/aggregation；CoT 则形成了高度明确的 k→k targeted retrieval trace，并让 trace 强烈影响最终答案。最强的新发现不是“CoT 有一个 counting head”，而是 <b>CoT 把计数拆成了可对齐的顺序检索，再通过多层、多头的分布式 readout 得到最终 count</b>。仍未闭合的两条因果链是 marker_k 如何触发 index_{'{'}k+1{'}'}，以及 trace residual 如何在 <code>&lt;Ans&gt;</code> 处被聚合成最终数字。</p>

  <footer>本报告为单文件 HTML；所有 PNG 均以 base64 内嵌。原始 CSV、notebook 与 figures 保留在 <code>{esc(root.name)}</code> 目录中。退化 mask 输出已在正文中标记为不可解释，而未被包装成负结果。</footer>
</main>
</body>
</html>
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained v2.2 mechanism report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "syn_v2.2_report.html")).resolve()
    build_report(root, output)
    print(output)


if __name__ == "__main__":
    main()
