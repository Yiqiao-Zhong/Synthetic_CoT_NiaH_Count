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
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def mean(values: Iterable[object]) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return fmean(vals) if vals else math.nan


def fmt(value: object, digits: int = 3) -> str:
    val = as_float(value)
    if math.isnan(val):
        return html.escape(str(value))
    return f"{val:.{digits}f}"


def pct(value: object) -> str:
    val = as_float(value)
    if math.isnan(val):
        return html.escape(str(value))
    return f"{100 * val:.1f}%"


def group_mean(rows: list[dict[str, str]], keys: list[str], values: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        for value in values:
            val = as_float(row.get(value))
            if not math.isnan(val):
                grouped[key][value].append(val)
    out: list[dict[str, object]] = []
    for key, vals in grouped.items():
        item: dict[str, object] = dict(zip(keys, key))
        item["n"] = max((len(v) for v in vals.values()), default=0)
        for value in values:
            item[value] = fmean(vals[value]) if vals.get(value) else math.nan
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


def code(text: object) -> str:
    return f"<code>{html.escape(str(text))}</code>"


def head_list(heads: object) -> str:
    if not isinstance(heads, list):
        return html.escape(str(heads))
    labels = []
    for pair in heads:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            labels.append(f"L{pair[0]}H{pair[1]}")
    return ", ".join(labels) if labels else html.escape(str(heads))


def setting_box(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{value}</td></tr>"
        for label, value in rows
    )
    return f"""
    <div class="setting">
      <div class="setting-title">{html.escape(title)}</div>
      <table class="setting-table"><tbody>{body}</tbody></table>
    </div>
    """


def find_attention_dir(causal_result_dir: Path) -> Path | None:
    candidates = [
        causal_result_dir.parent / "v3_v2_attention_deepdive_seed1234_20260708_053824",
        Path("colab_results/v3_v2_attention_deepdive_seed1234_20260708_053824"),
    ]
    for candidate in candidates:
        if (candidate / "analysis" / "tables" / "last_index_head_summary.csv").exists():
            return candidate
    return None


def load_attention_recap(attention_dir: Path | None) -> dict[str, object]:
    if attention_dir is None:
        return {}
    table_dir = attention_dir / "analysis" / "tables"
    last = read_rows(table_dir / "last_index_head_summary.csv")
    ablation = read_rows(table_dir / "head_ablation_results.csv")
    if not last:
        return {}
    for row in last:
        row["head_id"] = f"L{row['layer']}H{row['head']}"
    top_retrieval = sorted(last, key=lambda r: as_float(r["correct_prompt_needle_mass"]), reverse=True)
    top_plus = sorted(last, key=lambda r: as_float(r["plus_one_score"]), reverse=True)
    return {
        "dir": attention_dir,
        "top_retrieval": top_retrieval,
        "top_plus": top_plus,
        "ablation": ablation,
    }


def selected_necessity_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_mean(
        rows,
        ["condition", "position_scope"],
        ["answer_accuracy", "gold_logit_margin", "count_shift"],
    )
    keep_positions = {"all_positions", "index_token_last", "trace_positions"}
    keep_conditions = {
        "baseline_no_ablation",
        "retrieval_L3H3_or_top1",
        "retrieval_L3H1_or_top2",
        "retrieval_top2",
        "retrieval_top4",
        "plus_one_top1",
        "plus_one_top3",
        "retrieval_plus_one_top",
        "low_score_controls",
    }
    filtered = [
        row
        for row in grouped
        if row["position_scope"] in keep_positions and row["condition"] in keep_conditions
    ]
    order = {
        "baseline_no_ablation": 0,
        "retrieval_L3H3_or_top1": 1,
        "retrieval_L3H1_or_top2": 2,
        "retrieval_top2": 3,
        "retrieval_top4": 4,
        "plus_one_top1": 5,
        "plus_one_top3": 6,
        "retrieval_plus_one_top": 7,
        "low_score_controls": 8,
    }
    pos_order = {"index_token_last": 0, "trace_positions": 1, "all_positions": 2}
    return sorted(filtered, key=lambda r: (order.get(str(r["condition"]), 99), pos_order.get(str(r["position_scope"]), 99)))


def selected_dose_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_mean(
        rows,
        ["condition", "position_scope", "alpha"],
        ["answer_accuracy", "gold_logit_margin", "count_shift", "count_expectation"],
    )
    keep_conditions = {"L3H3_or_top_retrieval", "retrieval_top4", "L2H3_or_top_plus_one", "control_top2"}
    keep_alphas = {"-1.0", "0.0", "1.5"}
    filtered = [
        row
        for row in grouped
        if row["position_scope"] == "index_token_last"
        and row["condition"] in keep_conditions
        and row["alpha"] in keep_alphas
    ]
    order = {"L3H3_or_top_retrieval": 0, "retrieval_top4": 1, "L2H3_or_top_plus_one": 2, "control_top2": 3}
    return sorted(filtered, key=lambda r: (order.get(str(r["condition"]), 99), as_float(r["alpha"])))


def selected_local_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_mean(
        rows,
        ["target_type", "site_type", "intervention_name", "position_scope"],
        ["normalized_recovery", "patched_margin"],
    )
    keep = []
    for row in grouped:
        target = row["target_type"]
        site = row["site_type"]
        name = row["intervention_name"]
        if site == "head_output" and name in {
            "retrieval_top1",
            "retrieval_top2",
            "retrieval_top4",
            "plus_one_top1_index",
            "plus_one_top1_pre_index",
            "control_top2",
        }:
            keep.append(row)
        elif site == "resid_after_block" and (
            (target == "marker_after_final_index" and name in {"resid_L2_index_last", "resid_L3_index_last", "resid_L4_index_last"})
            or (target == "answer_after_ans" and name in {"resid_L2_ans", "resid_L3_ans", "resid_L4_ans"})
        ):
            keep.append(row)
    return sorted(
        keep,
        key=lambda r: (
            str(r["target_type"]),
            0 if r["site_type"] == "head_output" else 1,
            -as_float(r["normalized_recovery"]),
        ),
    )


def selected_counterfactual_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_mean(
        rows,
        ["edit_type"],
        ["trace_minus_prompt_logit", "all_prompt_needles_mass", "last_prompt_needle_mass", "prompt_noise_mass"],
    )
    return sorted(grouped, key=lambda r: str(r["edit_type"]))


def selected_path_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_mean(
        rows,
        ["target_type", "path_name", "intervention_name"],
        ["normalized_recovery", "patched_margin"],
    )
    return sorted(grouped, key=lambda r: (str(r["target_type"]), -as_float(r["normalized_recovery"])))


def build_report(result_dir: Path, attention_dir: Path | None = None) -> str:
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    causal_dir = result_dir / "v3_2_causal"
    table_dir = causal_dir / "tables"
    fig_dir = causal_dir / "figures"

    attention = load_attention_recap(attention_dir or find_attention_dir(result_dir))
    top_attention = attention.get("top_retrieval", [])
    top_plus = attention.get("top_plus", [])

    necessity = read_rows(table_dir / "head_necessity_results.csv")
    dose = read_rows(table_dir / "head_dose_response.csv")
    local = read_rows(table_dir / "local_causal_patching_results.csv")
    activation = read_rows(table_dir / "activation_patching_results.csv")
    path_rows = read_rows(table_dir / "path_patching_results.csv")
    counter = read_rows(table_dir / "counterfactual_edit_results.csv")
    residual_direction = read_rows(table_dir / "residual_direction_results.csv")

    necessity_table = selected_necessity_rows(necessity)
    dose_table = selected_dose_rows(dose)
    local_table = selected_local_rows(local)
    path_table = selected_path_rows(path_rows)
    counter_table = selected_counterfactual_rows(counter)
    activation_table = sorted(
        group_mean(activation, ["site_type", "intervention_name", "position_scope"], ["normalized_recovery", "patched_margin"]),
        key=lambda r: -as_float(r["normalized_recovery"]),
    )[:10]

    local_lookup = {
        (r["target_type"], r["site_type"], r["intervention_name"], r["position_scope"]): r
        for r in group_mean(
            local,
            ["target_type", "site_type", "intervention_name", "position_scope"],
            ["normalized_recovery", "patched_margin"],
        )
    }
    marker_top1 = local_lookup.get(("marker_after_final_index", "head_output", "retrieval_top1", "index_token_last"), {})
    marker_top2 = local_lookup.get(("marker_after_final_index", "head_output", "retrieval_top2", "index_token_last"), {})
    marker_top4 = local_lookup.get(("marker_after_final_index", "head_output", "retrieval_top4", "index_token_last"), {})
    marker_plus = local_lookup.get(("marker_after_final_index", "head_output", "plus_one_top1_index", "index_token_last"), {})
    marker_control = local_lookup.get(("marker_after_final_index", "head_output", "control_top2", "index_token_last"), {})
    answer_top4 = local_lookup.get(("answer_after_ans", "head_output", "retrieval_top4", "index_token_last"), {})

    necessity_lookup = {
        (r["condition"], r["position_scope"]): r
        for r in group_mean(
            necessity,
            ["condition", "position_scope"],
            ["answer_accuracy", "gold_logit_margin", "count_shift"],
        )
    }
    baseline = necessity_lookup.get(("baseline_no_ablation", "index_token_last"), {})
    retrieval_top4_all = necessity_lookup.get(("retrieval_top4", "all_positions"), {})
    control_all = necessity_lookup.get(("low_score_controls", "all_positions"), {})
    retrieval_top1_last = necessity_lookup.get(("retrieval_L3H3_or_top1", "index_token_last"), {})

    counter_lookup = {r["edit_type"]: r for r in counter_table}
    delete_last = counter_lookup.get("prompt_delete_last_keep_trace", {})
    insert_last = counter_lookup.get("prompt_insert_last_keep_trace", {})
    wrong_index = counter_lookup.get("wrong_final_index_keep_prompt", {})
    wrong_prev_marker = counter_lookup.get("wrong_previous_marker_keep_prompt", {})

    if top_attention:
        best = top_attention[0]
        second = top_attention[1]
        plus = top_plus[0]
        attention_recap = f"""
        <p>
          原 v3 attention deep-dive 的关键背景是：最后一个 trace index token 上，最强 retrieval head 是
          <b>{best["head_id"]}</b>，对最后一个 prompt needle 的 attention mass 为
          <b>{fmt(best["correct_prompt_needle_mass"])}</b>，top-1 retrieval rate 为
          <b>{fmt(best["correct_prompt_needle_top1"])}</b>；第二 retrieval head 是
          <b>{second["head_id"]}</b>，correct needle mass 为 <b>{fmt(second["correct_prompt_needle_mass"])}</b>。
          最强 local / plus-one head 是 <b>{plus["head_id"]}</b>，plus-one score 为
          <b>{fmt(plus["plus_one_score"])}</b>。
        </p>
        """
    else:
        attention_recap = """
        <p>
          这个结果包只包含 v3.2 causal tests，没有附带原 v3 attention deep-dive 表格。
          报告仍然会使用 manifest 中记录的 retrieval/plus/control head 组进行 causal 解读。
        </p>
        """

    residual_note = "not run"
    if residual_direction and residual_direction[0].get("status") == "skipped":
        residual_note = f"skipped: {residual_direction[0].get('reason', '')}"

    retrieval_heads = head_list(manifest.get("retrieval_heads", []))
    plus_one_heads = head_list(manifest.get("plus_one_heads", []))
    control_heads = head_list(manifest.get("control_heads", []))
    examples_per_count = manifest.get("examples_per_count")
    patch_pairs_per_count = manifest.get("patch_pairs_per_count")
    common_setting = setting_box(
        "共同实验设置",
        [
            ("模型", f"直接读取 v2 main run 的 final <b>thinking</b> checkpoint；v3.2 不重新训练模型，只做 causal tests。"),
            ("数据范围", f"沿用 v2 的 count=1..10 评估构造；每个 count 使用 {code(examples_per_count)} 个行为/ablation 样本，patching 每个 count 使用 {code(patch_pairs_per_count)} 个 clean/corrupt pair。"),
            ("clean prompt", "正常 prompt + 正常 teacher-forced thinking trace；trace 形如逐步 marker/index 序列，最后接 final answer。"),
            ("corrupt prompt", "按实验需要修改 prompt 或 trace，使 prompt count、trace count、最后 marker/readout 发生可控冲突。"),
            ("retrieval heads", f"{code(retrieval_heads)}。这些来自 v3 attention deep-dive：在最后 trace index token 上最强地指向 prompt 中对应 needle/marker。"),
            ("plus-one heads", f"{code(plus_one_heads)}。这些来自 v3 attention deep-dive：更像局部 trace 内的 index/marker 递推或 +1 相关 head。"),
            ("control heads", f"{code(control_heads)}。这些是低分或非目标 head，用来检查 patch/ablation 是否只是任意扰动都会产生效果。"),
            ("核心指标", f"{code('normalized_recovery = (patched_margin - corrupt_base_margin) / (clean_base_margin - corrupt_base_margin)')}；接近 1 表示 patch 足以恢复 clean 行为，接近 0 表示没有恢复，负数表示往反方向推。"),
        ],
    )
    notation_setting = setting_box(
        "本文图表里的位置和 target 定义",
        [
            ("index_token_last", "thinking trace 里的最后一个计数/index token，也就是模型准备从最后一次 retrieval 进入 final marker/readout 的位置。"),
            ("pre_index_last", "最后一个 index token 前一位；用于检验 local +1 / 前驱 token 是否比 retrieval head 更关键。"),
            ("trace_positions", "thinking span 内所有 trace token 位置。"),
            ("all_positions", "整个 prefix 内所有可干预 token 位置；这是强扰动 sanity check，不是精细定位。"),
            ("ans_token", f"最终答案前的 {code('<Ans>')} token；patch 这里通常直接影响最终 readout。"),
            ("marker_after_final_index", "在最后 trace index 之后预测下一个 marker token 的局部 target；这是判断 retrieval head 是否读回最后 prompt marker 的主要因果测试。"),
            ("answer_after_ans", f"在 {code('<Ans>')} 之后预测最终 count token 的 target；它更下游、更饱和，主要作为 final readout 对照。"),
        ],
    )
    necessity_setting = setting_box(
        "Experiment 1: head necessity / zero ablation 设置",
        [
            ("问题", "如果把某组 head output 置零，最终答案是否会坏掉？这测试 necessity，但 final-answer readout 很饱和，所以只能作为粗粒度证据。"),
            ("输入", f"clean prompt + teacher-forced clean trace；共 {code(len(necessity))} 条逐样本记录。"),
            ("干预", "把指定 head group 在指定 position_scope 的 head output 置零，然后继续读 final answer logits。"),
            ("head groups", f"retrieval_top1/top2/top4 使用 {code(retrieval_heads)} 的前 1/2/4 个；plus_one_top1/top3 使用 {code(plus_one_heads)} 的前 1/3 个；low_score_controls 使用 {code(control_heads)}。"),
            ("position scopes", f"{code('index_token_last')}、{code('trace_positions')}、{code('all_positions')} 等。注意 all_positions 是强破坏对照。"),
            ("报告指标", f"final-answer {code('answer_accuracy')}、gold count 的 {code('gold_logit_margin')}、平均 {code('count_shift')}。"),
        ],
    )
    dose_setting = setting_box(
        "Experiment 2: head dose response / scaling 设置",
        [
            ("问题", "如果只在最后 index token 缩放某组 head output，final answer 是否随 alpha 单调变化？"),
            ("输入", f"clean prompt + teacher-forced clean trace；共 {code(len(dose))} 条逐样本记录。"),
            ("干预", f"在 {code('index_token_last')} 把选中 head output 乘以 {code('alpha')}；alpha=1 是原模型，alpha=0 是删除该 output，alpha<0 是反向缩放。"),
            ("alpha grid", "main run 使用多个 alpha；报告表格突出 -1.0、0.0、1.5，图中保留完整曲线。"),
            ("解释边界", "这个实验只看最终答案 readout。若结果很稳，说明 final answer 已经有冗余/饱和，不等价于 retrieval head 没有局部作用。"),
        ],
    )
    activation_setting = setting_box(
        "Experiment 3: global activation patching 设置",
        [
            ("问题", "把 clean activation patch 到 corrupt run，是否能恢复最终 count？"),
            ("clean/corrupt pair", "主要使用 prompt 少一个或 trace/prompt 不一致的 pair，让 clean/corrupt 的 final-answer margin 不同。"),
            ("patch 对象", f"head_output 或 residual stream；位置包括 {code('index_token_last')}、{code('ans_token')} 等。"),
            ("target", f"直接看 {code('answer_after_ans')} 的 final count margin。"),
            ("为什么粗", f"{code('<Ans>')} 附近 residual/readout 已经很强，head-output patch 可能接近 0，而 residual patch 可能接近或超过 1；所以这里主要是诊断，不是最终机制证据。"),
        ],
    )
    local_setting = setting_box(
        "Experiment 4: local causal patching 设置",
        [
            ("核心问题", "clean retrieval head output 是否足以把 corrupt prompt 下的“最后 marker”拉回正确值？"),
            ("clean run", "prompt 和 trace 都完整，count 为 n，最后 index 后应预测第 n 个 marker。"),
            ("corrupt run", f"{code('prompt_delete_last_keep_clean_trace')}：prompt 删除最后一个 needle/marker，但 teacher-forced trace 仍保留 clean 的完整 trace。这样 prompt count 变成 n-1，而 trace 仍声称 n。"),
            ("patch 方法", "先缓存 clean run 的 head output 或 residual，再把它替换到 corrupt run 的同一位置/同一 head。"),
            ("主要 patch", f"在 {code('index_token_last')} patch retrieval_top1/top2/top4；对照 patch plus_one_top1_index、plus_one_top1_pre_index、control_top2。"),
            ("两个 target", f"{code('marker_after_final_index')}：最后 index 后的 marker logit margin；{code('answer_after_ans')}：最终 count logit margin。前者是主测试，后者是下游对照。"),
            ("记录量", f"共 {code(len(local))} 条逐 pair/target/intervention 记录。"),
        ],
    )
    path_setting = setting_box(
        "Experiment 5: minimal path patching 设置",
        [
            ("问题", "把 local patching 结果按机制假设组织：是 prompt-final-needle → retrieval head → marker/readout，还是 trace-local +1 → plus-one head → marker/readout？"),
            ("clean/corrupt", "沿用 local causal patching 的 clean/corrupt pair 和 recovery 公式。"),
            ("path hypotheses", f"{code('final_prompt_needle_to_retrieval_head_to_marker_or_answer')}、{code('local_trace_to_plus_one_head_to_marker_or_answer')}、{code('control_path')}。"),
            ("patched component", "retrieval_top1/top2/top4、plus_one_top1_index、plus_one_top1_pre_index、control_top2 等。"),
            ("解释", "如果 retrieval path 在 marker_after_final_index 上接近 1，而 plus-one/control 接近 0，就支持 targeted retrieval 是局部 marker 恢复的主要路径。"),
        ],
    )
    counter_setting = setting_box(
        "Experiment 6: counterfactual prompt/trace edits 设置",
        [
            ("问题", "当 prompt 中实际 needle 数和 teacher-forced trace 给出的 count 冲突时，模型 final answer 更跟 prompt 还是 trace？"),
            ("prompt_delete_last_keep_trace", "删除 prompt 最后一个 needle，但保留原 trace；prompt count=n-1，trace count=n。"),
            ("prompt_insert_last_keep_trace", "额外插入一个 prompt needle，但保留原 trace；prompt count=n+1，trace count=n。"),
            ("wrong_final_index_keep_prompt", "prompt 不变，但把 trace 最后一个 index token 改成错误 count。"),
            ("wrong_previous_marker_keep_prompt", "prompt 和 count 不变，只改前一个 marker identity；这是 marker-level 干扰，不一定改变 final count。"),
            ("主要指标", f"{code('trace_minus_prompt_logit = logit(trace_count) - logit(prompt_count)')}；正值表示 final answer 更跟 trace count。"),
            ("attention 指标", "同时记录 L3H3 对 all prompt needles、last prompt needle、prompt noise 的 attention mass，判断它是不是盲目看物理最后一个 needle。"),
            ("记录量", f"共 {code(len(counter))} 条 counterfactual 记录。"),
        ],
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trace Count v3.2 Causal Tests Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #586278;
      --line: #dce2ec;
      --band: #f6f8fb;
      --accent: #2454d6;
      --accent-soft: #eaf0ff;
      --warn: #8a4b05;
      --warn-soft: #fff5df;
      --good: #117a39;
      --good-soft: #eaf8ef;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.66;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 28px 72px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin-top: 42px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 24px; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; }}
    p, li {{ font-size: 16px; }}
    code {{ padding: 1px 5px; border-radius: 4px; background: #eef1f7; font-family: "SFMono-Regular", Consolas, monospace; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
    th {{ background: var(--band); text-align: left; }}
    .setting {{ margin: 18px 0 22px; padding: 14px 16px; border: 1px solid var(--line); border-radius: 10px; background: #fbfcff; }}
    .setting-title {{ font-weight: 800; margin-bottom: 8px; font-size: 16px; }}
    .setting-table {{ margin: 0; font-size: 13.5px; }}
    .setting-table th {{ width: 190px; background: #f0f3fa; }}
    .setting-table td {{ color: #263149; }}
    .subtitle {{ color: var(--muted); margin-bottom: 22px; font-size: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0 28px; }}
    .card {{ padding: 14px 16px; border: 1px solid var(--line); border-radius: 10px; background: var(--band); }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    .value {{ font-weight: 700; font-size: 20px; }}
    .callout {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--accent); border-radius: 8px; background: var(--accent-soft); }}
    .warning {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--warn); border-radius: 8px; background: var(--warn-soft); }}
    .positive {{ margin: 20px 0; padding: 14px 18px; border-left: 5px solid var(--good); border-radius: 8px; background: var(--good-soft); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
    .figure {{ margin: 22px 0; padding: 16px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
    .figure img {{ width: 100%; max-height: 560px; object-fit: contain; display: block; margin: 0 auto; }}
    .figure.wide img {{ max-height: 660px; }}
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
  <h1>Trace Count v3.2: Causal Tests for the v2 Attention Mechanism</h1>
  <div class="subtitle">
    目标：把 v3 中“最后 trace index 有 targeted retrieval head”的 attention 发现，推进到更强的 causal evidence。
  </div>

  <div class="cards">
    <div class="card"><div class="label">preset</div><div class="value">{html.escape(str(manifest.get("preset")))}</div></div>
    <div class="card"><div class="label">examples/count</div><div class="value">{manifest.get("examples_per_count")}</div></div>
    <div class="card"><div class="label">patch pairs/count</div><div class="value">{manifest.get("patch_pairs_per_count")}</div></div>
    <div class="card"><div class="label">saved_at</div><div class="value">{html.escape(str(manifest.get("saved_at")))}</div></div>
  </div>

  {common_setting}
  {notation_setting}

  <div class="callout">
    <b>一句话结论。</b>
    v3.2 的结果明显加强了 v3 的解释：retrieval heads 不是只“看起来像 retrieval”。
    在更局部、更敏感的 marker-after-final-index patching 测试中，clean retrieval head output 可以把 corrupt prompt 的最后 marker 预测恢复：
    L3H3/retrieval_top1 recovery = <b>{fmt(marker_top1.get("normalized_recovery"))}</b>，
    retrieval_top2 = <b>{fmt(marker_top2.get("normalized_recovery"))}</b>，
    retrieval_top4 = <b>{fmt(marker_top4.get("normalized_recovery"))}</b>。
    相比之下 plus-one head 在同一位置的 recovery = <b>{fmt(marker_plus.get("normalized_recovery"))}</b>，
    control heads = <b>{fmt(marker_control.get("normalized_recovery"))}</b>。
    这支持“最后 trace index 通过 retrieval heads 读取 prompt 中对应/最后 needle marker”的因果路径。
  </div>

  <h2>1. 背景：v3 attention 发现</h2>
  {attention_recap}
  <p>
    v3.2 不重新训练模型，而是使用同一个 v2 thinking checkpoint，在构造好的 clean/corrupt prompts 上做
    ablation、activation patching、local patching、path patching 和 counterfactual edit。
    这份报告把重点放在三类问题：哪些 head 是必要的、哪些 head 足以恢复局部 marker/readout、prompt 与 trace 冲突时模型跟谁走。
  </p>

  <h2>2. Head necessity：单点 ablation 不是最终答案瓶颈</h2>
  {necessity_setting}
  {figure(
        fig_dir / "head_necessity_answer_margin.png",
        "Figure 1. Zero-ablation effect on final-answer margin",
        "横轴是 ablated head group / token scope，纵轴是 final-answer gold logit margin 或相关 readout 指标。这里主要看 teacher-forced final readout 的稳健性：只在最后 index token ablate retrieval / plus-one heads 几乎不影响答案；大范围 all-positions ablation 才明显伤害某些 head group。",
        wide=True,
    )}
  <p>
    最后 index token 上单独 ablate retrieval_top1/L3H3，answer accuracy 仍为
    <b>{fmt(retrieval_top1_last.get("answer_accuracy"))}</b>，margin 仍为
    <b>{fmt(retrieval_top1_last.get("gold_logit_margin"))}</b>；baseline margin 为
    <b>{fmt(baseline.get("gold_logit_margin"))}</b>。
    这说明 final answer readout 很饱和，不能把“单 head ablation 不掉点”解释成“该 head 没有机制作用”。
  </p>
  <p>
    当 ablation 扩大到 all positions，retrieval_top4 accuracy 降到
    <b>{fmt(retrieval_top4_all.get("answer_accuracy"))}</b>，control heads all-positions accuracy 降到
    <b>{fmt(control_all.get("answer_accuracy"))}</b>。这个结果说明大范围 ablation 会引入广泛破坏，因此它适合做 sanity check，不适合单独定位精细路径。
  </p>
  {table_html(necessity_table, [
      ("condition", "condition"),
      ("position_scope", "scope"),
      ("answer_accuracy", "answer acc"),
      ("gold_logit_margin", "gold margin"),
      ("count_shift", "mean count shift"),
      ("n", "n"),
  ], max_rows=28)}

  <h2>3. Dose response：final answer 对 head scaling 也很稳</h2>
  {dose_setting}
  {figure(
        fig_dir / "head_dose_response_answer_margin.png",
        "Figure 2. Scaling selected head outputs at the final index",
        "横轴是 alpha，表示把指定 head output 做缩放/插值；纵轴是 final-answer margin 或相关 readout。retrieval 和 plus-one heads 的缩放几乎不改变答案，说明 final-answer readout 已经很饱和；control_top2 在 alpha=-1 时会伤害结果，提示该控制组不是完全无关的无害 head。",
        wide=True,
    )}
  {table_html(dose_table, [
      ("condition", "condition"),
      ("alpha", "alpha"),
      ("answer_accuracy", "answer acc"),
      ("gold_logit_margin", "gold margin"),
      ("count_shift", "count shift"),
      ("count_expectation", "count expectation"),
      ("n", "n"),
  ])}

  <h2>4. Activation patching：final-answer target 太粗，local target 更有信息</h2>
  {activation_setting}
  <div class="warning">
    <b>为什么早期 patching 图看起来“不对”？</b>
    如果 patch 的目标直接设成 <code>&lt;Ans&gt;</code> 后的 final count，模型已经有很强的 teacher-forced trace readout，
    因此很多 head-output patch 的 normalized recovery 接近 0；而 patch <code>&lt;Ans&gt;</code> token 的 residual stream 又会接近 1，
    这更多是在测试 readout residual，而不是测试 retrieval head 是否把 prompt needle 读出来。
    所以 v3.2 的更关键结果是下一节的 local causal patching：在最后 trace index 后预测 marker。
  </div>
  {table_html(activation_table, [
      ("site_type", "site"),
      ("intervention_name", "intervention"),
      ("position_scope", "scope"),
      ("normalized_recovery", "recovery"),
      ("patched_margin", "patched margin"),
      ("n", "n"),
  ])}

  <h2>5. Local causal patching：retrieval heads 足以恢复最后 marker</h2>
  {local_setting}
  {figure(
        fig_dir / "local_patching_marker_and_answer_recovery_by_head_group.png",
        "Figure 3. Local causal patching: marker-after-index vs final-answer readout",
        "横轴是 patched head group 和 token position，纵轴是 normalized recovery。蓝/不同面板分别比较 marker_after_final_index 与 answer_after_ans。marker_after_final_index 是更干净的局部 causal target：它问 clean prompt 的 retrieval head output 是否能把 corrupt prompt 下一个 marker 拉回正确值。",
        wide=True,
    )}
  <div class="positive">
    <b>最重要的结果。</b>
    对 <code>marker_after_final_index</code>，retrieval_top1/L3H3 的 recovery 是
    <b>{fmt(marker_top1.get("normalized_recovery"))}</b>，retrieval_top2 是
    <b>{fmt(marker_top2.get("normalized_recovery"))}</b>，retrieval_top4 是
    <b>{fmt(marker_top4.get("normalized_recovery"))}</b>。
    这几乎就是从 clean prompt 恢复最后 marker 的 causal signature。
    同一目标下 plus_one_top1_index recovery 是 <b>{fmt(marker_plus.get("normalized_recovery"))}</b>，
    control_top2 是 <b>{fmt(marker_control.get("normalized_recovery"))}</b>，说明不是任意 head patch 都能恢复。
  </div>
  <p>
    对 <code>answer_after_ans</code>，retrieval_top4 也有 recovery
    <b>{fmt(answer_top4.get("normalized_recovery"))}</b>，但弱于 marker target。
    这符合预期：final answer 会混入更下游的 residual/readout 路径，而 marker-after-index 更接近 targeted retrieval 本身。
  </p>
  {table_html(local_table, [
      ("target_type", "target"),
      ("site_type", "site"),
      ("intervention_name", "intervention"),
      ("position_scope", "scope"),
      ("normalized_recovery", "recovery"),
      ("patched_margin", "patched margin"),
      ("n", "n"),
  ], max_rows=24)}

  {figure(
        fig_dir / "patching_recovery_by_layer_position.png",
        "Figure 4. Local residual patching by target, layer, and position",
        "横轴是 patched residual stream after layer，纵轴是 patched token position，颜色是 normalized recovery。index token 的 residual 在后层恢复 marker，<Ans> token 的 residual 恢复 final readout。注意 answer_after_ans 上大于 1 的值表示 overshoot，不应被解释成更强的 clean causal path，只说明该 residual 位置直接携带强 readout 信息。",
        wide=True,
    )}

  <h2>6. Minimal path patching：prompt-final-needle → retrieval head → marker/readout</h2>
  {path_setting}
  {figure(
        fig_dir / "path_patching_recovery_heatmap.png",
        "Figure 5. Minimal path patching recovery heatmap",
        "横轴是 patched component，纵轴是 causal path hypothesis，颜色是 normalized recovery。最显著的路径是 final_prompt_needle_to_retrieval_head_to_marker_or_answer，尤其在 marker_after_final_index target 上，retrieval_top1/top2/top4 都有强恢复；local_trace_to_plus_one_head 路径恢复很弱。",
        wide=True,
    )}
  {table_html(path_table, [
      ("target_type", "target"),
      ("path_name", "path hypothesis"),
      ("intervention_name", "component"),
      ("normalized_recovery", "recovery"),
      ("patched_margin", "patched margin"),
      ("n", "n"),
  ], max_rows=18)}

  <h2>7. Counterfactual edits：prompt 和 trace 冲突时模型更跟 trace count</h2>
  {counter_setting}
  <div class="grid2">
    {figure(
        fig_dir / "counterfactual_prompt_vs_trace_logits.png",
        "Figure 6. Prompt-vs-trace counterfactual logits",
        "这里构造 prompt count 与 teacher-forced trace count 不一致的情况。纵轴/柱高反映 trace-count logit 相对 prompt-count logit 的优势。正值表示模型 final answer 更跟 trace count。",
    )}
    {figure(
        fig_dir / "counterfactual_l3h3_attention_shift.png",
        "Figure 7. L3H3 attention under counterfactual edits",
        "比较 L3H3 对 all prompt needles、last prompt needle 和 prompt noise 的 mass。prompt_insert_last_keep_trace 中，新增 prompt needle 不是 trace 对应对象，所以 last_prompt_needle_mass 很低；这支持 L3H3 不是盲目看物理最后一个 needle，而更像看 trace-index 对齐的 needle。",
    )}
  </div>
  <p>
    删除最后 prompt needle 但保留原 trace 时，trace-minus-prompt logit = <b>{fmt(delete_last.get("trace_minus_prompt_logit"))}</b>；
    插入额外 prompt needle 但保留原 trace 时，trace-minus-prompt logit = <b>{fmt(insert_last.get("trace_minus_prompt_logit"))}</b>；
    改错 final index 但保留 prompt 时，trace-minus-prompt logit = <b>{fmt(wrong_index.get("trace_minus_prompt_logit"))}</b>。
    这些都说明最终答案更跟 teacher-forced trace count，而不是直接重新数 prompt。
  </p>
  <p>
    但注意，wrong_previous_marker_keep_prompt 的 trace-minus-prompt logit = <b>{fmt(wrong_prev_marker.get("trace_minus_prompt_logit"))}</b>，
    因为它不改变 count 本身，只改变局部 marker identity。这类 edit 更适合用 marker-level target 而不是 final count target 来解释。
  </p>
  {table_html(counter_table, [
      ("edit_type", "edit"),
      ("trace_minus_prompt_logit", "trace - prompt logit"),
      ("all_prompt_needles_mass", "all needle mass"),
      ("last_prompt_needle_mass", "last needle mass"),
      ("prompt_noise_mass", "prompt noise mass"),
      ("n", "n"),
  ])}

  <h2>8. 综合解释</h2>
  <div class="callout">
    <p><b>当前最稳妥的机制叙述：</b></p>
    <p>
      v2 thinking model 并不是纯粹靠“上一个数字 +1”生成最后计数。
      它至少包含一个 strong targeted retrieval circuit：最后 trace index token 通过 L3H3/L3H1 等 retrieval heads 读取 prompt 中对应/最后 needle 的 marker；
      这个 retrieval output 对“下一步 marker 是什么”有强 causal effect。
      local plus-one heads 也存在，但在 v3.2 的 local patching 中，它们不能替代 retrieval heads 恢复最后 marker。
    </p>
    <p>
      对 final answer 来说，模型更依赖已经生成/teacher-forced 的 trace count，且 readout 很饱和。
      所以 final-answer accuracy 对单 head ablation 和 scaling 不敏感；这不是反证 retrieval，而是说明 final answer 是更下游、更冗余的 readout。
    </p>
  </div>
  <ul>
    <li><b>支持的结论：</b> L3H3/retrieval head group 对最后 marker retrieval 有强因果作用。</li>
    <li><b>不应过度 claim：</b> 单个 L3H3 不是最终答案的唯一瓶颈；final answer readout 有冗余和饱和。</li>
    <li><b>下一步：</b> 用 v6 separator trace 检查去掉显式数字 index 后是否仍形成 diagonal retrieval；再做 top-k retrieval ablation 和 value/path patching。</li>
  </ul>

  <h2>9. 文件与复现信息</h2>
  <table>
    <tbody>
      <tr><th>result_dir</th><td><code>{html.escape(str(result_dir))}</code></td></tr>
      <tr><th>source_v2_run_dir</th><td><code>{html.escape(str(manifest.get("source_v2_run_dir", "")))}</code></td></tr>
      <tr><th>thinking_model_dir</th><td><code>{html.escape(str(manifest.get("thinking_model_dir", "")))}</code></td></tr>
      <tr><th>causal_dir</th><td><code>{html.escape(str(manifest.get("causal_dir", "")))}</code></td></tr>
      <tr><th>retrieval_heads</th><td><code>{html.escape(str(manifest.get("retrieval_heads", "")))}</code></td></tr>
      <tr><th>plus_one_heads</th><td><code>{html.escape(str(manifest.get("plus_one_heads", "")))}</code></td></tr>
      <tr><th>control_heads</th><td><code>{html.escape(str(manifest.get("control_heads", "")))}</code></td></tr>
      <tr><th>residual directions</th><td><code>{html.escape(residual_note)}</code></td></tr>
    </tbody>
  </table>
  <p class="small">
    Generated from <code>v3_2_causal/tables/*.csv</code> and <code>v3_2_causal/figures/*.png</code>.
    The report intentionally distinguishes attention evidence, sufficiency-style patching evidence, and final-answer behavioral robustness.
  </p>
</main>
</body>
</html>
"""
    return html_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report for Trace Count v3.2 causal tests.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--attention-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.result_dir / "syn_v3_report.html"
    out.write_text(build_report(args.result_dir, args.attention_dir), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
