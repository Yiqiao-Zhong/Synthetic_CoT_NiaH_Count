from __future__ import annotations

import csv
import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(
    r"colab_results\v2_2_attention_diagnostics_seed1234_20260709_170849"
)
TABLE_DIR = ROOT / "analysis" / "tables"
FIG_DIR = ROOT / "analysis" / "figures"


def read_csv(name: str) -> list[dict[str, str]]:
    path = TABLE_DIR / name
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 3) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "NA"
    return f"{100 * x:.{digits}f}%"


def esc(value: object) -> str:
    return html.escape(str(value))


def head_label(row: dict[str, str]) -> str:
    return f"L{int(as_float(row['layer']))}H{int(as_float(row['head']))}"


def row_for(rows: list[dict[str, str]], layer: int, head: int) -> dict[str, str]:
    for row in rows:
        if int(as_float(row.get("layer"))) == layer and int(as_float(row.get("head"))) == head:
            return row
    raise KeyError((layer, head))


def sort_top(rows: list[dict[str, str]], key: str, n: int = 8, reverse: bool = True) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: as_float(r.get(key)), reverse=reverse)[:n]


def table_html(rows: list[dict[str, object]], columns: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    th = "".join(f"<th>{esc(label)}</th>" for _, label, _ in columns)
    body = []
    for row in rows:
        cells = []
        for key, _, kind in columns:
            value = row.get(key, "")
            if kind == "num":
                text = fmt(value)
            elif kind == "pct":
                text = pct(value)
            elif kind == "int":
                text = str(int(as_float(value))) if not math.isnan(as_float(value)) else "NA"
            elif kind == "code":
                text = f"<code>{esc(value)}</code>"
            else:
                text = esc(value)
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def figure(filename: str, title: str, caption: str, css: str = "") -> str:
    path = FIG_DIR / filename
    if not path.exists():
        return (
            f"<div class='figure missing'><h4>{esc(title)}</h4>"
            f"<p>Missing figure: <code>{esc(filename)}</code></p></div>"
        )
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    src = f"data:image/png;base64,{payload}"
    return f"""
    <figure class="figure {css}">
      <h4>{esc(title)}</h4>
      <img src="{esc(src)}" alt="{esc(title)}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def metric_card(title: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-title">{esc(title)}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def mechanism_svg_non_thinking() -> str:
    return """
    <svg class="mechanism-svg" viewBox="0 0 980 270" role="img" aria-label="non-thinking mechanism diagram">
      <defs>
        <marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
        </marker>
        <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
        </marker>
        <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#16a34a" />
        </marker>
      </defs>
      <text x="24" y="34" class="svg-title">Non-thinking: direct final-answer readout</text>
      <rect x="30" y="72" width="620" height="86" rx="12" class="svg-box prompt" />
      <text x="50" y="99" class="svg-label">Prompt body: noise tokens + counted marker tokens</text>
      <g class="tokens">
        <rect x="54" y="118" width="38" height="24" rx="5" class="noise" /><text x="65" y="135">N</text>
        <rect x="104" y="118" width="38" height="24" rx="5" class="needle" /><text x="114" y="135">M</text>
        <rect x="154" y="118" width="38" height="24" rx="5" class="noise" /><text x="165" y="135">N</text>
        <rect x="204" y="118" width="38" height="24" rx="5" class="needle" /><text x="214" y="135">M</text>
        <rect x="254" y="118" width="38" height="24" rx="5" class="noise" /><text x="265" y="135">N</text>
        <rect x="304" y="118" width="38" height="24" rx="5" class="needle" /><text x="314" y="135">M</text>
        <text x="362" y="136" class="svg-muted">...</text>
        <rect x="430" y="118" width="38" height="24" rx="5" class="noise" /><text x="441" y="135">N</text>
        <rect x="480" y="118" width="38" height="24" rx="5" class="needle" /><text x="490" y="135">M</text>
        <rect x="530" y="118" width="38" height="24" rx="5" class="noise" /><text x="541" y="135">N</text>
      </g>
      <rect x="750" y="92" width="110" height="56" rx="12" class="svg-box answer" />
      <text x="784" y="126" class="svg-label">&lt;Ans&gt;</text>
      <rect x="886" y="92" width="64" height="56" rx="12" class="svg-box output" />
      <text x="910" y="126" class="svg-label">n</text>
      <path d="M520,118 C625,60 690,74 748,102" class="arrow gray thin" marker-end="url(#arrow-gray)" />
      <path d="M160,118 C440,25 635,42 750,98" class="arrow green" marker-end="url(#arrow-green)" />
      <path d="M260,118 C475,42 640,58 750,112" class="arrow green" marker-end="url(#arrow-green)" />
      <path d="M360,118 C520,62 652,72 750,126" class="arrow green" marker-end="url(#arrow-green)" />
      <path d="M840,120 L884,120" class="arrow blue" marker-end="url(#arrow-blue)" />
      <text x="78" y="202" class="svg-note">Evidence: no large BOS sink; early heads spread mass over prompt, with marker enrichment above noise.</text>
      <text x="78" y="230" class="svg-note">Interpretation: a direct, distributed prompt-scan/readout path rather than an explicit sequential scratchpad.</text>
    </svg>
    """


def mechanism_svg_thinking() -> str:
    return """
    <svg class="mechanism-svg" viewBox="0 0 980 360" role="img" aria-label="thinking mechanism diagram">
      <defs>
        <marker id="arrow-t-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
        </marker>
        <marker id="arrow-t-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#16a34a" />
        </marker>
        <marker id="arrow-t-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
          <path d="M0,0 L0,6 L9,3 z" fill="#f97316" />
        </marker>
      </defs>
      <text x="24" y="34" class="svg-title">CoT / thinking: targeted retrieval + trace-mediated final answer</text>
      <rect x="30" y="68" width="430" height="92" rx="12" class="svg-box prompt" />
      <text x="50" y="96" class="svg-label">Prompt body</text>
      <rect x="72" y="118" width="34" height="24" rx="5" class="needle" /><text x="82" y="135">M₁</text>
      <rect x="150" y="118" width="34" height="24" rx="5" class="needle" /><text x="160" y="135">M₂</text>
      <rect x="228" y="118" width="34" height="24" rx="5" class="needle" /><text x="238" y="135">M₃</text>
      <text x="298" y="136" class="svg-muted">...</text>
      <rect x="360" y="118" width="34" height="24" rx="5" class="needle" /><text x="370" y="135">Mₙ</text>
      <rect x="540" y="70" width="350" height="92" rx="12" class="svg-box trace" />
      <text x="560" y="96" class="svg-label">&lt;think&gt; trace</text>
      <rect x="570" y="118" width="34" height="24" rx="5" class="index" /><text x="582" y="135">1</text>
      <rect x="610" y="118" width="34" height="24" rx="5" class="needle" /><text x="620" y="135">M₁</text>
      <rect x="658" y="118" width="34" height="24" rx="5" class="index" /><text x="670" y="135">2</text>
      <rect x="698" y="118" width="34" height="24" rx="5" class="needle" /><text x="708" y="135">M₂</text>
      <text x="742" y="136" class="svg-muted">...</text>
      <rect x="800" y="118" width="34" height="24" rx="5" class="index" /><text x="812" y="135">n</text>
      <rect x="840" y="118" width="34" height="24" rx="5" class="needle" /><text x="850" y="135">Mₙ</text>
      <path d="M588,118 C482,22 186,28 92,115" class="arrow green" marker-end="url(#arrow-t-green)" />
      <path d="M676,118 C520,22 264,28 170,115" class="arrow green" marker-end="url(#arrow-t-green)" />
      <path d="M818,118 C622,18 454,44 378,116" class="arrow green" marker-end="url(#arrow-t-green)" />
      <rect x="580" y="238" width="118" height="58" rx="12" class="svg-box answer" />
      <text x="612" y="273" class="svg-label">&lt;Ans&gt;</text>
      <rect x="728" y="238" width="72" height="58" rx="12" class="svg-box output" />
      <text x="754" y="273" class="svg-label">n</text>
      <path d="M840,144 C820,192 720,212 652,238" class="arrow orange" marker-end="url(#arrow-t-orange)" />
      <path d="M640,144 C642,188 642,210 642,238" class="arrow orange" marker-end="url(#arrow-t-orange)" />
      <path d="M380,145 C470,198 550,214 610,240" class="arrow blue thin" marker-end="url(#arrow-t-blue)" />
      <path d="M696,267 L726,267" class="arrow blue" marker-end="url(#arrow-t-blue)" />
      <text x="55" y="222" class="svg-note">Trace generation: index-token k retrieves prompt needle k. L3H3/L3H1 are the clearest targeted heads.</text>
      <text x="55" y="250" class="svg-note">Final answer: conflict tests show the model follows the trace when prompt count and trace count disagree.</text>
      <text x="55" y="278" class="svg-note">Ablations: no single head is the full circuit; multi-head/layer masks reveal distributed aggregation and readout.</text>
    </svg>
    """


def mechanism_svg_non_thinking() -> str:
    return """
    <div class="flow-title">Non-thinking: direct prompt scan -> final count</div>
    <div class="flow-row">
      <div class="flow-box prompt">
        <div class="flow-label">1. Prompt body</div>
        <div class="token-strip">
          <span class="tok">N</span><span class="tok needle">M</span><span class="tok">N</span>
          <span class="tok needle">M</span><span class="tok">N</span><span class="tok needle">M</span>
          <span class="tok">...</span><span class="tok needle">M</span><span class="tok">N</span>
        </div>
        <p class="flow-subtitle">N = noise token; M = counted marker/needle.</p>
      </div>
      <div class="flow-arrow green">==></div>
      <div class="flow-box answer">
        <div class="flow-label">2. Final-answer query</div>
        <div class="token-strip"><span class="tok answer">&lt;Ans&gt;</span></div>
        <p class="flow-subtitle">Query position is the token immediately before predicting the final count.</p>
      </div>
      <div class="flow-arrow">==></div>
      <div class="flow-box output">
        <div class="flow-label">3. Count logit</div>
        <div class="token-strip"><span class="tok index">n</span></div>
        <p class="flow-subtitle">No explicit trace tokens are available.</p>
      </div>
    </div>
    <div class="evidence-list">
      <div class="evidence-item"><b>Broad prompt mass.</b> Most attention stays on prompt body; prompt entropy is high.</div>
      <div class="evidence-item"><b>Marker enrichment.</b> Some L1 heads rank needles above noise, but raw needle mass is small.</div>
      <div class="evidence-item"><b>Not BOS sink.</b> Max BOS mass is around 0.09, so BOS is not the dominant storage site.</div>
    </div>
    """


def mechanism_svg_thinking() -> str:
    return """
    <div class="flow-title">CoT / thinking: targeted retrieval -> trace -> final count</div>
    <div class="flow-row long">
      <div class="flow-box prompt">
        <div class="flow-label">1. Prompt needles</div>
        <div class="token-strip">
          <span class="tok needle">M1</span><span class="tok needle">M2</span>
          <span class="tok needle">M3</span><span class="tok">...</span><span class="tok needle">Mk</span>
        </div>
        <p class="flow-subtitle">Needles are ordered by occurrence in the prompt.</p>
      </div>
      <div class="flow-arrow green">==></div>
      <div class="flow-box trace">
        <div class="flow-label">2. Trace generation</div>
        <div class="token-strip">
          <span class="tok index">1</span><span class="tok needle">M1</span>
          <span class="tok index">2</span><span class="tok needle">M2</span>
          <span class="tok">...</span><span class="tok index">k</span><span class="tok needle">Mk</span>
        </div>
        <p class="flow-subtitle">At index token k, L3H3/L3H1 retrieve prompt needle k.</p>
      </div>
      <div class="flow-arrow orange">==></div>
      <div class="flow-box answer">
        <div class="flow-label">3. Final-answer readout</div>
        <div class="token-strip"><span class="tok answer">&lt;Ans&gt;</span></div>
        <p class="flow-subtitle">Conflict tests show the final answer follows the trace.</p>
      </div>
      <div class="flow-arrow">==></div>
      <div class="flow-box output">
        <div class="flow-label">4. Count</div>
        <div class="token-strip"><span class="tok index">n</span></div>
      </div>
    </div>
    <div class="evidence-list">
      <div class="evidence-item"><b>Targeted retrieval.</b> L3H3 has near-perfect top-1 and high correct-needle mass.</div>
      <div class="evidence-item"><b>Not local +1.</b> L3H3/L3H1 have very low plus-one score.</div>
      <div class="evidence-item"><b>Distributed final readout.</b> Broad prompt heads and trace-readout heads both affect margins.</div>
    </div>
    """


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    non = read_csv("nonthinking_head_summary.csv")
    idx = read_csv("thinking_index_head_summary.csv")
    ans = read_csv("thinking_answer_head_summary.csv")
    conflict = read_csv("thinking_answer_prompt_trace_conflict_summary.csv")
    ablation = read_csv("thinking_head_output_multi_ablation.csv")
    l3 = read_csv("thinking_L3_head_summary.csv")

    best_non_needles = sort_top(non, "prompt_needles_mass", 1)[0]
    best_non_bos = sort_top(non, "bos_mass", 1)[0]
    best_non_topn = sort_top(non, "top_n_retrieval_recall", 4)

    l3h3 = row_for(idx, 3, 3)
    l3h1 = row_for(idx, 3, 1)
    l3h2 = row_for(idx, 3, 2)
    best_plus = sort_top(idx, "plus_one_score", 1)[0]

    top_broad = sort_top(ans, "broad_prompt_aggregate_score", 1)[0]
    top_trace = sort_top(ans, "trace_readout_score", 1)[0]
    top_enrich = sort_top(ans, "needle_enrichment", 1)[0]

    base = next(r for r in ablation if r["condition"] == "none")
    base_ans_acc = as_float(base["answer_acc"])
    base_ans_margin = as_float(base["answer_count_margin"])
    base_marker_acc = as_float(base["trace_marker_acc"])
    base_marker_margin = as_float(base["trace_marker_margin"])
    abl_rows: list[dict[str, object]] = []
    for row in ablation:
        out: dict[str, object] = dict(row)
        out["d_answer_acc"] = as_float(row["answer_acc"]) - base_ans_acc
        out["d_answer_margin"] = as_float(row["answer_count_margin"]) - base_ans_margin
        out["d_marker_acc"] = as_float(row["trace_marker_acc"]) - base_marker_acc
        out["d_marker_margin"] = as_float(row["trace_marker_margin"]) - base_marker_margin
        abl_rows.append(out)

    selected_conditions = [
        "mask_L3H3_main_retrieval",
        "mask_L3H1_H3_retrieval",
        "mask_L3_all",
        "mask_index_top8_correct",
        "mask_layer1_all",
        "mask_top4_answer_broad_prompt",
        "mask_answer_broad_top4_plus_trace_top4",
        "mask_all_16_heads_sanity",
    ]
    selected_abl = [r for cond in selected_conditions for r in abl_rows if r["condition"] == cond]
    worst_marker = sorted(
        [r for r in abl_rows if r["condition"] != "none"],
        key=lambda r: as_float(r["d_marker_margin"]),
    )[:8]
    worst_answer = sorted(
        [r for r in abl_rows if r["condition"] != "none"],
        key=lambda r: as_float(r["d_answer_margin"]),
    )[:8]

    conflict_rows = [
        {
            "condition": r["condition"],
            "n": r["n"],
            "trace_follow_rate": r["trace_follow_rate"],
            "prompt_follow_rate": r["prompt_follow_rate"],
            "mean_trace_minus_prompt_logit": r["mean_trace_minus_prompt_logit"],
        }
        for r in conflict
    ]

    non_table = []
    for row in sort_top(non, "prompt_needles_mass", 6):
        non_table.append(
            {
                "head": head_label(row),
                "top_n_retrieval_recall": row["top_n_retrieval_recall"],
                "prompt_needles_mass": row["prompt_needles_mass"],
                "prompt_noise_mass": row["prompt_noise_mass"],
                "bos_mass": row["bos_mass"],
                "prompt_entropy_normalized": row["prompt_entropy_normalized"],
                "needle_enrichment": as_float(row["needle_per_token_mass"]) / max(as_float(row["noise_per_token_mass"]), 1e-12),
            }
        )

    idx_table = []
    for row in sorted(l3, key=lambda r: int(as_float(r["head"]))):
        idx_table.append(
            {
                "head": head_label(row),
                "correct_top1_rate": row["correct_top1_rate"],
                "diag_share_of_needle_mass": row["diag_share_of_needle_mass"],
                "correct_prompt_needle_mass": row["correct_prompt_needle_mass"],
                "all_prompt_needles_mass": row["all_prompt_needles_mass"],
                "prompt_noise_mass": row["prompt_noise_mass"],
                "plus_one_score": row["plus_one_score"],
            }
        )

    answer_table = []
    answer_heads = []
    for row in sort_top(ans, "broad_prompt_aggregate_score", 4):
        answer_heads.append(row)
    for row in sort_top(ans, "trace_readout_score", 4):
        if row not in answer_heads:
            answer_heads.append(row)
    for row in answer_heads:
        answer_table.append(
            {
                "head": head_label(row),
                "prompt_mass": row["prompt_mass"],
                "prompt_needles_mass": row["prompt_needles_mass"],
                "prompt_noise_mass": row["prompt_noise_mass"],
                "trace_mass": row["trace_mass"],
                "needle_enrichment": row["needle_enrichment"],
                "broad_prompt_aggregate_score": row["broad_prompt_aggregate_score"],
                "trace_readout_score": row["trace_readout_score"],
            }
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v2.2 Attention Diagnostics Report</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #59657a;
      --line: #dce3ef;
      --soft: #f6f8fb;
      --blue: #2563eb;
      --green: #16a34a;
      --orange: #f97316;
      --red: #dc2626;
      --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #eef3fb 0%, #f8fafc 260px);
      font-family: "Segoe UI", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
      line-height: 1.62;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 28px 58px;
    }}
    .hero {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 32px 34px;
      box-shadow: 0 18px 50px rgba(31, 45, 75, 0.08);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 34px;
      letter-spacing: 0;
      line-height: 1.16;
    }}
    h2 {{
      margin: 42px 0 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      font-size: 25px;
    }}
    h3 {{ margin: 28px 0 10px; font-size: 19px; }}
    h4 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 8px 0 12px; }}
    code {{
      background: #eef2f7;
      padding: 2px 5px;
      border-radius: 5px;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.94em;
    }}
    .muted {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 14px;
      margin: 22px 0 4px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      background: #fbfdff;
      border-radius: 14px;
      padding: 16px 16px 14px;
    }}
    .metric-title {{ color: var(--muted); font-size: 13px; font-weight: 650; }}
    .metric-value {{ margin: 8px 0 5px; font-size: 23px; font-weight: 800; }}
    .metric-note {{ color: var(--muted); font-size: 13px; }}
    .section-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      margin: 18px 0;
    }}
    .callout {{
      border-left: 5px solid var(--blue);
      background: #eef5ff;
      border-radius: 12px;
      padding: 14px 16px;
      margin: 16px 0;
    }}
    .callout.green {{ border-left-color: var(--green); background: #f0fdf4; }}
    .callout.orange {{ border-left-color: var(--orange); background: #fff7ed; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      align-items: stretch;
      margin: 16px 0;
    }}
    .fig-grid.three {{
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    .figure {{
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 16px;
      padding: 15px;
      margin: 16px 0;
    }}
    .figure img {{
      display: block;
      width: 100%;
      max-height: 500px;
      object-fit: contain;
      border-radius: 10px;
      background: white;
    }}
    .figure.compact img {{ max-height: 410px; }}
    .figure.wide img {{ max-height: 500px; }}
    figcaption {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      background: white;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      padding: 9px 10px;
    }}
    th {{
      background: #f1f5f9;
      font-weight: 750;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: start;
    }}
    .mechanism {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin: 16px 0;
    }}
    .flow-title {{
      font-size: 19px;
      font-weight: 850;
      margin: 2px 0 14px;
    }}
    .flow-subtitle {{
      color: var(--muted);
      font-size: 14px;
      margin: 6px 0 0;
    }}
    .flow-row {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto minmax(220px, 1fr) auto minmax(130px, 0.6fr);
      gap: 12px;
      align-items: center;
      margin: 14px 0;
    }}
    .flow-row.long {{
      grid-template-columns: minmax(240px, 1fr) auto minmax(260px, 1fr) auto minmax(190px, 0.75fr) auto minmax(110px, 0.45fr);
    }}
    .flow-box {{
      border: 1px solid #cbd5e1;
      border-radius: 14px;
      padding: 14px;
      min-height: 96px;
      background: #f8fafc;
    }}
    .flow-box.prompt {{ background: #eff6ff; }}
    .flow-box.trace {{ background: #f0fdf4; }}
    .flow-box.answer {{ background: #fff7ed; }}
    .flow-box.output {{ background: #fef2f2; }}
    .flow-arrow {{
      color: var(--blue);
      font-size: 28px;
      font-weight: 900;
      text-align: center;
      white-space: nowrap;
    }}
    .flow-arrow.green {{ color: var(--green); }}
    .flow-arrow.orange {{ color: var(--orange); }}
    .flow-label {{
      font-weight: 800;
      margin-bottom: 8px;
    }}
    .token-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
    }}
    .tok {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 32px;
      min-height: 28px;
      padding: 2px 8px;
      border-radius: 8px;
      border: 1px solid #94a3b8;
      background: #e2e8f0;
      font-weight: 750;
      font-family: Consolas, "SFMono-Regular", monospace;
    }}
    .tok.needle {{ background: #bbf7d0; border-color: #16a34a; color: #14532d; }}
    .tok.index {{ background: #dbeafe; border-color: #2563eb; color: #1e3a8a; }}
    .tok.answer {{ background: #fed7aa; border-color: #f97316; color: #7c2d12; }}
    .evidence-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .evidence-item {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 14px;
    }}
    .mechanism-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .svg-title {{ font-size: 21px; font-weight: 800; fill: #172033; }}
    .svg-label {{ font-size: 16px; font-weight: 700; fill: #172033; }}
    .svg-muted {{ font-size: 18px; fill: #64748b; }}
    .svg-note {{ font-size: 15px; fill: #475569; }}
    .svg-box {{ stroke: #cbd5e1; stroke-width: 1.5; }}
    .svg-box.prompt {{ fill: #eff6ff; }}
    .svg-box.trace {{ fill: #f0fdf4; }}
    .svg-box.answer {{ fill: #fff7ed; }}
    .svg-box.output {{ fill: #fef2f2; }}
    .noise {{ fill: #cbd5e1; stroke: #94a3b8; }}
    .needle {{ fill: #bbf7d0; stroke: #16a34a; }}
    .index {{ fill: #dbeafe; stroke: #2563eb; }}
    .tokens text, svg text {{ font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }}
    .arrow {{ fill: none; stroke-width: 3; }}
    .arrow.thin {{ stroke-width: 1.8; opacity: 0.72; }}
    .arrow.gray {{ stroke: #94a3b8; }}
    .arrow.blue {{ stroke: #2563eb; }}
    .arrow.green {{ stroke: #16a34a; }}
    .arrow.orange {{ stroke: #f97316; }}
    ul {{ margin-top: 8px; }}
    li {{ margin: 7px 0; }}
    @media (max-width: 820px) {{
      main {{ padding: 18px 12px 34px; }}
      .hero {{ padding: 22px 18px; }}
      h1 {{ font-size: 27px; }}
      .two-col, .fig-grid {{ grid-template-columns: 1fr; }}
      .flow-row, .flow-row.long {{ grid-template-columns: 1fr; }}
      .flow-arrow {{ transform: rotate(90deg); }}
      .figure img {{ max-height: 420px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>v2.2 Attention Diagnostics：synthetic counting 的注意力与机制分析</h1>
    <p class="muted">
      结果目录：<code>{esc(str(ROOT))}</code><br>
      报告生成时间：{esc(generated_at)}；分析 bundle 生成时间：<code>{esc(manifest.get("created_at", "NA"))}</code>。
    </p>
    <p>
      这份报告整合 v2.2 补充实验，核心问题是：non-thinking 模型到底把注意力放在哪里；
      CoT/thinking 模型的 L3 heads 是否真的在做 targeted retrieval；
      最终答案阶段是否依赖 trace；以及 multi-head mask ablation 是否支持
      “targeted retrieval + aggregate/readout”的 counting mechanism。
    </p>
    <div class="cards">
      {metric_card("Non-thinking 不是 BOS sink", f"{head_label(best_non_bos)} BOS={fmt(best_non_bos['bos_mass'])}", "最大 BOS attention 也只有约 0.09；主要质量仍在 prompt body。")}
      {metric_card("Non-thinking 最强 marker-enriched head", f"{head_label(best_non_needles)} needle={fmt(best_non_needles['prompt_needles_mass'])}", f"同一 head 的 noise mass={fmt(best_non_needles['prompt_noise_mass'])}，说明是宽分布扫描，不是只盯 needle。")}
      {metric_card("CoT 最强 targeted retrieval", f"L3H3 top-1={pct(l3h3['correct_top1_rate'])}", f"correct needle mass={fmt(l3h3['correct_prompt_needle_mass'])}，plus-one score={fmt(l3h3['plus_one_score'])}。")}
      {metric_card("最终答案跟随 trace", "trace-follow=100%", "两类 prompt/trace conflict 条件下，最终答案都跟随 trace 而不是 prompt。")}
    </div>
  </section>

  <h2>1. 实验设定</h2>
  <div class="section-card">
    <table>
      <tbody>
        <tr><th>来源模型</th><td>v2 main run：<code>v2_marker_trace_main_seed1234_20260706_215757</code> final checkpoint。</td></tr>
        <tr><th>模型架构</th><td>GPT-2 style causal Transformer，4 layers × 4 heads，hidden size 256，learned absolute position embeddings；这里不是 RoPE。</td></tr>
        <tr><th>任务</th><td>synthetic NIAH-style counting。Prompt 中混合 noise token 和 marker/needle token；模型输出最终 count。v2 主要控制的是 marker token 类型集合相同、数量不同。</td></tr>
        <tr><th>count 范围</th><td>gold count 为 1 到 10；本分析每个 count 采样 <code>{esc(manifest.get("examples_per_count", "NA"))}</code> 个样本。</td></tr>
        <tr><th>两个模型条件</th><td><b>non-thinking</b>：prompt 后直接在 <code>&lt;Ans&gt;</code> 预测 count。<b>thinking/CoT</b>：prompt 后生成 <code>&lt;think&gt;</code> trace，例如 index/marker 交替序列，然后在 <code>&lt;Ans&gt;</code> 预测 count。</td></tr>
        <tr><th>本报告分析对象</th><td>final checkpoint 的 attention 分布、prompt-vs-trace conflict、以及 teacher-forced head-output masking ablation。</td></tr>
      </tbody>
    </table>
    <p>
      注意：这里的 ablation 是 <b>teacher-forced forward pass</b> 中的 head-output mask。
      它能说明某些 head/layer 对当前 token 的 logit margin 与 accuracy 有局部因果影响，
      但不完全等价于 autoregressive generation 中的连锁错误。
    </p>
  </div>

  <h2>2. 指标定义</h2>
  <div class="section-card">
    <table>
      <thead><tr><th>指标</th><th>如何计算</th><th>解释</th></tr></thead>
      <tbody>
        <tr><td><code>top_n_retrieval_recall</code></td><td>对 non-thinking 的 <code>&lt;Ans&gt;</code> query，取 prompt body 中 attention 最高的 n 个位置，n 为真实 count；看其中有多少是真实 needle，除以 n。</td><td>衡量最终答案 token 是否能在 prompt 中把所有 needles 排到最前面。</td></tr>
        <tr><td><code>correct_top1_rate</code></td><td>对 thinking trace 的 <code>index_token_k</code> query，找 attention 最高的 prompt needle；判断它是否是第 k 个 needle。</td><td>衡量 targeted retrieval 是否按 trace index 对齐到对应 needle。</td></tr>
        <tr><td><code>diag_share_of_needle_mass</code></td><td>在所有 prompt-needle attention mass 中，落在正确第 k 个 needle 上的比例。</td><td>越高说明 needle attention 更“对角线化”。</td></tr>
        <tr><td><code>plus_one_score</code></td><td><code>previous_index_token_mass + previous_marker_token_mass</code>。</td><td>用于检测 trace token 是否主要看前一个 trace 数字/marker，从而更像局部 <code>+1</code> continuation。</td></tr>
        <tr><td><code>needle_enrichment</code></td><td>每个 needle token 的平均 attention mass / 每个 noise token 的平均 attention mass。</td><td>消除 needle 数量少、noise 数量多带来的基数差异。</td></tr>
        <tr><td><code>broad_prompt_aggregate_score</code></td><td><code>prompt_mass × prompt_entropy_normalized × |corr_count_prompt_needles_mass|</code>。</td><td>一个启发式分数：高值表示 head 在最终答案处广泛扫 prompt，同时对 count-relevant prompt needles 有信号。</td></tr>
        <tr><td><code>trace_readout_score</code></td><td><code>trace_mass × |corr_count_trace_mass|</code>。</td><td>一个启发式分数：高值表示 head 在最终答案处更多读 trace，并且 trace 上的 count-relevant mass 有信号。</td></tr>
        <tr><td><code>logit margin</code></td><td>正确 count/marker token 的 logit 减去主要竞争 token 的 logit。</td><td>accuracy 可能仍为 1，但 margin 下降表示电路余量变小。</td></tr>
      </tbody>
    </table>
    <h3>Attention 统计的共同计算流程</h3>
    <ol>
      <li>对每个样本做一次 teacher-forced forward pass，并要求模型返回所有 layer/head 的 attention tensor。</li>
      <li>选定 query position：non-thinking 使用最终 <code>&lt;Ans&gt;</code> token；thinking trace 使用每个 trace <code>index_token_k</code>；thinking final answer 使用最终 <code>&lt;Ans&gt;</code> token。</li>
      <li>对每个 query、每个 layer/head，取它对所有 previous tokens 的 attention weight。因为是 causal Transformer，所以每一行 attention 在可见 token 上求和为 1。</li>
      <li>把 previous tokens 按角色分组，例如 <code>BOS</code>、<code>prompt_needles</code>、<code>prompt_noise</code>、<code>trace_index</code>、<code>trace_marker</code>、<code>previous_index_token</code>、<code>previous_marker_token</code> 等。</li>
      <li>对一个类别的 mass 是该 head 对该类别所有 token positions 的 attention weight 求和；图里的数值通常是先在样本内求类别 mass，再对 query tokens、counts、examples 求平均。</li>
    </ol>
    <h3>每张图的横纵坐标和数值定义</h3>
    <table>
      <thead><tr><th>图</th><th>query / 样本单位</th><th>横轴</th><th>纵轴</th><th>颜色或柱高如何计算</th></tr></thead>
      <tbody>
        <tr><td>Figure 1</td><td>non-thinking 的最终 <code>&lt;Ans&gt;</code> query；每个样本一行 attention。</td><td>token category：BOS、answer self、last prompt token、prompt needles、prompt noise 等。</td><td>16 个 heads，记作 LxHy。</td><td>该 head 对该 category 的 attention mass；先对类别内 token 求和，再对样本平均。</td></tr>
        <tr><td>Figure 2</td><td>non-thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>top_n_retrieval_recall</code>：在 prompt body 里取 attention 最高的 n 个位置，n=gold count，计算其中真实 needles 的比例。</td></tr>
        <tr><td>Figure 3</td><td>non-thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td>对所有 prompt needle positions 的 attention mass 之和，再对样本平均。</td></tr>
        <tr><td>Figure 4</td><td>non-thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td>attention 到 BOS position 的 mass，再对样本平均。</td></tr>
        <tr><td>Figure 5</td><td>thinking trace 中的每个 <code>index_token_k</code> query；每个样本贡献 k=1..count 多个 query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>correct_top1_rate</code>：最高 attention 的 prompt needle 是否等于第 k 个 needle；对所有 index queries 平均。</td></tr>
        <tr><td>Figure 6</td><td>thinking trace 的 <code>index_token_k</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td>attention 到正确第 k 个 prompt needle 的 raw mass；对所有 index queries 平均。</td></tr>
        <tr><td>Figure 7</td><td>thinking trace 的 <code>index_token_k</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>plus_one_score = previous_index_token_mass + previous_marker_token_mass</code>，用于测试是否主要看本地上一个 trace token。</td></tr>
        <tr><td>Figure 8</td><td>thinking trace 的 <code>index_token_k</code> query，只看第 3 层四个 heads。</td><td>attention category。</td><td>L3H0-L3H3。</td><td>每个 L3 head 对各 category 的平均 attention mass。</td></tr>
        <tr><td>Figure 9</td><td>thinking trace 的 <code>index_token_k</code> query，只看 L3。</td><td>L3 head。</td><td>top-attended role 的比例或 mass 汇总。</td><td>对每个 query 找最高 attention token 的角色，再统计该角色出现频率；用于区分 top-1 角色和 raw mass。</td></tr>
        <tr><td>Figure 10</td><td>thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>token category：prompt needles/noise、trace index/marker、BOS、think boundary 等。</td><td>16 个 heads，记作 LxHy。</td><td>该 head 对每个 category 的 attention mass，类别内求和后对样本平均。</td></tr>
        <tr><td>Figure 11</td><td>prompt/trace conflict 样本；人为让 prompt count 与 trace count 不一致。</td><td>conflict condition。</td><td>follow rate 或 logit preference。</td><td>看最终预测 count 是否等于 prompt count 或 trace count；logit preference 是 trace count logit 减 prompt count logit。</td></tr>
        <tr><td>Figure 12</td><td>thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>prompt_mass × prompt_entropy_normalized × |corr_count_prompt_needles_mass|</code>。</td></tr>
        <tr><td>Figure 13</td><td>thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>trace_mass × |corr_count_trace_mass|</code>。</td></tr>
        <tr><td>Figure 14</td><td>thinking 的最终 <code>&lt;Ans&gt;</code> query。</td><td>head index 0-3。</td><td>layer index 1-4。</td><td><code>needle_enrichment = needle_per_token_mass / noise_per_token_mass</code>。</td></tr>
        <tr><td>Figure 15</td><td>teacher-forced head-output mask ablation。</td><td>被 mask 的 head group / condition。</td><td>accuracy 或 logit-margin change。</td><td>把指定 heads 的 attention output 置零，重新计算 final answer 与 trace marker 的 accuracy/margin，相对 no-mask baseline 做差。</td></tr>
      </tbody>
    </table>
  </div>

  <h2>3. Non-thinking：不是 BOS sink，而是宽 prompt scan + marker enrichment</h2>
  <div class="callout">
    <b>结论。</b>non-thinking 最终答案不是主要靠 BOS attention sink，也不是只盯少数 needle。
    它的早期 heads 能把 needles 排到 top-n 里，但 raw attention mass 仍大量分布在 prompt noise 上，
    更像一种直接、分布式的 prompt-scan/readout。
  </div>
  <div class="mechanism">{mechanism_svg_non_thinking()}</div>

  <div class="fig-grid">
    {figure(
        "nonthinking_16head_category_signature.png",
        "Figure 1. Non-thinking 16 heads 的注意力类别签名",
        "横轴是 attention category，包括 BOS、answer self、last prompt token、prompt needles、prompt noise 等；纵轴是 head（LxHy）；颜色是该 head 在 <code>&lt;Ans&gt;</code> query 处分给该类别的平均 attention mass。图中可见大部分质量在 prompt body，尤其是 prompt noise，而不是 BOS。",
        "wide",
    )}
    {figure(
        "nonthinking_topn_recall_by_head.png",
        "Figure 2. Non-thinking top-n needle retrieval recall",
        "横轴是 head，纵轴是 layer，颜色是 <code>top_n_retrieval_recall</code>。虽然 L1 heads 可达到 1.0，表示它们能把真实 needles 排进 top-n prompt positions，但这不等于 attention mass 只集中在 needles。",
        "compact",
    )}
  </div>
  <div class="fig-grid">
    {figure(
        "nonthinking_prompt_needles_mass_by_head.png",
        "Figure 3. Non-thinking prompt needles mass",
        "横轴是 head，纵轴是 layer，颜色是 <code>&lt;Ans&gt;</code> query 给所有 prompt needles 的 raw attention mass。最高约为 0.11，说明 needle signal 存在但并不独占注意力。",
        "compact",
    )}
    {figure(
        "nonthinking_bos_mass_by_head.png",
        "Figure 4. Non-thinking BOS mass",
        "横轴是 head，纵轴是 layer，颜色是 attention 到 BOS 的质量。最大值约为 {fmt(best_non_bos['bos_mass'])}，不足以解释最终计数，因而不支持主要是 BOS sink 的解释。",
        "compact",
    )}
  </div>
  <h3>Non-thinking top heads</h3>
  {table_html(non_table, [
      ("head", "head", "text"),
      ("top_n_retrieval_recall", "top-n recall", "num"),
      ("prompt_needles_mass", "needle mass", "num"),
      ("prompt_noise_mass", "noise mass", "num"),
      ("bos_mass", "BOS mass", "num"),
      ("prompt_entropy_normalized", "prompt entropy", "num"),
      ("needle_enrichment", "needle/noise enrichment", "num"),
  ])}
  <p>
    关键读法：L1H2/L1H1/L1H3 都有很高的 top-n recall，但它们的 prompt noise mass 约 0.88。
    这说明它们确实“知道”哪些位置更像 needle，但最终不是一个稀疏 pinpoint retrieval；
    它更像在 learned absolute position + token identity 表示上做一个宽分布聚合。
  </p>

  <h2>4. CoT trace generation：L3H3/L3H1 是 targeted retrieval，不是单纯 +1</h2>
  <div class="callout green">
    <b>结论。</b>在 thinking trace 的 <code>index_token_k</code> 处，L3H3 是最强 targeted retrieval head：
    correct top-1 为 {pct(l3h3['correct_top1_rate'])}，correct needle mass 为 {fmt(l3h3['correct_prompt_needle_mass'])}。
    它的 <code>plus_one_score</code> 只有 {fmt(l3h3['plus_one_score'])}，说明它不是主要看前一个 trace 数字再做 +1。
  </div>
  <div class="mechanism">{mechanism_svg_thinking()}</div>

  <div class="fig-grid three">
    {figure(
        "thinking_index_correct_top1_by_head.png",
        "Figure 5. Thinking index-token correct top-1 retrieval",
        "横轴是 head，纵轴是 layer，颜色是 <code>correct_top1_rate</code>。query 是 trace 中的第 k 个 index token；如果最高 attention 的 prompt needle 正好是第 k 个 needle，则记为正确。L3H3 和 L3H1 最突出。",
        "compact",
    )}
    {figure(
        "thinking_index_correct_needle_mass_by_head.png",
        "Figure 6. Thinking index-token correct needle mass",
        "横轴是 head，纵轴是 layer，颜色是 raw attention mass 到正确第 k 个 prompt needle。这个图比 top-1 更严格，因为 top-1 高但 mass 小可能只是排序上的弱偏好。",
        "compact",
    )}
    {figure(
        "thinking_index_plus_one_score_by_head.png",
        "Figure 7. Thinking index-token plus-one score",
        "横轴是 head，纵轴是 layer，颜色是 previous index token + previous marker token 的 attention mass。若模型只是在 trace 内部做局部 +1，这里应当很高；但 L3H3/L3H1 的 targeted retrieval 与 plus-one score 不匹配。",
        "compact",
    )}
  </div>

  <div class="fig-grid">
    {figure(
        "thinking_L3_all_heads_category_mass.png",
        "Figure 8. L3 四个 heads 的类别分配",
        "横轴是 attention category，纵轴是 L3H0-L3H3，颜色是平均 attention mass。这个图专门回答 L3H2 的问题：L3H2 top-1 看起来不低，但大部分 mass 仍在 prompt noise，而不是正确 needle、BOS 或 previous trace。",
        "wide",
    )}
    {figure(
        "thinking_L3_top_roles_by_head.png",
        "Figure 9. L3 heads 的 top attention role",
        "横轴是 L3 head，颜色/堆叠表示最高 attention token 的类别分布。它帮助区分“top-1 选中了正确 needle”与“head 实际大部分质量放在哪里”这两件事。",
        "wide",
    )}
  </div>

  <h3>L3 head summary</h3>
  {table_html(idx_table, [
      ("head", "head", "text"),
      ("correct_top1_rate", "correct top-1", "num"),
      ("diag_share_of_needle_mass", "diagonal share", "num"),
      ("correct_prompt_needle_mass", "correct needle mass", "num"),
      ("all_prompt_needles_mass", "all needle mass", "num"),
      ("prompt_noise_mass", "noise mass", "num"),
      ("plus_one_score", "plus-one score", "num"),
  ])}
  <p>
    L3H3 和 L3H1 同时具备高 top-1、高 diagonal share、高 correct needle mass、低 plus-one score。
    L3H2 的 <code>correct_top1_rate</code> 只有 {fmt(l3h2['correct_top1_rate'])}，
    correct needle mass 只有 {fmt(l3h2['correct_prompt_needle_mass'])}，
    prompt noise mass 却有 {fmt(l3h2['prompt_noise_mass'])}。
    因此它不应被解释为强 targeted retrieval head；它更像一个以 prompt background/noise 为主的 head，
    top-1 数值受到低 needle mass 下排序波动影响。
  </p>

  <h2>5. CoT final answer：最终 count 更跟随 trace，而不是 prompt</h2>
  <div class="callout orange">
    <b>最强证据来自 prompt-vs-trace conflict。</b>当 prompt count 与 teacher-forced trace count 被人为设成冲突时，
    最终答案 100% 跟随 trace。也就是说 CoT trace 不是装饰性文本；它形成了最终 readout 使用的中间状态。
  </div>

  <div class="fig-grid">
    {figure(
        "thinking_answer_16head_category_signature.png",
        "Figure 10. Thinking final-answer 16 heads 的类别签名",
        "横轴是 attention category，纵轴是 head（LxHy），颜色是 <code>&lt;Ans&gt;</code> query 分给各类别的平均 mass。这里同时能看到 broad prompt heads 与 trace/marker readout heads，说明最终答案阶段不是单一 head 决定。",
        "wide",
    )}
    {figure(
        "thinking_answer_prompt_trace_conflict.png",
        "Figure 11. Prompt-vs-trace conflict：最终答案跟谁走",
        "横轴是 conflict condition；纵轴是 follow rate 或 logit preference（取决于图中 panel）。实验人为让 prompt count 与 trace count 不一致，然后看最终预测 count 更接近哪一个。结果显示 trace-follow rate 为 1.0，prompt-follow rate 为 0。",
        "wide",
    )}
  </div>
  <div class="fig-grid three">
    {figure(
        "thinking_answer_broad_prompt_score_by_head.png",
        "Figure 12. Final-answer broad prompt aggregate score",
        "横轴是 head，纵轴是 layer，颜色是 <code>broad_prompt_aggregate_score</code>。高值表示该 head 在 <code>&lt;Ans&gt;</code> 处广泛扫 prompt，且对 count-relevant prompt needle 有信号。",
        "compact",
    )}
    {figure(
        "thinking_answer_trace_readout_score_by_head.png",
        "Figure 13. Final-answer trace readout score",
        "横轴是 head，纵轴是 layer，颜色是 <code>trace_readout_score</code>。高值表示该 head 更多读 trace，并且读到的 trace 位置与 count 相关。",
        "compact",
    )}
    {figure(
        "thinking_answer_needle_enrichment_by_head.png",
        "Figure 14. Final-answer needle/noise enrichment",
        "横轴是 head，纵轴是 layer，颜色是每个 needle token 的平均 attention mass 相对于每个 noise token 的平均 attention mass。这个指标避免把 noise token 数量多误读成模型偏好 noise。",
        "compact",
    )}
  </div>

  <h3>Prompt/trace conflict summary</h3>
  {table_html(conflict_rows, [
      ("condition", "condition", "code"),
      ("n", "n", "int"),
      ("trace_follow_rate", "trace follow", "num"),
      ("prompt_follow_rate", "prompt follow", "num"),
      ("mean_trace_minus_prompt_logit", "trace - prompt logit", "num"),
  ])}

  <h3>Final-answer candidate heads</h3>
  {table_html(answer_table, [
      ("head", "head", "text"),
      ("prompt_mass", "prompt mass", "num"),
      ("prompt_needles_mass", "prompt needle mass", "num"),
      ("prompt_noise_mass", "prompt noise mass", "num"),
      ("trace_mass", "trace mass", "num"),
      ("needle_enrichment", "needle enrichment", "num"),
      ("broad_prompt_aggregate_score", "broad prompt score", "num"),
      ("trace_readout_score", "trace readout score", "num"),
  ])}
  <p>
    读法上要分清两个层面：broad prompt heads（如 {head_label(top_broad)}）说明最终答案仍保留 prompt-wide 聚合；
    trace readout heads（如 {head_label(top_trace)}）说明最终答案也读取 trace。
    但是 conflict test 更强：当 prompt 与 trace 不一致时，答案跟随 trace。
    因此目前最合理的解释是：prompt aggregation 与 trace readout 共同提供信号，但 final count 的决定性中间变量是 trace。
  </p>

  <h2>6. Multi-head mask ablation：支持 distributed circuit，而不是单头故事</h2>
  <div class="section-card">
    <p>
      Head-output mask ablation 的做法是在 teacher-forced forward 中把指定 attention heads 的输出置零，
      再重新计算最终答案 accuracy/margin 与 trace marker accuracy/margin。
      这里的 <code>d_answer_margin</code>、<code>d_marker_margin</code> 是相对于 no-ablation baseline 的变化；
      负数越大，说明该 mask 对相应功能破坏越强。
    </p>
    <p>
      Baseline：answer_acc={fmt(base['answer_acc'])}，answer_count_margin={fmt(base['answer_count_margin'])}；
      trace_marker_acc={fmt(base['trace_marker_acc'])}，trace_marker_margin={fmt(base['trace_marker_margin'])}。
    </p>
  </div>
  {figure(
      "thinking_head_output_multi_ablation.png",
      "Figure 15. Thinking head-output multi-ablation",
      "横轴是 ablation condition，即被 mask 的 head group；纵轴通常是 accuracy 或 logit margin change（图中 panel 标注）。这张图比较单 head、L3 all、layer all、answer broad heads、trace readout heads、以及组合 mask。它显示单个 retrieval head 的 mask 主要降低 margin；多 head / layer mask 才明显打穿 accuracy。",
      "wide",
  )}

  <h3>Selected ablation conditions</h3>
  {table_html(selected_abl, [
      ("condition", "condition", "code"),
      ("mask_heads", "masked heads", "code"),
      ("answer_acc", "answer acc", "num"),
      ("d_answer_margin", "Δ answer margin", "num"),
      ("trace_marker_acc", "marker acc", "num"),
      ("d_marker_margin", "Δ marker margin", "num"),
  ])}

  <div class="two-col">
    <div>
      <h3>Largest trace-marker margin drops</h3>
      {table_html(worst_marker, [
          ("condition", "condition", "code"),
          ("answer_acc", "answer acc", "num"),
          ("trace_marker_acc", "marker acc", "num"),
          ("d_marker_margin", "Δ marker margin", "num"),
      ])}
    </div>
    <div>
      <h3>Largest final-answer margin drops</h3>
      {table_html(worst_answer, [
          ("condition", "condition", "code"),
          ("answer_acc", "answer acc", "num"),
          ("d_answer_margin", "Δ answer margin", "num"),
          ("trace_marker_acc", "marker acc", "num"),
      ])}
    </div>
  </div>

  <p>
    Ablation 的主要信息不是“某一个 head 就是 counting circuit”，而是：
    L3 targeted retrieval heads 对 trace marker margin 很重要；
    layer 1 和 answer broad prompt heads 对最终答案 margin/accuracy 很重要；
    trace-readout / needle-enrichment heads 单独 mask 时 accuracy 不一定掉，但 margin 会下降。
    这符合一个分布式机制：targeted retrieval 产生或维护 trace，broad prompt/trace readout heads 聚合并把结果送到最终 count logits。
  </p>

  <h2>7. 当前结果支持的 counting task mechanisms</h2>
  <div class="section-card">
    <h3>Non-thinking mechanism</h3>
    <ul>
      <li><b>输入到输出路径短：</b>prompt 后直接在 <code>&lt;Ans&gt;</code> 预测 count，没有显式 trace。</li>
      <li><b>注意力分布宽：</b>大部分 attention mass 在 prompt body/noise 上，prompt entropy 高，不是 BOS sink。</li>
      <li><b>仍有 marker signal：</b>L1 heads 的 top-n retrieval recall 可达 1.0，needle/noise enrichment 大于 1，说明模型不是完全忽略 needles。</li>
      <li><b>机制解释：</b>更像 direct prompt scan + distributed aggregation，而不是一步步 retrieval 到每个 needle。</li>
    </ul>

    <h3>CoT / thinking mechanism</h3>
    <ul>
      <li><b>Trace generation 阶段：</b>L3H3/L3H1 对 <code>index_token_k</code> 做 targeted retrieval，指向第 k 个 prompt needle。</li>
      <li><b>不是单纯局部 +1：</b>最强 retrieval heads 的 <code>plus_one_score</code> 很低；L3H2 虽然有一定 top-1，但主要 mass 在 prompt noise。</li>
      <li><b>Final answer 阶段：</b>prompt-vs-trace conflict 显示 final count 跟随 trace；final-answer heads 中存在 broad prompt aggregation 和 trace readout 两类信号。</li>
      <li><b>多头分布式：</b>multi-head mask 才显著破坏 accuracy，单个 head mask 更多表现为 margin 下降。</li>
    </ul>
  </div>

  <h2>8. 限制与下一步</h2>
  <div class="section-card">
    <ul>
      <li><b>Teacher-forced limitation：</b>本轮 ablation 在 trace tokens 已给定时做，因此可能低估 autoregressive trace 错误的级联影响。</li>
      <li><b>Attention 不是完整因果证据：</b>attention heatmap 说明信息流候选路径；真正因果需要 activation patching、path patching 或 generation-time ablation。</li>
      <li><b>建议下一步：</b>对 L3H3/L3H1、layer1 broad heads、answer trace-readout heads 做 autoregressive generation mask；同时做 clean/corrupt path patching，测试“prompt needle → L3 retrieval → trace marker → final answer”的最小路径。</li>
      <li><b>对 NIAH 类比：</b>CoT 模型中 L3H3 的 targeted retrieval 与 NIAH HTML 报告中的 “CoT token targeted retrieval 到对应 prompt needle” 很接近；non-thinking 则更像弱定位的 prompt-wide aggregation。</li>
    </ul>
  </div>
</main>
</body>
</html>
"""

    report = ROOT / "report.html"
    report.write_text(html_doc, encoding="utf-8")
    for stale in [
        ROOT / "v2_2_attention_diagnostics_report.html",
        ROOT / "v2_2_attention_diagnostics_report_embedded.html",
    ]:
        if stale.exists():
            stale.unlink()
    print(report)


if __name__ == "__main__":
    main()
