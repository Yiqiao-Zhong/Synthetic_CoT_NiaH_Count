from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742")
RUNS = {
    "think_trace": ROOT / "runs" / "small_main" / "think_trace_full_sequence_seed0",
    "answer_only": ROOT / "runs" / "small_main" / "answer_only_full_sequence_seed0",
}
DATA = {
    "think_trace": ROOT / "data_metadata_and_examples" / "think_trace",
    "answer_only": ROOT / "data_metadata_and_examples" / "answer_only",
}
ASSET_DIR = ROOT / "report_assets"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def value_range(values: list[int]) -> str:
    values = [int(v) for v in values]
    if not values:
        return "NA"
    values = sorted(values)
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(v) for v in values)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def acc(rows: list[dict], pred_key: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.get(pred_key) == row["true_count"]) / len(rows)


def valid_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if row.get("ar_format_valid")) / len(rows)


def pred_counter(rows: list[dict], key: str) -> Counter:
    return Counter(row.get(key) for row in rows)


def per_group(rows: list[dict], key: str) -> list[dict]:
    groups: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = []
    for group, group_rows in sorted(groups.items()):
        out.append(
            {
                key: group,
                "n": len(group_rows),
                "tf_acc": acc(group_rows, "tf_pred_count"),
                "ar_acc": acc(group_rows, "ar_pred_count"),
                "format_valid": valid_rate(group_rows),
                "tf_pred_mode": pred_counter(group_rows, "tf_pred_count").most_common(1)[0][0],
                "ar_pred_mode": pred_counter(group_rows, "ar_pred_count").most_common(1)[0][0],
            }
        )
    return out


def ols_r2(features: list[list[float]], y: list[float]) -> float:
    if not features or len(features) != len(y):
        return float("nan")
    n = len(y)
    cols = len(features[0]) + 1
    xtx = [[0.0 for _ in range(cols)] for _ in range(cols)]
    xty = [0.0 for _ in range(cols)]
    for row, target in zip(features, y):
        x = [1.0] + [float(v) for v in row]
        for i in range(cols):
            xty[i] += x[i] * target
            for j in range(cols):
                xtx[i][j] += x[i] * x[j]
    # small ridge term keeps duplicated columns/invariants numerically stable.
    for i in range(cols):
        xtx[i][i] += 1e-8
    beta = solve_linear(xtx, xty)
    pred = []
    for row in features:
        x = [1.0] + [float(v) for v in row]
        pred.append(sum(b * xv for b, xv in zip(beta, x)))
    y_mean = sum(y) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    ss_tot = sum((a - y_mean) ** 2 for a in y)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        if abs(denom) < 1e-12:
            continue
        for j in range(col, n + 1):
            aug[col][j] /= denom
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def bar_svg(path: Path, title: str, labels: list[str], series: list[tuple[str, str, list[float]]], ylabel: str) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 90, 40, 70, 135
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max([1.0] + [v for _, _, vals in series for v in vals if v is not None])
    ymax = max(1.0, math.ceil(ymax * 10) / 10)
    group_w = plot_w / max(len(labels), 1)
    bar_w = min(36, group_w / (len(series) + 1))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="26" font-weight="700">{title}</text>',
        f'<text x="22" y="{top + plot_h/2}" font-family="Arial" font-size="18" transform="rotate(-90 22 {top + plot_h/2})">{ylabel}</text>',
    ]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - (t / ymax) * plot_h
        svg.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d7dee8"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#334155">{t:.2f}</text>')
    svg.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#64748b"/>')
    for i, label in enumerate(labels):
        center = left + group_w * (i + 0.5)
        svg.append(
            f'<text x="{center}" y="{top+plot_h+30}" text-anchor="middle" font-family="Arial" font-size="14" fill="#1f2937">{label}</text>'
        )
        for si, (_, color, vals) in enumerate(series):
            value = vals[i]
            if value is None or math.isnan(value):
                continue
            x = center - (len(series) * bar_w) / 2 + si * bar_w
            h = (value / ymax) * plot_h
            y = top + plot_h - h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}" opacity="0.92"/>')
            svg.append(f'<text x="{x+bar_w*0.41:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
    legend_x = left
    legend_y = height - 42
    for name, color, _ in series:
        svg.append(f'<rect x="{legend_x}" y="{legend_y-14}" width="16" height="16" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+22}" y="{legend_y}" font-family="Arial" font-size="15">{name}</text>')
        legend_x += 260
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def line_svg(path: Path, title: str, rows: list[dict], x_key: str, y_series: list[tuple[str, str, str]], ylabel: str) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 90, 55, 70, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({float(row[x_key]) for row in rows})
    ymin = 0.0
    ymax = max([1.0] + [float(row[key]) for row in rows for _, _, key in y_series if row.get(key) is not None])
    xmax = max(xs) if xs else 1.0
    xmin = min(xs) if xs else 0.0
    if xmax == xmin:
        xmax = xmin + 1.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="26" font-weight="700">{title}</text>',
        f'<text x="22" y="{top + plot_h/2}" font-family="Arial" font-size="18" transform="rotate(-90 22 {top + plot_h/2})">{ylabel}</text>',
    ]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - ((t - ymin) / (ymax - ymin)) * plot_h
        svg.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d7dee8"/>')
        svg.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="14" fill="#334155">{t:.2f}</text>')
    svg.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{top+plot_h}" y2="{top+plot_h}" stroke="#64748b"/>')
    svg.append(f'<text x="{left+plot_w/2}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="18">{x_key}</text>')
    for name, color, key in y_series:
        pts = []
        for row in sorted(rows, key=lambda r: float(r[x_key])):
            x = left + ((float(row[x_key]) - xmin) / (xmax - xmin)) * plot_w
            y = top + plot_h - ((float(row[key]) - ymin) / (ymax - ymin)) * plot_h
            pts.append((x, y))
        if not pts:
            continue
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    legend_x = left
    legend_y = height - 62
    for name, color, _ in y_series:
        svg.append(f'<line x1="{legend_x}" x2="{legend_x+28}" y1="{legend_y}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<text x="{legend_x+36}" y="{legend_y+5}" font-family="Arial" font-size="15">{name}</text>')
        legend_x += 240
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    predictions: dict[tuple[str, str], list[dict]] = {}
    for variant, run_dir in RUNS.items():
        for split in ["val_id", "val_count_ood"]:
            predictions[(variant, split)] = read_jsonl(run_dir / "eval" / f"predictions_{split}.jsonl")

    manifest = read_json(ROOT / "manifest.json")
    run_config = read_json(RUNS["think_trace"] / "config.json")
    model_config = run_config["model"]
    model_setting_table = md_table(
        ["项目", "设置"],
        [
            ["结果包", ROOT],
            ["experiment", manifest.get("experiment_name")],
            ["model_name", model_config.get("model_name")],
            ["architecture", "GPT-2 style decoder-only LM"],
            ["layers / heads / hidden", f"{model_config.get('n_layer')} layers / {model_config.get('n_head')} heads / d_model={model_config.get('n_embd')}"],
            ["MLP inner dim", model_config.get("n_inner")],
            ["context length", model_config.get("n_positions")],
            ["dropout", f"attn={model_config.get('attn_pdrop')}, embd={model_config.get('embd_pdrop')}, resid={model_config.get('resid_pdrop')}"],
            ["loss objective", f"{run_config.get('loss_mask')} / all-token next-token prediction"],
            ["final count weight", run_config.get("final_weight")],
            ["precision", run_config.get("precision")],
            ["seed", run_config.get("seed")],
            ["max_steps / batch_size", f"{run_config.get('max_steps')} / {run_config.get('batch_size')}"],
            ["learning_rate / warmup / weight_decay", f"{run_config.get('learning_rate')} / {run_config.get('warmup_steps')} / {run_config.get('weight_decay')}"],
        ],
    )

    data_setting_rows = []
    for variant, data_dir in DATA.items():
        metadata = read_json(data_dir / "dataset_metadata.json")
        for split in ["train", "val_id", "val_count_ood"]:
            spec = metadata["split_specs"][split]
            data_setting_rows.append(
                [
                    variant,
                    split,
                    value_range(spec["lengths"]),
                    value_range(spec["counts"]),
                    spec["examples_per_pair"],
                    metadata["split_counts"][split],
                    metadata["max_count"],
                    metadata["noise_vocab_size"],
                ]
            )
    data_setting_table = md_table(
        ["variant", "split", "lengths", "count range", "examples/pair", "n_total", "max_count token", "noise vocab"],
        data_setting_rows,
    )

    main_rows = []
    for variant in RUNS:
        id_rows = predictions[(variant, "val_id")]
        ood_rows = predictions[(variant, "val_count_ood")]
        strict_rows = [row for row in ood_rows if row["true_count"] > 5]
        main_rows.append(
            {
                "variant": variant,
                "id_tf": acc(id_rows, "tf_pred_count"),
                "id_ar": acc(id_rows, "ar_pred_count"),
                "id_format": valid_rate(id_rows),
                "ood_tf_incl5": acc(ood_rows, "tf_pred_count"),
                "ood_ar_incl5": acc(ood_rows, "ar_pred_count"),
                "ood_format": valid_rate(ood_rows),
                "strict_tf": acc(strict_rows, "tf_pred_count"),
                "strict_ar": acc(strict_rows, "ar_pred_count"),
                "strict_format": valid_rate(strict_rows),
                "strict_ar_pred_mode": pred_counter(strict_rows, "ar_pred_count").most_common(1)[0][0],
            }
        )

    ood_by_count = []
    for variant in RUNS:
        for row in per_group(predictions[(variant, "val_count_ood")], "true_count"):
            row["variant"] = variant
            ood_by_count.append(row)
    strict_ood_by_count = [row for row in ood_by_count if int(row["true_count"]) > 5]

    id_by_len = []
    for variant in RUNS:
        for row in per_group(predictions[(variant, "val_id")], "seq_len"):
            row["variant"] = variant
            id_by_len.append(row)

    position_rows = []
    for variant, data_dir in DATA.items():
        examples = read_jsonl(data_dir / "val_id.jsonl")
        y = [float(ex["count"]) for ex in examples]
        ans_idx = [[float(ex["spans"]["ans_idx"])] for ex in examples]
        ans_idx_seq = [[float(ex["spans"]["ans_idx"]), float(ex["seq_len"])] for ex in examples]
        position_rows.append(
            {
                "variant": variant,
                "ans_idx_r2": ols_r2(ans_idx, y),
                "ans_idx_plus_len_r2": ols_r2(ans_idx_seq, y),
            }
        )

    probe_rows = []
    for variant, run_dir in RUNS.items():
        for row in read_csv(run_dir / "directions" / "direction_summary.csv"):
            if row.get("ok") != "True":
                continue
            probe_rows.append(
                {
                    "variant": variant,
                    "layer": row["layer"],
                    "anchor": row["anchor"],
                    "target": row["target"],
                    "r2": float(row["r2"]),
                    "mae": float(row["mae"]),
                    "n": int(float(row["n"])),
                }
            )
    probe_highlights = []
    for variant in RUNS:
        rows = [row for row in probe_rows if row["variant"] == variant]
        for anchor, target in [("ans", "total_count"), ("source_marker", "running_count"), ("trace_index", "k"), ("trace_marker", "k")]:
            sub = [row for row in rows if row["anchor"] == anchor and row["target"] == target]
            if sub:
                best = max(sub, key=lambda r: r["r2"])
                final = next((r for r in sub if r["layer"] == "layer_4"), None)
                probe_highlights.append(
                    {
                        "variant": variant,
                        "anchor_target": f"{anchor}/{target}",
                        "best_layer": best["layer"],
                        "best_r2": best["r2"],
                        "final_r2": None if final is None else final["r2"],
                        "n": best["n"],
                    }
                )

    steering_rows = []
    for variant, run_dir in RUNS.items():
        for row in read_csv(run_dir / "steering" / "steering_summary.csv"):
            steering_rows.append(
                {
                    "variant": variant,
                    "alpha": float(row["alpha"]),
                    "accuracy": float(row["accuracy"]),
                    "mean_pred_count": float(row["mean_pred_count"]),
                    "mean_true_count": float(row["mean_true_count"]),
                    "mae": float(row["mae"]),
                }
            )

    train_summary = []
    for variant, run_dir in RUNS.items():
        logs = read_jsonl(run_dir / "train_log.jsonl")
        train = [row for row in logs if "total_weighted_loss" in row]
        evals = [row for row in logs if "val_total_weighted_loss" in row or "val_tf_count_acc" in row]
        last = train[-1] if train else {}
        train_summary.append(
            {
                "variant": variant,
                "first_loss": train[0]["total_weighted_loss"] if train else None,
                "last_loss": last.get("total_weighted_loss"),
                "source_loss": last.get("source_loss"),
                "answer_prefix_loss": last.get("answer_prefix_loss"),
                "count_loss": last.get("count_loss"),
                "trace_index_loss": last.get("trace_index_loss"),
                "trace_marker_loss": last.get("trace_marker_loss"),
                "think_boundary_loss": last.get("think_boundary_loss"),
                "last_step": last.get("step"),
                "train_minutes": last.get("elapsed_sec", 0) / 60 if train else None,
                "last_val_tf": evals[-1].get("val_tf_count_acc") if evals else None,
            }
        )

    bar_svg(
        ASSET_DIR / "v1_cn_main_accuracy.svg",
        "V1 final-count accuracy: ID vs strict OOD",
        ["think_trace", "answer_only"],
        [
            ("ID TF final count", "#2563eb", [row["id_tf"] for row in main_rows]),
            ("ID AR final count", "#16a34a", [row["id_ar"] for row in main_rows]),
            ("strict OOD AR >5", "#dc2626", [row["strict_ar"] for row in main_rows]),
        ],
        "accuracy",
    )
    count_rows_plot = [row for row in strict_ood_by_count if row["variant"] == "think_trace"]
    line_svg(
        ASSET_DIR / "v1_cn_think_ood_by_count.svg",
        "think_trace strict count-OOD by true count (6-10)",
        count_rows_plot,
        "true_count",
        [("TF final count", "#2563eb", "tf_acc"), ("AR final count", "#16a34a", "ar_acc"), ("format valid", "#9333ea", "format_valid")],
        "rate",
    )
    count_rows_plot = [row for row in strict_ood_by_count if row["variant"] == "answer_only"]
    line_svg(
        ASSET_DIR / "v1_cn_answer_ood_by_count.svg",
        "answer_only strict count-OOD by true count (6-10)",
        count_rows_plot,
        "true_count",
        [("TF final count", "#2563eb", "tf_acc"), ("AR final count", "#16a34a", "ar_acc"), ("format valid", "#9333ea", "format_valid")],
        "rate",
    )
    bar_svg(
        ASSET_DIR / "v1_cn_position_probe_baseline.svg",
        "Position-only baseline for total count on val_id",
        ["think_trace", "answer_only"],
        [
            ("ANS position only", "#f97316", [row["ans_idx_r2"] for row in position_rows]),
            ("ANS position + seq length", "#0f766e", [row["ans_idx_plus_len_r2"] for row in position_rows]),
        ],
        "linear R2",
    )
    bar_svg(
        ASSET_DIR / "v1_cn_probe_r2.svg",
        "Best ridge direction R2 by anchor/target",
        [row["anchor_target"].replace("_", " ") for row in probe_highlights if row["variant"] == "think_trace"],
        [
            (
                "think_trace best R2",
                "#2563eb",
                [row["best_r2"] for row in probe_highlights if row["variant"] == "think_trace"],
            ),
            (
                "answer_only best R2",
                "#16a34a",
                [
                    next(
                        (other["best_r2"] for other in probe_highlights if other["variant"] == "answer_only" and other["anchor_target"] == row["anchor_target"]),
                        None,
                    )
                    for row in probe_highlights
                    if row["variant"] == "think_trace"
                ],
            ),
        ],
        "R2",
    )

    summary = {
        "main_rows": main_rows,
        "ood_by_count": ood_by_count,
        "strict_ood_by_count": strict_ood_by_count,
        "id_by_len": id_by_len,
        "position_rows": position_rows,
        "probe_highlights": probe_highlights,
        "steering_rows": steering_rows,
        "train_summary": train_summary,
    }
    (ROOT / "v1_chinese_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    main_table = md_table(
        [
            "模型",
            "ID TF 最终计数",
            "ID AR 最终计数",
            "旧包OOD含5 TF",
            "旧包OOD含5 AR",
            "strict OOD 6-10 TF",
            "strict OOD 6-10 AR",
            "严格OOD最常AR预测",
        ],
        [
            [
                row["variant"],
                fnum(row["id_tf"]),
                fnum(row["id_ar"]),
                fnum(row["ood_tf_incl5"]),
                fnum(row["ood_ar_incl5"]),
                fnum(row["strict_tf"]),
                fnum(row["strict_ar"]),
                f"<C{row['strict_ar_pred_mode']}>" if row["strict_ar_pred_mode"] is not None else "invalid",
            ]
            for row in main_rows
        ],
    )
    ood_table = md_table(
        ["模型", "真实count", "n", "TF acc", "AR acc", "format valid", "TF众数", "AR众数"],
        [
            [
                row["variant"],
                row["true_count"],
                row["n"],
                fnum(row["tf_acc"]),
                fnum(row["ar_acc"]),
                fnum(row["format_valid"]),
                f"<C{row['tf_pred_mode']}>" if row["tf_pred_mode"] is not None else "invalid",
                f"<C{row['ar_pred_mode']}>" if row["ar_pred_mode"] is not None else "invalid",
            ]
            for row in strict_ood_by_count
        ],
    )
    len_table = md_table(
        ["模型", "ID seq_len", "n", "TF acc", "AR acc", "format valid", "AR众数"],
        [
            [
                row["variant"],
                row["seq_len"],
                row["n"],
                fnum(row["tf_acc"]),
                fnum(row["ar_acc"]),
                fnum(row["format_valid"]),
                f"<C{row['ar_pred_mode']}>" if row["ar_pred_mode"] is not None else "invalid",
            ]
            for row in id_by_len
        ],
    )
    pos_table = md_table(
        ["模型", "只用ANS位置预测count R2", "ANS位置+seq_len预测count R2"],
        [[row["variant"], fnum(row["ans_idx_r2"]), fnum(row["ans_idx_plus_len_r2"])] for row in position_rows],
    )
    probe_table = md_table(
        ["模型", "anchor/target", "best layer", "best R2", "final-layer R2", "n"],
        [
            [
                row["variant"],
                row["anchor_target"],
                row["best_layer"],
                fnum(row["best_r2"]),
                fnum(row["final_r2"]),
                row["n"],
            ]
            for row in probe_highlights
        ],
    )
    train_table = md_table(
        ["模型", "首个total loss", "最终total loss", "source loss", "answer loss", "count loss", "trace index loss", "trace marker loss", "训练分钟"],
        [
            [
                row["variant"],
                fnum(row["first_loss"]),
                fnum(row["last_loss"]),
                fnum(row["source_loss"]),
                fnum(row["answer_prefix_loss"]),
                fnum(row["count_loss"]),
                fnum(row["trace_index_loss"]),
                fnum(row["trace_marker_loss"]),
                fnum(row["train_minutes"], 2),
            ]
            for row in train_summary
        ],
    )

    md = f"""# V1 中文结果分析与补充实验

本节基于下载到本地的完整结果包重新审计：`{ROOT}`。我额外补了几项不需要重新训练的本地分析：严格 OOD 去掉边界 count=5、按真实 count 的预测塌缩、按长度分组、position-only probe baseline，以及 probe/steering 的中文解释。注意：本地 Python 当前没有 `transformers`，因此没有在本地重跑模型前向；需要模型前向的 attention 补跑仍建议在 Colab 里执行。

我也把后续需要的 `direction_projection` probe 接进了 repo/pipeline：它会把 ID 上 fit 出来的 ridge count direction 投影到 count-OOD hidden states 上，用来判断 counter direction 在 6-10 上是继续增长还是饱和到训练边界。这个 probe 需要 `transformers` 加载 checkpoint，本地环境缺依赖所以没有本地执行；在 Colab 里可直接设置 `PIPELINE_STAGE = "projection"` 补跑。

## 0. 实验设置

### 0.1 模型与训练设置

{model_setting_table}

### 0.2 数据生成设置

{data_setting_table}

**重要说明。** 这个下载下来的旧结果包里，`val_count_ood` 的 count range 是 `5-10`，其中 `5` 和训练 count `0-5` 重叠，只能算边界点，不是真正 OOD。下面所有 “strict OOD” 图和表都已经把 count=5 去掉，只分析 count=6-10。后续 notebook/pipeline 的默认 OOD 已改成 `6:10`。

### 0.3 先澄清：这里的“准确率”到底看什么？

`TF acc` 和 `AR acc` 都是 **最终计数 token `<Ck>` 的准确率**，不是 thinking trace 的准确率。

- `TF acc`：teacher-forced final-count accuracy。模型拿到 gold prefix，一直到 `<ANS>` 为止，然后只预测 `<ANS>` 后面的最终 count token。对 `think_trace` 来说，gold thinking trace 已经给了模型。
- `AR acc`：autoregressive final-count accuracy。模型只从 source prefix 开始自由生成；对 `think_trace`，它会自己生成 thinking trace 和 `<ANS> <Ck>`，但这里的准确率仍然只检查最终解析出的 `<Ck>` 是否等于真实 needle 数。
- `trace_exact`：只适用于 `think_trace`，检查自由生成的完整 thinking trace 是否和 gold trace 完全一致。它是过程监督质量，不是最终答案准确率。
- `format_valid`：自由生成能否解析出合法 answer skeleton。当前汇总里不要求 `<EOS>` 必须紧跟 count 后面。

## 1. 总体结论

{main_table}

![V1 final-count accuracy](../colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742/report_assets/v1_cn_main_accuracy.svg)

**怎么读这张图。** 横轴是两种训练格式：`think_trace` 表示 source 后要生成 counting trace 再答题；`answer_only` 表示 source 后直接生成 `<ANS> <Ck>`。纵轴是最终 count token 的准确率。蓝色是 ID teacher-forced，绿色是 ID autoregressive，红色是严格 OOD autoregressive，也就是只看 count=6-10。

**结论。** 两个模型都把 ID 任务学会了，尤其 teacher-forced 几乎满分。但严格 OOD，也就是训练没见过的 count=6-10，两个模型都是 0。`OOD含5` 看起来有 18%-25%，主要是因为 val_count_ood 里混入了 count=5，而 count=5 是训练上界，不是真正 extrapolation。

## 2. 严格 OOD：去掉 count=5 之后，模型系统性卡在训练边界

{ood_table}

![think_trace count-OOD](../colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742/report_assets/v1_cn_think_ood_by_count.svg)

![answer_only count-OOD](../colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742/report_assets/v1_cn_answer_ood_by_count.svg)

**怎么读这两张图。** 横轴是真实 needle 数，只显示严格 OOD 的 `6-10`，已经移除了旧结果包里混入的边界 count=5；纵轴是比例。蓝线是 teacher-forced 最终 count 准确率，绿线是 autoregressive 最终 count 准确率，紫线是自由生成格式合法率。

**结论。** 去掉 count=5 后，`TF acc` 和 `AR acc` 在 6-10 上都是 0。这个现象说明失败不只是“自由生成过程坏了”；即使给了 gold prefix，模型在最终答案位置也没有把 6-10 作为可外推计数读出来。`answer_only` 在 count=6-10 的 format valid 很高但 accuracy 为 0，说明它经常能生成一个形式上合法的 `<ANS> <Ck>`，只是答案被吸到训练边界 `<C5>` 附近。`think_trace` 到更大 count 时还会更频繁地产生 invalid generation，说明 trace 格式本身也开始崩。

## 3. ID 上有没有长度问题？

{len_table}

**结论。** ID split 里长度 50/100/200 都是训练见过的长度。teacher-forced 基本没有长度问题；autoregressive 下 `answer_only` 的 ID final-count accuracy 高于 `think_trace`，但这不能直接说明 thinking 有害，因为本结果包里两个 variant 的数据量不同：`answer_only` 训练样本是 `think_trace` 的 2 倍。这是当前比较里最大的混杂因素。

## 4. Probe：模型里确实有 count 信息，但要警惕位置泄漏

{probe_table}

![Probe R2](../colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742/report_assets/v1_cn_probe_r2.svg)

**怎么读这张图。** 横轴是 probe 的位置和目标，例如 `source_marker/running_count` 表示在 source 中每个 needle 位置读取“到这里为止已经数到第几个”；`ans/total_count` 表示在答案前的 `<ANS>` 位置读取总 count。纵轴是 ridge regression 的 held-out R2，越高说明线性可解码越强。

`think_trace` 的 `source_marker/running_count` 很强，说明 trace supervision 让模型在 source marker 位置形成了接近 counter 的线性方向；这是 thinking token 最积极的证据。可是这个 counter 没有变成 count extrapolation：严格 OOD 仍然是 0。

### 4.1 Position-only baseline：为什么 `ans/total_count` 不能单独当强证据

{pos_table}

![Position baseline](../colab_results/trace_count_v1_seed0_full_colab_small_main_steps10000_20260706_140742/report_assets/v1_cn_position_probe_baseline.svg)

**怎么读这张图。** 横轴是两种数据格式；纵轴是只用位置变量预测真实 count 的线性 R2。橙色只用 `<ANS>` 的 token index；绿色用 `<ANS>` index 加 source length。

**结论。** 对 `think_trace`，`<ANS>` 出现的位置本身就携带 count 信息，因为 trace 长度大约随 `2 * count` 增长。只用 `<ANS>` 位置加 source length 就能几乎完美恢复 count。因此 `ans/total_count` probe 高，不一定说明模型内部真的形成了抽象 counter；更干净的证据是 `source_marker/running_count`，因为它在 source marker 位置读“局部累计数量”，受 answer 位置泄漏影响小。

## 5. Steering：当前方向是描述性的，不是强因果控制

已有 steering 结果显示，把 final answer hidden state 沿 count direction 推动时，平均预测 count 只轻微变化，仍停在训练边界附近。也就是说，当前 `ans/total_count` 方向更像一个可读出的相关方向，而不是可以直接修复 OOD 的因果旋钮。下一步应该尝试：

1. 在 `source_marker/running_count` 方向上做更局部的 intervention，而不是只改最终 `<ANS>` state。
2. 对 trace token 内部的 `trace_index/k` 或 `trace_marker/k` 做分层 steering，看能不能改变中间计数轨迹。
3. 把 alpha sweep 分开报告 count=6、7、8、9、10，避免平均值掩盖边界吸附。

## 6. Attention：当前结果包没有可解释 attention 数据

两个 variant 的 `attention_summary.csv` 都只有 header，没有 rows。因此这次不能下结论说 thinking token 是否改变了 attention-to-needle 分布。代码已经改成 eager attention 并把 header-only 文件视为 incomplete；建议在 Colab 只补跑 attention stage，不需要重训：

```python
PIPELINE_STAGE = "attention"
SKIP_COMPLETED = True
ATTENTION_LIMIT = 512
ATTENTION_SPLITS = "val_id,val_count_ood"
ATTENTION_QUERY_ANCHORS = "ans,think_close"
```

补跑后重点看两个量：`marker_enrichment` 和 `top_source_marker_rate`。如果 thinking 真的帮助 targeted retrieval，我们希望 `think_trace` 在 `think_close` 或 `ans` query 上，对 source marker 的 per-token attention 明显高于 noise token，且这个差异在 count-OOD 上不崩。

## 7. 我建议补的实验与已经补好的 probe

**已经改进/补好。**

1. **严格 OOD 默认值已改成 `6:10`。** 以后 notebook 和 `scripts/run_v1_niah_like.py` 默认都会使用不和训练 count=0-5 重叠的 OOD。
2. **position-only baseline 已本地跑完。** 结果见 4.1，说明 `think_trace` 的 `ans/total_count` probe 必须做位置控制。
3. **OOD direction projection probe 已实现。** 在 Colab 里运行 `PIPELINE_STAGE = "projection"` 会生成 `direction_projection/direction_projection_summary.csv`，用来检查 ID count direction 到 OOD 上是否饱和。

**仍需要重训/补跑的实验。**

1. **两组模型数据量完全对齐后重训。** 当前下载结果里 `answer_only` 样本量是 `think_trace` 的 2 倍，不能把差异完全归因于 thinking token。当前 pipeline 用相同 `examples_per_pair` 生成两个 variant；重跑时请换新的 `DATA_ROOT/OUT_ROOT`，避免 `SKIP_COMPLETED=True` 跳过旧数据。
2. **多 seed。** 至少 seed 0/1/2。现在只有一个 seed，只能当现象观察。
3. **attention 补跑。** 当前结果包 attention 是空表；用 eager attention 补跑 `PIPELINE_STAGE = "attention"`。

**Probe/analysis 建议。**

1. **position-controlled probe。** 对 `ans/total_count`，把 hidden state probe 和 position-only baseline 并列报告；或者回归时加入位置特征作为 nuisance control。
2. **source-marker counter probe 作为主证据。** 重点报告 `source_marker/running_count`，少依赖 `ans/total_count`。
3. **OOD hidden-state probe。** 已实现为 `trace_counting.direction_projection`；下一步是在 Colab 上跑出图表并加入报告。
4. **causal intervention 换位置。** 优先在 source marker 或 trace index 位置 steering，而不是只动 `<ANS>`。
5. **attention 补跑。** 用 eager attention 看 marker enrichment/top-source-marker-rate，尤其比较 `think_close` 和 `ans`。

## 8. 训练本身

{train_table}

V1 按你的要求用了 `full_sequence` / all-token next-token prediction。这里要特别小心：source 里很多 noise token 是随机采样的，下一 token 本来就很难预测，所以 `source_loss` 会长期很高，`total loss` 也不会像 completion-only 那样接近 0。因此 **total loss 不是判断 counting 是否学会的好指标**。

更应该看任务相关 segment：`count_loss`、`answer_prefix_loss`、`trace_index_loss`、`trace_marker_loss`。这些 loss 在 `think_trace` 里已经接近 0，说明模型学会了训练分布内的 trace/answer 模式；`answer_only` 的 `count_loss` 也接近 0，说明最终计数 token 在 ID 上可学。真正的问题不是训练没收敛，而是模型把 count 限制在训练区间附近，没有学到 count extrapolation。下一版实验应先把任务拆成两个层级：`ID count 0-5 + strict OOD 6-10` 作为 NiaH-like 主实验，再保留更大 count range 作为压力测试。
"""
    (ROOT / "v1_chinese_analysis.md").write_text(md, encoding="utf-8")
    print(f"wrote {ROOT / 'v1_chinese_analysis.md'}")
    print(f"wrote {ROOT / 'v1_chinese_analysis_summary.json'}")


if __name__ == "__main__":
    main()
