from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from synthetic_niah_v5.model import GPT2LMHeadModel


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, 11)]


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        tokens = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS
        return cls({tok: i for i, tok in enumerate(tokens)}, tokens)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls(dict(obj["token_to_id"]), list(obj["id_to_token"]))

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    def count_to_token(self, count: int) -> str:
        return f"<{int(count)}>"

    @property
    def numeric_ids(self) -> list[int]:
        return [self.token_to_id[t] for t in NUMBER_TOKENS]


@dataclass
class BaseExample:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None


def sample_base_example(seq_len: int, rng: random.Random, *, count: int, seed: int | None = None) -> BaseExample:
    positions = sorted(rng.sample(range(seq_len), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in positions]
    seq = [rng.choice(NOISE_TOKENS) for _ in range(seq_len)]
    for pos, marker in zip(positions, markers):
        seq[pos] = marker
    return BaseExample(seq, int(count), positions, markers, seed)


def balanced_examples(seq_len: int, examples_per_count: int, seed: int) -> list[BaseExample]:
    rng = random.Random(seed)
    examples: list[BaseExample] = []
    for count in range(1, 11):
        for i in range(examples_per_count):
            examples.append(sample_base_example(seq_len, rng, count=count, seed=seed + 1000 * count + i))
    rng.shuffle(examples)
    return examples


def render_thinking(ex: BaseExample, vocab: Vocab, *, trace_count_override: int | None = None) -> dict[str, Any]:
    trace: list[str] = []
    index_positions: list[int] = []
    marker_positions: list[int] = []
    pre_index_positions: list[int] = []
    think_start_pos = 1 + len(ex.seq_tokens)
    pos = think_start_pos + 1
    prev_marker_pos = think_start_pos
    trace_count = ex.count if trace_count_override is None else int(trace_count_override)
    for k in range(1, trace_count + 1):
        marker = ex.needle_markers[(k - 1) % len(ex.needle_markers)]
        pre_index_positions.append(prev_marker_pos)
        trace.extend([vocab.count_to_token(k), marker])
        index_positions.append(pos)
        marker_positions.append(pos + 1)
        prev_marker_pos = pos + 1
        pos += 2
    tokens = ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace + ["</Think>", "<Ans>", vocab.count_to_token(trace_count), "<EOS>"]
    think_end_pos = think_start_pos + 1 + len(trace)
    ans_pos = think_end_pos + 1
    return {
        "tokens": tokens,
        "input_ids": vocab.encode(tokens),
        "count": ex.count,
        "trace_count": trace_count,
        "needle_markers": ex.needle_markers,
        "prompt_needle_positions": [1 + p for p in ex.needle_positions],
        "anchors": {
            "prompt_start": 1,
            "prompt_end_exclusive": 1 + len(ex.seq_tokens),
            "think_start": think_start_pos,
            "think_end": think_end_pos,
            "ans_token": ans_pos,
            "index_positions": index_positions,
            "marker_positions": marker_positions,
            "pre_index_positions": pre_index_positions,
        },
    }


def load_v2_thinking_model(v2_run_dir: str | Path, device: str):
    v2_run_dir = Path(v2_run_dir)
    ckpt = v2_run_dir / "checkpoints" / "final" / "thinking"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing v2 thinking checkpoint: {ckpt}")
    try:
        model = GPT2LMHeadModel.from_pretrained(ckpt, attn_implementation="eager")
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(ckpt)
        model.config._attn_implementation = "eager"
    model.to(device)
    model.eval()
    vocab_path = v2_run_dir / "checkpoints" / "final" / "vocab.json"
    vocab = Vocab.load(vocab_path) if vocab_path.exists() else Vocab.build()
    cfg = {"seq_len": 256, "seed": 1234}
    cfg_path = v2_run_dir / "config.json"
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    return model, vocab, cfg


def _attention_entropy(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0 or len(weights) <= 1:
        return math.nan
    p = weights / total
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(len(weights)))


def _margin(logits: torch.Tensor, target_id: int, competitor_ids: list[int]) -> float:
    values = logits.detach().float().cpu()
    competitors = [idx for idx in competitor_ids if int(idx) != int(target_id)]
    if not competitors:
        return math.nan
    return float(values[target_id] - values[competitors].max())


@torch.no_grad()
def run_v2_2_followup(
    v2_run_dir: str | Path,
    *,
    examples_per_count: int = 50,
    device: str | None = None,
    seed_offset: int = 88_000,
) -> dict[str, pd.DataFrame]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, cfg = load_v2_thinking_model(v2_run_dir, device)
    examples = balanced_examples(int(cfg.get("seq_len", 256)), examples_per_count, int(cfg.get("seed", 1234)) + seed_offset)
    out_dir = Path(v2_run_dir) / "v2_2_followup_mechanism"
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    transition_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    close_id = vocab.token_to_id["</Think>"]
    number_ids = vocab.numeric_ids

    for ex_idx, ex in enumerate(examples):
        rendered = render_thinking(ex, vocab)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        out = model(input_ids=ids, output_attentions=True)
        logits = out.logits[0]
        attns = [a[0].detach().cpu().numpy() for a in out.attentions or []]
        a = rendered["anchors"]
        prompt_needles = rendered["prompt_needle_positions"]
        prompt_set = set(prompt_needles)
        prompt_noise = [p for p in range(a["prompt_start"], a["prompt_end_exclusive"]) if p not in prompt_set]
        index_positions = a["index_positions"]
        marker_positions = a["marker_positions"]
        ans_pos = a["ans_token"]
        for k, marker_pos in enumerate(marker_positions, start=1):
            target_token = vocab.count_to_token(k + 1) if k < len(marker_positions) else "</Think>"
            target_id = vocab.token_to_id[target_token]
            competitors = number_ids + [close_id]
            for layer, probs in enumerate(attns, start=1):
                for head in range(probs.shape[0]):
                    row = probs[head, marker_pos]
                    next_prompt = prompt_needles[k] if k < len(prompt_needles) else None
                    transition_rows.append(
                        {
                            "example_idx": ex_idx,
                            "count": ex.count,
                            "k": k,
                            "is_final_marker": float(k == ex.count),
                            "layer": layer,
                            "head": head,
                            "target_next_token": target_token,
                            "next_token_margin": _margin(logits[marker_pos], target_id, competitors),
                            "current_index_mass": float(row[index_positions[k - 1]]),
                            "current_marker_self_mass": float(row[marker_pos]),
                            "previous_marker_mass": float(row[marker_positions[k - 2]]) if k > 1 else 0.0,
                            "next_prompt_needle_mass": float(row[next_prompt]) if next_prompt is not None else 0.0,
                            "all_prompt_needles_mass": float(row[prompt_needles].sum()) if prompt_needles else 0.0,
                            "prompt_noise_mass": float(row[prompt_noise].sum()) if prompt_noise else 0.0,
                        }
                    )
        for layer, probs in enumerate(attns, start=1):
            for head in range(probs.shape[0]):
                row = probs[head, ans_pos]
                marker_weights = row[marker_positions]
                index_weights = row[index_positions]
                answer_rows.append(
                    {
                        "example_idx": ex_idx,
                        "count": ex.count,
                        "layer": layer,
                        "head": head,
                        "all_trace_marker_mass": float(marker_weights.sum()),
                        "all_trace_index_mass": float(index_weights.sum()),
                        "last_trace_marker_mass": float(marker_weights[-1]) if len(marker_weights) else 0.0,
                        "last_trace_index_mass": float(index_weights[-1]) if len(index_weights) else 0.0,
                        "trace_marker_entropy": _attention_entropy(marker_weights),
                        "trace_index_entropy": _attention_entropy(index_weights),
                        "prompt_needles_mass": float(row[prompt_needles].sum()) if prompt_needles else 0.0,
                        "prompt_noise_mass": float(row[prompt_noise].sum()) if prompt_noise else 0.0,
                    }
                )
        if ex_idx < max(20, examples_per_count):
            for trace_count in range(1, 11):
                r = render_thinking(ex, vocab, trace_count_override=trace_count)
                ids2 = torch.tensor([r["input_ids"]], dtype=torch.long, device=device)
                logits2 = model(input_ids=ids2).logits[0, r["anchors"]["ans_token"]]
                pred_offset = int(logits2[vocab.numeric_ids].argmax().item())
                pred_count = pred_offset + 1
                override_rows.append(
                    {
                        "example_idx": ex_idx,
                        "prompt_count": ex.count,
                        "trace_count": trace_count,
                        "pred_count": pred_count,
                        "follows_trace": float(pred_count == trace_count),
                        "follows_prompt": float(pred_count == ex.count),
                    }
                )

    transition = pd.DataFrame(transition_rows)
    answer = pd.DataFrame(answer_rows)
    override = pd.DataFrame(override_rows)
    transition_head = transition.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)
    answer_head = answer.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)
    override_summary = override.groupby(["prompt_count", "trace_count"], as_index=False).mean(numeric_only=True)
    outputs = {
        "successor_transition_rows": transition,
        "successor_transition_head_summary": transition_head,
        "answer_trace_attention_rows": answer,
        "answer_trace_attention_head_summary": answer_head,
        "trace_length_override": override,
        "trace_length_override_summary": override_summary,
    }
    for name, df in outputs.items():
        df.to_csv(tables / f"{name}.csv", index=False)
    make_plots(outputs, out_dir)
    make_report(outputs, out_dir)
    return outputs


def _heatmap(df: pd.DataFrame, metric: str, title: str, path: Path, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    mat = df.pivot(index="layer", columns="head", values=metric)
    values = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), labels=[str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), labels=[str(i) for i in mat.index])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmean(finite)) if finite.size else 0.5
    for y, row in enumerate(values):
        for x, val in enumerate(row):
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < midpoint else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figs = out_dir / "figures"
    transition = outputs["successor_transition_head_summary"]
    answer = outputs["answer_trace_attention_head_summary"]
    _heatmap(transition, "next_token_margin", "Successor transition: next-index/close logit margin", figs / "successor_next_token_margin.png")
    _heatmap(transition, "current_marker_self_mass", "Successor transition: current marker self mass", figs / "successor_current_marker_self_mass.png", vmin=0)
    _heatmap(transition, "next_prompt_needle_mass", "Successor transition: next prompt needle mass", figs / "successor_next_prompt_needle_mass.png", vmin=0)
    _heatmap(answer, "all_trace_marker_mass", "Final answer: all trace marker mass", figs / "answer_all_trace_marker_mass.png", vmin=0)
    _heatmap(answer, "last_trace_marker_mass", "Final answer: last trace marker mass", figs / "answer_last_trace_marker_mass.png", vmin=0)
    override = outputs["trace_length_override_summary"]
    if not override.empty:
        mat = override.pivot(index="prompt_count", columns="trace_count", values="follows_trace")
        values = mat.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(mat.columns)), labels=[str(c) for c in mat.columns])
        ax.set_yticks(range(len(mat.index)), labels=[str(i) for i in mat.index])
        ax.set_xlabel("forced trace count")
        ax.set_ylabel("prompt count")
        for y, row in enumerate(values):
            for x, val in enumerate(row):
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Final answer follows overridden trace length")
        fig.tight_layout()
        fig.savefig(figs / "trace_length_override_follows_trace.png", dpi=180)
        plt.close(fig)


def _img(name: str) -> str:
    return f"<img src='../figures/{name}' style='max-width:100%;border:1px solid #ddd;border-radius:8px'>"


def make_report(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    trans = outputs["successor_transition_head_summary"].sort_values("next_token_margin", ascending=False)
    ans = outputs["answer_trace_attention_head_summary"].sort_values("all_trace_marker_mass", ascending=False)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v2.2 follow-up mechanism</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;line-height:1.55;max-width:1150px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f4f7fb}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}</style>
</head><body>
<h1>v2.2 follow-up: successor localization and aggregation</h1>
<p>This report asks what happens after L3 targeted retrieval: marker_k predicts the next index/close token, then the next index token retrieves the next prompt needle. It also tests whether final answer follows trace length when trace length is overridden.</p>
<div class="grid"><div>{_img('successor_next_token_margin.png')}</div><div>{_img('successor_current_marker_self_mass.png')}</div><div>{_img('successor_next_prompt_needle_mass.png')}</div><div>{_img('answer_all_trace_marker_mass.png')}</div><div>{_img('answer_last_trace_marker_mass.png')}</div><div>{_img('trace_length_override_follows_trace.png')}</div></div>
<h2>Top successor-transition heads</h2>{trans.head(12).to_html(index=False)}
<h2>Top final-answer trace-attention heads</h2>{ans.head(12).to_html(index=False)}
</body></html>"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2.2 follow-up mechanism diagnostics")
    p.add_argument("--v2-run-dir", required=True)
    p.add_argument("--examples-per-count", type=int, default=50)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_v2_2_followup(args.v2_run_dir, examples_per_count=args.examples_per_count, device=args.device)


if __name__ == "__main__":
    main()
