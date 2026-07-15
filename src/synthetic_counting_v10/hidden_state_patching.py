from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import V10Config
from .core import Example, Vocab, count_bin, count_prediction, margin, render
from .report_followups import load_run, nested_example_pair
from .training import load_final_model


def _post_block_states(
    model,
    input_ids: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Capture the actual post-block residual used by a block-output patch.

    Hugging Face's final ``hidden_states`` entry is after the final layer norm,
    whereas a GPT-2 block forward hook sees the pre-final-LN residual. Capturing
    with the same hook location used for patching keeps all layers in a
    representation-matched space.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[:, int(position), :].detach().clone()

        return hook

    for layer, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(layer)))
    try:
        logits = model(input_ids=input_ids).logits[0].detach()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(range(len(model.transformer.h))) - set(captured))
    if missing:
        raise RuntimeError(f"Failed to capture post-block residuals for layers {missing}")
    return logits, [captured[layer] for layer in range(len(model.transformer.h))]


def _patched_forward(
    model,
    receiver_ids: torch.Tensor,
    layer: int,
    receiver_position: int,
    donor_state: torch.Tensor,
) -> torch.Tensor:
    """Replace one token's post-block residual and return all-position logits."""

    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        hidden[:, int(receiver_position), :] = donor_state.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = model.transformer.h[int(layer)].register_forward_hook(hook)
    try:
        return model(input_ids=receiver_ids).logits[0].detach()
    finally:
        handle.remove()


def _pair_margin(logits: torch.Tensor, vocab: Vocab, donor: int, receiver: int) -> float:
    if int(donor) == int(receiver):
        return math.nan
    return float(logits[vocab.number_id(donor)] - logits[vocab.number_id(receiver)])


def _normalized_recovery(
    patched: float,
    receiver: float,
    donor: float,
) -> float:
    denominator = float(donor) - float(receiver)
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return math.nan
    return (float(patched) - float(receiver)) / denominator


def _prefix_nested_pair(
    cfg: V10Config,
    vocab: Vocab,
    short_count: int,
    long_count: int,
    seed: int,
) -> tuple[Example, Example]:
    """Create count-m/count-n prompts whose ordered trace is prefix-aligned."""
    short_count, long_count = int(short_count), int(long_count)
    if not cfg.count_min <= short_count < long_count <= cfg.count_max:
        raise ValueError("prefix pair requires count_min <= short_count < long_count <= count_max")
    rng = random.Random(int(seed))
    positions = sorted(rng.sample(range(cfg.seq_len), long_count))
    markers = [rng.choice(vocab.markers) for _ in positions]
    base_noise = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]

    def build(count: int) -> Example:
        selected_positions = positions[:count]
        selected_markers = markers[:count]
        tokens = list(base_noise)
        for position, marker in zip(selected_positions, selected_markers):
            tokens[position] = marker
        return Example(tokens, count, list(selected_positions), list(selected_markers), int(seed))

    return build(short_count), build(long_count)


def _truncated_thinking_ids(
    example: Example,
    vocab: Vocab,
    trace_count: int,
) -> tuple[list[int], int, int]:
    """Render prompt + first m trace pairs + forced close + Ans, without gold tail."""
    trace: list[str] = []
    for k in range(1, int(trace_count) + 1):
        trace.extend([vocab.number_token(k), example.needle_markers[k - 1]])
    tokens = [
        "<BOS>",
        *example.seq_tokens,
        "<Think>",
        *trace,
        "</Think>",
        "<Ans>",
    ]
    marker_position = 1 + len(example.seq_tokens) + 2 * int(trace_count)
    ans_position = len(tokens) - 1
    return vocab.encode(tokens), marker_position, ans_position


def _regression_summary(
    frame: pd.DataFrame,
    groups: list[str],
    *,
    x_column: str = "donor_offset",
    y_column: str = "causal_expected_shift",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(groups, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        finite = part[np.isfinite(part[x_column]) & np.isfinite(part[y_column])]
        if len(finite) < 2 or finite[x_column].nunique() < 2:
            slope = intercept = r2 = math.nan
        else:
            x = finite[x_column].to_numpy(dtype=float)
            y = finite[y_column].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ beta
            denominator = float(((y - y.mean()) ** 2).sum())
            slope, intercept = float(beta[1]), float(beta[0])
            r2 = (
                1.0 - float(((y - prediction) ** 2).sum()) / denominator
                if denominator > 1e-12
                else math.nan
            )
        row = {
            **dict(zip(groups, keys)),
            "n_rows": int(len(part)),
            "transport_slope": slope,
            "transport_intercept": intercept,
            "transport_r2": r2,
        }
        for column in (
            "follows_donor",
            "follows_receiver",
            "normalized_recovery",
            "close_margin_shift",
            "close_predicted",
        ):
            if column in part:
                row[f"mean_{column}"] = float(part[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def run_final_answer_patching(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_specs = [
        (receiver, int(offset))
        for receiver in range(cfg.count_min, cfg.count_max + 1)
        for offset in (0, *cfg.patch_offsets)
        if cfg.count_min <= receiver + int(offset) <= cfg.count_max
    ]
    total = len(cfg.modes) * len(pair_specs) * int(examples_per_pair) * cfg.n_layer
    progress = tqdm(total=total, desc="v10 hidden patch: final Ans")
    for mode in cfg.modes:
        model = models[mode]
        device = next(model.parameters()).device
        for receiver_count, offset in pair_specs:
            donor_count = receiver_count + offset
            for example_index in range(int(examples_per_pair)):
                seed = cfg.seed + 1_210_000 + receiver_count * 10_000 + (offset + 30) * 100 + example_index
                receiver, donor = nested_example_pair(
                    cfg, vocab, receiver_count, donor_count, seed
                )
                receiver_item = render(receiver, vocab, mode)
                donor_item = render(donor, vocab, mode)
                receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
                donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
                receiver_logits_all, receiver_states = _post_block_states(
                    model, receiver_ids, receiver_item.spans.ans_pos
                )
                donor_logits_all, donor_states = _post_block_states(
                    model, donor_ids, donor_item.spans.ans_pos
                )
                receiver_logits = receiver_logits_all[receiver_item.spans.ans_pos]
                donor_logits = donor_logits_all[donor_item.spans.ans_pos]
                receiver_pred, receiver_expected, _ = count_prediction(receiver_logits, vocab)
                donor_pred, donor_expected, _ = count_prediction(donor_logits, vocab)
                receiver_pair_margin = _pair_margin(
                    receiver_logits, vocab, donor_count, receiver_count
                )
                donor_pair_margin = _pair_margin(
                    donor_logits, vocab, donor_count, receiver_count
                )
                for layer, donor_state in enumerate(donor_states):
                    patched_all = _patched_forward(
                        model,
                        receiver_ids,
                        layer,
                        receiver_item.spans.ans_pos,
                        donor_state,
                    )
                    patched_logits = patched_all[receiver_item.spans.ans_pos]
                    patched_pred, patched_expected, _ = count_prediction(patched_logits, vocab)
                    patched_pair_margin = _pair_margin(
                        patched_logits, vocab, donor_count, receiver_count
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "site": "final_answer_query",
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "count_bin": count_bin(receiver_count),
                            "donor_offset": offset,
                            "example_index": example_index,
                            "layer": layer + 1,
                            "receiver_position": receiver_item.spans.ans_pos,
                            "donor_position": donor_item.spans.ans_pos,
                            "position_delta": donor_item.spans.ans_pos - receiver_item.spans.ans_pos,
                            "receiver_pred": receiver_pred,
                            "receiver_expected": receiver_expected,
                            "donor_pred": donor_pred,
                            "donor_expected": donor_expected,
                            "patched_pred": patched_pred,
                            "patched_expected": patched_expected,
                            "causal_expected_shift": patched_expected - receiver_expected,
                            "follows_donor": float(patched_pred == donor_count),
                            "follows_receiver": float(patched_pred == receiver_count),
                            "receiver_pair_margin": receiver_pair_margin,
                            "donor_pair_margin": donor_pair_margin,
                            "patched_pair_margin": patched_pair_margin,
                            "normalized_recovery": _normalized_recovery(
                                patched_pair_margin, receiver_pair_margin, donor_pair_margin
                            ),
                            "donor_receiver_state_l2": float(
                                torch.linalg.vector_norm(
                                    donor_state - receiver_states[layer]
                                ).item()
                            ),
                        }
                    )
                    progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


@torch.no_grad()
def run_trace_final_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_specs = [
        (receiver, int(offset))
        for receiver in range(cfg.count_min, cfg.count_max + 1)
        for offset in cfg.patch_offsets
        if cfg.count_min <= receiver + int(offset) <= cfg.count_max
    ]
    progress = tqdm(
        total=len(pair_specs) * int(examples_per_pair) * cfg.n_layer,
        desc="v10 hidden patch: trace final-to-final",
    )
    device = next(model.parameters()).device
    for receiver_count, offset in pair_specs:
        donor_count = receiver_count + offset
        for example_index in range(int(examples_per_pair)):
            seed = cfg.seed + 1_220_000 + receiver_count * 10_000 + (offset + 30) * 100 + example_index
            receiver, donor = _prefix_nested_pair(
                cfg,
                vocab,
                min(receiver_count, donor_count),
                max(receiver_count, donor_count),
                seed,
            )
            if receiver.count != receiver_count:
                receiver, donor = donor, receiver
            receiver_item = render(receiver, vocab, "thinking")
            donor_item = render(donor, vocab, "thinking")
            receiver_pos = receiver_item.spans.trace_marker_positions[-1]
            donor_pos = donor_item.spans.trace_marker_positions[-1]
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            receiver_all, _ = _post_block_states(model, receiver_ids, receiver_pos)
            donor_all, donor_states = _post_block_states(model, donor_ids, donor_pos)
            receiver_ans = receiver_all[receiver_item.spans.ans_pos]
            donor_ans = donor_all[donor_item.spans.ans_pos]
            receiver_pred, receiver_expected, _ = count_prediction(receiver_ans, vocab)
            donor_pred, donor_expected, _ = count_prediction(donor_ans, vocab)
            receiver_pair_margin = _pair_margin(receiver_ans, vocab, donor_count, receiver_count)
            donor_pair_margin = _pair_margin(donor_ans, vocab, donor_count, receiver_count)
            baseline_close_margin = margin(
                receiver_all[receiver_pos], vocab.think_close_id, vocab.number_ids
            )
            for layer, donor_state in enumerate(donor_states):
                patched_all = _patched_forward(
                    model, receiver_ids, layer, receiver_pos, donor_state
                )
                patched_ans = patched_all[receiver_item.spans.ans_pos]
                patched_pred, patched_expected, _ = count_prediction(patched_ans, vocab)
                patched_pair_margin = _pair_margin(
                    patched_ans, vocab, donor_count, receiver_count
                )
                patched_close_margin = margin(
                    patched_all[receiver_pos], vocab.think_close_id, vocab.number_ids
                )
                rows.append(
                    {
                        "mode": "thinking",
                        "site": "trace_final_marker_to_final_marker",
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "count_bin": count_bin(receiver_count),
                        "donor_offset": offset,
                        "example_index": example_index,
                        "layer": layer + 1,
                        "receiver_position": receiver_pos,
                        "donor_position": donor_pos,
                        "position_delta": donor_pos - receiver_pos,
                        "receiver_pred": receiver_pred,
                        "receiver_expected": receiver_expected,
                        "donor_pred": donor_pred,
                        "donor_expected": donor_expected,
                        "patched_pred": patched_pred,
                        "patched_expected": patched_expected,
                        "causal_expected_shift": patched_expected - receiver_expected,
                        "follows_donor": float(patched_pred == donor_count),
                        "follows_receiver": float(patched_pred == receiver_count),
                        "receiver_pair_margin": receiver_pair_margin,
                        "donor_pair_margin": donor_pair_margin,
                        "patched_pair_margin": patched_pair_margin,
                        "normalized_recovery": _normalized_recovery(
                            patched_pair_margin, receiver_pair_margin, donor_pair_margin
                        ),
                        "baseline_close_margin": baseline_close_margin,
                        "patched_close_margin": patched_close_margin,
                        "close_margin_shift": patched_close_margin - baseline_close_margin,
                    }
                )
                progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


@torch.no_grad()
def run_trace_early_stop_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    positive_offsets = sorted({int(value) for value in cfg.patch_offsets if int(value) > 0})
    pairs = [
        (short, short + offset)
        for short in range(cfg.count_min, cfg.count_max)
        for offset in positive_offsets
        if short + offset <= cfg.count_max
    ]
    progress = tqdm(
        total=len(pairs) * int(examples_per_pair) * cfg.n_layer,
        desc="v10 hidden patch: trace early stop",
    )
    device = next(model.parameters()).device
    for donor_count, receiver_count in pairs:
        for example_index in range(int(examples_per_pair)):
            seed = cfg.seed + 1_230_000 + receiver_count * 10_000 + donor_count * 100 + example_index
            donor, receiver = _prefix_nested_pair(
                cfg, vocab, donor_count, receiver_count, seed
            )
            donor_item = render(donor, vocab, "thinking")
            receiver_item = render(receiver, vocab, "thinking")
            donor_pos = donor_item.spans.trace_marker_positions[-1]
            receiver_pos = receiver_item.spans.trace_marker_positions[donor_count - 1]
            if donor_pos != receiver_pos:
                raise AssertionError("prefix-aligned traces must share the M_m token position")
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            truncated_tokens, truncated_pos, truncated_ans_pos = _truncated_thinking_ids(
                receiver, vocab, donor_count
            )
            if truncated_pos != receiver_pos:
                raise AssertionError("truncated receiver must preserve the M_m position")
            truncated_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)

            donor_all, donor_states = _post_block_states(model, donor_ids, donor_pos)
            receiver_all, _ = _post_block_states(model, receiver_ids, receiver_pos)
            truncated_all, _ = _post_block_states(model, truncated_ids, truncated_pos)
            next_id = vocab.number_id(donor_count + 1)
            baseline_close_margin = float(
                receiver_all[receiver_pos, vocab.think_close_id]
                - receiver_all[receiver_pos, next_id]
            )
            donor_close_margin = float(
                donor_all[donor_pos, vocab.think_close_id]
                - donor_all[donor_pos, next_id]
            )
            truncated_ans = truncated_all[truncated_ans_pos]
            truncated_pred, truncated_expected, _ = count_prediction(truncated_ans, vocab)
            donor_ans = donor_all[donor_item.spans.ans_pos]
            donor_pred, donor_expected, _ = count_prediction(donor_ans, vocab)
            baseline_pair_margin = _pair_margin(
                truncated_ans, vocab, donor_count, receiver_count
            )
            donor_pair_margin = _pair_margin(donor_ans, vocab, donor_count, receiver_count)
            for layer, donor_state in enumerate(donor_states):
                patched_full = _patched_forward(
                    model, receiver_ids, layer, receiver_pos, donor_state
                )
                patched_close_margin = float(
                    patched_full[receiver_pos, vocab.think_close_id]
                    - patched_full[receiver_pos, next_id]
                )
                patched_local_id = int(patched_full[receiver_pos].argmax().item())

                patched_truncated = _patched_forward(
                    model, truncated_ids, layer, truncated_pos, donor_state
                )
                patched_ans = patched_truncated[truncated_ans_pos]
                patched_pred, patched_expected, _ = count_prediction(patched_ans, vocab)
                patched_pair_margin = _pair_margin(
                    patched_ans, vocab, donor_count, receiver_count
                )
                rows.append(
                    {
                        "mode": "thinking",
                        "site": "trace_prefix_early_stop",
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "count_bin": count_bin(receiver_count),
                        "donor_offset": donor_count - receiver_count,
                        "trace_gap": receiver_count - donor_count,
                        "example_index": example_index,
                        "layer": layer + 1,
                        "receiver_position": receiver_pos,
                        "donor_position": donor_pos,
                        "position_delta": donor_pos - receiver_pos,
                        "baseline_close_margin": baseline_close_margin,
                        "donor_close_margin": donor_close_margin,
                        "patched_close_margin": patched_close_margin,
                        "close_margin_shift": patched_close_margin - baseline_close_margin,
                        "close_recovery": _normalized_recovery(
                            patched_close_margin, baseline_close_margin, donor_close_margin
                        ),
                        "close_predicted": float(patched_local_id == vocab.think_close_id),
                        "baseline_truncated_pred": truncated_pred,
                        "baseline_truncated_expected": truncated_expected,
                        "donor_pred": donor_pred,
                        "donor_expected": donor_expected,
                        "patched_pred": patched_pred,
                        "patched_expected": patched_expected,
                        "causal_expected_shift": patched_expected - truncated_expected,
                        "follows_donor": float(patched_pred == donor_count),
                        "follows_receiver": float(patched_pred == receiver_count),
                        "receiver_pair_margin": baseline_pair_margin,
                        "donor_pair_margin": donor_pair_margin,
                        "patched_pair_margin": patched_pair_margin,
                        "normalized_recovery": _normalized_recovery(
                            patched_pair_margin, baseline_pair_margin, donor_pair_margin
                        ),
                        "uses_gold_trace_tail": False,
                        "forced_close_and_ans_tokens": True,
                    }
                )
                progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


def _plot_layer_lines(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    y_column: str,
    ylabel: str,
    hue: str | None = None,
) -> None:
    bins = [value for value in ("1-10", "11-20", "21-30") if value in set(summary.count_bin)]
    fig, axes = plt.subplots(1, len(bins), figsize=(5.0 * len(bins), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bin_name in zip(axes, bins):
        part = summary[summary.count_bin == bin_name]
        if hue and hue in part:
            for label, group in part.groupby(hue, sort=False):
                ax.plot(group.layer, group[y_column], marker="o", linewidth=2, label=str(label))
        else:
            ax.plot(part.layer, part[y_column], marker="o", linewidth=2, color="#2563eb")
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(f"receiver count {bin_name}")
        ax.set_xlabel("patched post-Layer residual")
        ax.set_xticks(sorted(part.layer.unique()))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    if hue:
        axes[-1].legend(title=hue, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_hidden_state_patching_plots(
    output_dir: Path,
    final_summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    early_summary: pd.DataFrame,
) -> list[Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    specs = [
        (
            final_summary,
            "final_answer_transport_by_layer.png",
            "Cross-count residual transplant at the final <Ans> query",
            "transport_slope",
            "expected-count transport slope",
            "mode",
        ),
        (
            trace_summary,
            "trace_final_transport_by_layer.png",
            "CoT final-marker m-to-n residual transplant",
            "transport_slope",
            "final-answer expected-count transport slope",
            None,
        ),
        (
            early_summary,
            "trace_early_stop_close_margin.png",
            "Early-stop test: donor final-marker state at receiver interior marker",
            "mean_close_margin_shift",
            "change in logit(</Think>) - logit(<m+1>)",
            None,
        ),
        (
            early_summary,
            "trace_early_stop_count_transport.png",
            "Forced-close readout without a gold trace tail",
            "transport_slope",
            "expected-count transport slope",
            None,
        ),
    ]
    for frame, name, title, column, ylabel, hue in specs:
        path = figures / name
        _plot_layer_lines(
            frame,
            path,
            title=title,
            y_column=column,
            ylabel=ylabel,
            hue=hue,
        )
        paths.append(path)
    return paths


def run_hidden_state_patching(
    run_dir: str | Path,
    *,
    device: str | None = None,
    examples_per_pair: int | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    cfg, vocab = load_run(run_dir, device=device)
    examples_per_pair = int(
        examples_per_pair
        if examples_per_pair is not None
        else (1 if cfg.preset == "debug" else 2)
    )
    output_dir = run_dir / "analysis" / "hidden_state_patching"
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    expected = {
        "final_detail": tables / "final_answer_patching.csv",
        "final_summary": tables / "final_answer_patching_summary.csv",
        "trace_detail": tables / "trace_final_patching.csv",
        "trace_summary": tables / "trace_final_patching_summary.csv",
        "early_detail": tables / "trace_early_stop_patching.csv",
        "early_summary": tables / "trace_early_stop_patching_summary.csv",
    }
    if not overwrite and all(path.exists() and path.stat().st_size > 0 for path in expected.values()):
        outputs = {name: pd.read_csv(path) for name, path in expected.items()}
        make_hidden_state_patching_plots(
            output_dir,
            outputs["final_summary"],
            outputs["trace_summary"],
            outputs["early_summary"],
        )
        return outputs

    models = {
        mode: load_final_model(cfg, vocab, run_dir, mode)
        for mode in cfg.modes
    }
    final_detail, final_summary = run_final_answer_patching(
        models,
        cfg,
        vocab,
        examples_per_pair=examples_per_pair,
    )
    trace_detail, trace_summary = run_trace_final_patching(
        models["thinking"],
        cfg,
        vocab,
        examples_per_pair=examples_per_pair,
    )
    early_detail, early_summary = run_trace_early_stop_patching(
        models["thinking"],
        cfg,
        vocab,
        examples_per_pair=examples_per_pair,
    )
    outputs = {
        "final_detail": final_detail,
        "final_summary": final_summary,
        "trace_detail": trace_detail,
        "trace_summary": trace_summary,
        "early_detail": early_detail,
        "early_summary": early_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(expected[name], index=False)
    manifest = {
        "experiment": "v10_hidden_state_patching",
        "representation_space": "post-transformer-layer residual captured and patched at the same block-output hook",
        "examples_per_pair": examples_per_pair,
        "final_answer_modes": list(cfg.modes),
        "trace_tests": [
            "final_marker_m_to_final_marker_n",
            "prefix_M_m_early_close_teacher_forced",
            "prefix_M_m_forced_close_readout_without_gold_trace_tail",
        ],
        "teacher_forcing_note": (
            "The early-close local metric uses the full receiver sequence only to score the token "
            "immediately after M_m. Its downstream readout uses a separate truncated input that "
            "contains forced </Think><Ans> but no later gold trace or answer token."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_hidden_state_patching_plots(
        output_dir, final_summary, trace_summary, early_summary
    )
    return outputs
