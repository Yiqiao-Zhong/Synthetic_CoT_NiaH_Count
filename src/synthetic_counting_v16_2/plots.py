from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import V16_2Config


sns.set_theme(style="whitegrid", context="notebook")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_v16_2_loss_suites(cfg: V16_2Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_loss_curves.csv")
    if frame.empty:
        return
    suites = [suite for suite in ("raw", "task", "mixture") if suite in set(frame["suite"])]
    positions = [value for value in cfg.position_encodings if value in set(frame["position_encoding"])]
    modes = [value for value in ("nonthinking", "thinking") if value in set(frame["mode"])]
    colors = {"train": "#2864b7", "heldout": "#d1543f"}
    styles = {"nonthinking": "-", "thinking": "--"}
    figure, axes = plt.subplots(
        len(positions), len(suites),
        figsize=(5.0 * len(suites), 4.0 * len(positions)),
        squeeze=False,
    )
    for row, position in enumerate(positions):
        for column, suite in enumerate(suites):
            axis = axes[row, column]
            subset = frame[(frame.position_encoding == position) & (frame.suite == suite)]
            for source in ("train", "heldout"):
                for mode in modes:
                    line = subset[(subset.curve_source == source) & (subset["mode"] == mode)].sort_values("step")
                    if line.empty:
                        continue
                    axis.plot(
                        line.step,
                        line.example_mean_cross_entropy,
                        color=colors[source],
                        linestyle=styles[mode],
                        marker="o",
                        markersize=3,
                        label=f"{source} | {mode}",
                    )
            axis.set_title(f"{position.upper()} — {suite}")
            axis.set_xlabel("training step")
            axis.set_ylabel("mean per-sequence cross-entropy")
            axis.legend(fontsize=8)
    figure.suptitle("v16_2 fixed-suite loss: train versus held-out/test (validation region)", y=1.01)
    _save(figure, run_dir / "figures" / "learning_loss_suites_train_vs_heldout.png")


def plot_v16_2_token_weighted_suites(cfg: V16_2Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_loss_curves.csv")
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, suite in zip(axes[0], ("raw", "task", "mixture")):
        subset = frame[frame.suite == suite]
        if not subset.empty:
            sns.lineplot(
                data=subset,
                x="step",
                y="token_weighted_cross_entropy",
                hue="curve_source",
                style="mode",
                units="position_encoding",
                estimator=None,
                ax=axis,
            )
        axis.set_title(suite)
        axis.set_ylabel("token-weighted cross-entropy")
    figure.suptitle("Secondary token-weighted loss diagnostic", y=1.02)
    _save(figure, run_dir / "figures" / "learning_loss_suites_token_weighted.png")


def plot_v16_2_training(cfg: V16_2Config, run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "train_metrics.csv")
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.lineplot(data=frame, x="step", y="train_total_loss", hue="mode", style="position_encoding", ax=axes[0])
    axes[0].set_title("Stochastic optimizer minibatch loss")
    sns.lineplot(
        data=frame,
        x="step",
        y="batch_realized_task_example_ratio",
        hue="mode",
        style="position_encoding",
        ax=axes[1],
    )
    axes[1].axhline(cfg.task_occurrence_ratio, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Realized task-example ratio")
    _save(figure, run_dir / "figures" / "training_diagnostics.png")


def plot_v16_2_accuracy(run_dir: Path) -> None:
    frame = _read(run_dir / "tables" / "eval_by_count.csv")
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(
        data=frame,
        x="step",
        y="tf_final_accuracy",
        hue="mode",
        style="position_encoding",
        marker="o",
        ax=axis,
    )
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Held-out teacher-forced counting accuracy")
    _save(figure, run_dir / "figures" / "learning_count_accuracy.png")


def plot_v16_2_analysis(run_dir: Path) -> None:
    attention = _read(run_dir / "tables" / "attention_summary.csv")
    if not attention.empty:
        final = attention[attention.query_kind == "final_answer"]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(
            data=final,
            x="layer",
            y="prompt_needles_mass",
            hue="mode",
            style="position_encoding",
            marker="o",
            ax=axis,
        )
        axis.set_title("Final-answer attention mass on matching prompt characters")
        _save(figure, run_dir / "figures" / "attention_needles_by_layer.png")
    states = _read(run_dir / "tables" / "state_probe_summary.csv")
    if not states.empty:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        sns.lineplot(
            data=states[states.site == "final_answer"],
            x="layer",
            y="nearest_centroid_accuracy",
            hue="mode",
            style="position_encoding",
            marker="o",
            ax=axis,
        )
        axis.set_title("Held-out count decoding from final-answer states")
        _save(figure, run_dir / "figures" / "state_count_decoding.png")


def make_all_v16_2_plots(cfg: V16_2Config, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_v16_2_loss_suites(cfg, run_dir)
    plot_v16_2_token_weighted_suites(cfg, run_dir)
    plot_v16_2_training(cfg, run_dir)
    plot_v16_2_accuracy(run_dir)
    plot_v16_2_analysis(run_dir)
