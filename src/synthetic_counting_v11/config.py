from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SUPPORTED_VERSIONS = ("v11", "v12", "v13", "v14")
SUPPORTED_POSITION_ENCODINGS = ("ape", "rope", "rpe")


@dataclass(frozen=True)
class ExperimentConfig:
    version: str = "v11"
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    count_min: int = 1
    count_max: int = 30
    noise_vocab_size: int = 64
    marker_vocab_size: int = 10
    noise_source: str = "uniform"
    training_data_mode: str = "streaming"
    fixed_train_examples_per_count: int = 512
    position_encodings: tuple[str, ...] = ("ape", "rope", "rpe")

    train_steps: int = 10_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_examples_per_count: int = 100
    ar_examples_per_count: int = 10

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_inner: int = 256
    n_positions: int = 384
    max_relative_distance: int = 256

    attention_examples_per_count: int = 20
    state_train_examples_per_count: int = 40
    state_eval_examples_per_count: int = 15
    analysis_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def modes(self) -> tuple[str, str]:
        return ("nonthinking", "thinking")

    @property
    def max_render_len(self) -> int:
        return int(self.seq_len) + 2 * int(self.count_max) + 6

    @property
    def model_variants(self) -> tuple[tuple[str, str], ...]:
        return tuple((position, mode) for position in self.position_encodings for mode in self.modes)

    @property
    def count_bins(self) -> tuple[tuple[int, int], ...]:
        bins: list[tuple[int, int]] = []
        start = int(self.count_min)
        while start <= int(self.count_max):
            end = min(start + 9, int(self.count_max))
            bins.append((start, end))
            start = end + 1
        return tuple(bins)

    def count_bin(self, count: int) -> str:
        value = int(count)
        for lo, hi in self.count_bins:
            if lo <= value <= hi:
                return f"{lo}-{hi}"
        raise ValueError(f"count {value} is outside {self.count_min}..{self.count_max}")

    def validate(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {self.version}")
        if self.count_min < 1 or self.count_max < self.count_min:
            raise ValueError("count range must satisfy 1 <= count_min <= count_max")
        if self.count_max > self.seq_len:
            raise ValueError("count_max cannot exceed seq_len")
        if self.n_layer != 4 or self.n_head != 4 or self.n_embd != 64:
            raise ValueError("v11-v14 use exactly 4 layers, 4 heads, and hidden size 64")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )
        invalid = sorted(set(self.position_encodings) - set(SUPPORTED_POSITION_ENCODINGS))
        if invalid:
            raise ValueError(f"Unsupported position encodings: {invalid}")
        if self.noise_source not in {"uniform", "shakespeare_char"}:
            raise ValueError(f"Unsupported noise source: {self.noise_source}")
        if self.training_data_mode not in {"streaming", "fixed"}:
            raise ValueError(f"Unsupported training data mode: {self.training_data_mode}")
        if self.version == "v11" and set(self.position_encodings) != set(SUPPORTED_POSITION_ENCODINGS):
            raise ValueError("v11 must compare APE, RoPE, and RPE")
        if self.version in {"v12", "v13", "v14"} and self.position_encodings != ("ape",):
            raise ValueError(f"{self.version} must use APE only")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_encodings"] = list(self.position_encodings)
        result["count_bins"] = [list(pair) for pair in self.count_bins]
        result["architecture"] = (
            "v10-style random-init GPT-2/pre-LN causal Transformer core; "
            "4 layers; 4 heads; d_model=64; MLP=256; "
            "tied token embedding/unembedding"
        )
        result["architecture_note"] = (
            "v12-v14 reuse the v10-style computation and rendering path, but never "
            "the v10 width: every v11-v14 model is fixed at hidden size 64"
        )
        result["rpe_definition"] = "learned per-layer, per-head causal relative-distance attention bias"
        result["separate_transformers"] = True
        result["shared_trace_and_answer_numbers"] = True
        return result


def _main_config(version: str) -> ExperimentConfig:
    if version == "v11":
        return ExperimentConfig(version="v11", preset="main")
    if version == "v12":
        return ExperimentConfig(
            version="v12",
            preset="main",
            seq_len=512,
            count_max=50,
            n_positions=640,
            max_relative_distance=512,
            position_encodings=("ape",),
        )
    if version == "v13":
        return ExperimentConfig(
            version="v13",
            preset="main",
            training_data_mode="fixed",
            position_encodings=("ape",),
        )
    if version == "v14":
        return ExperimentConfig(
            version="v14",
            preset="main",
            noise_source="shakespeare_char",
            position_encodings=("ape",),
        )
    raise ValueError(f"Unknown experiment version: {version}")


def preset_config(version: str, preset: str, **overrides: Any) -> ExperimentConfig:
    cfg = _main_config(version)
    if preset == "debug":
        debug_count_max = 6 if version == "v12" else 4
        cfg = replace(
            cfg,
            preset="debug",
            seq_len=80 if version == "v12" else 48,
            count_max=debug_count_max,
            n_positions=128 if version == "v12" else 96,
            max_relative_distance=96,
            train_steps=6,
            batch_size=4,
            warmup_steps=2,
            log_every=1,
            eval_every=3,
            ar_eval_every=3,
            checkpoint_every=3,
            eval_examples_per_count=2,
            ar_examples_per_count=1,
            attention_examples_per_count=1,
            state_train_examples_per_count=2,
            state_eval_examples_per_count=1,
            fixed_train_examples_per_count=4,
            analysis_batch_size=8,
        )
    elif preset != "main":
        raise ValueError(f"Unknown preset: {preset}")
    known = set(cfg.__dataclass_fields__)
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(f"Unknown ExperimentConfig overrides: {unknown}")
    if "position_encodings" in overrides:
        overrides["position_encodings"] = tuple(overrides["position_encodings"])
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    data = dict(values)
    for derived in (
        "architecture",
        "architecture_note",
        "rpe_definition",
        "separate_transformers",
        "shared_trace_and_answer_numbers",
        "count_bins",
    ):
        data.pop(derived, None)
    data["position_encodings"] = tuple(data["position_encodings"])
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg


def default_run_name(cfg: ExperimentConfig) -> str:
    positions = "-".join(cfg.position_encodings)
    data_tag = "fixed" if cfg.training_data_mode == "fixed" else cfg.noise_source
    return (
        f"{cfg.version}_{cfg.preset}_L{cfg.seq_len}_count{cfg.count_min}-{cfg.count_max}_"
        f"4L4H_d64_{positions}_{data_tag}_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    path = Path(out_root) / (run_name or default_run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs", "data"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
