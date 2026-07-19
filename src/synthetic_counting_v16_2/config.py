from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


SUPPORTED_POSITION_ENCODINGS = ("rope", "rpe")


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class V16_2Config:
    """Configuration whose count_max property is a read-only compatibility alias."""

    version: str = "v16_2"
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    needle_set_size: int = 3
    needle_pool_size: int = 100
    needle_pool_frequency_threshold: float = 0.04
    needle_pool_frequency_bins: int = 20
    needle_pool_seed: int | None = None
    count_max_threshold: int = 10
    task_occurrence_ratio: float = 1.0
    corpus_train_fraction: float = 0.80
    corpus_validation_fraction: float = 0.10
    candidate_filter_max_attempts: int = 100_000
    shuffle_needle_set_order: bool = True
    position_encodings: tuple[str, ...] = ("rope", "rpe")

    train_steps: int = 10_000
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    warmup_steps: int = 500
    grad_clip: float = 1.0
    precision: str = "float32"
    log_every: int = 50
    eval_every: int = 500
    ar_eval_every: int = 1_000
    checkpoint_every: int = 1_000
    eval_examples_per_count: int = 100
    ar_examples_per_count: int = 10

    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int = 1024
    n_positions: int = 384
    max_relative_distance: int = 256
    rope_base: float = 10_000.0

    attention_examples_per_count: int = 20
    state_train_examples_per_count: int = 40
    state_eval_examples_per_count: int = 15
    analysis_batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model/data interface metadata. These are deliberately immutable in v16_2.
    noise_source: str = "shakespeare_char"
    task_type: str = "target_character_set"
    loss_scope: str = "all_sequence"

    @property
    def count_min(self) -> int:
        return 1

    @property
    def count_max(self) -> int:
        """Compatibility alias; count_max_threshold is the only stored setting."""

        return int(self.count_max_threshold)

    @property
    def effective_needle_pool_seed(self) -> int:
        return int(self.seed + 20_000 if self.needle_pool_seed is None else self.needle_pool_seed)

    @property
    def corpus_test_fraction(self) -> float:
        return 1.0 - float(self.corpus_train_fraction) - float(self.corpus_validation_fraction)

    @property
    def modes(self) -> tuple[str, str]:
        return ("nonthinking", "thinking")

    @property
    def model_variants(self) -> tuple[tuple[str, str], ...]:
        return tuple((position, mode) for position in self.position_encodings for mode in self.modes)

    @property
    def max_render_len(self) -> int:
        # BOS + (CountChar, set members, Sep) + prompt + Think/trace/close/Ans/count/EOS.
        return 1 + (2 + self.needle_set_size) + self.seq_len + 5 + 2 * self.count_max_threshold

    @property
    def count_bins(self) -> tuple[tuple[int, int], ...]:
        return ((1, self.count_max_threshold),)

    def count_bin(self, count: int) -> str:
        value = int(count)
        if not 1 <= value <= self.count_max_threshold:
            raise ValueError(f"count {value} is outside 1..{self.count_max_threshold}")
        return f"1-{self.count_max_threshold}"

    def validate(self) -> None:
        if self.version != "v16_2":
            raise ValueError("V16_2Config.version must be 'v16_2'")
        if self.needle_set_size != 3:
            raise ValueError("v16_2 requires exactly three distinct characters per needle set")
        if self.needle_pool_size <= 0 or self.needle_pool_frequency_bins <= 0:
            raise ValueError("needle pool size and number of bins must be positive")
        if not 0.0 < self.needle_pool_frequency_threshold <= 1.0:
            raise ValueError("needle_pool_frequency_threshold must be in (0, 1]")
        if not 1 <= self.count_max_threshold <= self.seq_len:
            raise ValueError("count_max_threshold must satisfy 1 <= threshold <= seq_len")
        if not 0.0 <= self.task_occurrence_ratio <= 1.0:
            raise ValueError("task_occurrence_ratio must be in [0, 1]")
        if self.corpus_train_fraction <= 0 or self.corpus_validation_fraction <= 0:
            raise ValueError("corpus train and validation fractions must be positive")
        if self.corpus_train_fraction + self.corpus_validation_fraction >= 1:
            raise ValueError("train + validation fractions must be less than one")
        if self.candidate_filter_max_attempts <= 0:
            raise ValueError("candidate_filter_max_attempts must be positive")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least two")
        if (self.n_layer, self.n_head, self.n_embd, self.n_inner) != (4, 4, 256, 1024):
            raise ValueError("v16_2 requires 4 layers, 4 heads, d_model=256, MLP=1024")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.max_render_len > self.n_positions:
            raise ValueError(
                f"max rendered length {self.max_render_len} exceeds n_positions={self.n_positions}"
            )
        if not self.position_encodings:
            raise ValueError("at least one position encoding is required")
        invalid = sorted(set(self.position_encodings) - set(SUPPORTED_POSITION_ENCODINGS))
        if invalid:
            raise ValueError(f"unsupported position encodings: {invalid}")
        if self.noise_source != "shakespeare_char" or self.task_type != "target_character_set":
            raise ValueError("v16_2 requires the Shakespeare target-character-set task")
        if self.loss_scope != "all_sequence":
            raise ValueError("v16_2 requires all-sequence next-token loss")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be float32 or bf16")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        for name in (
            "train_steps",
            "batch_size",
            "log_every",
            "eval_every",
            "ar_eval_every",
            "checkpoint_every",
            "eval_examples_per_count",
            "analysis_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["position_encodings"] = list(self.position_encodings)
        result["count_max"] = self.count_max
        result["count_max_alias"] = "read-only alias of count_max_threshold"
        result["effective_needle_pool_seed"] = self.effective_needle_pool_seed
        result["corpus_test_fraction"] = self.corpus_test_fraction
        result["training_objective"] = (
            "teacher-forced next-token cross-entropy over every non-padding token"
        )
        result["task_occurrence_ratio_definition"] = (
            "example-level probability of formatting a training corpus window as a counting task"
        )
        return result


def preset_config(preset: str = "debug", **overrides: Any) -> V16_2Config:
    cfg = V16_2Config(preset="main")
    if preset == "debug":
        cfg = replace(
            cfg,
            preset="debug",
            seq_len=48,
            count_max_threshold=4,
            n_positions=96,
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
            analysis_batch_size=8,
        )
    elif preset != "main":
        raise ValueError(f"unknown preset: {preset}")
    unknown = sorted(set(overrides) - set(cfg.__dataclass_fields__))
    if unknown:
        raise TypeError(f"unknown V16_2Config overrides: {unknown}")
    if "position_encodings" in overrides:
        overrides["position_encodings"] = tuple(overrides["position_encodings"])
    cfg = replace(cfg, **overrides)
    cfg.validate()
    return cfg


def config_from_dict(values: dict[str, Any]) -> V16_2Config:
    data = dict(values)
    alias = data.pop("count_max", None)
    threshold = int(data["count_max_threshold"])
    if alias is not None and int(alias) != threshold:
        raise ValueError("serialized count_max alias disagrees with count_max_threshold")
    for derived in (
        "count_max_alias",
        "effective_needle_pool_seed",
        "corpus_test_fraction",
        "training_objective",
        "task_occurrence_ratio_definition",
    ):
        data.pop(derived, None)
    data["position_encodings"] = tuple(data["position_encodings"])
    cfg = V16_2Config(**data)
    cfg.validate()
    return cfg


def default_run_name(cfg: V16_2Config) -> str:
    positions = "-".join(cfg.position_encodings)
    return (
        f"v16_2_{cfg.preset}_L{cfg.seq_len}_pool{cfg.needle_pool_size}x{cfg.needle_set_size}_"
        f"pf{_float_tag(cfg.needle_pool_frequency_threshold)}_count1-{cfg.count_max_threshold}_"
        f"taskr{_float_tag(cfg.task_occurrence_ratio)}_{positions}_all_sequence_seed{cfg.seed}"
    )


def prepare_run_dir(out_root: str | Path, cfg: V16_2Config, run_name: str | None = None) -> Path:
    path = Path(out_root) / (run_name or default_run_name(cfg))
    for subdir in ("tables", "figures", "checkpoints", "analysis", "logs", "data"):
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path
