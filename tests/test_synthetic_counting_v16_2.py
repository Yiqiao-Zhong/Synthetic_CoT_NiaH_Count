from __future__ import annotations

import importlib.util
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_counting_v16_2.config import V16_2Config, config_from_dict, default_run_name, preset_config
from synthetic_counting_v16_2.data import (
    V16_2Vocab,
    balanced_v16_2_examples,
    build_corpus_split,
    build_loss_suite_manifests,
    build_test_suite_manifests,
    character_token,
    collate_v16_2,
    load_corpus_text,
    make_training_example,
    make_v16_2_example,
    render_v16_2,
    shifted_v16_2_token_losses,
)
from synthetic_counting_v16_2.needle_pool import build_needle_pool
from synthetic_counting_v16_2.plots import plot_v16_2_loss_suites
from synthetic_counting_v16_2.training import evaluate_loss_suite


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    cfg = preset_config("debug", device="cpu")
    text = load_corpus_text()
    split = build_corpus_split(cfg, text)
    vocab = V16_2Vocab.build(cfg, text)
    pool = build_needle_pool(cfg, text, split, vocab.fingerprint)
    return cfg, text, split, vocab, pool


def test_config_alias_ratio_validation_and_run_identity():
    cfg = preset_config("main", task_occurrence_ratio=0.25, count_max_threshold=10)
    assert cfg.count_max == cfg.count_max_threshold == 10
    assert "taskr0p25" in default_run_name(cfg)
    assert "pool100x3" in default_run_name(cfg)
    serialized = cfg.to_dict()
    assert config_from_dict(serialized) == cfg
    serialized["count_max"] = 9
    with pytest.raises(ValueError, match="disagrees"):
        config_from_dict(serialized)
    for ratio in (-0.01, 1.01):
        with pytest.raises(ValueError, match="task_occurrence_ratio"):
            preset_config("debug", task_occurrence_ratio=ratio)


def test_guarded_corpus_regions_are_disjoint(prepared):
    cfg, text, split, _, _ = prepared
    assert split.train.end + cfg.seq_len - 1 == split.validation.start
    assert split.validation.end + cfg.seq_len - 1 == split.test.start
    assert split.test.end == len(text)
    for region in (split.train, split.validation, split.test):
        assert region.length >= cfg.seq_len


def test_pool_contains_unique_distinct_triples_under_threshold(prepared):
    cfg, text, split, _, pool = prepared
    assert len(pool) == cfg.needle_pool_size
    sets = [item.characters for item in pool.sets]
    assert len(set(sets)) == len(sets)
    for item in pool.sets:
        assert len(item.characters) == len(set(item.characters)) == 3
        assert item.characters == tuple(sorted(item.characters, key=ord))
        assert item.frequency_sum <= cfg.needle_pool_frequency_threshold + 1e-12
        expected = sum(text[split.train.start : split.train.end].count(char) for char in item.characters)
        assert math.isclose(item.frequency_sum, expected / split.train.length, abs_tol=1e-12)


def test_task_example_preserves_window_and_renders_three_character_prefix(prepared):
    cfg, text, split, vocab, pool = prepared
    example = make_v16_2_example(cfg, vocab, text, split, pool, random.Random(4), region_name="train")
    source = text[example.corpus_start : example.corpus_end]
    assert example.seq_tokens == [character_token(char) for char in source]
    assert len(set(example.needle_characters)) == 3
    expected_positions = tuple(
        index for index, char in enumerate(source) if char in set(example.needle_characters)
    )
    assert example.needle_positions == expected_positions
    assert example.count == len(expected_positions)
    nonthinking = render_v16_2(example, vocab, "nonthinking")
    thinking = render_v16_2(example, vocab, "thinking")
    assert nonthinking.spans.prompt_start == 6
    assert nonthinking.tokens[1] == "<CountChar>"
    assert nonthinking.tokens[5] == "<Sep>"
    assert len(nonthinking.tokens[2:5]) == 3
    assert len(nonthinking.tokens) == cfg.seq_len + 9
    assert len(thinking.tokens) == cfg.seq_len + 11 + 2 * example.count
    trace_markers = [thinking.tokens[position] for position in thinking.spans.trace_marker_positions]
    assert trace_markers == list(example.needle_markers)
    assert nonthinking.labels == nonthinking.input_ids
    assert thinking.labels == thinking.input_ids


def test_ratio_zero_and_one_boundaries(prepared):
    base, text, split, vocab, pool = prepared
    zero = replace(base, task_occurrence_ratio=0.0)
    rng_zero = random.Random(22)
    raw = [make_training_example(zero, vocab, text, split, pool, rng_zero) for _ in range(12)]
    assert all(item.example_kind == "raw_lm" for item in raw)
    for example in raw:
        rendered_n = render_v16_2(example, vocab, "nonthinking")
        rendered_t = render_v16_2(example, vocab, "thinking")
        assert rendered_n.tokens == rendered_t.tokens == example.seq_tokens
        assert len(rendered_n.tokens) == zero.seq_len
        assert all(not token.startswith("<Count") for token in rendered_n.tokens)

    one = replace(base, task_occurrence_ratio=1.0)
    rng_one = random.Random(22)
    task = [make_training_example(one, vocab, text, split, pool, rng_one) for _ in range(12)]
    assert all(item.example_kind == "counting_task" for item in task)
    assert all(1 <= int(item.count) <= one.count_max_threshold for item in task)


def test_balanced_examples_and_fixed_suite_composition(prepared):
    cfg, text, split, vocab, pool = prepared
    balanced = balanced_v16_2_examples(
        cfg, vocab, text, split, pool, 2, 91, region_name="validation"
    )
    assert {count: sum(item.count == count for item in balanced) for count in range(1, 5)} == {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
    }
    curves = build_loss_suite_manifests(cfg, vocab, text, split, pool)
    test = build_test_suite_manifests(cfg, vocab, text, split, pool)
    expected = cfg.eval_examples_per_count * cfg.count_max_threshold
    for source, expected_region in (("train", "train"), ("heldout", "validation")):
        assert set(curves[source]) == {"raw", "task", "mixture"}
        assert all(len(values) == expected for values in curves[source].values())
        assert all(item.example_kind == "raw_lm" for item in curves[source]["raw"])
        assert all(item.example_kind == "counting_task" for item in curves[source]["task"])
        assert all(item.corpus_region == expected_region for values in curves[source].values() for item in values)
        task_in_mixture = sum(item.example_kind == "counting_task" for item in curves[source]["mixture"])
        assert task_in_mixture == round(cfg.task_occurrence_ratio * expected)
    assert all(item.corpus_region == "test" for values in test.values() for item in values)


class _CheatingLossModel(torch.nn.Module):
    """Makes deterministic position-dependent losses for aggregation tests."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        for row in range(batch):
            for position in range(length - 1):
                confidence = 0.1 + 0.01 * position
                logits[row, position, input_ids[row, position + 1]] = confidence
        return SimpleNamespace(logits=logits)


def test_suite_loss_is_equal_mean_over_input_sequences_and_batch_invariant(prepared):
    cfg, text, split, vocab, pool = prepared
    raw_cfg = replace(cfg, task_occurrence_ratio=0.5)
    raw = make_training_example(replace(raw_cfg, task_occurrence_ratio=0), vocab, text, split, pool, random.Random(5))
    task = make_v16_2_example(raw_cfg, vocab, text, split, pool, random.Random(6), region_name="train")
    examples = [raw, task]
    model = _CheatingLossModel(len(vocab.id_to_token))
    row, _ = evaluate_loss_suite(
        model,
        replace(raw_cfg, analysis_batch_size=2),
        vocab,
        examples,
        position_encoding="rope",
        mode="thinking",
        step=0,
        curve_source="train",
        suite="mixture",
    )
    rendered = [render_v16_2(example, vocab, "thinking") for example in examples]
    ids, labels, mask = collate_v16_2(rendered, vocab, "cpu")
    output = model(ids, mask)
    _, losses, active = shifted_v16_2_token_losses(output.logits, labels)
    per_sequence = [
        float((losses[index] * active[index]).sum() / active[index].sum())
        for index in range(len(examples))
    ]
    assert row["example_mean_cross_entropy"] == pytest.approx(np.mean(per_sequence))
    token_weighted = float((losses * active).sum() / active.sum())
    assert row["token_weighted_cross_entropy"] == pytest.approx(token_weighted)

    one_at_a_time, _ = evaluate_loss_suite(
        model,
        replace(raw_cfg, analysis_batch_size=1),
        vocab,
        examples,
        position_encoding="rope",
        mode="thinking",
        step=0,
        curve_source="train",
        suite="mixture",
    )
    assert one_at_a_time["example_mean_cross_entropy"] == pytest.approx(
        row["example_mean_cross_entropy"]
    )
    assert one_at_a_time["token_weighted_cross_entropy"] == pytest.approx(
        row["token_weighted_cross_entropy"]
    )


def test_plot_supports_each_mode_individually_and_jointly(tmp_path):
    cfg = replace(preset_config("debug", device="cpu"), position_encodings=("rope",))
    rows = []
    for mode in ("nonthinking", "thinking"):
        for source in ("train", "heldout"):
            for suite in ("raw", "task", "mixture"):
                for step in (0, 3, 6):
                    rows.append(
                        {
                            "step": step,
                            "position_encoding": "rope",
                            "mode": mode,
                            "curve_source": source,
                            "suite": suite,
                            "example_mean_cross_entropy": 4 - 0.1 * step + (source == "heldout") * 0.2,
                            "token_weighted_cross_entropy": 4,
                        }
                    )
    tables = tmp_path / "tables"
    tables.mkdir()
    figures = tmp_path / "figures"
    figures.mkdir()
    frame = pd.DataFrame(rows)
    for modes in (("nonthinking",), ("thinking",), ("nonthinking", "thinking")):
        frame[frame["mode"].isin(modes)].to_csv(tables / "eval_loss_curves.csv", index=False)
        plot_v16_2_loss_suites(cfg, tmp_path)
        assert (figures / "learning_loss_suites_train_vs_heldout.png").stat().st_size > 0


def test_v16_2_notebook_compiles_and_legacy_v16_runner_is_isolated(tmp_path):
    notebook_path = ROOT / "notebooks" / "Trace_Count_v16_2_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "drive.mount" in "".join(code_cells[0]["source"])
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "Colab Notebooks/NIAH_synthetic" in source
    assert '"--no-deps"' in source
    assert 'sys.path.insert(0, src_root)' in source
    assert "TASK_OCCURRENCE_RATIO =" in source
    assert '"--task-occurrence-ratio", str(TASK_OCCURRENCE_RATIO)' in source
    assert "--stage\", \"prepare" in source
    for cell in code_cells:
        compile("".join(cell["source"]), str(notebook_path), "exec")

    builder_path = ROOT / "scripts" / "build_v16_2_notebook.py"
    spec = importlib.util.spec_from_file_location("build_v16_2_notebook", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.OUTPUT = tmp_path / notebook_path.name
    generated_path = builder.build()
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    assert generated == notebook

    legacy_runner = (ROOT / "src" / "synthetic_counting_v16" / "run_v16.py").read_text(encoding="utf-8")
    assert "synthetic_counting_v16_2" not in legacy_runner
