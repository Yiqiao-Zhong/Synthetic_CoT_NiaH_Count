from __future__ import annotations

import random

from trace_counting.generate_data import make_example, validate_example


def test_generated_example_schema_and_trace_order() -> None:
    example = make_example(
        split="train",
        seed=0,
        seq_len=16,
        count=5,
        rng=random.Random(123),
        example_index=0,
        max_count=64,
    )
    validate_example(example)
    full = example["full_tokens"]
    source_markers = [full[pair["source_idx"]] for pair in example["spans"]["trace_pairs"]]
    trace_markers = [pair["marker"] for pair in example["spans"]["trace_pairs"]]
    assert trace_markers == source_markers
    assert example["trace_tokens"] == [tok for pair in example["spans"]["trace_pairs"] for tok in (f"<I{pair['k']}>", pair["marker"])]


def test_zero_count_renders_adjacent_think_tokens() -> None:
    example = make_example(
        split="val_id",
        seed=0,
        seq_len=8,
        count=0,
        rng=random.Random(99),
        example_index=0,
        max_count=64,
    )
    validate_example(example)
    spans = example["spans"]
    assert example["trace_tokens"] == []
    assert spans["trace_start"] == spans["trace_end_exclusive"]
    assert spans["think_close_idx"] == spans["think_open_idx"] + 1
    assert example["full_tokens"][spans["think_open_idx"] : spans["eos_idx"] + 1] == [
        "<Think>",
        "<Think>",
        "<ANS>",
        "<C0>",
        "<EOS>",
    ]


def test_answer_only_format_has_no_think_trace_tokens() -> None:
    example = make_example(
        split="val_id",
        seed=0,
        seq_len=8,
        count=3,
        rng=random.Random(7),
        example_index=0,
        max_count=10,
        task_format="answer_only",
    )
    validate_example(example)
    spans = example["spans"]
    assert example["task_format"] == "answer_only"
    assert spans["think_open_idx"] is None
    assert spans["think_close_idx"] is None
    assert spans["trace_start"] == spans["trace_end_exclusive"]
    assert example["full_tokens"][spans["source_end_exclusive"] :] == ["<ANS>", "<C3>", "<EOS>"]


def test_repeat_count_think_format_uses_seen_token_types_for_ood_counts() -> None:
    example = make_example(
        split="val_count_ood",
        seed=0,
        seq_len=16,
        count=7,
        rng=random.Random(17),
        example_index=0,
        max_count=10,
        task_format="think_trace_repeat_count",
    )
    validate_example(example)
    spans = example["spans"]
    assert example["trace_tokens"][0::2] == ["<TICK>"] * 7
    assert example["answer_tokens"] == ["<CNT>"] * 7
    assert example["full_tokens"][spans["count_start_idx"] : spans["count_end_exclusive"]] == ["<CNT>"] * 7
    assert "<C7>" not in example["full_tokens"]
    assert "<I7>" not in example["full_tokens"]


def test_repeat_count_answer_only_zero_count_has_empty_count_span() -> None:
    example = make_example(
        split="val_id",
        seed=0,
        seq_len=8,
        count=0,
        rng=random.Random(31),
        example_index=0,
        max_count=10,
        task_format="answer_only_repeat_count",
    )
    validate_example(example)
    spans = example["spans"]
    assert example["answer_tokens"] == []
    assert spans["count_start_idx"] == spans["count_end_exclusive"] == spans["eos_idx"]
    assert example["full_tokens"][spans["source_end_exclusive"] :] == ["<ANS>", "<EOS>"]
