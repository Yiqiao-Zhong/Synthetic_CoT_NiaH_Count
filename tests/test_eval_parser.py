from __future__ import annotations

import pytest

pytest.importorskip("torch")

from trace_counting.eval import invalid_trace_metrics, parse_generation, trace_metrics


def test_parse_valid_generation() -> None:
    parsed = parse_generation(["<Think>", "<I1>", "X", "<Think>", "<ANS>", "<C1>", "<EOS>"])
    assert parsed["format_valid"] is True
    assert parsed["pred_count"] == 1
    assert parsed["trace_tokens"] == ["<I1>", "X"]
    assert parsed["eos_after_count"] is True


def test_parse_answer_only_generation() -> None:
    parsed = parse_generation(["<ANS>", "<C3>", "<EOS>"], task_format="answer_only")
    assert parsed["format_valid"] is True
    assert parsed["pred_count"] == 3
    assert parsed["trace_tokens"] == []
    assert parsed["eos_after_count"] is True


def test_parse_repeat_count_answer_only_generation() -> None:
    parsed = parse_generation(["<ANS>", "<CNT>", "<CNT>", "<CNT>", "<EOS>"], task_format="answer_only_repeat_count")
    assert parsed["format_valid"] is True
    assert parsed["pred_count"] == 3
    assert parsed["answer_tokens"] == ["<CNT>", "<CNT>", "<CNT>"]
    assert parsed["trace_tokens"] == []


def test_parse_repeat_count_think_generation() -> None:
    parsed = parse_generation(
        ["<Think>", "<TICK>", "X", "<TICK>", "Y", "<Think>", "<ANS>", "<CNT>", "<CNT>", "<EOS>"],
        task_format="think_trace_repeat_count",
    )
    assert parsed["format_valid"] is True
    assert parsed["pred_count"] == 2
    assert parsed["trace_tokens"] == ["<TICK>", "X", "<TICK>", "Y"]


def test_parse_repeat_count_rejects_invalid_answer_token() -> None:
    parsed = parse_generation(["<ANS>", "<CNT>", "<C2>", "<EOS>"], task_format="answer_only_repeat_count")
    assert parsed["format_valid"] is False
    assert parsed["invalid_reason"] == "invalid_count_token"
    assert parsed["pred_count"] is None


@pytest.mark.parametrize(
    ("tokens", "reason"),
    [
        (["<I1>", "X", "<ANS>", "<C1>"], "missing_first_think"),
        (["<Think>", "<I1>", "X", "<ANS>", "<C1>"], "missing_second_think"),
        (["<Think>", "<I1>", "X", "<Think>", "<C1>"], "missing_ans"),
        (["<Think>", "<Think>", "<ANS>"], "missing_count_after_ans"),
        (["<Think>", "<Think>", "<ANS>", "X"], "invalid_count_token"),
    ],
)
def test_parse_invalid_generation(tokens: list[str], reason: str) -> None:
    parsed = parse_generation(tokens)
    assert parsed["format_valid"] is False
    assert parsed["invalid_reason"] == reason


def test_trace_metrics_zero_count() -> None:
    metrics = trace_metrics([], [])
    assert metrics["trace_exact_match"] is True
    assert metrics["trace_index_accuracy"] == 1.0
    assert metrics["trace_marker_precision"] == 1.0
    assert metrics["trace_marker_recall"] == 1.0


def test_invalid_trace_metrics_do_not_credit_empty_trace() -> None:
    metrics = invalid_trace_metrics([], [])
    assert metrics["trace_exact_match"] is False
    assert metrics["trace_index_accuracy"] == 0.0
