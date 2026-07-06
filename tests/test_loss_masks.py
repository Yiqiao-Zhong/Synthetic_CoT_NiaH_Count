from __future__ import annotations

from trace_counting.loss_masks import build_labels_and_weights, token_segment
from trace_counting.tokenizer import build_default_tokenizer


def canonical_example() -> dict:
    full_tokens = [
        "<BOS>",
        "N4",
        "X",
        "N9",
        "Y",
        "N2",
        "Z",
        "N7",
        "<Think>",
        "<I1>",
        "X",
        "<I2>",
        "Y",
        "<I3>",
        "Z",
        "<Think>",
        "<ANS>",
        "<C3>",
        "<EOS>",
    ]
    return {
        "example_id": "canonical",
        "split": "train",
        "seed": 0,
        "seq_len": 7,
        "count": 3,
        "source_tokens": full_tokens[1:8],
        "positive_positions_source": [1, 3, 5],
        "positive_markers": ["X", "Y", "Z"],
        "trace_tokens": full_tokens[9:15],
        "answer_token": "<C3>",
        "full_tokens": full_tokens,
        "spans": {
            "source_start": 1,
            "source_end_exclusive": 8,
            "think_open_idx": 8,
            "trace_start": 9,
            "trace_end_exclusive": 15,
            "think_close_idx": 15,
            "ans_idx": 16,
            "count_idx": 17,
            "eos_idx": 18,
            "trace_pairs": [
                {"k": 1, "index_idx": 9, "marker_idx": 10, "marker": "X", "source_idx": 2},
                {"k": 2, "index_idx": 11, "marker_idx": 12, "marker": "Y", "source_idx": 4},
                {"k": 3, "index_idx": 13, "marker_idx": 14, "marker": "Z", "source_idx": 6},
            ],
        },
    }


def supervised_set(labels: list[int]) -> set[int]:
    return {idx for idx, label in enumerate(labels) if label != -100}


def test_loss_mask_supervised_indices() -> None:
    tokenizer = build_default_tokenizer()
    example = canonical_example()
    expected = {
        "full_sequence": set(range(1, 19)),
        "full_sequence_final_weighted": set(range(1, 19)),
        "completion_only": set(range(8, 19)),
        "completion_final_weighted": set(range(8, 19)),
        "final_count_only": {17},
    }
    for loss_mask, expected_indices in expected.items():
        labels, weights = build_labels_and_weights(example, tokenizer, loss_mask=loss_mask, final_weight=10.0)
        assert supervised_set(labels) == expected_indices
        assert {idx for idx, weight in enumerate(weights) if weight > 0} == expected_indices


def test_weighted_regimes_only_upweight_count_token() -> None:
    tokenizer = build_default_tokenizer()
    example = canonical_example()
    for loss_mask in ["full_sequence_final_weighted", "completion_final_weighted"]:
        labels, weights = build_labels_and_weights(example, tokenizer, loss_mask=loss_mask, final_weight=7.0, eos_weight=1.5)
        assert labels[17] != -100
        assert weights[17] == 7.0
        assert weights[18] == 1.5
        for idx, weight in enumerate(weights):
            if idx in {0, 17, 18}:
                continue
            if labels[idx] != -100:
                assert weight == 1.0


def repeat_count_example() -> dict:
    full_tokens = [
        "<BOS>",
        "N4",
        "X",
        "N9",
        "Y",
        "N2",
        "<Think>",
        "<TICK>",
        "X",
        "<TICK>",
        "Y",
        "<Think>",
        "<ANS>",
        "<CNT>",
        "<CNT>",
        "<EOS>",
    ]
    return {
        "example_id": "repeat",
        "split": "train",
        "seed": 0,
        "seq_len": 5,
        "count": 2,
        "source_tokens": full_tokens[1:6],
        "positive_positions_source": [1, 3],
        "positive_markers": ["X", "Y"],
        "trace_tokens": full_tokens[7:11],
        "answer_token": "<CNT>",
        "answer_tokens": ["<CNT>", "<CNT>"],
        "full_tokens": full_tokens,
        "task_format": "think_trace_repeat_count",
        "spans": {
            "source_start": 1,
            "source_end_exclusive": 6,
            "think_open_idx": 6,
            "trace_start": 7,
            "trace_end_exclusive": 11,
            "think_close_idx": 11,
            "ans_idx": 12,
            "count_start_idx": 13,
            "count_end_exclusive": 15,
            "count_idx": 13,
            "eos_idx": 15,
            "trace_pairs": [
                {"k": 1, "index_idx": 7, "marker_idx": 8, "marker": "X", "source_idx": 2},
                {"k": 2, "index_idx": 9, "marker_idx": 10, "marker": "Y", "source_idx": 4},
            ],
        },
    }


def test_repeat_count_masks_supervise_every_count_unit() -> None:
    tokenizer = build_default_tokenizer(max_count=10)
    example = repeat_count_example()
    labels, weights = build_labels_and_weights(
        example,
        tokenizer,
        loss_mask="completion_final_weighted",
        final_weight=5.0,
    )
    assert supervised_set(labels) == set(range(6, 16))
    assert labels[13] != -100
    assert labels[14] != -100
    assert weights[13] == 5.0
    assert weights[14] == 5.0
    assert weights[15] == 1.0


def test_repeat_count_tick_tokens_are_trace_index_segment() -> None:
    example = repeat_count_example()
    assert token_segment(example, 7) == "trace_index_loss"
    assert token_segment(example, 8) == "trace_marker_loss"
    assert token_segment(example, 9) == "trace_index_loss"
    assert token_segment(example, 13) == "count_loss"
    assert token_segment(example, 14) == "count_loss"
