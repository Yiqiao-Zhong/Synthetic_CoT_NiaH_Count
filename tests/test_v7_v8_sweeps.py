from __future__ import annotations

from synthetic_counting_extensions.v7_v8_sweeps import (
    SweepConfig,
    Vocab,
    preset_configs,
    render,
    validation_split,
)
from synthetic_counting_extensions.v7_v8_sweeps import Example


def test_numeric_tokens_are_shared_between_trace_and_answer() -> None:
    cfg = SweepConfig(experiment="v8", train_count_max=30, eval_count_max=30)
    vocab = Vocab(cfg)

    assert vocab.count_token(7) == "<7>"
    assert vocab.index_token(7) == "<7>"
    assert vocab.token_to_id[vocab.count_token(7)] == vocab.token_to_id[vocab.index_token(7)]
    assert "<C7>" not in vocab.token_to_id
    assert "<I7>" not in vocab.token_to_id
    assert len(vocab.id_to_token) == 6 + 64 + 10 + 30


def test_thinking_render_reuses_the_final_number_token() -> None:
    cfg = SweepConfig(
        experiment="v8",
        seq_len=4,
        train_count_max=3,
        eval_count_max=3,
    )
    vocab = Vocab(cfg)
    example = Example(
        seq_tokens=["<M0>", "<N0>", "<M1>", "<M2>"],
        count=3,
        needle_positions=[0, 2, 3],
        needle_markers=["<M0>", "<M1>", "<M2>"],
    )

    rendered = render(example, vocab, "thinking")
    assert rendered["tokens"] == [
        "<BOS>",
        "<M0>",
        "<N0>",
        "<M1>",
        "<M2>",
        "<Think>",
        "<1>",
        "<M0>",
        "<2>",
        "<M1>",
        "<3>",
        "<M2>",
        "</Think>",
        "<Ans>",
        "<3>",
        "<EOS>",
    ]


def test_v7_main_is_two_length_only_settings() -> None:
    configs = preset_configs("v7", "main")

    assert [cfg.seq_len for cfg in configs] == [1024, 2048]
    assert all(cfg.train_count_min == 1 and cfg.train_count_max == 10 for cfg in configs)
    assert all(cfg.eval_count_min == 1 and cfg.eval_count_max == 10 for cfg in configs)
    assert all(cfg.train_steps == 10000 for cfg in configs)
    assert all(cfg.effective_batch_size == 128 for cfg in configs)


def test_v8_main_is_one_many_needle_setting_with_three_validation_ranges() -> None:
    configs = preset_configs("v8", "main")

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.seq_len == 256
    assert (cfg.train_count_min, cfg.train_count_max) == (1, 30)
    assert (cfg.eval_count_min, cfg.eval_count_max) == (1, 30)
    assert cfg.train_steps == 10000
    assert cfg.effective_batch_size == 128
    assert validation_split(1) == "val_1_10"
    assert validation_split(10) == "val_1_10"
    assert validation_split(11) == "val_11_20"
    assert validation_split(20) == "val_11_20"
    assert validation_split(21) == "val_21_30"
    assert validation_split(30) == "val_21_30"


def test_v8_debug_exercises_all_three_validation_ranges() -> None:
    cfg = preset_configs("v8", "debug")[0]
    assert cfg.seq_len == 48
    assert cfg.train_count_max == 30
    assert cfg.eval_count_max == 30
