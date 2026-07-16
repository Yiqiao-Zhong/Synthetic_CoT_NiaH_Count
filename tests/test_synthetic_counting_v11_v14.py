from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from synthetic_counting_v11.config import preset_config
from synthetic_counting_v11.data import (
    TINY_SHAKESPEARE_URL,
    Vocab,
    _noise_sequence,
    load_or_create_fixed_pool,
    make_example,
    render,
    shakespeare_text,
)
from synthetic_counting_v11.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_v11_v14_architecture_is_always_four_layers_four_heads_and_d64():
    for version in ("v11", "v12", "v13", "v14"):
        cfg = preset_config(version, "main")
        assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 64, 256)
        assert "hidden size 64" in cfg.to_dict()["architecture_note"]


def test_version_protocols_match_the_requested_controlled_differences():
    v11 = preset_config("v11", "main")
    assert v11.position_encodings == ("ape", "rope", "rpe")
    assert (v11.seq_len, v11.count_min, v11.count_max) == (256, 1, 30)
    assert len(v11.model_variants) == 6

    v12 = preset_config("v12", "main")
    assert v12.position_encodings == ("ape",)
    assert (v12.seq_len, v12.count_min, v12.count_max) == (512, 1, 50)
    assert v12.count_bins == ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50))
    assert len(v12.model_variants) == 2

    v13 = preset_config("v13", "main")
    assert v13.training_data_mode == "fixed"
    assert v13.position_encodings == ("ape",)

    v14 = preset_config("v14", "main")
    assert v14.noise_source == "shakespeare_char"
    assert v14.position_encodings == ("ape",)


def test_v11_uses_v2_v10_rendering_and_shared_number_tokens():
    cfg = preset_config("v11", "debug")
    vocab = Vocab.build(cfg)
    example = make_example(cfg, vocab, random.Random(11), count=4)
    direct = render(example, vocab, "nonthinking")
    thinking = render(example, vocab, "thinking")

    assert direct.tokens[-3:] == ["<Ans>", "<4>", "<EOS>"]
    assert thinking.tokens[-4:] == ["</Think>", "<Ans>", "<4>", "<EOS>"]
    assert [thinking.tokens[position] for position in thinking.spans.trace_index_positions] == [
        "<1>",
        "<2>",
        "<3>",
        "<4>",
    ]
    assert thinking.tokens[thinking.spans.count_pos] == "<4>"
    assert thinking.labels[thinking.spans.trace_index_positions[0]] == vocab.number_id(1)
    assert direct.labels[direct.spans.count_pos] == vocab.number_id(4)


def test_all_three_position_encodings_forward_with_the_same_small_width():
    cfg = preset_config("v11", "debug", device="cpu")
    vocab = Vocab.build(cfg)
    item = render(make_example(cfg, vocab, random.Random(12), count=3), vocab, "thinking")
    ids = torch.tensor([item.input_ids], dtype=torch.long)

    parameter_counts = {}
    for position_encoding in ("ape", "rope", "rpe"):
        model = build_model(cfg, vocab, position_encoding, "cpu").eval()
        with torch.no_grad():
            output = model(ids, output_attentions=True, output_hidden_states=True)
        assert output.logits.shape == (1, len(item.input_ids), len(vocab.id_to_token))
        assert len(output.attentions or ()) == 4
        assert len(output.hidden_states or ()) == 5
        assert model.config.n_embd == 64
        assert model.layers[0].attention.head_dim == 16
        parameter_counts[position_encoding] = model.parameter_count()

    assert parameter_counts["ape"] > parameter_counts["rope"]
    assert parameter_counts["rpe"] > parameter_counts["rope"]


def test_v13_fixed_pool_is_persisted_and_reloaded_exactly(tmp_path):
    cfg = preset_config("v13", "debug")
    vocab = Vocab.build(cfg)
    path = tmp_path / "fixed_train_pool.npz"
    first = load_or_create_fixed_pool(path, cfg, vocab)
    second = load_or_create_fixed_pool(path, cfg, vocab)

    assert len(first) == cfg.fixed_train_examples_per_count * cfg.count_max
    assert np.array_equal(first.prompt_ids, second.prompt_ids)
    assert np.array_equal(first.counts, second.counts)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["training_semantics"].startswith("finite pool")
    assert len(metadata["dataset_sha256"]) == 64


def test_v14_haystack_is_a_contiguous_shakespeare_character_window():
    cfg = preset_config("v14", "debug")
    vocab = Vocab.build(cfg)
    sequence = _noise_sequence(cfg, vocab, random.Random(13))
    recovered = "".join(chr(int(token[4:-1], 16)) for token in sequence)
    corpus = shakespeare_text()

    assert "karpathy/char-rnn" in TINY_SHAKESPEARE_URL
    assert "First Citizen:" in corpus
    assert len(corpus) > 1_000_000
    assert recovered in corpus
    assert cfg.noise_vocab_size == 64  # ignored by the character-level vocabulary
    assert len(vocab.noise) == len(set(corpus))


def test_colab_notebooks_expose_src_to_the_current_kernel_before_runtime_import():
    for version in range(11, 15):
        path = ROOT / "notebooks" / f"Trace_Count_v{version}_Colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}

        setup = cells["environment-setup"]
        runtime = cells["runtime-settings"]
        assert 'src_root = (repo / "src").resolve()' in setup
        assert 'sys.path.insert(0, str(src_root))' in setup
        assert "from synthetic_counting_v11.config import preset_config as _import_probe" in setup
        assert "PLANNED_CONFIG.n_embd" in runtime
