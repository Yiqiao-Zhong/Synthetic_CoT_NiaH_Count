from __future__ import annotations

import random

import numpy as np
import torch

from synthetic_counting_v10.attention_causal import (
    _head_mask,
    attention_categories,
    collect_attention,
)
from synthetic_counting_v10.config import preset_config
from synthetic_counting_v10.core import (
    Vocab,
    build_model,
    component_target_positions,
    make_example,
    render,
)
from synthetic_counting_v10.state_causal import (
    SITE_NONTHINKING_FINAL,
    _fit_pca,
    fit_count_directions,
)
from synthetic_counting_v10.report_followups import (
    _forward_with_position_local_head_ablation,
    _layer_matched_order,
    _local_query_positions,
)
from synthetic_counting_v10.successor_patching import (
    DIRECTION_CLOSE,
    DIRECTION_CONTINUE,
    run_successor_patching_rows,
    successor_nested_example_pair,
)
from synthetic_counting_v10.successor_conversion import (
    INTERVENTIONS,
    run_successor_conversion_rows,
    summarize_successor_conversion,
    summarize_successor_logit_lens,
)
from synthetic_counting_v10.successor_mlp_features import (
    _analysis_layers,
    evaluate_mlp_feature_patching,
    fit_mlp_feature_statistics,
    summarize_feature_concentration,
    summarize_mlp_feature_patching,
)
from synthetic_counting_v10.geometry_path_steering import (
    centroid_chord_point,
    centroid_polyline_point,
)


def tiny_setup():
    cfg = preset_config("debug", count_max=6)
    vocab = Vocab.build(cfg)
    model = build_model(cfg, vocab, "cpu").eval()
    return cfg, vocab, model


def test_v10_uses_separate_v2_formats_and_shared_numeric_tokens():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(1), count=6)
    direct = render(example, vocab, "nonthinking")
    thinking = render(example, vocab, "thinking")

    assert direct.tokens[-3:] == ["<Ans>", "<6>", "<EOS>"]
    assert thinking.tokens[-4:] == ["</Think>", "<Ans>", "<6>", "<EOS>"]
    assert [thinking.tokens[pos] for pos in thinking.spans.trace_index_positions] == [
        f"<{value}>" for value in range(1, 7)
    ]
    assert thinking.tokens[thinking.spans.count_pos] == thinking.tokens[thinking.spans.trace_index_positions[-1]]
    assert len(thinking.tokens) <= cfg.n_positions
    assert component_target_positions(direct)["final_count"] == [direct.spans.count_pos]
    assert len(component_target_positions(thinking)["trace_marker"]) == 6


def test_attention_categories_partition_the_causal_row():
    cfg, vocab, _ = tiny_setup()
    example = make_example(cfg, vocab, random.Random(2), count=4)
    item = render(example, vocab, "thinking")
    row = np.zeros(len(item.tokens), dtype=float)
    row[: item.spans.ans_pos + 1] = 1.0 / (item.spans.ans_pos + 1)
    categories = attention_categories(item, row)
    assert abs(categories["classified_mass"] + categories["other_or_query_self_mass"] - 1.0) < 1e-6
    assert categories["prompt_needles_mass"] > 0
    assert 0 <= categories["broad_attention_score"] <= categories["prompt_needles_mass"]


def test_tiny_attention_collection_and_head_mask():
    cfg, vocab, model = tiny_setup()
    examples = [make_example(cfg, vocab, random.Random(3), count=3)]
    detail, summary = collect_attention(model, vocab, examples, "thinking", "cpu")
    assert not detail.empty
    assert {"final_count_query", "targeted_retrieval_query", "successor_query"}.issubset(
        set(summary.query_kind)
    )
    mask = _head_mask(model, [(0, 1), (1, 0)], "cpu")
    assert mask.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_pca_and_direction_geometry_recover_a_known_count_axis():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(1, 7), 8)
    values = rng.normal(scale=0.01, size=(len(labels), 12))
    values[:, 0] += labels * 2.0
    _, coordinates, variance = _fit_pca(values, 6)
    assert coordinates.shape == (len(labels), 6)
    assert variance[:6].sum() <= 1.0 + 1e-8

    states = {SITE_NONTHINKING_FINAL: [values, values + 0.1]}
    eval_values = values + rng.normal(scale=0.005, size=values.shape)
    directions, geometry, centroids = fit_count_directions(
        states,
        {SITE_NONTHINKING_FINAL: labels},
        {SITE_NONTHINKING_FINAL: [eval_values, eval_values + 0.1]},
        {SITE_NONTHINKING_FINAL: labels},
    )
    assert (SITE_NONTHINKING_FINAL, 0, "adjacent_mean") in directions
    assert geometry.projection_r2_heldout.max() > 0.99
    assert (SITE_NONTHINKING_FINAL, 0, 6) in centroids


def test_v10_main_matches_requested_v2_architecture_and_count_range():
    cfg = preset_config("main")
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_inner) == (4, 4, 256, 1024)
    assert (cfg.seq_len, cfg.count_min, cfg.count_max) == (256, 1, 30)
    assert cfg.early_stop_patience == 0
    assert cfg.max_render_len == 322
    assert cfg.n_positions >= cfg.max_render_len


def test_layer_matched_controls_preserve_the_ranked_layer_sequence():
    ranking = [(2, 1), (3, 0), (2, 3), (0, 2), (3, 2), (0, 1)]
    low = _layer_matched_order(ranking, reverse_within_layer=True)
    random_order = _layer_matched_order(ranking, seed=19)
    assert [layer for layer, _ in low] == [layer for layer, _ in ranking]
    assert [layer for layer, _ in random_order] == [layer for layer, _ in ranking]
    assert sorted(low) == sorted(ranking)
    assert sorted(random_order) == sorted(ranking)


def test_position_local_ablation_runs_at_trace_and_answer_queries_without_leaking_hooks():
    cfg, vocab, model = tiny_setup()
    example = make_example(cfg, vocab, random.Random(7), count=4)
    item = render(example, vocab, "thinking")
    ids = torch.tensor([item.input_ids], dtype=torch.long)
    mask = torch.ones_like(ids)
    baseline = model(input_ids=ids, attention_mask=mask).logits
    local = _forward_with_position_local_head_ablation(
        model,
        ids,
        mask,
        [(0, 0)],
        [_local_query_positions(item, "trace_index_tokens")],
    ).logits
    restored = model(input_ids=ids, attention_mask=mask).logits
    assert local.shape == baseline.shape
    assert not torch.allclose(local, baseline)
    assert torch.allclose(restored, baseline)
    assert _local_query_positions(item, "ans_token") == [item.spans.ans_pos]


def test_successor_pair_is_identical_through_marker_k_and_then_changes_target():
    cfg, vocab, _ = tiny_setup()
    short, long = successor_nested_example_pair(cfg, vocab, k=4, seed=81)
    assert short.count == 4
    assert long.count == 5
    assert short.needle_positions == long.needle_positions[:4]
    assert short.needle_markers == long.needle_markers[:4]
    assert long.needle_positions[-1] > short.needle_positions[-1]

    short_item = render(short, vocab, "thinking")
    long_item = render(long, vocab, "thinking")
    short_query = short_item.spans.trace_marker_positions[3]
    long_query = long_item.spans.trace_marker_positions[3]
    assert short_query == long_query
    assert (
        short_item.tokens[short_item.spans.think_pos : short_query + 1]
        == long_item.tokens[long_item.spans.think_pos : long_query + 1]
    )
    assert short_item.tokens[short_query + 1] == "</Think>"
    assert long_item.tokens[long_query + 1] == "<5>"


def test_successor_patching_smoke_records_both_directions_and_controls():
    cfg, vocab, model = tiny_setup()
    rankings = {
        "successor": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "targeted_retrieval": [(1, 1), (0, 1), (1, 0), (0, 0)],
    }
    detail = run_successor_patching_rows(
        model,
        cfg,
        vocab,
        rankings,
        examples_per_k=1,
        random_replicates=1,
        top_ns=(1, 2),
    )
    assert set(detail.direction) == {DIRECTION_CONTINUE, DIRECTION_CLOSE}
    assert set(detail.family) == {
        "successor_top",
        "targeted_top",
        "successor_wrong_row",
        "random",
    }
    assert set(detail.donor_row) == {"marker_query", "index_query"}
    assert detail.normalized_recovery.notna().any()


def test_successor_conversion_smoke_records_all_sublayers_and_interventions():
    cfg, vocab, model = tiny_setup()
    logit_detail, patch_detail = run_successor_conversion_rows(
        model,
        cfg,
        vocab,
        examples_per_k=1,
    )
    assert not logit_detail.empty
    assert not patch_detail.empty
    assert set(logit_detail.stage) == {
        "resid_pre",
        "attn_out",
        "post_attn",
        "mlp_out",
        "post_mlp",
    }
    assert set(patch_detail.intervention) == set(INTERVENTIONS)
    assert patch_detail.patched_margin.notna().all()
    assert not summarize_successor_logit_lens(logit_detail).empty
    assert not summarize_successor_conversion(patch_detail).empty


def test_successor_mlp_feature_fit_and_held_out_patch_smoke():
    cfg, vocab, model = tiny_setup()
    layers = _analysis_layers(cfg)
    assert layers == (0, 1)
    statistics, fitted = fit_mlp_feature_statistics(
        model,
        cfg,
        vocab,
        fit_examples_per_k=1,
        layers=layers,
    )
    assert not statistics.empty
    assert set(statistics.layer) == set(layers)
    assert statistics.groupby(["direction", "count_bin", "layer"]).size().min() == cfg.n_inner
    concentration = summarize_feature_concentration(statistics, (1, 4, cfg.n_inner))
    assert set(concentration.support_size) == {1, 4, cfg.n_inner}

    detail = evaluate_mlp_feature_patching(
        model,
        cfg,
        vocab,
        fitted,
        eval_examples_per_k=1,
        fit_examples_per_k=1,
        support_sizes=(1, 4, cfg.n_inner),
        random_replicates=1,
        layers=layers,
        patch_batch_size=8,
    )
    assert not detail.empty
    assert set(detail.family) == {
        "ranked_feature_replacement",
        "random_feature_replacement",
        "sparse_mean_direction",
        "random_sparse_mean_direction",
    }
    assert detail.patched_margin.notna().all()
    full = detail[
        (detail.family == "ranked_feature_replacement")
        & (detail.support_size == cfg.n_inner)
    ]
    assert not full.empty
    assert np.isfinite(full.normalized_recovery).all()
    assert not summarize_mlp_feature_patching(detail).empty


def test_centroid_curve_differs_midway_but_matches_chord_endpoint():
    centroids = {
        1: np.asarray([0.0, 0.0], dtype=np.float32),
        2: np.asarray([1.0, 0.0], dtype=np.float32),
        3: np.asarray([1.0, 2.0], dtype=np.float32),
    }
    chord_mid, chord_count = centroid_chord_point(centroids, 1, 3, 0.5)
    curve_mid, curve_count = centroid_polyline_point(centroids, 1, 3, 0.5)
    assert np.allclose(chord_mid, [0.5, 1.0])
    assert np.allclose(curve_mid, [1.0, 0.5])
    assert chord_count == 2.0
    assert curve_count == 2.25
    chord_end, chord_end_count = centroid_chord_point(centroids, 1, 3, 1.0)
    curve_end, curve_end_count = centroid_polyline_point(centroids, 1, 3, 1.0)
    assert np.allclose(chord_end, centroids[3])
    assert np.allclose(curve_end, centroids[3])
    assert chord_end_count == curve_end_count == 3.0


def test_centroid_curve_supports_reverse_count_transport():
    centroids = {
        1: np.asarray([0.0, 0.0], dtype=np.float32),
        2: np.asarray([1.0, 0.0], dtype=np.float32),
        3: np.asarray([1.0, 2.0], dtype=np.float32),
    }
    start, start_count = centroid_polyline_point(centroids, 3, 1, 0.0)
    end, end_count = centroid_polyline_point(centroids, 3, 1, 1.0)
    assert np.allclose(start, centroids[3])
    assert np.allclose(end, centroids[1])
    assert start_count == 3.0
    assert end_count == 1.0
