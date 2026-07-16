# v11-v14: Small-Architecture Controlled Counting Experiments

## Shared controls

All four experiments use the same small causal Transformer budget:

| field | value |
| --- | --- |
| layers | 4 |
| attention heads | 4 |
| hidden size (`d_model`) | 64 |
| head size | 16 |
| MLP hidden size | 256 |
| initialization | random |
| normalization | pre-LayerNorm |
| token embedding / unembedding | tied |
| models | separate `nonthinking` and `thinking` Transformers |

The v12-v14 implementation intentionally reuses the v10-style rendering, training,
evaluation, and analysis path, but **not** the v10 model width. Configuration validation
rejects any v11-v14 run that is not exactly `4 layers x 4 heads x d_model 64`.

The two modes share number tokens. A count `k` is represented by the same `<k>` token in
the thinking trace and in the final answer:

```text
nonthinking: <BOS> prompt <Ans> <n> <EOS>
thinking:    <BOS> prompt <Think> <1> M1 ... <n> Mn </Think> <Ans> <n> <EOS>
```

Both models receive completion-only next-token supervision. They use matched prompt
streams and matched shared-parameter initialization so comparisons are not confounded by
different examples or unrelated random initial weights.

## v11: positional-encoding comparison

- Prompt length: 256.
- Needle count: balanced `1..30`.
- Six training runs: `APE/RoPE/RPE x nonthinking/thinking`.
- APE: learned absolute position embeddings added at the input.
- RoPE: rotary position transformations applied to each query/key head.
- RPE: learned per-layer, per-head causal relative-distance attention bias.

The experiment stops after the first six report sections used in v10: learning dynamics,
descriptive attention, and descriptive hidden-state geometry. It does not run the later
v10 causal intervention suite.

## v12: longer context and more needles

- Position encoding: APE only.
- Prompt length: 512.
- Needle count: balanced `1..50`.
- Two training runs: `nonthinking` and `thinking`.
- Architecture remains `4L/4H/d64/MLP256`.

This isolates task scaling while keeping the small architecture fixed.

## v13: fixed-dataset training

- Position encoding: APE only.
- Prompt length: 256.
- Needle count: balanced `1..30`.
- Training data: a finite, balanced pool generated once and saved as
  `data/fixed_train_dataset.npz`.
- Main pool size: 512 examples per exact count.

The saved pool includes a vocabulary fingerprint and is reloaded exactly on resumed runs.
This distinguishes finite-dataset memorization/generalization dynamics from the streaming
fresh-sample regime used by v11, v12, and v14.

## v14: Shakespeare character haystack

- Position encoding: APE only.
- Prompt length: 256.
- Needle count: balanced `1..30`.
- Haystack: a random contiguous character window from the bundled public-domain
  Shakespeare excerpt, followed by random marker insertion.

The character sequence is contiguous before marker replacement. Therefore the model sees
natural local character correlations rather than i.i.d. uniform noise, while marker
placement and the counting target remain controlled.

## Recorded analyses

Every version writes:

- train loss and component losses;
- teacher-forced and autoregressive final-count accuracy by exact count and 10-count bin;
- final-answer broad-attention scores;
- thinking trace k-to-k retrieval mass, top-1 retrieval, and diagonal dominance;
- final-answer trace-marker readout mass;
- hidden-state nearest-centroid accuracy, position-only baseline, ridge R2, and MAE;
- count/progress centroid PCA coordinates and PC1-PC6 explained variance.

Checkpoints and result folders can be restored from and synchronized to Google Drive.
The notebooks mount Drive at the beginning, use resumable checkpoints, display results
directly, save the complete run bundle, and then optionally disconnect the Colab runtime.

## Commands

```bash
python -m synthetic_counting_v11.run_v11 --preset debug --stage all --device cpu
python -m synthetic_counting_v12.run_v12 --preset debug --stage all --device cpu
python -m synthetic_counting_v13.run_v13 --preset debug --stage all --device cpu
python -m synthetic_counting_v14.run_v14 --preset debug --stage all --device cpu
```

Replace `debug` with `main` and use `--device cuda` for the full Colab experiments.

