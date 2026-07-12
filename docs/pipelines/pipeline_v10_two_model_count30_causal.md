# Trace Count v10: two-model count-30 dynamics and causal mechanisms

## 1. Scientific objective

v10 keeps the controlled v2 counting task and model family, while extending the
needle count from `1..10` to `1..30`. It asks two separate questions:

1. When and where do separately trained non-thinking and thinking models learn the
   three count ranges `1..10`, `11..20`, and `21..30`?
2. Are the observed broad-attention, targeted-retrieval, and count-state signatures
   causally involved in prediction, or are they merely readable correlates?

There is no v5-style mode switch. Non-thinking and thinking are two independently
initialized and independently optimized Transformers.

## 2. Controlled task and architecture

Prompt length is fixed at 256. Each prompt contains 64 possible noise-token types,
10 possible marker-token types, and a uniformly sampled needle count from 1 through
30. Needle locations are sampled without replacement. Marker identities are sampled
with replacement, so the answer cannot be obtained by counting distinct marker types.

Both models use the v2 architecture:

- random-initialized GPT-2 decoder-only LM;
- 4 Transformer Layers, 4 attention heads per Layer;
- `d_model=256`, MLP width 1024;
- learned absolute-position embeddings;
- no dropout;
- context capacity 384 tokens.

The numeric tokens `<1>...<30>` are shared between trace indices and final answers.
This avoids introducing a separate index-token codebook.

```text
non-thinking:
<BOS> prompt <Ans> <n> <EOS>

thinking:
<BOS> prompt <Think> <1> M1 <2> M2 ... <n> Mn </Think> <Ans> <n> <EOS>
```

The non-thinking loss supervises the final numeric answer and EOS. The thinking loss
supervises the complete trace continuation, close token, answer marker, final numeric
answer, and EOS. Prompt tokens are context only.

## 3. Training dynamics

Main defaults are 10,000 optimizer steps, batch size 128, AdamW at `3e-4`, 500 warmup
steps, cosine decay, and seed 1234. Training logs are written every 50 steps.

Every 500 steps, v10 performs balanced teacher-forced evaluation. Every 1,000 steps it
also performs sampled autoregressive evaluation and writes a fully resumable checkpoint.
The following losses are saved separately:

- total supervised completion loss;
- trace-index loss;
- trace-marker loss;
- thinking-close loss;
- answer-marker loss;
- final-count loss;
- EOS loss.

Accuracy and MAE are saved by exact count and by the balanced bins `1-10`, `11-20`,
and `21-30`. Both teacher-forced final-count accuracy and autoregressive final-count
accuracy are retained. Thinking evaluation additionally retains trace-index accuracy,
trace-marker accuracy, exact trace rate, and marker recall.

Early stopping is implemented but disabled by default. The complete 10k trajectory is
the scientific object, and stopping the two models at different times would confound
their learning dynamics. A `best` checkpoint is still maintained alongside periodic and
final checkpoints.

## 4. Attention-head definitions

Let `q` be a query position and `p_1,...,p_n` the prompt needle positions. Let
`A_lh(q,p)` be the causal attention probability at Layer `l`, head `h`.

### Broad needle attention

For the final-count query `<Ans>`, define needle mass

`m_N = sum_j A_lh(q, p_j)`.

Normalize the attention weights within the needle subset as

`r_j = A_lh(q,p_j) / m_N`,

and define normalized entropy

`H_N = -sum_j r_j log(r_j) / log(n)`.

The broad score is

`B_lh = m_N * H_N`.

It is high only when a head assigns substantial total mass to needles and distributes
that mass across multiple needles. A head that sharply retrieves one needle has low
entropy and therefore a lower broad score.

### Targeted k-to-k retrieval

In the thinking trace, the numeric token `<k>` is the causal query that predicts marker
`M_k`. Its targeted score is

`T_lh = mean_examples,k A_lh(<k>, p_k)`.

The pipeline also records top-1 retrieval and diagonal dominance, where diagonal
dominance divides correct-needle mass by total prompt-needle mass.

### Causal tests

All 16 heads are ranked by each signature. Cumulative head-mask ablation is evaluated
for every `top_n` from 1 through 16, with a deterministic matched-random ranking as a
control. Effects are measured on final-count accuracy/margin, marker accuracy/margin,
and successor-index accuracy.

Clean-to-corrupt head-output patching changes one prompt marker identity while keeping
count and positions fixed, then restores selected clean head outputs at the matching
trace query. Recovery is

`(patched_margin - corrupt_margin) / (clean_margin - corrupt_margin)`.

Count-state head patching covers donor offsets `-10,-5,-3,-2,-1,+1,+2,+3,+5,+10`
rather than only adjacent counts. Tables always retain donor and receiver positions,
because the model uses learned absolute positions.

## 5. Hidden-state geometry and causality

Residual states are collected after every Transformer Layer at four anchors:

- non-thinking `<Ans>` final-count query;
- thinking `<Ans>` final-count query;
- thinking `<Ans>` after a fixed 15-pair trace canvas, which holds trace content and
  absolute answer position constant while prompt count varies;
- thinking trace index `<k>`, which predicts marker `M_k`;
- thinking trace marker `M_k`, which predicts `<k+1>` or `</Think>`.

For each anchor and Layer, independent train/eval samples are used to fit:

- count/progress centroids;
- adjacent-centroid mean directions;
- ridge count directions;
- 2-PC, 3-PC, and 6-PC PCA manifolds.

The report records held-out projection R-squared, MAE, adjacent-difference cosine,
projected step variance, and cumulative PCA variance. It produces separate 2D and 3D
plots for every Layer.

Readability is not treated as causality. The following interventions are therefore run:

1. geometry steering along one estimated count step at each Layer;
2. final-query full-residual transplant from donor count `m` to receiver count `n`;
3. within-trace transplant from marker state `M_m` to `M_n`;
4. final-to-earlier transplants that test induced early close;
5. earlier-to-final transplants that test induced continuation.

The trace experiment distinguishes a state that merely predicts the next token from a
state sufficient to transport progress/stop behavior. Because `M_m` and `M_n` occupy
different absolute positions, every row stores the position delta and conclusions must
retain this limitation.

## 6. Output contract

The run directory contains:

```text
config.json
vocab.json
checkpoints/{nonthinking,thinking}/{step_*,best,final}/
tables/                         training and evaluation dynamics
figures/training/               learning curves
analysis/attention_causal/      attention rows, rankings, ablation, patching
analysis/state_causal/          PCA, directions, steering, state transplants
```

Run locally or in Colab:

```bash
python -m synthetic_counting_v10.run_v10 \
  --preset main \
  --stage all \
  --device cuda \
  --out-root runs/synthetic_counting_v10 \
  --run-name v10_main_seed1234 \
  --skip-completed
```

For safer long Colab sessions, run stages separately in the order `train`, `attention`,
`state`, `plots`. The notebook mounts Google Drive before imports and can restore the
run from its checkpoint-sync directory after a runtime disconnect.
