# Synthetic NIAH Counting v5: One Transformer With Thinking Toggle

You are implementing v5 of the synthetic NIAH counting experiments. The goal is to train **one** small decoder-only transformer on a mixed dataset containing both thinking and non-thinking formats. At test time, the user controls the mode by either leaving the `<Think/>` block open or by immediately closing it.

This mimics the practical LLM interface:

```text
thinking enabled:
  seq_tokens <Think/>

non-thinking enabled:
  seq_tokens <Think/> </Think>
```

The model should learn both behaviors inside a single set of weights.

---

## 0. Core idea

For a sequence containing three needles, for example:

```text
... <X> ... <Y> ... <Z> ...
```

include both training variants:

```text
thinking variant:
  <BOS> ... <X> ... <Y> ... <Z> ... <Think/> <X> <Y> <Z> </Think> <C3> <EOS>

non-thinking variant:
  <BOS> ... <X> ... <Y> ... <Z> ... <Think/> </Think> <C3> <EOS>
```

At evaluation:

```text
thinking query:
  <BOS> ... <X> ... <Y> ... <Z> ... <Think/>
  model should generate: <X> <Y> <Z> </Think> <C3>

non-thinking query:
  <BOS> ... <X> ... <Y> ... <Z> ... <Think/> </Think>
  model should predict: <C3>
```

Use marker-only traces by default. Do not use `<I1> <I2> ...` index tokens in v5 unless a flag explicitly enables them.

---

## 1. Important ambiguity and required loss mask

If both variants are trained with full next-token loss, the prefix below is ambiguous:

```text
<BOS> seq_tokens <Think/>
```

For thinking examples, the next token should be the first marker.
For non-thinking examples, the next token would be `</Think>`.

That creates a 50/50 next-token conflict and can cause the model to close the thinking block immediately during thinking-mode generation.

Therefore, implement this rule:

```text
In non-thinking examples, treat </Think> as a provided control token, not as a token the model must learn to generate.
Mask the loss on the non-thinking </Think> token.
Only train the model to predict the count after </Think>.
```

This preserves the intended mode interface:

```text
thinking mode:
  user supplies <Think/>, model generates trace and </Think>

non-thinking mode:
  user supplies <Think/> </Think>, model skips trace and predicts count
```

Add an optional ablation flag:

```bash
--ablate-no-conflict-mask
```

When this flag is set, train full next-token loss on `</Think>` for non-thinking examples. This is expected to hurt thinking-mode trace generation and should be reported as a diagnostic, not the default.

---

## 2. Task definition

### Vocabulary

Use exactly this vocabulary unless a config override is supplied.

```text
special tokens:
  <BOS>, <EOS>, <Think/>, </Think>

noise tokens:
  <N0>, <N1>, ..., <N63>

countable marker tokens:
  <A>, <B>, <C>, <D>, <E>, <F>, <G>, <H>, <I>, <J>

answer count tokens:
  <C1>, <C2>, ..., <C10>

optional trace index tokens, disabled by default:
  <I1>, <I2>, ..., <I10>
```

There is no `<ANS>` token in v5 by default. The count token directly follows `</Think>`.

### Base example generation

Each base example has:

```python
seq_len: int = 256
count: int in {1, ..., 10}
needle_positions: sorted list[int] of length count
needle_markers: list[str] of length count
seq_tokens: list[str] of length seq_len
```

Sampling rules:

```text
count: uniform over 1..10
needle positions: choose count unique positions uniformly from 0..seq_len-1
needle marker type: each marker independently uniform over <A>..<J>
noise token type: each non-needle position independently uniform over <N0>..<N63>
```

Trace order is left-to-right by prompt position.

---

## 3. Rendered examples

For every base example, render two paired variants.

### 3.1 Thinking variant

Default marker-only trace:

```text
<BOS> seq_tokens <Think/> marker_1 marker_2 ... marker_n </Think> <C_n> <EOS>
```

Example:

```text
<BOS> <N4> <A> <N9> <B> <N2> <C> <N7>
<Think/> <A> <B> <C> </Think> <C3> <EOS>
```

If `--trace-indices` is enabled, use:

```text
<BOS> seq_tokens <Think/> <I1> marker_1 <I2> marker_2 ... <In> marker_n </Think> <C_n> <EOS>
```

But `--trace-indices` should default to false.

### 3.2 Non-thinking variant

```text
<BOS> seq_tokens <Think/> </Think> <C_n> <EOS>
```

Example:

```text
<BOS> <N4> <A> <N9> <B> <N2> <C> <N7>
<Think/> </Think> <C3> <EOS>
```

---

## 4. Training objective

Use standard causal next-token prediction with label masking.

### 4.1 Thinking labels

For thinking examples:

```text
prompt tokens:
  <BOS> seq_tokens <Think/>

supervised tokens:
  marker_1 marker_2 ... marker_n </Think> <C_n> <EOS>
```

Labels before the first trace marker are `-100`.

### 4.2 Non-thinking labels

For non-thinking examples:

```text
prompt/control tokens:
  <BOS> seq_tokens <Think/> </Think>

supervised tokens:
  <C_n> <EOS>
```

Labels up through and including `</Think>` are `-100`.

This is mandatory unless `--ablate-no-conflict-mask` is passed.

### 4.3 Batch composition

Use one single model and one optimizer.

Default batch mixture:

```yaml
train:
  thinking_fraction: 0.5
  nonthinking_fraction: 0.5
```

Use paired base examples when possible:

```text
For each sampled base example, choose the variant by Bernoulli(0.5), or include both variants in the same batch if batch construction makes this easy.
```

The report must show per-mode losses separately:

```text
thinking_trace_loss
thinking_final_count_loss
nonthinking_final_count_loss
```

---

## 5. Model

Train one small decoder-only transformer from scratch, using the same transformer family as v2: HuggingFace `GPT2LMHeadModel` built from `GPT2Config`.

Important: v5 should **not** use the v3 custom RoPE transformer. The v2 architecture uses learned absolute positional embeddings through GPT-2's `wpe` table. This keeps the mixed thinking-toggle experiment comparable to the previous v2 attention/trace results.

Default main config:

```yaml
model:
  architecture: gpt2_lm_head
  position_embedding: learned_absolute
  n_layer: 4
  n_head: 4
  n_embd: 256
  n_positions: 384
  dropout: 0.0
  activation_function: gelu_new
```

Debug config:

```yaml
model:
  architecture: gpt2_lm_head
  position_embedding: learned_absolute
  n_layer: 2
  n_head: 2
  n_embd: 128
  n_positions: 384
  dropout: 0.0
  activation_function: gelu_new
```

Training defaults:

```yaml
train:
  seq_len: 256
  count_min: 1
  count_max: 10
  steps: 10000
  batch_size: 128
  lr: 3e-4
  weight_decay: 0.01
  warmup_steps: 500
  grad_clip: 1.0
  eval_every: 500
  checkpoint_every: 1000
  seed: 1234
```

---

## 6. Required package structure

Create this package:

```text
synthetic_niah_v5/
  __init__.py
  config.py
  vocab.py
  data.py
  model.py
  train.py
  generation.py
  evaluation.py
  cache.py
  probes.py
  attention.py
  plots.py
  report.py
  run_v5.py
```

Main commands:

```bash
python -m synthetic_niah_v5.run_v5 --preset debug --stage all
python -m synthetic_niah_v5.run_v5 --preset main  --stage all
```

Supported stages:

```text
train
eval
cache
probe
attention
plots
report
all
```

---

## 7. Evaluation

Evaluate the same checkpoint in two modes.

### 7.1 Thinking-on evaluation

Prompt:

```text
<BOS> seq_tokens <Think/>
```

Greedy generate until one of these occurs:

```text
</Think>
<EOS>
max_new_tokens reached
```

After `</Think>`, continue generation for one count token and optional `<EOS>`.

Expected generation for count 3:

```text
<A> <B> <C> </Think> <C3> <EOS>
```

Metrics:

```text
final_count_accuracy
final_count_MAE
trace_exact
trace_marker_precision
trace_marker_recall
trace_duplicate_rate
premature_close_rate
missing_close_rate
invalid_count_rate
```

Definitions:

```text
trace_exact:
  generated trace markers exactly match gold markers left-to-right.

trace_marker_recall:
  fraction of gold prompt needles that appear in trace in the correct left-to-right order.

trace_marker_precision:
  fraction of generated trace markers that correspond to the gold trace sequence.

premature_close_rate:
  model emits </Think> before emitting all gold markers.
```

### 7.2 Non-thinking evaluation

Prompt:

```text
<BOS> seq_tokens <Think/> </Think>
```

Do not ask the model to generate the empty close. The close token is supplied by the query.

Evaluate the next-token logits restricted to count tokens `<C1>.. <C10>`.

Metrics:

```text
final_count_accuracy
final_count_MAE
undercount_rate
overcount_rate
invalid_count_rate
```

### 7.3 Ambiguous-prefix diagnostic

For the prefix:

```text
<BOS> seq_tokens <Think/>
```

measure:

```text
P(next token = </Think>)
P(next token is one of the marker tokens)
P(next token = gold first marker)
```

This is critical. With the default conflict-free mask, the model should prefer starting a trace. With the no-mask ablation, the model may prefer closing immediately.

### 7.4 Per-count reporting

Report all metrics by exact count and by bins:

```text
low  = 1,2,3
mid  = 4,5,6
high = 7,8,9,10
```

No OOD split in v5.

---

## 8. Optional hidden-state analysis

Keep this light in v5. The primary goal is to see whether one model can support both modes.

### 8.1 Anchors

Cache hidden states at:

```text
prompt_marker_k
think_open_pos
think_close_pos
pre_count_pos
count_pos
```

For thinking generated/oracle traces also cache:

```text
trace_marker_k
post_trace_marker_k
```

### 8.2 Probes

Fit simple probes:

```text
hidden -> final_count
hidden at trace_marker_k/post_trace_marker_k -> prefix_count
```

Report position-only and trace-length-only baselines. Do not over-interpret high R2.

### 8.3 Mode comparison

For the same base example, compare hidden states under two prefixes:

```text
thinking prefix:
  <BOS> seq_tokens <Think/> gold_trace </Think>

non-thinking prefix:
  <BOS> seq_tokens <Think/> </Think>
```

At the `</Think>` token and the pre-count position, report:

```text
cosine similarity between thinking and non-thinking hidden states
linear probe accuracy in each mode
count-logit distribution in each mode
```

This helps test whether the same model uses different internal routes depending on whether the thinking block contains a trace.

---

## 9. Attention analysis

For thinking mode, compute trace-to-prompt retrieval attention:

```text
query positions: generated trace marker_k positions
key positions: prompt needle positions
```

Metrics:

```text
correct_top1:
  whether trace marker k attends most to prompt needle k

diagonal_dominance:
  attention mass on k-to-k diagonal divided by total needle attention mass

needle_mass:
  total attention mass from trace queries to all prompt needles

needle_vs_noise_ratio:
  attention mass on prompt needles divided by attention mass on prompt noise tokens
```

For non-thinking mode, compute attention from `</Think>` and pre-count position to prompt needles.

The report should compare:

```text
thinking: sequential trace-to-prompt retrieval
non-thinking: direct readout after empty think block
```

---

## 10. Plots

Generate these figures:

```text
outputs/v5/figures/train_loss_by_step_and_mode.png
outputs/v5/figures/final_accuracy_by_step_mode.png
outputs/v5/figures/final_accuracy_by_count_mode.png
outputs/v5/figures/trace_metrics_by_count.png
outputs/v5/figures/ambiguous_prefix_probs_by_step.png
outputs/v5/figures/confusion_matrix_thinking.png
outputs/v5/figures/confusion_matrix_nonthinking.png
outputs/v5/figures/mode_hidden_similarity.png
outputs/v5/figures/attention_trace_to_prompt_best_head.png
```

At minimum, the debug run must produce the first five.

---

## 11. Required CSV schemas

### `train_log.csv`

```text
step, loss_total, loss_thinking_trace, loss_thinking_final_count,
loss_nonthinking_final_count, lr
```

### `eval_by_step.csv`

```text
step, mode, count, count_bin, n_examples,
final_accuracy, final_mae, undercount_rate, overcount_rate,
trace_exact, trace_marker_precision, trace_marker_recall,
premature_close_rate, missing_close_rate, invalid_count_rate
```

### `ambiguous_prefix.csv`

```text
step, count, count_bin, n_examples,
p_close_after_think, p_any_marker_after_think, p_gold_first_marker_after_think,
argmax_token_after_think, argmax_is_close, argmax_is_gold_first_marker
```

### `attention_metrics.csv`

```text
mode, layer, head, query_anchor,
correct_top1, diagonal_dominance, needle_mass, needle_vs_noise_ratio, entropy
```

### `probe_results.csv`

```text
mode, anchor_name, target, layer, hook_name,
probe_type, accuracy, r2, mae,
position_baseline_acc, trace_len_baseline_acc, leakage_prone
```

---

## 12. Report

Generate:

```text
outputs/v5/report.html
outputs/v5/report.md
```

The report must answer:

1. Can one transformer learn both thinking and non-thinking formats?
2. Does thinking-on generation produce the full marker trace before closing `</Think>`?
3. Does non-thinking query with `<Think/> </Think>` directly predict the count?
4. Does the conflict-free non-thinking loss mask prevent premature close in thinking mode?
5. Do the two modes show different attention/retrieval patterns inside the same model?

The conclusion should be one of:

```text
A. successful toggle:
   thinking mode generates traces and answers correctly; non-thinking mode answers correctly from empty think block.

B. partial toggle:
   one mode works but the other degrades.

C. mode collapse:
   model ignores the toggle or always chooses one behavior.

D. ambiguity failure:
   model closes immediately after <Think/> in thinking mode, likely due to no-mask training conflict.
```

---

## 13. Acceptance tests

Debug command:

```bash
python -m synthetic_niah_v5.run_v5 --preset debug --stage all
```

Must produce:

```text
outputs/v5/report.html
outputs/v5/tables/train_log.csv
outputs/v5/tables/eval_by_step.csv
outputs/v5/tables/ambiguous_prefix.csv
outputs/v5/figures/final_accuracy_by_count_mode.png
outputs/v5/figures/ambiguous_prefix_probs_by_step.png
```

Unit tests:

```text
1. tokenizer round-trip;
2. base generator count correctness;
3. thinking render correctness;
4. non-thinking render correctness;
5. thinking label mask correctness;
6. non-thinking label mask masks </Think> prediction by default;
7. --ablate-no-conflict-mask unmasks </Think> prediction;
8. thinking query starts from <Think/> and can generate trace;
9. non-thinking query supplies <Think/> </Think> and evaluates count next;
10. eval parser handles missing </Think>, premature </Think>, invalid count, and duplicate markers.
```

---

## 14. Main run

After debug passes, run:

```bash
python -m synthetic_niah_v5.run_v5 --preset main --stage all \
  --seq-len 256 \
  --count-min 1 \
  --count-max 10 \
  --train-steps 10000 \
  --batch-size 128 \
  --thinking-fraction 0.5 \
  --eval-examples-per-count 1000
```

Then optionally run the ambiguity ablation:

```bash
python -m synthetic_niah_v5.run_v5 --preset main --stage all \
  --run-name no_conflict_mask_ablation \
  --ablate-no-conflict-mask
```

Expected result for the default run:

```text
- thinking-on prompt should generate a non-empty marker trace;
- non-thinking prompt should answer after the provided empty think block;
- both final count accuracies should be reported separately;
- ambiguous-prefix probability should show low P(</Think>) immediately after <Think/> in the default run.
```

If default run collapses to closing immediately after `<Think/>`, inspect the non-thinking label mask first.

