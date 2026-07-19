# v16_2: three-character Tiny Shakespeare counting

`synthetic_counting_v16_2` is an isolated successor to v16. It does not change v16
data, rendering, checkpoints, or reports. The only reused implementation is the
task-independent causal Transformer core.

## Data preparation

Run `prepare` before training. Preparation fingerprints Tiny Shakespeare and creates
contiguous train/validation/test regions separated by `seq_len - 1` character guards.
It computes character frequencies from the training region only, enumerates distinct
three-character sets whose summed frequency is at most
`needle_pool_frequency_threshold`, and samples `needle_pool_size` unique sets across
approximately uniform frequency bins. The split, pool, and all fixed evaluation
manifests are persisted under the run's `data/` directory.

Because an extremely rare training character can be absent from a shorter held-out
region, evaluation samples sets uniformly from that region's viable pool subset (sets
with at least one native occurrence). Training still samples the complete pool uniformly.
`tables/regional_pool_viability.csv` records every set/region viability decision; no
held-out frequencies are used to construct the pool.

The stored `count_max_threshold` is the only mutable count maximum. `cfg.count_max` is
a read-only compatibility property and serialized alias; loading rejects disagreeing
values.

## Training examples

For each example, sample one untouched `seq_len`-character training window. With
probability `1 - task_occurrence_ratio`, use the raw window directly. Otherwise select
one set uniformly from the pool and accept the initial window—or resample only the
window—when its union count satisfies `1 <= n <= count_max_threshold`.

Raw examples contain exactly the corpus characters and provide `seq_len - 1` shifted
next-character targets. Task examples use:

```text
nonthinking: <BOS> <CountChar> A B C <Sep> prompt <Ans> <n> <EOS>
thinking:    <BOS> <CountChar> A B C <Sep> prompt
             <Think> <1> M_1 ... <n> M_n </Think> <Ans> <n> <EOS>
```

The prompt remains an untouched contiguous corpus slice. `M_k` is the actual kth
left-to-right prompt character belonging to the set. The set members are distinct;
their prefix order is shuffled by default while the canonical set ID remains stable.

## Weighted training objective

Training retains all-sequence next-token supervision but permits two task-relevant
weights:

- `final_count_loss_weight` applies to the final numeric answer token in both modes;
- `cot_trace_loss_weight` applies to thinking trace indices and marker characters only.

Control tokens (`<Think>`, `</Think>`, `<Ans>`, and `<EOS>`), task-prefix tokens, prompt
characters, and raw-language targets retain weight 1. The objective is

```text
sum(weight_i * CE_i) / sum(weight_i)
```

over non-padding shifted targets. Both defaults are 1.0, reproducing the prior v16_2
objective. Fixed-suite cross-entropies, perplexities, and component losses remain
unweighted for comparability; training tables separately identify the weighted objective
and the weighted shares assigned to final-count and trace targets.

## Optimizer regularization

Training uses AdamW with configurable `weight_decay`, defaulting to `0.01`. The generated
notebook exposes this as `WEIGHT_DECAY`; setting it to `0.0` disables decay. The current
optimizer passes all model parameters in one group, so the coefficient applies to every
trainable parameter, including embeddings, biases, and LayerNorm parameters. Excluding
biases or normalization parameters would change the optimizer definition and is outside
this revision.

Weight decay does not add dropout or early stopping. The existing `0.01` setting did not
prevent held-out language loss from deteriorating after approximately 1,000 steps in the
reported v16_2 runs. Checkpoint selection must therefore continue considering held-out
loss together with autoregressive task accuracy. A different default decay should be
chosen only after a controlled sweep.

## Evaluation and loss curves

Three fixed suites—raw, task, and ratio-matched mixture—are built from both the training
region and held-out validation region. They are evaluated at step 0, every `eval_every`,
and the final step for every available RoPE/RPE and nonthinking/thinking model.

`eval_examples_per_count` controls balanced fixed-suite size. Its main default is 100,
so counts 1–10 produce 1,000 examples per raw/task/mixture suite. A periodic checkpoint
evaluates three train suites, three held-out suites, and the held-out behavioral task
suite—approximately 7,000 teacher-forced sequences per enabled model at the default.
This setting also determines final-test suite size; it does not change the separately
configured autoregressive subset.

The primary curve is the equal-input-sequence mean:

```text
CE_i = total active next-token NLL in sequence i / active targets in sequence i
curve CE = mean_i CE_i
```

Token-weighted cross-entropy is also saved as a secondary training-objective diagnostic.
The primary figure contains raw, task, and mixture panels with train and held-out lines.
The corpus test region is evaluated only at the final checkpoint and saved separately;
repeatedly viewed “test curves” use the validation region and are labeled held-out/test.

## Commands

```bash
python -m synthetic_counting_v16_2.run_v16_2 --preset debug --stage all --device cpu

python -m synthetic_counting_v16_2.run_v16_2 \
  --preset main --stage all --device cuda \
  --task-occurrence-ratio 1.0 \
  --count-max-threshold 10 \
  --weight-decay 0.01 \
  --final-count-loss-weight 1.0 \
  --cot-trace-loss-weight 1.0 \
  --model-variant rope/thinking \
  --model-variant rpe/thinking \
  --train-steps 10000 \
  --eval-examples-per-count 100 \
  --skip-completed
```

The generated Colab notebook exposes both loss weights, `WEIGHT_DECAY`, four independent
model switches, the maximum steps per enabled model, and evaluation examples per count.
At least one model switch must be enabled. Legacy v16_2 configs without the newer fields
load with unit task weights, weight decay `0.01`, and their former position-encoding x
mode Cartesian product.

Stages are `prepare -> train -> attention -> state -> plots`. A train or analysis stage
refuses to start when the split, pool, or fixed-suite manifests are missing or when any
fingerprint differs from the configuration/checkpoint.
