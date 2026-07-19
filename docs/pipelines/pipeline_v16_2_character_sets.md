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

## Evaluation and loss curves

Three fixed suites—raw, task, and ratio-matched mixture—are built from both the training
region and held-out validation region. They are evaluated at step 0, every `eval_every`,
and the final step for every available RoPE/RPE and nonthinking/thinking model.

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
  --skip-completed
```

Stages are `prepare -> train -> attention -> state -> plots`. A train or analysis stage
refuses to start when the split, pool, or fixed-suite manifests are missing or when any
fingerprint differs from the configuration/checkpoint.
