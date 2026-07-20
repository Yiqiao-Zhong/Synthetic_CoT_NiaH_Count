# `synthetic_counting_v16`: workflow summary and extension plan

## Scope

This document summarizes the implemented workflow behind
`notebooks/Trace_Count_v16_Colab.ipynb` and identifies the most important issues to
address before drawing stronger mechanistic conclusions or extending the experiment.

The notebook itself is an unexecuted Colab driver: all code cells have empty outputs.
Consequently, this is a workflow and methodology analysis, not a summary of observed
v16 results.

## Where the behavior lives

`synthetic_counting_v16` is only a versioned entry point:

- `src/synthetic_counting_v16/run_v16.py` calls the shared v11 CLI with `version="v16"`.
- Configuration, data construction, model code, training, evaluation, attention
  summaries, state probes, and plots all live under `src/synthetic_counting_v11/`.
- `scripts/build_v15_v17_notebooks.py` generates the v16 notebook. Notebook edits that
  should persist must also be made in this builder, or regeneration will overwrite them.
- `docs/pipelines/pipeline_v15_v17_completion.md` is the main protocol description.
- `tests/test_synthetic_counting_v15_v17.py` checks the principal v16 invariants.

This sharing is convenient, but it also means that changes in the v11 implementation can
silently affect v11-v17. V16-specific follow-up analyses are safer in new modules under
`src/synthetic_counting_v16/`, using the shared checkpoint and data loaders where useful.

## Notebook workflow

The notebook performs the following operations in order:

1. Mount Google Drive and create a persistent results root.
2. Locate an existing repository or clone it into `/content`; on Colab, attempt a
   non-failing `git pull --ff-only`.
3. Probe the scientific Python stack. If its ABI is broken, reinstall pinned NumPy,
   pandas, SciPy, Matplotlib, and seaborn versions, then force a runtime restart.
4. Install this repository editable with `--no-deps`, import PyTorch, and report GPU
   information.
5. Run `tests/test_synthetic_counting_v15_v17.py` by default.
6. Construct and assert the planned v16 configuration.
7. Launch `python -m synthetic_counting_v16.run_v16` as a streaming subprocess with
   `stage=all`, live checkpoint sync to Drive, and completed-model skipping.
8. Display selected configuration tables, learning tables, and generated figures.
9. Copy the complete local run directory to a timestamped Drive directory.
10. Optionally commit and push broad repository changes; this is disabled by default.
11. Disconnect the Colab runtime only after the Drive copy is verified.

The default notebook run directory is
`runs/synthetic_counting_v16/v16_main_all_sequence_seed1234`. A pre-existing
`config.json` must exactly match the requested configuration, which protects this short
run name from mixing incompatible experiments.

## Corpus info

The bundled Tiny Shakespeare corpus contains 1,115,394 characters, 40,000 newline
characters, and 65 unique character types. The tables below count overlapping,
case-sensitive character n-grams over the complete corpus. Frequencies use
`count / (corpus_length - n + 1)`. In the displayed n-grams, `␠` denotes a space and
`\n` denotes a newline.

### Character 1-gram frequencies

There are 1,115,394 unigram observations. All 65 character types are shown.

| Character | Count | Frequency |
| --- | ---: | ---: |
| `␠` | 169,892 | 15.231568% |
| `e` | 94,611 | 8.482294% |
| `t` | 67,009 | 6.007653% |
| `o` | 65,798 | 5.899081% |
| `a` | 55,507 | 4.976448% |
| `h` | 51,310 | 4.600168% |
| `s` | 49,696 | 4.455466% |
| `r` | 48,889 | 4.383115% |
| `n` | 48,529 | 4.350839% |
| `i` | 45,537 | 4.082593% |
| `\n` | 40,000 | 3.586177% |
| `l` | 33,339 | 2.988989% |
| `d` | 31,358 | 2.811383% |
| `u` | 26,584 | 2.383373% |
| `m` | 22,243 | 1.994183% |
| `y` | 20,448 | 1.833254% |
| `,` | 19,846 | 1.779282% |
| `w` | 17,585 | 1.576573% |
| `f` | 15,770 | 1.413850% |
| `c` | 15,623 | 1.400671% |
| `g` | 13,356 | 1.197424% |
| `I` | 11,832 | 1.060791% |
| `b` | 11,321 | 1.014978% |
| `p` | 10,808 | 0.968985% |
| `:` | 10,316 | 0.924875% |
| `.` | 7,885 | 0.706925% |
| `A` | 7,819 | 0.701008% |
| `v` | 7,793 | 0.698677% |
| `k` | 7,088 | 0.635471% |
| `T` | 7,015 | 0.628926% |
| `'` | 6,187 | 0.554692% |
| `E` | 6,041 | 0.541602% |
| `O` | 5,481 | 0.491396% |
| `N` | 5,079 | 0.455355% |
| `R` | 4,869 | 0.436527% |
| `S` | 4,523 | 0.405507% |
| `L` | 3,876 | 0.347501% |
| `C` | 3,820 | 0.342480% |
| `;` | 3,628 | 0.325266% |
| `W` | 3,530 | 0.316480% |
| `U` | 3,313 | 0.297025% |
| `H` | 3,068 | 0.275060% |
| `M` | 2,840 | 0.254619% |
| `B` | 2,761 | 0.247536% |
| `?` | 2,462 | 0.220729% |
| `G` | 2,399 | 0.215081% |
| `!` | 2,172 | 0.194729% |
| `D` | 2,089 | 0.187288% |
| `-` | 1,897 | 0.170074% |
| `F` | 1,797 | 0.161109% |
| `Y` | 1,718 | 0.154026% |
| `P` | 1,641 | 0.147123% |
| `K` | 1,584 | 0.142013% |
| `V` | 798 | 0.071544% |
| `j` | 628 | 0.056303% |
| `q` | 609 | 0.054600% |
| `x` | 529 | 0.047427% |
| `z` | 356 | 0.031917% |
| `J` | 320 | 0.028689% |
| `Q` | 231 | 0.020710% |
| `Z` | 198 | 0.017752% |
| `X` | 112 | 0.010041% |
| `3` | 27 | 0.002421% |
| `&` | 3 | 0.000269% |
| `$` | 1 | 0.000090% |

### Top 10 character 2-grams

There are 1,115,393 bigram observations and 1,403 distinct bigram types.

| Bigram | Count | Frequency |
| --- | ---: | ---: |
| `e␠` | 27,643 | 2.478319% |
| `␠t` | 23,837 | 2.137094% |
| `th` | 22,739 | 2.038654% |
| `he` | 18,203 | 1.631981% |
| `t␠` | 16,508 | 1.480016% |
| `s␠` | 15,364 | 1.377452% |
| `d␠` | 14,165 | 1.269956% |
| `,␠` | 14,098 | 1.263949% |
| `␠a` | 13,541 | 1.214012% |
| `ou` | 12,730 | 1.141302% |

### Top 10 character 3-grams

There are 1,115,392 trigram observations and 11,556 distinct trigram types.

| Trigram | Count | Frequency |
| --- | ---: | ---: |
| `␠th` | 16,032 | 1.437342% |
| `the` | 10,495 | 0.940925% |
| `he␠` | 7,762 | 0.695899% |
| `nd␠` | 6,653 | 0.596472% |
| `.\n\n` | 5,018 | 0.449887% |
| `and` | 4,931 | 0.442087% |
| `is␠` | 4,913 | 0.440473% |
| `␠to` | 4,887 | 0.438142% |
| `you` | 4,649 | 0.416804% |
| `␠yo` | 4,625 | 0.414652% |

## Scientific workflow

### 1. Task and data generation

The corpus is the standard 1,115,394-character Tiny Shakespeare text. Its 65 distinct
characters are represented as tokens such as `<CH_0061>`.

For every training example:

1. Sample the requested count `n` uniformly from 1 through 30.
2. For each allowed target character, consult a precomputed index of all length-256
   corpus windows containing exactly `n` occurrences of that character.
3. Choose uniformly among target characters that are viable for this count.
4. Choose uniformly among that target's candidate window starts.
5. Return the untouched, contiguous corpus window and the positions of all native target
   occurrences. No marker is inserted or overwritten.

The case-sensitive target set is `S H A K E R s h a k e r`. Each prompt explicitly names
the target:

```text
<BOS> <CountChar> target-character <Sep> 256-character Shakespeare window ...
```

Training examples are generated online with replacement. Validation and analysis sets
use different random seeds and are balanced by requested count, but they are drawn from
the same corpus-window index; there is no disjoint text split or overlap exclusion.

The vocabulary contains 113 tokens:

- 8 special/task tokens;
- 65 Shakespeare character tokens;
- 10 `<M*>` marker tokens inherited from the shared implementation but unused by v16;
- 30 shared numeric tokens `<1>` through `<30>`.

The unused marker tokens still participate in the output softmax and can be generated.

### 2. Output formats and loss

Non-thinking format:

```text
<BOS> <CountChar> C <Sep> prompt <Ans> <n> <EOS>
```

This always contains 263 tokens, hence 262 shifted prediction targets.

Thinking format:

```text
<BOS> <CountChar> C <Sep> prompt
<Think> <1> C <2> C ... <n> C </Think> <Ans> <n> <EOS>
```

It contains `265 + 2n` tokens: 267 tokens for count 1 and 325 for count 30. The trace and
final answer deliberately share number tokens. Every trace marker is the same named
native character token `C`; occurrence order is represented by trace position and numeric
progress, not by marker identity.

V16 uses all-sequence teacher-forced causal language modeling. After shifting, every
non-padding token after `<BOS>` contributes to cross-entropy: task prefix, all 256 prompt
characters, trace, delimiters, final answer, and `<EOS>`. Padding uses label `-100`.

Important consequences:

- The final count is only 1 of 262 loss targets in non-thinking and 1 of 266-324 in
  thinking (about 0.31-0.38% of token weight).
- Most optimization pressure is ordinary character-level language modeling, not the
  final counting decision.
- Thinking examples with larger counts contain more supervised tokens, so uniform
  example-level count sampling is not uniform token-level weighting. A count-30 example
  contributes about 22% more targets than a count-1 example.
- Training is teacher forced. It does not expose the model to its own trace mistakes;
  free-running evaluation is needed to measure error propagation.

### 3. Model variants

Four models are trained:

| Position encoding | Output mode |
| --- | --- |
| RoPE | non-thinking |
| RoPE | thinking |
| learned relative-position bias (RPE) | non-thinking |
| RPE | thinking |

Each model is a causal pre-LayerNorm Transformer with 4 layers, 4 heads, hidden width
256, head width 64, MLP width 1024, GELU, final LayerNorm, and tied token
embedding/unembedding. There is no dropout.

- RoPE rotates queries and keys with base 10,000 and has approximately 3,188,480
  parameters with the current vocabulary.
- RPE adds a learned per-layer, per-head scalar bias indexed by causal distance and has
  approximately 3,192,592 parameters.
- RPE distances are clipped at 256 although a thinking sequence can be 325 tokens long;
  all larger distances share the final bias bucket. RoPE is not clipped this way. This is
  a substantive difference when comparing long-range behavior near the end of a trace.

The notebook says that every combination has an independent random initialization, but
the implementation is more controlled: `_paired_model` recreates the same seed and
copies all shape-compatible weights from a canonical RoPE model. Thus all four runs
begin with identical shared weights; RPE alone has its additional zero-initialized bias.
The models and optimizers are independent after initialization. The Python sampling RNG
is also reset for every variant, so corresponding optimization steps see the same prompt
stream (with mode-specific rendering). This paired design should be documented accurately.

### 4. Optimization, evaluation, and resumption

The main preset uses:

- 10,000 optimizer steps and batch size 128;
- AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, weight decay `0.01`;
- 500-step linear warmup followed by cosine decay to zero;
- global gradient clipping at 1.0 and float32 training;
- logging every 50 steps;
- teacher-forced evaluation every 500 steps;
- autoregressive evaluation and checkpointing every 1,000 steps.

A fixed balanced teacher-forced set contains 100 examples per count. Autoregressive
evaluation generates a new balanced set of 10 examples per count at each applicable
step. The same examples are used across model variants, enabling paired comparisons.

Checkpoint payloads contain model and optimizer states, configuration, vocabulary
fingerprint, Python RNG state, CPU Torch RNG state, and CUDA RNG states. A rerun can
restore Drive checkpoints, resume from the greatest numeric step, or skip a variant with
a final checkpoint. Table writes and checkpoints are mostly atomic, and rows are
deduplicated by identifying columns.

CUDA deterministic algorithms are not enabled, so the fixed seed and restored RNG state
do not by themselves guarantee bitwise reproducibility on every GPU/software stack.

### 5. Recorded behavioral metrics

Training and evaluation record total and component losses plus count-bin summaries for
1-10, 11-20, and 21-30.

Teacher-forced metrics include:

- final-count accuracy at the `<Ans>` position;
- thinking trace-index accuracy;
- thinking trace-marker accuracy.

`tf_final_accuracy` is computed by restricting argmax to the 30 number tokens. It does
not require the gold count to beat every other vocabulary token. It is therefore a
conditional count classifier metric, not full-vocabulary next-token accuracy.

Autoregressive evaluation greedily generates from `<Ans>` for non-thinking or from
`<Think>` for thinking. It reports final-count accuracy and absolute error, and for
thinking also exact-trace accuracy and trace-marker recall. This metric uses global
vocabulary generation and is the stronger end-to-end test.

`time_to_99.csv` records the first evaluation step at which teacher-forced final,
trace-marker, or trace-index accuracy reaches 0.99 in each count bin.

### 6. Descriptive attention analysis

Final checkpoints are evaluated on 20 balanced examples per count. For every layer and
head, the code inspects:

- the final-answer query in both modes;
- every trace-index query in thinking mode.

Attention mass is partitioned among BOS, task prefix, target occurrences, other prompt
characters, trace indices, trace markers, delimiters, answer token, and remaining
context. The main derived quantities are:

- broad target attention: total mass on target occurrences multiplied by normalized
  entropy across them;
- raw kth-to-kth occurrence mass for each trace step;
- whether the kth occurrence is top-attended among target occurrences;
- kth mass divided by total target-occurrence mass;
- final-query mass on trace markers.

These are routing descriptions, not causal effects. In v16 all occurrence tokens have
identical identity, so a kth-to-kth signature necessarily relies on positional structure
and trace progress rather than distinct marker content.

### 7. Hidden-state analysis

The state stage gathers embeddings and post-layer residual states from independently
sampled balanced train/evaluation sets. Sites are:

- final-answer position in both modes, labeled by final count;
- trace-index and trace-marker positions in thinking mode, labeled by trace progress `k`.

At every residual depth (embedding plus four layers), it fits or evaluates:

- standardized nearest-centroid classification;
- ridge regression with R-squared and mean absolute error;
- a token-position-only nearest-mean baseline;
- PCA of per-label state centroids.

The position baseline is essential: the thinking answer position is a deterministic
function of count, and trace-site position strongly correlates with progress. Above-baseline
decodability is evidence that state contains label information, but it still does not show
that the model uses that information causally.

### 8. Output bundle

The run directory contains:

- immutable `config.json` and fingerprinted `vocab.json`;
- `manifest.json` with per-stage running/failed/complete status;
- resumable and final checkpoints for each variant;
- detailed and aggregated CSV tables for training, evaluation, autoregressive generation,
  attention, and state probes;
- learning, count-distribution, attention-heatmap, retrieval, probe, and PCA figures.

Stages run in the fixed order `train -> attention -> state -> plots` for `stage=all`.
Selecting a later stage directly assumes its prerequisite checkpoints or tables already
exist.

## Methodological issues to resolve

### Target-count confounding

The requested count is uniform, but target identity is not independent of count. The
number of viable targets shrinks as count rises:

| Counts | Viable target characters |
| --- | --- |
| 1-8 | all 12 targets |
| 9-11 | 11 targets |
| 12 | 9 targets |
| 13-14 | 8 targets |
| 15-16 | 7 targets |
| 17-18 | 6 targets |
| 19-29 | only `s h a e r` |
| 30 | only `h a e` |

The sampler first chooses a viable target uniformly, not a corpus window uniformly.
Target/count cells with few candidate windows are consequently oversampled and repeated.
A model can exploit target identity and target-conditioned language statistics as a count
prior, especially at high counts, without implementing a target-occurrence counter.

### Corpus overlap and memorization

Training, validation, autoregressive evaluation, attention analysis, and state analysis
all sample overlapping length-256 windows from the same text. Different RNG seeds do not
create an out-of-distribution or even necessarily disjoint split. Adjacent starts share up
to 255 characters, so exact-window deduplication alone is insufficient. Reported held-out
metrics can mix counting generalization with corpus and local-context memorization.

### Objective dilution and task leakage

All-sequence loss heavily rewards modeling Shakespeare. Exact-count conditioning changes
the language distribution of the selected windows, and the target prefix itself carries
a count prior because viability changes with count. Total loss can improve while the
counting computation does not. Total losses across modes also average over sequences with
different supervised lengths.

### Metric interpretation

- Restricted-number teacher-forced accuracy can exceed actual next-token generation
  accuracy.
- Trace-marker accuracy is not evidence of kth-occurrence retrieval: every correct marker
  is the same target token, already shown in the task prefix and repeated in the trace.
- Attention mass is correlational and can be redistributed without changing behavior.
- Linear/centroid decodability does not imply causal use and is vulnerable to absolute
  token-position leakage.
- A single seed provides no estimate of run-to-run uncertainty.

## Recommended extension plan

### Phase A: audit and strengthen analysis without retraining

1. Add provenance fields to `Example` and downstream tables: corpus start/end, target
   character, number of candidate windows for the target/count cell, and a stable prompt
   hash.
2. Emit a `target_count_support.csv` table with candidate-window counts and realized
   sampling frequencies. Plot target-by-count support and compare it with the realized
   train/evaluation distribution.
3. Add simple baselines:
   - target-only count classifier;
   - target plus character-frequency/context statistics;
   - sequence length / answer-position only;
   - bag-of-characters linear count regressor or classifier.
4. Stratify every behavioral metric by target x count, and report target-macro as well as
   example-micro averages. High-count aggregate accuracy should not be reported without
   showing which targets are represented.
5. Add full-vocabulary final-token rank, accuracy, probability, margin, and calibration in
   addition to the current number-restricted accuracy.
6. Normalize retrieval metrics against count-dependent chance (`1/n`) and include matched
   controls: non-target prompt positions, other occurrences of the same character, shuffled
   occurrence order, and distance-matched prompt positions.
7. Add causal tests on existing checkpoints: head ablation, activation patching across
   examples matched on target and count, trace-state patching, and counterfactual target
   prefix swaps. Report changes in full-vocabulary count probability and autoregressive
   answer, not just attention redistribution.
8. Run the new analysis on the existing final checkpoints before changing training. This
   separates conclusions about the current experiment from conclusions about a redesigned
   dataset.

Suggested v16-specific modules are `data_audit.py`, `behavioral.py`, `attention_controls.py`,
and `causal.py`, with raw CSV output kept separate from plotting/report code.

### Phase B: redesign data splits and retrain

1. Split Tiny Shakespeare into contiguous train/validation/test regions before indexing
   windows. Leave at least a 255-character guard band so windows cannot overlap across
   splits.
2. Save fixed dataset manifests with corpus coordinates and hashes. Enforce disjointness
   in tests and report unique-window and overlap statistics.
3. Remove target-count support confounding. Candidate designs include:
   - use a fully crossed target x count range with a minimum unique-window threshold;
   - use a smaller target set that supports the complete range;
   - sample equal quotas for every supported target x count cell and explicitly mark
     unsupported cells;
   - create synthetic held-out text while retaining native-character target identity.
4. Decide whether markers should remain in the v16 vocabulary. Removing unused `<M*>`
   tokens makes the output space match the task; retaining them should be an explicit
   cross-version control.
5. Compare objectives under matched data and initialization:
   - current all-sequence loss;
   - completion-only loss;
   - weighted multitask loss with separately reported prompt-LM, trace, and final-count
     terms;
   - final-count upweighting while preserving prompt modeling.
6. Make example-level versus token-level weighting explicit, especially for thinking mode.
7. Run at least several independent seeds. Preserve paired seeds across RoPE/RPE and modes,
   then report paired differences with confidence intervals.
8. Either increase RPE's distance range to cover the maximum rendered sequence or add a
   controlled ablation showing the effect of clipping at 256.

### Phase C: test the counting mechanism

The strongest follow-up should distinguish three hypotheses:

1. **Target/count prior:** the model predicts from target identity and corpus statistics.
2. **Parallel aggregation:** the final state pools evidence from many occurrences directly.
3. **Sequential trace computation:** thinking retrieves occurrences in order, advances a
   progress state, and reads the final count from that state.

Useful discriminating experiments are:

- counterfactual target-prefix swaps while holding the window fixed;
- insertion/deletion of one native target occurrence with minimally changed context;
- matched examples with the same target/count but different occurrence positions;
- matched examples with the same target and surface statistics but different count;
- trace truncation, trace corruption, and forced wrong trace indices;
- causal patching of final-count and trace-progress states between carefully matched donor
  and receiver examples;
- head ablations chosen on one split and confirmed on a disjoint test split;
- generalization to held-out targets, corpus regions, prompt lengths, and counts.

Evidence for a mechanism should require a controlled behavioral effect under intervention,
not only high probe accuracy or an interpretable heatmap.

## Implementation order

1. Add provenance and support audits plus tests.
2. Add target-stratified and full-vocabulary behavioral metrics.
3. Reanalyze existing checkpoints.
4. Add matched causal interventions and negative controls.
5. Introduce guarded corpus splits and target/count balancing.
6. Retrain objective and positional-encoding ablations across multiple paired seeds.
7. Update the notebook builder, protocol documentation, and notebook displays only after
   the underlying tables and tests are stable.

# `synthetic_counting_v16_2`: revision plan

### Goal and compatibility boundary

Create `v16_2` as a new experiment version rather than changing v16 in place. V16 must
continue to reproduce its existing single-target data, sequence format, checkpoints, and
tables. V16_2 will preserve the same Tiny Shakespeare corpus, untouched contiguous
windows, model architecture, RoPE/RPE comparison, non-thinking/thinking modes, and
all-sequence causal objective, but will replace the single target with a three-character
needle set drawn from a precomputed pool. Training will additionally mix raw Shakespeare
language-model examples and counting-formatted examples through a configurable
`task_occurrence_ratio`.

**Isolation requirement:** do not add v16_2 branches to legacy v16 data generation,
rendering, training, evaluation, analysis, plotting, CLI, or notebook-builder functions.
Do not rename, replace, or overwrite existing implementations. Every function whose
behavior differs must live in a new v16_2 file and use a v16_2-specific name. Changes to
existing repository files should be limited to additive registration or documentation,
such as listing the new package in `pyproject.toml` and adding a new README section.

V16_2 may import genuinely unchanged, low-level components such as the Transformer model,
but its config, vocabulary/data classes, pool construction, sampler, renderer, training
driver, analyses, plots, pipeline, CLI, and notebook builder should be separate. This
intentional isolation is preferable to minimizing duplicated code because it protects
legacy results and makes the new semantics easier to test and document.

Use `synthetic_counting_v16_2` as the Python package and `v16_2` as the serialized version
name. Its run name must contain `v16_2`, the pool size, set size, pool-frequency threshold,
count threshold, task occurrence ratio, objective, and seed so that it cannot collide with
v16 checkpoints or another ratio sweep.

### 1. New configuration

Add the following fields to the new `V16_2Config`, its serialization, the v16_2 CLI, and
the v16_2 main/debug presets. Use the correctly spelled `threshold` in code even though
the initial proposal used `pool_treshhold`.

| Config field | Default | Meaning |
| --- | ---: | --- |
| `needle_set_size` | 3 | Number of distinct accepted characters in each set |
| `needle_pool_size` | 100 | Number of unique character sets in the persisted pool (`N_set`) |
| `needle_pool_frequency_threshold` | 0.04 | Maximum sum of the three marginal corpus frequencies |
| `needle_pool_frequency_bins` | 20 | Equal-width bins used to make pool frequency sums approximately uniform |
| `needle_pool_seed` | experiment seed plus a fixed offset | Reproducible pool sampling seed |
| `count_max_threshold` | 10 | Single source of truth for the accepted-count and output-number upper bound |
| `task_occurrence_ratio` | 1.0 | Probability that a sampled training prompt is converted to counting-task format |
| `corpus_train_fraction` | 0.80 | Leading contiguous fraction reserved for training candidates |
| `corpus_validation_fraction` | 0.10 | Following contiguous fraction used for fixed training-curve evaluation suites |
| `candidate_filter_max_attempts` | 100,000 | Fail clearly rather than looping forever during rejection sampling |
| `shuffle_needle_set_order` | `true` | Randomize prefix order so a set is not given a canonical first member |

In v16_2, `count_max` and `count_max_threshold` would be duplicates if both were mutable
config fields. Avoid two independent values: store only `count_max_threshold` and expose a
read-only `count_max` compatibility property that returns it. Existing-style utilities
can read `cfg.count_max`, while candidate filtering, vocabulary construction, trace bounds,
balanced evaluation, and plots all derive from the same stored threshold. The v16_2 CLI
should expose only `--count-max-threshold`.

For transparency, v16_2 `config.json` may serialize both
`"count_max_threshold": 10` and derived `"count_max": 10`, but it must label `count_max`
as a compatibility alias and reject a loaded artifact in which they differ. Explain this
alias explicitly in `src/synthetic_counting_v16_2/config.py`, CLI help, the v16_2 pipeline
document, and `README.md`. Use the correctly spelled `count_max_threshold`; do not create
the misspelled `count_max_trreshold` as another alias.

Define `V16_2Config` and the `target_character_set` task identifier in the new
`src/synthetic_counting_v16_2/config.py`; do not add v16_2 semantics to the shared v11
`ExperimentConfig`. Validate that v16_2 uses Shakespeare characters, a set size of exactly
three, a positive pool size, a frequency threshold in `(0, 1]`, and
`1 <= count_max_threshold <= seq_len`. Validate
`0.0 <= task_occurrence_ratio <= 1.0`, serialize it, include it in the run name, and expose
it directly as `--task-occurrence-ratio` in the v16_2 CLI. The notebook should define an
easy-to-edit `TASK_OCCURRENCE_RATIO = 1.0` setting and pass it to both the planned config
and the subprocess command. Validate that the train and validation fractions are positive
and sum to less than one; the remaining fraction is the test region. Expose split
fractions in config/CLI but keep their notebook defaults prominent and stable.

### 2. Build and persist the needle pool before training

Add an explicit `prepare` stage before `train`:

```text
prepare -> train -> attention -> state -> plots
```

The notebook should run and display the `prepare` result before launching training.
`stage=all` may include preparation automatically, but the train stage must refuse to run
if the pool is missing, malformed, or inconsistent with the current config/corpus. It
must never silently rebuild a different pool inside a resumed run.

Pool construction should be deterministic for a corpus fingerprint, config, and pool
seed. The `prepare` stage must materialize the corpus split described below first, and
pool frequencies must be computed from the training region only so validation/test
statistics do not influence pool selection:

**Hard pool invariant:** every pool entry is an unordered set of exactly three
pairwise-distinct corpus characters. Repeated members such as `{a, a, b}` are invalid,
and duplicate triples are invalid even if their displayed order differs.

1. Count every case-sensitive character in the training corpus region and compute
   `p_train(c) = count_train(c) / train_region_length`.
2. Enumerate all unordered triples of distinct corpus characters. With 65 character
   types, this is only `C(65, 3) = 43,680` candidates.
3. For each triple `S`, compute `p_train(S) = sum(p_train(c) for c in S)` and retain it only when
   `p(S) <= needle_pool_frequency_threshold`.
4. Divide `[0, threshold]` into 20 equal-width bins and allocate the 100-set quota as
   evenly as possible across bins (five per bin under the defaults).
5. Sample unique triples without replacement within each bin using `needle_pool_seed`.
   If a bin lacks enough candidates, carry its deficit to the nearest bins with unused
   candidates and record that redistribution.
6. Canonicalize each stored set by character code point, assign a stable `set_id`, and
   verify that all sets are unique and contain exactly three distinct characters.

This produces a roughly uniform distribution of frequency sums over the feasible portion
of 0-4%. Exact zero is impossible because every corpus character has positive frequency;
the pool report should show the achieved range and histogram rather than claiming exact
continuous uniformity.

Save both:

- `data/needle_pool.json`: complete reproducibility artifact containing config values,
  full-corpus and split fingerprints, pool seed, vocabulary fingerprint, training-region
  character counts/frequencies, sets, and a pool fingerprint;
- `tables/needle_pool.csv`: one row per set with `set_id`, readable characters, character
  tokens, individual frequencies, summed frequency, and frequency bin.

Generate `figures/needle_pool_frequency_distribution.png` and display it in the notebook
before training. The all-character policy includes letters, punctuation, whitespace, and
newlines when they satisfy the threshold; if a letters-only experiment is later desired,
it should be a separate explicit config policy rather than an undocumented filter.

### 3. Corpus split and evaluation isolation

The current v16 code does **not** reserve untouched text for evaluation: training,
teacher-forced evaluation, autoregressive evaluation, attention analysis, and state
analysis all sample from the same full Tiny Shakespeare corpus with different RNG seeds.
V16_2 should fix this without modifying v16.

Create one deterministic contiguous split during `prepare`. Let
`guard = cfg.seq_len - 1` and first remove space for two guard bands. Divide the remaining
usable characters according to the configured 80% train, 10% validation, and derived 10%
test fractions, then lay them out as:

```text
[ contiguous train region ]
[ guard of cfg.seq_len - 1 characters ]
[ contiguous validation region ]
[ guard of cfg.seq_len - 1 characters ]
[ contiguous test region ]
```

A candidate window belongs to a split only when its complete `cfg.seq_len` corpus slice
lies inside that split's region. The guard bands ensure that no training window shares a
character with a validation or test window. This is a region-level holdout, not merely a
different sampling seed.

Save `data/corpus_split.json` with the full-corpus SHA-256, exact half-open coordinates,
region hashes/lengths, guard coordinates, fractions, and `seq_len`. Refuse to reuse a
pool, evaluation manifest, or checkpoint if the split fingerprint changes.

Use the regions as follows:

- **train:** the only source of streaming raw and task training windows, and the only
  source used to calculate frequencies for needle-pool construction;
- **validation:** three fixed held-out evaluation suites used repeatedly for the
  train-versus-held-out loss curves and periodic behavioral evaluation during training;
- **test:** independently sampled fixed suites used only for final-checkpoint reporting.

Persist train-curve, validation, and test example manifests, including corpus coordinates
and hashes, before training. Raw and fully task-formatted manifests should be independent of
`task_occurrence_ratio` and shared across model variants and ratio sweeps that use the
same split/pool seed. Ratio-specific mixture manifests should be deterministic compositions
of those fixed component suites. Do not regenerate any evaluation suite at each training
step.

### 4. Candidate-window sampling and filtering

Replace v16's exact-count window index with the requested proposal-and-filter sampler.
This sampler is called only after an example has been routed to the counting-task branch.
The router supplies an initial uniformly sampled window and a needle set selected uniformly
from the persisted pool. Hold that set fixed while resampling invalid windows; otherwise
set-dependent acceptance rates would distort uniform set sampling.

```text
choose one needle set uniformly from the persisted pool
candidate_window = the router's initial window
repeat up to candidate_filter_max_attempts:
    find every window position whose character belongs to the set
    n = number of matched positions
    accept if 1 <= n <= count_max_threshold
    otherwise choose a new corpus start uniformly and replace candidate_window
```

Use `cfg.seq_len` for the corpus-window length everywhere; 256 is only its default. Sample
starts only from the active split region and require `start + cfg.seq_len <= region_end`.
Training filtering uses the train region, periodic evaluation uses validation, and final
evaluation uses test.

On acceptance:

- keep the original window unchanged;
- sort matched positions from left to right;
- store the actual matched character token at every position;
- store `set_id`, the canonical set, rendered set order, corpus start/end, count, and
  prompt hash in `Example`;
- define `count = len(needle_positions)` and assert it equals the number of characters in
  the window that belong to the set.

The three-character set, not the individual character, is sampled uniformly from the
pool. The resulting training count distribution is not guaranteed to be uniform. Record
both proposed and accepted counts, rejection reasons, attempts per accepted example, and
the realized distribution by count, set-frequency bin, and `set_id`.

Keep evaluation balanced by count without restoring the old exact-window index. Repeatedly
draw candidates from the relevant held-out region using the same random-window/random-set proposal and place accepted
examples into count buckets 1-10 until every bucket reaches its requested quota. Use
separate deterministic seeds for teacher-forced evaluation, autoregressive evaluation,
attention, and state analysis. Raise a diagnostic error listing unfilled buckets if the
attempt limit is reached.

An extremely rare set constructed from training-only statistics can have no member in a
shorter validation or test region. Such a set can never pass the held-out filter, no
matter how many windows are resampled. Therefore training samples uniformly from the full
pool, while each held-out sampler first identifies sets with at least one native
occurrence in that region and samples uniformly from that viable subset. Save every
set/region viability decision to `tables/regional_pool_viability.csv`. This availability
check must not feed back into pool construction or use held-out frequencies.

### 5. Task-occurrence mixture and raw-language examples

`task_occurrence_ratio` is an example-level Bernoulli mixture applied independently to
every training example. Construct an example in this order:

```text
1. Sample a contiguous segment of length cfg.seq_len uniformly from the training region.
2. Draw is_task ~ Bernoulli(task_occurrence_ratio).
3. If is_task is false:
       return the segment directly as a raw-language example.
4. If is_task is true:
       choose a needle set uniformly from the persisted pool;
       test the initially sampled segment;
       if its union count is invalid, resample only the segment until
       1 <= n <= count_max_threshold;
       render the accepted example in the appropriate counting format.
```

For a raw-language example, the token sequence must be exactly the corpus segment:

```text
character_1 character_2 ... character_seq_len
```

Do not prepend `<BOS>`, append `<EOS>`, select a needle set, run the count filter, or add
any task tokens. Under the current shifted-label implementation, a raw segment of length
`seq_len` supplies `seq_len - 1` next-character targets. Both the non-thinking and thinking
models receive the same raw-example format.

Boundary behavior must be exact:

- `task_occurrence_ratio = 1.0`: every training example is counting-formatted, recovering
  the previously specified v16_2 workflow;
- `task_occurrence_ratio = 0.0`: every training input is an untouched Shakespeare segment,
  recovering the requested NanoGPT-like raw-Shakespeare input format;
- values strictly between zero and one produce a stochastic mixture, reproducible from
  the training RNG state and checkpoint.

Keep one common vocabulary and model architecture across ratio sweeps so checkpoints and
capacity remain comparable. Consequently, ratio zero recovers raw NanoGPT-style *input
sequences and next-character training*, but not an exact stock NanoGPT vocabulary: unused
task, number, and delimiter tokens remain in the output softmax. If an exact character-only
NanoGPT baseline is needed, implement it as a separately named baseline rather than
silently changing the ratio-zero model shape.

The configured ratio is a probability over examples, not a direct fraction of loss
tokens. Counting-formatted examples are longer than raw examples, and all-sequence loss
also predicts the `cfg.seq_len` Shakespeare characters inside task prompts. Therefore
record:

- configured and realized task-example ratios globally and per logged batch;
- raw versus task active-token counts and the realized task-token ratio;
- separate raw-language, task-prefix, task-prompt, trace, final-count, and EOS losses;
- rejection attempts only for task-routed examples.

This instrumentation is necessary for the scientific goal: `task_occurrence_ratio`
controls how often the counting format occurs, but it does not by itself make language
modeling loss and counting-specific loss contribute equal weight.

Training uses the mixture, but primary counting evaluation should remain fully
counting-formatted and balanced over counts 1 through `count_max_threshold`, independent
of the training ratio. Add a separate raw-language validation set of untouched held-out
windows and report next-character cross-entropy/perplexity. Attention and count-state
analyses should use task-formatted examples; raw examples have no task anchors.

The pool may still be prepared when the ratio is zero so the same run can perform
zero-shot counting evaluation and remain comparable with other ratio settings. Pool
construction must not affect the raw training-example stream.

### 6. Revised sequence formats

Represent the set as three ordinary Shakespeare character tokens between `<CountChar>`
and `<Sep>`. No new per-set vocabulary tokens are needed.

For a rendered set `(A, B, C)`, non-thinking becomes:

```text
<BOS> <CountChar> A B C <Sep> prompt <Ans> <n> <EOS>
```

Thinking becomes:

```text
<BOS> <CountChar> A B C <Sep> prompt
<Think> <1> M_1 <2> M_2 ... <n> M_n </Think> <Ans> <n> <EOS>
```

Here `M_k` is the actual character at the kth left-to-right prompt position that belongs
to `{A, B, C}`. Each `M_k` can be any of the three accepted characters, and characters may
repeat. The trace must not cycle through the three set members. For example, if the
matched prompt characters in order are `B, B, A, C`, the gold trace is:

```text
<Think> <1> B <2> B <3> A <4> C </Think> <Ans> <4> <EOS>
```

The final answer is the union count across all three characters. A position matching one
set member contributes exactly one because the set contains distinct characters.

When `shuffle_needle_set_order=true`, shuffle `(A, B, C)` for each rendered example while
retaining the canonical `set_id`. This makes prefix order semantically irrelevant and
enables a permutation-invariance evaluation. The same rendered order must be reproducible
from the example seed and saved in detailed tables.

The task prefix now has five tokens after `<BOS>`:

```text
<CountChar> A B C <Sep>
```

Therefore `prompt_start` changes from 4 in v16 to 6 in v16_2. With a 256-character prompt:

- non-thinking length is 265 tokens;
- thinking length is `267 + 2n`, at most 287 tokens for `n=10`;
- the existing `n_positions=384` remains sufficient.

Compute these lengths from `needle_set_size` rather than hard-coding them. Update
`max_render_len`, task-prefix spans, component losses, attention categories, and every
test that assumes the old three-token task prefix.

### 7. Data structures and code organization

Add a self-contained v16_2 package:

```text
src/synthetic_counting_v16_2/
    __init__.py
    config.py
    cli.py
    run_v16_2.py
    needle_pool.py
    data.py
    model.py
    training.py
    analysis.py
    plots.py
    pipeline.py
```

`needle_pool.py` should own the pool dataclasses, construction, validation, fingerprinting,
serialization, and pool-specific plots/tables. `data.py` should define independent
`V16_2Vocab`, `V16_2Example`, `V16_2Spans`, and `V16_2Rendered` classes plus functions such
as `make_v16_2_example`, `balanced_v16_2_examples`, `render_v16_2`, and
`collate_v16_2`. Do not add fields or dispatch branches to v11's `Example`, `Vocab`,
`make_example`, `balanced_examples`, or `render`.

`V16_2Example` should have an explicit `example_kind` equal to `raw_lm` or
`counting_task`. Raw examples contain corpus coordinates and tokens but no set, count,
needle positions, trace, or task spans. Task examples contain the complete set/count
metadata described above. The v16_2 collator and loss-component code must support both
kinds in the same batch while excluding raw examples from task-only metrics.

Likewise, use v16_2-specific entry points such as `train_v16_2_models`,
`run_v16_2_attention_analysis`, `run_v16_2_state_analysis`, and `run_v16_2_pipeline`.
These may be forked from the current implementation and then revised, leaving the source
functions untouched. The new `model.py` may wrap or re-export the unchanged causal
Transformer implementation, provided the wrapper translates `V16_2Config` explicitly and
the legacy model module requires no edits.

Do not use mutable module-global pool state. `run_v16_2_pipeline` should prepare/load the
pool once and pass the validated object through v16_2 training and analysis functions.
Checkpoints should record the pool fingerprint, and checkpoint loading/resumption should
fail when it differs from the run's pool file.

Create separate workflow surfaces rather than extending old ones:

- `scripts/build_v16_2_notebook.py`, not a v16_2 branch in
  `scripts/build_v15_v17_notebooks.py`;
- `notebooks/Trace_Count_v16_2_Colab.ipynb`;
- `docs/pipelines/pipeline_v16_2_character_sets.md`;
- `tests/test_synthetic_counting_v16_2.py`.

Only additive edits should be made outside the new files:

- add `src/synthetic_counting_v16_2` to the package list in `pyproject.toml`;
- add v16_2 commands plus the `count_max` compatibility-alias and
  `task_occurrence_ratio` semantics to `README.md`;
- optionally add links to the new notebook/document from existing documentation indexes.

The legacy v16 runner must continue importing only its current v11 implementation and
must never import v16_2. Add a regression test that constructs and renders a seeded v16
example and confirms its original single-character prefix and sequence lengths remain
unchanged.

### 8. Metrics and analyses to retain or add

The separate v16_2 final-count, autoregressive, attention, and state implementations
should preserve the existing metric definitions where they remain applicable, because
those definitions operate on `needle_positions` and `needle_markers`. Do not call or
modify the legacy analysis functions directly. Under v16_2, `needle_markers[k]` is simply
the actual member of the three-character set found at the kth matched position.

#### Three fixed loss-evaluation suites on train and held-out sequences

At model initialization (step 0), every `eval_every` steps, and at the final step, compute
teacher-forced next-token cross-entropy on the same three suite definitions for **both** a
fixed train-region sample and a fixed held-out validation-region sample:

1. **Raw-language suite:** only untouched corpus windows of length `cfg.seq_len`, with no
   BOS/EOS or task tokens.
2. **Task-formatted suite:** only valid counting-formatted examples, balanced over counts
   `1..count_max_threshold`; use the non-thinking or thinking format matching the model
   being evaluated.
3. **Ratio-matched mixture suite:** a deterministic mixture of raw and task examples with
   exactly `round(task_occurrence_ratio * N_eval)` task examples and the remaining examples
   raw, where `N_eval = eval_examples_per_count * count_max_threshold` by default.

Call the two curve sources `train` and `heldout` in tables and plots. `heldout` means the
validation corpus region and is the user's requested test-sequence curve for detecting a
train/held-out generalization gap. Keep the separate corpus `test` region final-only so
that repeatedly viewing curves does not turn the nominal test set into validation data.
If user-facing text uses “test loss curve,” label it `held-out/test curve (validation
region)` and never imply that it is the untouched final test result.

Persist six component manifests before training: raw, task, and mixture for each of the
train and held-out sources. They must be fixed across training steps, position encodings,
modes, and compatible ratio sweeps. Train-curve manifests must be sampled independently
of the stochastic optimization minibatches but from the same train corpus region;
held-out manifests must use only the validation region. Build each mixture
deterministically from the corresponding raw/task manifests instead of making fresh
Bernoulli draws at each evaluation. For task examples, non-thinking and thinking
renderings must use the same underlying window, set, positions, and count. Use the same
number of examples and the same count-balancing policy in corresponding train and
held-out suites so their gaps are directly comparable.

The loss shown in the three train-versus-held-out figures must be averaged equally over
input sequences. For sequence `i`, let `T_i` be the number of active next-token targets
after shifting (padding and ignored targets do not count), and define:

```text
sequence_cross_entropy_i = sum_t NLL(i, t) / T_i
example_mean_cross_entropy = sum_i sequence_cross_entropy_i / N_examples
```

Accumulate each sequence's NLL and active-target count across all minibatches, then take
the suite mean. Do not average batch means, and do not let the last partial batch or
variable thinking-trace length change an example's weight. This is the primary plotted
metric requested for loss curves. Also save the training-objective-compatible secondary
metric
`token_weighted_cross_entropy = sum_i sum_t NLL(i, t) / sum_i T_i`. The online optimizer
minibatch loss in `train_metrics.csv` remains useful for debugging but is not a comparable
train curve and must not be substituted for the fixed train-suite result.

Save both cross-entropies, `example_mean_perplexity = exp(example_mean_cross_entropy)`,
`token_weighted_perplexity = exp(token_weighted_cross_entropy)`, number of examples,
number of active tokens, requested/realized task-example ratio, and realized task-token
ratio.

Write one long-form `tables/eval_loss_curves.csv` with at least:

```text
step, position_encoding, mode, curve_source, source_region, suite,
task_occurrence_ratio, realized_task_example_ratio, realized_task_token_ratio,
num_examples, active_tokens, token_weighted_cross_entropy,
example_mean_cross_entropy, token_weighted_perplexity, example_mean_perplexity
```

Use the uniqueness key `(step, position_encoding, mode, curve_source, suite,
task_occurrence_ratio)` when appending rows so train and held-out results cannot overwrite
one another. Write `tables/eval_loss_components.csv` for the decomposed raw-language, task-prefix,
task-prompt, think-open, trace-index, trace-marker, final-count, and EOS losses. Raw rows
have only the raw-language component; task rows retain all applicable components; mixture
rows report both component families.

Plot `figures/learning_loss_suites_train_vs_heldout.png` from
`example_mean_cross_entropy`, with separate raw, task, and mixture panels. In every panel,
draw the fixed train-suite and held-out-suite curves at identical checkpoint steps. Facet
by position encoding when more than one is present, and use a consistent mode encoding
(`nonthinking` solid and `thinking` dashed) plus a consistent curve-source color. The
plotting function must work when a run contains only non-thinking, only thinking, or both
modes; it should discover available groups from the table instead of assuming both exist.
Do not smooth the primary curves. Keep each suite's y-axis independently scaled, while
using the same y-scale for train and held-out lines within that panel. Optionally create a
second token-weighted diagnostic figure, clearly labeled as such.

These curves answer different questions: raw loss tracks language modeling, task loss
tracks the complete formatted task objective, the mixture loss tracks the configured
deployment/training mixture, and the separation between the train and held-out lines
tracks potential overfitting. Because raw sequences do not depend on output format, their
underlying manifests are identical across non-thinking and thinking models; their plotted
losses can still differ because the trained model parameters differ.

Construct analogous fixed suites from the untouched test region, but evaluate them only
after selecting the final checkpoint. Save these results to
`tables/test_loss_summary.csv`; do not use test losses for training monitoring, checkpoint
selection, early stopping, or hyperparameter choices.

Add the following columns to detailed tables:

- `needle_set_id` and the three canonical/rendered characters;
- each character's corpus frequency and their sum;
- frequency bin, corpus window start/end, and prompt hash;
- per-character counts within the window and their union count;
- number of rejected candidates before acceptance.

Add summaries by count, pool-frequency bin, and set. Also report:

- prefix permutation consistency: prediction agreement across all six orderings of the
  same set;
- per-character trace-marker accuracy as well as aggregate marker accuracy;
- a set-frequency-only baseline for predicting count;
- accepted-count distribution versus the intended pool-frequency distribution;
- rejection rate and sampling cost.

The set's summed corpus frequency is deliberately informative about expected count
(`approximately cfg.seq_len * summed_frequency` before filtering). Therefore aggregate
accuracy alone cannot demonstrate counting. Always compare against the set-frequency-only
baseline and evaluate windows with the same set but different realized counts.

### 9. Tests and acceptance criteria

Add a dedicated v16_2 test module covering:

1. **Pool validity:** exactly 100 unique sets; exactly three distinct in-corpus characters
   per set; summed frequency at most 0.04 within floating-point tolerance.
2. **Pool distribution:** deterministic 20-bin occupancy is approximately balanced, with
   any quota redistribution explicitly represented in metadata.
3. **Pool reproducibility:** identical corpus/config/seed gives an identical fingerprint;
   changing any of them invalidates reuse.
4. **Candidate filtering:** every accepted example has length `cfg.seq_len` and count in
   `1..count_max_threshold`; rejected zero and over-threshold candidates are recorded.
5. **Untouched windows:** `seq_tokens` exactly equals the corpus slice at the saved
   coordinates.
6. **Occurrence truth:** positions equal all and only locations whose characters belong
   to the selected set; marker tokens match those positions in left-to-right order.
7. **Rendering:** prefix contains exactly three target tokens, `prompt_start == 6`, trace
   indices/characters are correct, and final answer equals the union count.
8. **Set-order invariance protocol:** all permutations preserve positions, trace, count,
   and `set_id` even though prefix order changes.
9. **Balanced evaluation:** the filter/bucket sampler returns the configured number of
   examples for every count in `1..count_max_threshold` or fails with an informative
   attempt-limit error.
10. **Training smoke test:** all four RoPE/RPE x mode variants complete the debug
    `prepare,train,attention,state,plots` pipeline.
11. **Resume safety:** a missing or mismatched pool fingerprint prevents checkpoint load.
12. **Notebook test:** the generated v16_2 notebook compiles, prepares the pool before
    training, displays its histogram, and uses a v16_2-specific run name.
13. **Legacy isolation:** the existing v16 tests still pass; seeded v16 rendering remains
    byte-for-byte unchanged; and no legacy v16 module imports `synthetic_counting_v16_2`.
14. **Count alias:** `cfg.count_max == cfg.count_max_threshold`, the CLI has one mutable
    threshold option, serialized aliases agree, and mismatched saved values are rejected.
15. **Ratio validation and exposure:** values below zero or above one fail; CLI and
    notebook overrides round-trip into `config.json`; run names differ across ratios.
16. **Ratio-zero boundary:** every training example is a raw corpus slice of exactly
    `cfg.seq_len`, contains no BOS/EOS/task tokens, selects no set, and invokes no filter;
    non-thinking and thinking raw formats are identical.
17. **Ratio-one boundary:** every training example is task-formatted and satisfies
    `1 <= n <= count_max_threshold`.
18. **Intermediate mixture:** a sufficiently large seeded sample has a realized task
    fraction within a statistical tolerance of the configured probability; rerunning from
    the same RNG state reproduces the same routing decisions and windows.
19. **Mixed-batch training:** raw and variable-length task examples collate together,
    shifted labels are correct, task-only metrics ignore raw rows, and separate component
    losses/token counts are finite.
20. **Evaluation separation:** counting evaluation is fully formatted and count-balanced
    at every training ratio, while raw validation contains only untouched windows and
    reports next-character loss/perplexity.
21. **Corpus isolation:** train, validation, test, and guard coordinates are disjoint;
    every saved window lies wholly inside its declared region; no window from different
    regions shares a corpus character.
22. **Fixed manifests:** repeated evaluation steps and model variants use identical raw
    and task example IDs/coordinates within each curve source; train and held-out
    manifests come from their declared disjoint regions, while final-test manifests are
    distinct from both.
23. **Loss-suite composition:** raw and task suites are pure; the mixture has exactly the
    deterministic rounded example ratio for 0, intermediate, and 1 settings; task examples
    are count-balanced.
24. **Loss aggregation:** hand-computed variable-length toy sequences match saved
    token-weighted and equal-example-mean cross-entropies, active-token counts, component
    losses, and both perplexities; changing evaluation batch size or the size of the final
    partial batch does not change suite-level results.
25. **Train-versus-held-out coverage:** every evaluation checkpoint has exactly one train
    and one held-out row for each available `(position_encoding, mode, suite, ratio)`;
    neither source can overwrite the other in the long-form table.
26. **Plot coverage:** plotting succeeds for non-thinking-only, thinking-only, and joint
    runs; each available model variant has three panels with both train and held-out lines,
    and the plotted y-values equal `example_mean_cross_entropy` from the CSV.
27. **Test discipline:** periodic held-out curves use the validation region, not the
    untouched final-test region; final-test loss files are produced only from final
    checkpoints.

### 10. Implementation order

1. Create the separate v16_2 package, config, validation, entry point, and run naming
   without editing legacy implementations.
2. Implement deterministic guarded corpus splitting and persist/fingerprint the split.
3. Implement training-region-only pool construction, persistence, diagnostics, and the
   remainder of the `prepare` stage.
4. Implement independent v16_2 data classes and the random-window/random-set filter
   sampler.
5. Implement the Bernoulli task router, exact raw-language rendering, ratio diagnostics,
   and mixed-example collation.
6. Implement v16_2 task rendering and span calculations for a three-token set prefix.
7. Build and persist fixed train/validation/test raw, task, and deterministic mixture
   manifests, with train and validation manifests paired for curve comparisons.
8. Implement the separate v16_2 training, three train-versus-held-out loss curves using
   equal-example averaging, balanced behavioral evaluation, final-only test evaluation,
   and analysis pipeline; pass the validated pool explicitly.
9. Add proposal/acceptance, mixture, component-loss, loss-suite, and set-stratified tables
   and plots.
10. Add v16_2 tests plus legacy-isolation regression tests and run ratio-zero, intermediate,
   and ratio-one debug pipelines.
11. Generate the v16_2 Colab notebook with its separate builder and add new documentation
   without changing the old notebook generator.
12. Document the `count_max` compatibility alias, `task_occurrence_ratio`, corpus split,
    and three loss suites in
    code, CLI help, the pipeline document, and README.
13. Only after the debug artifact passes all checks, launch main ratio sweeps and save the
    immutable pool alongside every result bundle.

# revision plan 2: minor optimization of workflow

## Scope and compatibility requirements

Make these changes only in `synthetic_counting_v16_2` and its v16_2 CLI, notebook
builder, generated notebook, tests, and documentation. Do not change the v16 objective,
notebook, checkpoint schema, or behavior. The revision has two purposes:

1. allow the final-count and thinking-trace targets to contribute more strongly than the
   long raw/task-prompt portions of an all-sequence training example; and
2. let a Colab user choose any subset of the four RoPE/RPE x non-thinking/thinking models
   and set the training-step limit without editing package code; and
3. expose the size of the fixed periodic evaluation suites so evaluation cost can be
   reduced explicitly without weakening count balance by accident.

Use backward-compatible package defaults. A v16_2 config that does not contain the new
loss-weight or model-selection fields must load as the current behavior: both position
encodings, both modes, 10,000 training steps, and unit weight on every active token.
Existing checkpoints must remain readable. New settings must be part of `config.json`
and checkpoint config metadata so a run cannot silently resume under a different
objective or model selection.

## 1. Add two task-relevant loss weights

Add the following stored fields to `V16_2Config`:

```python
final_count_loss_weight: float = 1.0
cot_trace_loss_weight: float = 1.0
```

`final_count_loss_weight` applies to the single gold number token at
`spans.count_pos` in both non-thinking and thinking task examples.
`cot_trace_loss_weight` applies only to the thinking trace body: all positions in
`spans.trace_index_positions` and `spans.trace_marker_positions`. It does not implicitly
weight `<Think>`, `</Think>`, `<Ans>`, `<EOS>`, the task prefix, or the Shakespeare
prompt. Raw-language examples have weight 1 everywhere. The two target sets are
disjoint, so weights do not multiply or stack.

Require both values to be finite and strictly positive. Unit weights must reproduce the
old unweighted objective, within deterministic floating-point tolerance. Keep `1.0` as
the package/config default for backward compatibility; expose comments in the notebook
showing that values greater than one upweight those targets. Do not silently choose a
large experimental default without a controlled sweep.

Implement a target-aligned weight tensor with the same unshifted shape as `labels`.
Initialize active targets to 1, overwrite final-count and trace positions from the saved
spans, leave padding at zero/ignored, and shift the tensor exactly as labels are shifted.
Extend the token-loss helper to calculate

```text
weighted_loss = sum_i(weight_i * cross_entropy_i) / sum_i(weight_i)
```

over active targets. Normalizing by the sum of weights, rather than by the number of
tokens, keeps the overall optimizer scale stable when weights change. Continue returning
the ordinary per-token cross-entropies and active mask so component diagnostics remain
interpretable.

Use the weighted loss for `backward()` during training. Preserve the existing unweighted
`token_weighted_cross_entropy`, `example_mean_cross_entropy`, perplexities, and component
losses in fixed train/validation/test suites so old and new runs remain directly
comparable. Add explicit training diagnostics such as `train_weighted_objective_loss`,
`batch_active_weight_sum`, and realized weighted shares for final-count and trace targets;
do not relabel the weighted objective as ordinary cross-entropy. If a weighted validation
objective is saved, give it a separate column and never overwrite the existing loss
columns.

Propagate and persist both fields through all relevant surfaces:

- `src/synthetic_counting_v16_2/config.py`: dataclass fields, validation, `to_dict`,
  legacy-aware `config_from_dict`, and run-name tags;
- `src/synthetic_counting_v16_2/cli.py`: `--final-count-loss-weight` and
  `--cot-trace-loss-weight`, including help text and override forwarding;
- `src/synthetic_counting_v16_2/data.py` and/or `training.py`: target-aligned weight
  construction, shifted weighted reduction, and training diagnostics;
- `scripts/build_v16_2_notebook.py`: the source of truth for notebook settings and CLI
  construction;
- regenerated `notebooks/Trace_Count_v16_2_Colab.ipynb`;
- `README.md` and `docs/pipelines/pipeline_v16_2_character_sets.md`: mathematical
  definition, defaults, affected token groups, and compatibility note.

Include both weights in the default run name, for example `fcw1_cotw1`, so two objectives
cannot reuse the same directory when `RUN_NAME=None`. The existing config equality check
must continue rejecting a manually reused directory whose saved weights differ.

In the notebook's **Easy-to-edit settings** cell, add:

```python
FINAL_COUNT_LOSS_WEIGHT = 1.0  # >1 upweights the final numeric answer target
COT_TRACE_LOSS_WEIGHT = 1.0    # >1 upweights thinking trace indices and markers
```

Pass both values to `preset_config(...)` and the subprocess CLI. Print the effective
values before preparation begins.

## 2. Add independent switches for the four model variants

Expose four clear booleans in the same notebook settings cell:

```python
RUN_ROPE_NONTHINKING = True
RUN_ROPE_THINKING = True
RUN_RPE_NONTHINKING = True
RUN_RPE_THINKING = True
```

Derive one canonical ordered selection from these booleans:

```text
rope/nonthinking, rope/thinking, rpe/nonthinking, rpe/thinking
```

Add a stored config field such as `enabled_model_variants`, with the four canonical
strings above as its default. Keep `cfg.model_variants` as the tuple-of-tuples interface
used by training and analysis, but derive it from the stored selection. Validate that the
selection is non-empty, contains no duplicates, and contains only supported
position-encoding/mode pairs. For legacy v16_2 configs without this field, derive the old
Cartesian product from `position_encodings` and both modes.

Add a repeatable CLI option such as
`--model-variant rope/nonthinking --model-variant rpe/thinking`. The notebook builder
should append one argument per enabled boolean. Persist the effective ordered selection
in `config.json`, checkpoint config metadata, and the run name. Continue retaining or
deriving `position_encodings` as needed for legacy loading, but do not let it contradict
the explicit enabled variants.

Audit every loop and plot that currently assumes all four models. Training, final test,
permutation evaluation, attention collection, state probing, summary tables, and plots
must iterate only over `cfg.model_variants`. A one-model run must not attempt to load
three missing checkpoints. Plot legends and line styles must be discovered from the
available rows, as they already are for several v16_2 plots.

## 3. Expose the existing training-step control in the notebook

The package already has `train_steps=10_000` and the CLI already supports
`--train-steps`; no new underlying optimization parameter is needed. Add this explicit
notebook setting:

```python
MAX_TRAIN_STEPS = 10_000
```

Pass it as `train_steps=MAX_TRAIN_STEPS` to `preset_config(...)` and as
`--train-steps MAX_TRAIN_STEPS` to the subprocess. Include a `steps10000` tag in the
default run name so short/debug and full runs cannot collide. Retain the current positive
integer validation. The existing loop already evaluates and saves a final checkpoint at
`train_steps` even when the value is not a multiple of `eval_every` or
`checkpoint_every`; add a regression test for that boundary.

Before launching the pipeline, print a compact planned-run summary containing the
enabled variants, number of models, maximum steps per model, total planned optimizer
steps, task ratio, and both loss weights. Fail in the settings cell if all four model
switches are false rather than starting an empty run.

## 4. Expose periodic evaluation-suite size in the notebook

The current main config already stores `eval_examples_per_count=100`. With
`count_max_threshold=10`, this produces 1,000 examples in each fixed raw, task, and
mixture suite. At every 500-step evaluation, each model evaluates all three train suites,
all three held-out suites, and the 1,000-example held-out teacher-forced behavioral set.
Therefore the current periodic workload is approximately 7,000 teacher-forced sequences
per model per evaluation checkpoint, not 1,000 sequences total. At every 1,000 steps,
the separate autoregressive subset adds `ar_examples_per_count * count_max_threshold`
generated examples.

Expose the existing balanced-suite control in the **Easy-to-edit settings** cell:

```python
EVAL_EXAMPLES_PER_COUNT = 100  # 100 x counts 1..10 = 1,000 examples per fixed suite
```

Prefer the per-count setting over a bare total because task evaluation must remain
exactly balanced across every accepted count. In the planned-run summary, calculate and
print:

```python
EVAL_EXAMPLES_PER_SUITE = EVAL_EXAMPLES_PER_COUNT * COUNT_MAX_THRESHOLD
PERIODIC_TF_EXAMPLES_PER_MODEL = 7 * EVAL_EXAMPLES_PER_SUITE
```

The second value is an explanatory workload estimate for the present six loss suites
plus one behavioral task evaluation; it is not a new config field. Keep the evaluation
cadence at `eval_every=500` in this minor revision. Also state explicitly that this
setting does not change `ar_examples_per_count=10`, which controls generation evaluation
at the separate 1,000-step cadence.

No new core config field is required because `eval_examples_per_count` already exists and
is serialized in `config.json` and checkpoint metadata. Add the missing CLI option
`--eval-examples-per-count`, forward it through `preset_config`, and pass it from
`scripts/build_v16_2_notebook.py` into both `PLANNED_CONFIG` and `base_cmd`. Regenerate
`notebooks/Trace_Count_v16_2_Colab.ipynb` from the builder. Validate it as a positive
integer before data preparation.

This existing config controls the sizes of all fixed train, validation, and final-test
raw/task/mixture manifests, as well as the periodic teacher-forced task set. Document
that scope rather than implying that it affects only one validation table. Include an
`evaln1000` tag, based on the derived per-suite total, in the default run name because
changing this value changes persisted data manifests and evaluation precision. The run
directory config-equality and manifest-fingerprint checks must continue preventing reuse
of fixed suites created at a different size.

Reducing this setting trades statistical precision for speed. For example, with counts
1–10, `EVAL_EXAMPLES_PER_COUNT=20` yields 200 examples per suite and roughly 1,400
teacher-forced sequences per model at each 500-step checkpoint. Do not permit a total
that drops or unevenly samples count classes; every suite must still contain the same
number of task examples for each count.

## 5. Tests and acceptance criteria

Extend `tests/test_synthetic_counting_v16_2.py` with the following checks:

1. A hand-computed toy batch verifies shifted alignment and weighted normalization for
   final-count, trace-index, trace-marker, ordinary, and padded targets.
2. `final_count_loss_weight` affects both modes; `cot_trace_loss_weight` affects only the
   thinking trace body and leaves non-thinking/raw targets unchanged.
3. Unit weights reproduce the previous scalar loss and gradients within deterministic
   tolerance.
4. Invalid, non-finite, zero, and negative weights fail before a run directory is used.
5. Both weights round-trip through `to_dict`/`config_from_dict`, CLI parsing,
   `config.json`, checkpoint metadata, and notebook construction. A legacy config missing
   them loads with unit weights.
6. Each individual model switch and representative mixed subsets produce exactly the
   requested checkpoints/tables; attention, state, test, permutation, summary, and plots
   do not look for disabled variants.
7. An empty model selection, duplicate CLI variants, and unsupported variants fail with
   informative messages. Legacy configs still select all formerly implied variants.
8. `MAX_TRAIN_STEPS` reaches the config and CLI, stops at the exact requested step, and
   produces the final evaluation/checkpoint for a non-cadence-aligned value.
9. `EVAL_EXAMPLES_PER_COUNT` round-trips through config, CLI, notebook, and persisted
   manifests; values zero or below fail before preparation.
10. With a small value, every raw/task/mixture train, validation, and test suite has the
    derived total size; every task suite remains exactly balanced by count; the periodic
    detail table contains that same derived number of examples per model/checkpoint.
11. Changing evaluation size changes the default run name and suite-manifest fingerprint,
    while rerunning the same seeded size reproduces the same fixed examples.
12. Default run names differ when loss weights, enabled variants, training steps, or
    evaluation-suite size differ.
13. The generated notebook contains all eight easy controls (two weights, four model
    switches, maximum steps, and periodic examples per count), passes them to both
    `PLANNED_CONFIG` and `base_cmd`, prints the derived per-suite/workload totals, and
    remains executable from a fresh Colab runtime.
14. Existing v16 tests and untouched v16 behavior continue to pass.

Run the fast unit/config/notebook tests first, then a CPU debug pipeline with one enabled
variant, then a two-variant mixed selection, and finally the existing all-four debug
pipeline with unit weights. Do not launch a new main run until those artifacts confirm
that disabled variants are absent, weighted shares are recorded, resumption remains
safe, and unit-weight behavior matches the current v16_2 baseline.

## 6. Implementation order

1. Add and validate the two loss-weight fields plus legacy config loading and run-name
   tags.
2. Implement target-aligned weight construction and the normalized weighted training
   reduction while preserving unweighted evaluation metrics.
3. Add the canonical enabled-variant config/CLI representation and make every pipeline
   stage subset-safe.
4. Add the CLI plumbing and run-name tag for the existing
   `eval_examples_per_count` setting.
5. Expose `MAX_TRAIN_STEPS`, `EVAL_EXAMPLES_PER_COUNT`, the four switches, and both weights in
   `scripts/build_v16_2_notebook.py`; regenerate the notebook rather than editing only the
   generated `.ipynb`.
6. Add unit, CLI, subset-pipeline, evaluation-size, notebook-generation, and
   legacy-compatibility tests.
7. Update README and the v16_2 pipeline document with the new objective and controls.
8. Run the staged debug acceptance sequence, inspect the resulting config/tables/run
   names, and only then choose non-unit weights for a main experiment.

# revision plan 3: expose v16_2 weight decay in the notebook

## 1. Scope and compatibility contract

Expose the existing v16_2 `weight_decay` optimizer setting through the Colab notebook
without changing its default or its mathematical behavior. The default must remain
`0.01`, matching the current `V16_2Config` and all completed v16_2 runs. This revision is
configuration plumbing only: it must not add dropout, alter AdamW parameter groups,
exclude biases or LayerNorm parameters from decay, change the learning-rate schedule, or
modify checkpoint contents beyond the already serialized config value.

The current optimizer construction passes `model.parameters()` directly to AdamW, so
the configured decay applies to every trainable parameter. Document that fact rather
than implying the notebook setting uses the more elaborate matrix-only decay convention
found in some language-model implementations. Any future parameter-group redesign should
be proposed and tested separately because it would change the meaning of the same
numeric value.

## 2. Add the easy-to-edit notebook control

Add the following setting to the v16_2 notebook's **Easy-to-edit settings** cell, beside
the other optimizer/objective controls:

```python
WEIGHT_DECAY = 0.01  # AdamW decay applied to all trainable parameters; set 0.0 to disable
```

Pass it to the planned config:

```python
PLANNED_CONFIG = preset_config(
    ...,
    weight_decay=WEIGHT_DECAY,
)
```

and to the subprocess command:

```python
"--weight-decay", str(WEIGHT_DECAY),
```

The planned-run summary should display the effective value explicitly so a user can
confirm it before preparation or training begins. Do not change any of the user's other
current easy-to-edit choices when adding this line.

Implement the notebook change first in `scripts/build_v16_2_notebook.py`, which defines
the stable generated-notebook default, and add the same control and wiring to
`notebooks/Trace_Count_v16_2_Colab.ipynb`. Because the checked-in notebook's experiment
values may intentionally differ from the builder defaults, do not regenerate it in a way
that silently resets the user's model switches, task ratio, step count, evaluation size,
or loss weights. Notebook-generation tests should continue permitting user edits inside
the runtime-settings cell.

## 3. Add CLI forwarding and validation

Add a v16_2 CLI argument:

```text
--weight-decay FLOAT
```

with help text stating that it is the AdamW decoupled weight-decay coefficient and that
`0` disables decay. Forward a supplied value through the existing override dictionary to
`preset_config`; when omitted, retain the config default of `0.01`.

The core config field already exists and is already stored by `to_dict`, included in
`config.json`, and embedded in checkpoint config metadata. Add explicit validation that
`weight_decay` is finite and nonnegative. Permit exactly `0.0`; reject negative, NaN, and
infinite values before creating or reusing a run directory. Do not impose an arbitrary
upper limit at the config layer, although documentation should warn that large values can
substantially change optimization.

## 4. Make optimizer changes part of run identity

Add a weight-decay tag to the default v16_2 run name, for example:

```text
wd0p01
```

using the existing stable float-tag formatting. This prevents runs with `0`, `0.01`, or
`0.1` decay from sharing a default directory when `RUN_NAME=None`. The saved-config
equality check must still reject a manually reused custom run name if its stored decay
differs.

Changing the default run-name format affects only the names of newly created v16_2 run
directories. It must not prevent loading an older config or checkpoint whose config
already contains `weight_decay=0.01`, and it must not rename or mutate existing results.
Legacy v16_2 configs that genuinely omit the field should load with the dataclass default
`0.01`, consistent with the historical optimizer setting.

## 5. Documentation

Update `README.md` and `docs/pipelines/pipeline_v16_2_character_sets.md` to state:

- the notebook exposes `WEIGHT_DECAY` and defaults it to `0.01`;
- `0.0` disables AdamW decay;
- the current implementation applies decay to all trainable parameters;
- weight decay regularizes parameter magnitude but does not provide dropout or early
  stopping and did not prevent the observed held-out language-loss deterioration after
  roughly 1,000 steps;
- checkpoint selection using autoregressive task validation and held-out loss remains
  necessary even when decay is enabled.

Keep the documentation descriptive. Do not recommend a new default coefficient until a
controlled sweep compares both language generalization and autoregressive counting.

## 6. Tests and acceptance criteria

Extend `tests/test_synthetic_counting_v16_2.py` with checks that:

1. CLI parsing accepts `--weight-decay 0`, `--weight-decay 0.01`, and another positive
   finite value and forwards the exact float into the config.
2. Negative, NaN, and infinite values fail config validation before pipeline artifacts
   are created.
3. `weight_decay` round-trips through `to_dict` and `config_from_dict`; a legacy config
   missing the field receives `0.01`.
4. Different decay values produce different default run names, while identical values
   reproduce the same name.
5. The builder-generated notebook contains `WEIGHT_DECAY = 0.01`, passes it to both
   `PLANNED_CONFIG` and `base_cmd`, and reports the effective value.
6. The checked-in notebook contains the editable control and command wiring but tests do
   not require the user's current experiment value to equal the builder default.
7. A CPU debug run records the selected decay in `config.json` and in checkpoint config
   metadata, and optimizer construction uses that value.
8. Unit-weight loss behavior, enabled-model selection, resumption, and untouched v16
   tests remain unchanged.

After editing, run the focused v16_2 tests, legacy v16/v11-v14 regression tests, notebook
code-cell compilation, Ruff, and `git diff --check`. Perform one short CPU debug smoke run
with `WEIGHT_DECAY=0.0` and one with `0.01`; verify distinct run directories and otherwise
matching seeded data/model configuration. Update README and pipeline documentation before
handing the changes back for commit.

# revision plan 4: transition from language prediction to task-output-only loss

## 1. Objective and exact boundary semantics

Add a v16_2 training-schedule parameter:

```python
max_steps_for_language_pred: int = 1500
```

Training steps `1` through `max_steps_for_language_pred`, inclusive, retain the current
weighted all-sequence next-token objective. Starting at step
`max_steps_for_language_pred + 1`, exclude the language-modeling prefix and optimize only
the complete task-output region. The starting target position depends on mode and is
inclusive:

```text
nonthinking: <Ans> <numeric-count> <EOS>
thinking:    <Think> <trace indices and markers> </Think> <Ans> <numeric-count> <EOS>
```

Thus the nonthinking task-output mask spans `ans_pos:eos_pos + 1`, while the thinking
task-output mask spans `think_pos:eos_pos + 1`. The start delimiter is itself an active
prediction target: `<Ans>` is predicted from the final prompt token in nonthinking mode,
and `<Think>` is predicted from the final prompt token in thinking mode. The count target
keeps `final_count_loss_weight`; trace indices and trace markers keep
`cot_trace_loss_weight`; all other active task-output targets retain unit weight.

Apply the mask to unshifted target positions before `shifted_v16_2_token_losses` performs
its existing one-token causal shift. In particular, the first output delimiter is
predicted by the logits immediately before it, the numeric count at `count_pos` is
predicted from the logits at `ans_pos`, and EOS at `eos_pos` is predicted from the logits
at `count_pos`. Add alignment-focused unit tests for both modes to prevent off-by-one
errors at either start boundary.

Interpret the requested “sum” as restricting the numerator to these task-output token
losses while retaining the existing normalization by the active weight sum. Do not use
an unnormalized batch sum: the number of active output tokens differs substantially
between thinking and nonthinking examples and across counts, so a raw sum would make
gradient scale depend on trace length and batch composition. The learning-rate schedule,
optimizer state, weight decay, gradient clipping, and global step continue uninterrupted
across the boundary.

Allow `0`, meaning task-output-only training from step 1. Allow values equal to or above
`train_steps`, meaning no switch occurs during that run. Require a nonnegative integer
and reject booleans, negative values, and non-integral values before creating artifacts.

## 2. Raw examples and mode-specific consequences

Raw-language examples have no task-output span and therefore contribute zero objective
weight after the transition. Continue sampling and recording them only if preserving the
configured example stream is required for deterministic compatibility, but do not let an
all-raw post-transition batch silently perform an AdamW-only update. Add a deterministic
guard that resamples such a batch until at least one counting task is present, with a
clear validation error when `task_occurrence_ratio == 0` and the switch would occur before
training ends. This is particularly important for the builder default ratio `0.05`, for
which an all-raw batch is uncommon but not negligible over thousands of steps.

In nonthinking mode the second phase concentrates learning on entering the answer region,
predicting the count, and terminating. In thinking mode it preserves the entire CoT
generation objective, including the decisions to start and stop thinking, while removing
loss from the Shakespeare prompt and task prefix. This is aligned with the observed
failure: the thinking model already maps a generated trace to its answer reliably, but
needs better autoregressive trace generation. `cot_trace_loss_weight` therefore remains
meaningful after the transition and must not be zeroed by the phase mask.

The objective remains teacher-forced, so this change reduces competition from language
prediction but does not by itself eliminate exposure bias. Continue using autoregressive
trace exactness and final-count accuracy to judge whether the thinking model actually
benefits.

## 3. Config, serialization, CLI, and run identity

Add `max_steps_for_language_pred=1500` to `V16_2Config`, include it in validation,
`to_dict`, `config.json`, checkpoint config metadata, and the human-readable derived
training-objective description. Add a derived phase description that reports:

```text
steps 1-1500: weighted all-sequence loss
steps 1501-5000: weighted task-output-only loss
  nonthinking starts at <Ans>; thinking starts at <Think>
```

Do not leave metadata claiming that the entire run uses only `all_sequence`. Either add a
separate explicit `training_loss_schedule` metadata field or broaden the existing
`loss_scope` validation so new scheduled runs and historical all-sequence runs are
distinguishable without changing model/data formats.

Add a CLI option:

```text
--max-steps-for-language-pred INT
```

and forward it through the existing override dictionary to `preset_config`. Include the
effective boundary in default run identity, for example `langsteps1500`, so scheduled and
historical all-sequence runs cannot share a default output directory. Saved-config
equality checks must continue rejecting custom run-directory reuse when the schedule
differs.

Backward compatibility needs special handling because historical v16_2 configs do not
contain this field. New configs and presets default to `1500`, as requested. When
`config_from_dict` loads a genuinely legacy config missing the field, set its effective
boundary to that saved config's `train_steps`, preserving its historical all-sequence
behavior through the end of the run. This prevents loading or resuming an old checkpoint
from silently changing its objective. Do not rename or mutate existing run directories or
checkpoints.

## 4. Training-mask implementation

Extend `collate_v16_2_loss_weights` or add a narrowly named training-objective mask helper
that accepts the absolute optimizer step. Keep target rendering and labels unchanged.
The helper should:

1. Build the existing all-sequence weights at or before the threshold.
2. After the threshold, initialize every position to zero.
3. For a nonthinking counting example, activate every target from `ans_pos` through
   `eos_pos`, inclusive.
4. For a thinking counting example, activate every target from `think_pos` through
   `eos_pos`, inclusive.
5. Within the active span, apply `final_count_loss_weight` at `count_pos` and
   `cot_trace_loss_weight` at each trace index and trace marker; leave delimiters and EOS
   at unit weight.
6. Leave every position in a raw example and every padding position at zero while
   preserving device/dtype behavior.

Pass the current absolute step from `train_v16_2_variant`, including after checkpoint
resume. Derive the phase entirely from `step` and the serialized config rather than
storing mutable phase state, making a resumed step 1,501 identical to an uninterrupted
step 1,501.

Keep fixed-suite evaluation unchanged and unweighted. Continue reporting full-sequence
train/held-out/test cross-entropies so the run reveals whether language loss stabilizes,
worsens, or is forgotten after its gradients stop. Autoregressive validation also remains
unchanged and is the primary criterion for whether the task-output phase improves
counting and trace generation.

## 5. Metrics and observability

Extend `train_metrics.csv` with fields such as:

- `training_loss_phase`: `all_sequence` or `task_output`;
- `batch_objective_active_tokens`: number of targets with nonzero objective weight;
- `batch_task_output_examples`: counting examples contributing output targets;
- `language_prediction_enabled`: boolean boundary indicator.

Keep the existing `batch_active_tokens` and cumulative sampling statistics defined over
the rendered data stream for backward comparability. Compute
`batch_final_count_weight_share` and `batch_cot_trace_weight_share` from the effective
step-specific objective weights. After the transition, trace share must remain zero for
nonthinking but must reflect the configured trace weights for thinking; final-count share
must reflect its fraction of the full mode-specific task-output span. Continue logging
unweighted component losses as diagnostics even for components whose training weight is
zero, but clearly distinguish them from the optimized objective.

Record the scheduled boundary in the planned notebook summary and make the console emit a
single phase-transition message at the first task-output-only step. Avoid printing it
again on every step or after a resume already beyond the boundary.

## 6. Notebook block 3 and builder

Add the following control to code block 3, **Easy-to-edit settings**, in both
`scripts/build_v16_2_notebook.py` and the checked-in v16_2 notebook:

```python
MAX_STEPS_FOR_LANGUAGE_PRED = 1500  # through this step use all-token LM loss; afterward train task output only
```

Pass it to `preset_config` as `max_steps_for_language_pred`, display the two derived step
ranges in `PLANNED_CONFIG` output, and add this CLI wiring to `base_cmd`:

```python
"--max-steps-for-language-pred", str(MAX_STEPS_FOR_LANGUAGE_PRED),
```

Do not reset the user's current task ratio, model switches, weights, training steps,
evaluation size, weight decay, output root, or checkpoint choices while editing the
generated notebook. If `MAX_STEPS_FOR_LANGUAGE_PRED >= MAX_TRAIN_STEPS`, print that the
task-output-only phase will not run rather than presenting an empty or inverted range.

## 7. Documentation

Update `README.md` and `docs/pipelines/pipeline_v16_2_character_sets.md` with:

- the default boundary and exact inclusive/exclusive step semantics;
- the inclusive mode-specific starts: `<Ans>` for nonthinking and `<Think>` for thinking;
- the exact active output spans, including delimiters, trace, count, and EOS as applicable;
- the retained normalized weighted reduction and final-count weight interaction;
- raw examples' zero contribution after the switch;
- continued CoT-trace supervision and `cot_trace_loss_weight` behavior in thinking mode;
- unchanged full-sequence evaluation curves and autoregressive checkpoint selection;
- the special legacy-loading rule that preserves old all-sequence runs.

Include a short example for a 5,000-step run: all-sequence optimization through 1,500,
then task-output-only optimization for steps 1,501-5,000. State that this changes training
behavior from earlier v16_2 runs even though tokenization, architecture, corpus split,
needle pool, and evaluation suites remain unchanged.

## 8. Tests and acceptance criteria

Extend `tests/test_synthetic_counting_v16_2.py` to verify:

1. Config, CLI, serialization, notebook, and builder round-trip the default `1500` and
   arbitrary valid boundaries; invalid values fail early.
2. A legacy config missing the field loads with its saved `train_steps`, reproducing the
   historical full-sequence schedule.
3. Default run names differ when only the language-prediction boundary differs.
4. At step equal to the threshold, raw, prompt, trace, answer-delimiter, count, and EOS
   targets have the existing weights.
5. At the first step after the threshold, a nonthinking example has nonzero weights
   exactly from `ans_pos` through `eos_pos`, a thinking example has nonzero weights
   exactly from `think_pos` through `eos_pos`, and every raw-example weight is zero.
6. Shifted-logit gradients are zero for all excluded prefix targets and nonzero for the
   correctly aligned `<Ans>`/`<Think>` start predictor and subsequent output predictors.
7. The task-output weighted loss uses the existing normalized reduction, honors both
   configured task weights in their applicable modes, and remains finite for mixed
   batches and variable-length thinking traces.
8. An all-raw post-transition batch is resampled or rejected without applying an
   optimizer/weight-decay-only step; a zero task ratio with an active second phase fails
   validation.
9. Resuming immediately before, at, and after the threshold produces the same phase,
   objective, metrics, and parameter update as an uninterrupted run.
10. Periodic train/held-out/test loss suites and autoregressive evaluations retain their
    prior definitions.
11. Training metrics show all-sequence phase rows through the boundary and task-output
    rows afterward, with the expected mode-specific active-token totals and trace shares.
12. Unit-weight pre-threshold behavior and untouched v16/v11-v14 behavior remain exactly
    unchanged.

Run focused unit and notebook-generation tests, legacy regression tests, notebook cell
compilation, Ruff, and `git diff --check`. Then run a deterministic CPU debug experiment
with one nonthinking model, four training steps, and boundary `2`; inspect steps 2 and 3
to confirm the exact mask transition, save/resume at the boundary, and verify that
autoregressive evaluation and full-sequence evaluation still execute. Run a second short
thinking smoke test to confirm `<Think>`, trace, `</Think>`, `<Ans>`, count, and EOS remain
active after the boundary while the task prefix and Shakespeare prompt are excluded.

# revision plan 5: various attention / hidden states metrics

## 1. Goal, scope, and compatibility boundary

Add a v16_2-specific post-training checkpoint-dynamics analysis that explains when the
internal counting mechanisms emerge, how they change around the language-loss boundary,
and why thinking and nonthinking behave differently. Adapt the useful design of
`scripts/analyze_cot_learning_stages.py`—fixed examples, sequential checkpoint loading,
fixed-final-head versus best-current-head tracking, checkpoint attention tables,
hidden-state geometry tables, and dynamics figures—but do not directly reuse its
v11-v14 config, data, model, table, or run-discovery assumptions.

The implementation must remain isolated under v16_2. It must not change v11-v14 or v16
checkpoint formats, analyses, notebooks, or plots. Preserve the current v16_2 final-only
`attention` and `state` stages and their existing tables for backward compatibility. The
new analysis is additive and operates on saved numeric checkpoints after training has
completed. It must support any valid subset of RoPE/RPE and thinking/nonthinking models
recorded in `enabled_model_variants` without looking for disabled variants.

Attention weights and probe decodability are descriptive rather than causal. Every table
and figure must distinguish:

- teacher-forced thinking diagnostics, which see the gold trace before `<Ans>`;
- generated-trace diagnostics, which condition on the model's own greedy trace;
- nonthinking diagnostics, whose `<Ans>` context contains only the prefix and prompt;
- layer 0 input embeddings from post-transformer residual layers 1 through 4.

Do not describe a high attention score or probe accuracy as proof that the model uses the
feature. Provide causal head ablation or activation patching only as a separately labeled
optional extension; the core revision should first produce reproducible descriptive
dynamics and behavior-linked evidence.

## 2. Save checkpoints every 500 steps, including initialization

Change the new v16_2 default checkpoint cadence from 1,000 to 500 optimizer steps while
leaving `eval_every` and `ar_eval_every` unchanged unless explicitly configured. Expose
the cadence through config, CLI, serialization, checkpoint metadata, the planned-run
summary, and the notebook:

```python
CHECKPOINT_EVERY_STEPS = 500  # retain internal-state snapshots every 500 optimizer steps
```

Add or expose `--checkpoint-every 500` and pass it to `preset_config` and the training
subprocess. Require a positive integer. Save `step_000000/checkpoint.pt` immediately
after paired initialization and before the first optimizer update, then save numeric
checkpoints at every configured cadence and at the exact final step even when it is not
cadence-aligned. Continue writing the `final/checkpoint.pt` artifact expected by the
existing final-only stages. The dynamics checkpoint iterator must deduplicate the numeric
final checkpoint and `final/` alias by optimizer step.

For the current 5,000-step design, the intended sequence is:

```text
0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
```

Step 1,500 is the final all-sequence update and therefore the critical boundary snapshot;
step 2,000 is the first saved model after 500 task-output-only updates. Plot a vertical
line at `max_steps_for_language_pred` in every checkpoint-dynamics figure and shade or
label the two objective phases. If the boundary is not cadence-aligned, save an additional
numeric checkpoint exactly at the boundary so the pre-transition state is never missing.

Saving twice as often approximately doubles intermediate-checkpoint storage relative to
the previous 1,000-step cadence. The notebook summary must print the expected checkpoint
steps, number of files per enabled model, and a storage warning before training. Do not
delete or compact checkpoints automatically; the user controls Drive retention.

## 3. New v16_2 dynamics module and checkpoint loader

Add a package module such as
`src/synthetic_counting_v16_2/checkpoint_dynamics.py` and a thin runnable entry point such
as `scripts/analyze_v16_2_checkpoint_dynamics.py`. The package module should own metric
definitions and table schemas; the script should only parse arguments, locate one run,
and invoke the analysis. Do not embed the scientific implementation solely in the
notebook.

### 3.1 Explicit reuse and orchestration strategy

Implement a **v16_2-specific orchestration layer**, not a new copy of the complete
scientific stack and not a direct invocation of the hard-coded v11-v14 script. The three
boundaries must remain explicit:

1. `scripts/analyze_cot_learning_stages.py` remains an unchanged reference implementation
   for v11-v14. Reuse its proven design and port small pure numerical/plotting helpers
   where appropriate, but do not import its v11 config, run discovery, renderer, model,
   or table assumptions into v16_2.
2. Existing v16_2 collectors remain the source of truth for v16_2 sequence semantics.
   Call `collect_v16_2_attention(...)` directly with each loaded checkpoint model. Refactor
   the existing hidden-state collection, nearest-centroid, and ridge helpers into reusable
   v16_2 functions without changing the current final-only `attention` and `state` wrapper
   outputs.
3. The new `checkpoint_dynamics.py` layer owns only cross-checkpoint orchestration and new
   dynamics-specific metrics: checkpoint inventory and loading, fixed-suite reuse,
   variant/step loops, gold/generated/counterfactual contexts, aggregation, caching,
   manifests, behavior joins, and calls into dynamics plotting functions.

The intended call structure is:

```text
notebook analysis cell / thin CLI script
    -> v16_2 checkpoint-dynamics orchestrator
        -> validated arbitrary-checkpoint loader
        -> existing/refactored v16_2 attention and state collectors
        -> new dynamics-only metric reducers
        -> saved checkpoint_* tables
        -> table-only dynamics plotting functions
```

This boundary is also the debugging contract. A failure must be attributable to one of
four named phases printed in the console and recorded in the manifest:

```text
inventory -> load/validate -> collect -> aggregate/plot
```

Every error should include the model variant, checkpoint step, phase, and relevant input
path. The orchestrator must support a single `--model-variant`, a single
`--checkpoint-step`, and individual metric-family switches so a failing checkpoint can
be reproduced without rerunning all models or metrics. Do not hide collector exceptions
inside a broad skip; mark that partition failed and preserve already completed
partitions.

Expected direct reuse is therefore substantial: v16_2 rendering/collation, attention
category extraction, semantic state sites, hidden-state forwards, centroid/ridge probes,
model construction, and artifact fingerprint checks should not be reimplemented. New
code should concentrate on the checkpoint loop and the additional metrics that do not
yet exist—nonthinking coverage, generated/counterfactual states, cross-site transfer,
CKA, head stability, behavior linkage, resumable tables, and dynamics plots.

Add an arbitrary-checkpoint loader alongside the final-checkpoint loader. It must:

1. discover and numerically sort `step_*/checkpoint.pt` files for each enabled variant;
2. validate saved config, vocabulary, pool, and corpus-split fingerprints before a model
   forward;
3. verify that the payload's position encoding, mode, and step match its directory;
4. load one model at a time, use `eval()` and `torch.no_grad()`, then release it and clear
   CUDA cache before loading the next checkpoint;
5. refuse mixed or incomplete checkpoint families with a concise inventory of the
   missing/mismatched steps;
6. permit a requested subset of checkpoint steps for debugging;
7. deduplicate the numeric final checkpoint and `final/` alias;
8. never restore or mutate optimizer/RNG state during analysis.

Use fixed persisted v16_2 task examples rather than resampling at each checkpoint. Reuse
the stored balanced train and held-out task manifests and the same per-count ordering used
by final attention/state analysis. Store a diagnostic-suite fingerprint in every dynamics
manifest. All variants and checkpoints must see byte-identical rendered prompts within a
mode; thinking and nonthinking may differ only by their defined output rendering.

Write results under the existing run directory with unambiguous `checkpoint_` prefixes,
and write atomically. Maintain
`analysis/checkpoint_dynamics_manifest.json` with config/suite fingerprints, requested
metrics, discovered checkpoint inventory, per-variant/per-step status, and timestamps.
An interrupted analysis should resume from completed checkpoint/metric partitions rather
than recomputing the entire run. A `--force` option may invalidate only dynamics outputs;
it must never delete checkpoints, training tables, or final-only analysis artifacts.

## 4. Attention metrics across checkpoints

Use the same query positions and token categories as the current v16_2 analysis, while
adding metrics that are appropriate to each mode and retaining example-level rows for
behavior linkage.

### 4.1 Thinking kth-occurrence retrieval

At every trace-index query `<k>`, where the current position predicts the marker for the
kth occurrence, record for every layer and head:

- raw attention mass on the correct kth prompt occurrence;
- total attention mass on all prompt needle occurrences;
- correct top-1 within the true needle positions;
- diagonal dominance, defined as correct-k mass divided by total needle mass;
- normalized entropy over needle positions;
- attention mass on the task prefix, prompt non-needles, previous trace indices, and
  previous trace markers;
- count-dependent chance top-1 `1 / count` and observed-minus-chance/top-1-over-chance;
- stratification by true count and ordinal `k`.

This analysis should show when a stable kth-to-kth retrieval circuit emerges. For each
metric, plot both (a) the fixed head selected by a predeclared final-checkpoint rule and
(b) the best head at each checkpoint. Select the fixed final head on one designated
selection subset and report it on a disjoint held-out analysis subset so head selection
does not inflate the displayed curve.

### 4.2 Final-answer information routing

At the `<Ans>` query, partition attention into BOS, task prefix, matching prompt
characters, nonmatching prompt characters, trace indices, trace markers, and remaining
control/output tokens. Track these categories by checkpoint, layer, head, and mode.

For thinking, measure the transition from prompt-directed computation to trace readout.
For nonthinking, measure how target-set prefix heads and prompt-scanning heads divide the
work when no trace exists. Include the combined trace-readout mass
`trace_indices_mass + trace_markers_mass` and the ratio of trace-readout mass to direct
prompt-needle mass, with a safe definition when the denominator is near zero.

### 4.3 Nonthinking needle coverage and selectivity

Because nonthinking must aggregate all matches rather than retrieve one ordinal, add:

- needle attention enrichment relative to uniform attention within the 256-token prompt;
- top-n retrieval recall, using `n = true count` highest-attended prompt positions;
- precision among those top-n positions;
- total matching-character attention mass;
- normalized entropy/coverage across all matching positions;
- count-stratified versions of every metric.

Use enrichment and top-n recall rather than applying thinking's kth-occurrence top-1
metric to nonthinking. Plot whether matching-character detection emerges before, after,
or without reliable final counting.

### 4.4 Head specialization and stability

At every checkpoint, rank heads for candidate roles: task-prefix routing, broad needle
detection, kth-occurrence retrieval, previous-trace tracking, and final trace readout.
Record final-head rank over time, best-current-head identity, rank correlation across
adjacent checkpoints, and attention-map similarity on the fixed examples. This must make
head switching visible rather than producing a deceptively smooth best-of-16 curve.

### 4.5 Attention-behavior coupling

On a smaller fixed generated subset, run greedy autoregressive evaluation at every saved
checkpoint and join it to the matching example-level teacher-forced attention rows.
Report correlations or grouped differences between attention metrics and final accuracy,
absolute error, correct trace length, exact trace, and trace-marker recall. Control or
stratify by true count because retrieval is inherently easier at small counts. Label
these associations exploratory and non-causal.

Persist at least:

```text
tables/checkpoint_attention_detail.csv
tables/checkpoint_attention_summary.csv
tables/checkpoint_attention_by_count.csv
tables/checkpoint_attention_by_k.csv
tables/checkpoint_head_stability.csv
tables/checkpoint_attention_behavior_link.csv
```

## 5. Hidden-state metrics across checkpoints

Reuse the current balanced train/held-out split for fitting and evaluating probes. Fit a
new probe independently at each checkpoint and layer unless a metric explicitly tests
cross-checkpoint or cross-site transfer. Standardize from the probe-training states only;
never use held-out states to choose normalization, ridge coefficients, PCA axes, heads, or
thresholds.

### 5.1 Final-answer count decodability

At the `<Ans>` position for both modes, record nearest-centroid exact-count accuracy,
ridge MAE, ridge R-squared, and per-count prediction bias for embedding layer 0 and every
transformer layer. Create checkpoint-by-layer heatmaps and layer trajectories. For
thinking, report two contexts separately:

- `gold_trace`: the existing teacher-forced rendering;
- `generated_trace`: the model's own greedily generated trace, re-forwarded to collect a
  comparable final-answer residual when a valid `<Ans>` is reached.

Do not combine these two contexts in one accuracy number. Their gap is a direct measure
of trace exposure bias and error propagation.

### 5.2 Trace-progress state

At trace-index and trace-marker positions, decode ordinal progress `k` using the same
nearest-centroid and ridge metrics. Also measure whether the marker state preserves `k`
until the next trace index. Treat perfect layer-0 trace-index decoding as a lexical-token
baseline because the input token is literally `<k>`; emphasize trace-marker and later
residual layers when interpreting learned progress state.

Stratify by `k` and true count to determine whether a counter is learned first for early
trace steps and later extends to longer traces.

### 5.3 Shared counter direction across sites

At each checkpoint and layer, fit a standardized ridge direction on trace-marker states
labeled by progress `k` and apply it, without refitting, to final-answer states labeled by
count `n`. Run the reverse transfer as well. Record cross-site MAE, R-squared, slope,
intercept, and direction cosine. Successful transfer would indicate reuse of a common
counter axis rather than two independently decodable representations.

Keep train/evaluation examples disjoint and fit any affine calibration only on the probe
training split. Include same-site train/test probes as upper controls and shuffled-label
transfer as a negative control.

### 5.4 Counterfactual trace-length versus prompt-count states

For thinking counts 2 through the configured maximum, construct a paired, deterministic
counterfactual by deleting only the final `<k> marker` pair from the gold trace while
leaving the task prefix and prompt unchanged. The resulting supplied trace length is
`m = n - 1`; close the trace normally and retain `<Ans>` so the answer state is well
defined. At each checkpoint and layer, probe or read out both true prompt count `n` and
supplied trace length `m`, and save the unrestricted count-token logits.

Compare whether the hidden state and model answer follow the prompt or the shortened
trace. This provides a precise, reproducible version of the prompt-count-versus-trace
question without inventing an extra marker or a nonexistent count-zero token. Keep these
counterfactuals out of training and label them as interventions rather than natural
examples.

### 5.5 Ordered geometry and representation stability

For every site, checkpoint, and layer, compute per-label centroids followed by:

- PC1 correlation-squared with count/progress;
- adjacent-centroid direction consistency;
- PC1 and PC1-to-PC6 explained variance;
- effective dimensionality;
- adjacent-count distances and monotonic ordering violations.

Use these metrics to distinguish a one-dimensional ordered counter from ten unrelated
but decodable count clusters. In addition, compute linear CKA on the same centered
example-state matrices between adjacent checkpoints, plus cosine similarity of fitted
count/progress directions. These stability metrics should identify representational
reorganization around the step-1,500 objective transition even when probe accuracy stays
flat.

Persist at least:

```text
tables/checkpoint_state_probe_summary.csv
tables/checkpoint_state_by_count.csv
tables/checkpoint_state_geometry.csv
tables/checkpoint_state_cross_site.csv
tables/checkpoint_state_counterfactual_trace.csv
tables/checkpoint_state_similarity.csv
```

## 6. Dynamics visualizations

Add v16_2-specific plotting functions that consume the saved checkpoint tables and do
not reload models. Every plot must support one model, an arbitrary model subset, missing
optional generated diagnostics, and non-5,000-step runs. Discover steps from the tables,
draw the exact language-loss boundary from config, and label layers as embedding `L0`
followed by transformer `L1-L4`.

Produce at least:

1. `checkpoint_attention_retrieval_emergence.png`: fixed-final retrieval heads and
   best-current heads, showing correct mass, top-1, diagonal dominance, and chance.
2. `checkpoint_answer_routing.png`: checkpoint-by-layer/head routing from `<Ans>` to
   prefix, prompt needles, trace indices, and trace markers, faceted by mode.
3. `checkpoint_nonthinking_needle_coverage.png`: enrichment, top-n recall, entropy, and
   autoregressive accuracy on aligned axes.
4. `checkpoint_head_role_stability.png`: role rankings and head identities across steps.
5. `checkpoint_final_count_probe_heatmap.png`: checkpoint-by-layer nearest-centroid
   accuracy and ridge R-squared for thinking/nonthinking and gold/generated contexts.
6. `checkpoint_trace_progress_probe_heatmap.png`: trace-index/marker progress decoding.
7. `checkpoint_cross_site_counter_transfer.png`: trace-to-answer and answer-to-trace
   transfer by checkpoint and layer.
8. `checkpoint_counterfactual_trace_readout.png`: probability/output preference for
   prompt count `n` versus shortened trace length `n-1`.
9. `checkpoint_state_geometry_emergence.png`: PC1 alignment, adjacent consistency, and
   effective dimension.
10. `checkpoint_representation_stability.png`: adjacent-checkpoint CKA and direction
    cosine, with the loss-phase boundary marked.
11. `checkpoint_mechanism_overview.png`: a compact summary combining behavioral AR
    accuracy, the strongest fixed retrieval head, final-answer trace readout, and final
    count-state decoding.

Avoid plotting only the strongest head at each checkpoint. Show fixed-head and
best-current-head curves together, include count-dependent chance baselines where
applicable, and retain tables behind every plotted aggregate. Use consistent colors for
thinking/nonthinking and consistent line styles for RoPE/RPE across figures.

## 7. New v16_2 notebook analysis block

Add a new code block to `scripts/build_v16_2_notebook.py` and the checked-in
`notebooks/Trace_Count_v16_2_Colab.ipynb` after training completes and before final result
display/copy. The notebook block must invoke the repository script rather than duplicating
analysis code. Include clearly editable controls such as:

```python
RUN_CHECKPOINT_DYNAMICS = True
DYNAMICS_EXAMPLES_PER_COUNT = 20       # teacher-forced attention/probe suite
DYNAMICS_AR_EXAMPLES_PER_COUNT = 10    # slower generated-trace/behavior-linked subset
DYNAMICS_DEVICE = DEVICE
FORCE_CHECKPOINT_DYNAMICS = False
```

The block should:

1. resolve the completed local run or its synchronized Google Drive copy;
2. print the discovered variants and ordered checkpoint steps before loading a model;
3. verify that the step-1,500/boundary checkpoint and final checkpoint exist;
4. run `scripts/analyze_v16_2_checkpoint_dynamics.py` with streamed stdout/stderr;
5. resume cached partitions by default and require an explicit force flag to recompute;
6. display the mechanism-overview figure followed by the main attention and state
   dynamics figures;
7. print links/paths to the detailed CSV tables and dynamics manifest;
8. fail clearly when checkpoints were deleted rather than silently plotting final-only
   tables;
9. avoid running pytest or retraining any model;
10. ensure all new outputs are included in the final Drive synchronization/copy.

When `RUN_CHECKPOINT_DYNAMICS=False`, print one concise skipped message and allow the
notebook to continue. The analysis block must be safely rerunnable in a fresh Colab
session after mounting Drive, without requiring the training cell to run again. Document
the expected extra runtime and make it clear that generated-trace collection is the
slowest optional component.

## 8. Structured runtime logging for debugging

Instrument the complete v16_2 workflow with monotonic wall-clock timing so slowdowns can
be assigned to training, evaluation, checkpoint I/O, model loading, individual dynamics
metric families, plotting, or Drive synchronization. Timing must be observational only:
it must not change RNG state, model outputs, checkpoint cadence, or resumption behavior.

Use `time.perf_counter()` for elapsed durations and UTC timestamps for human-readable
start/end records. Emit paired live console messages in a stable format:

```text
[timing:start] scope=checkpoint_dynamics variant=rope/thinking step=1500 block=attention
[timing:done]  scope=checkpoint_dynamics variant=rope/thinking step=1500 block=attention seconds=12.34 status=complete
```

Persist atomic rows in a common table such as `tables/runtime_events.csv` with at least:

```text
event_id, scope, block, position_encoding, mode, step,
started_at_utc, finished_at_utc, duration_seconds, status,
num_examples, num_batches, device, resumed_or_cached, error_type
```

Use blank nullable fields when an event is not model- or step-specific. Give every event
a deterministic ID so notebook reruns and resumed analyses update/deduplicate the same
logical event rather than double-counting it. Failed events must retain their elapsed
time and error type; a cached partition should be logged as cached with near-zero active
compute rather than being presented as a newly executed measurement.

During training, log:

- initialization and step-0 checkpoint writing per variant;
- optimizer-only training time between evaluation points, excluding evaluation and
  checkpoint I/O;
- periodic fixed-suite teacher-forced evaluation at each `eval_every` step;
- periodic autoregressive evaluation at each `ar_eval_every` step;
- checkpoint serialization and Drive checkpoint synchronization at every saved step;
- final test, prefix-permutation evaluation, and per-variant total training time;
- complete `prepare`, `train`, final-only `attention`, final-only `state`, and `plots`
  stage times.

The distinction between optimizer time and evaluation time is required: total wall time
alone cannot diagnose whether a run is slow because of training or because a larger
evaluation suite runs every 500 steps.

For checkpoint dynamics, log every variant/checkpoint combination separately for:

- checkpoint file read and model construction/state loading;
- teacher-forced attention collection;
- nonthinking coverage and head-stability reduction;
- gold-trace hidden-state collection and probes;
- generated-trace autoregressive generation and state collection;
- counterfactual trace construction/forward passes;
- cross-site transfer, geometry, and CKA reductions;
- partition/table writes;
- cross-checkpoint aggregation and each figure or figure family;
- per-checkpoint, per-variant, per-metric-family, and complete dynamics totals.

Where CUDA is used, synchronize immediately before starting and stopping a timed GPU
forward block so durations do not merely measure asynchronous kernel launch. Do not
synchronize inside every batch. Optionally record peak allocated CUDA memory for the
block after resetting peak statistics, but keep CPU execution valid when this field is
absent.

The notebook should time and print its main blocks—environment setup, preparation,
training subprocess, checkpoint dynamics, result display, local-to-Drive copy, and final
verification—and display a compact runtime summary grouped by scope/block/variant. Add a
stacked or grouped figure such as `figures/runtime_breakdown.png` showing optimizer,
teacher-forced evaluation, autoregressive evaluation, checkpoint I/O, attention
dynamics, state dynamics, generated/counterfactual analysis, and plotting. Preserve the
raw timing table behind the figure and document that Colab/Drive I/O timings depend on
the current runtime and network conditions.

## 9. README and protocol documentation

Update `README.md` and `docs/pipelines/pipeline_v16_2_character_sets.md` after coding.
Document:

- the 500-step checkpoint default, step-0 snapshot, exact-boundary snapshot, storage
  implications, and notebook control;
- how to run or rerun checkpoint dynamics from the notebook and command line;
- every attention metric, its query position, denominator, chance baseline, and whether
  it applies to thinking, nonthinking, or both;
- every hidden-state site, layer convention, probe train/evaluation split, standardization,
  geometry metric, and cross-site-transfer definition;
- the difference between gold-trace, generated-trace, and counterfactual-trace states;
- the teacher-forcing, positional leakage, multiple-head selection, correlational
  attention, and decodability-not-causality limitations;
- output table and figure names, so a user can archive derived metrics and later delete
  checkpoints;
- the runtime-event schema, console timing messages, optimizer-versus-evaluation timing
  distinction, runtime-breakdown figure, and CUDA synchronization convention;
- backward compatibility: historical v16_2 runs retain their saved 1,000-step cadence
  and can be analyzed only at checkpoints they actually contain.

Include a short example command pointing to one run directory, a `--skip-generated`
fast path, and a note that all checkpoint models are loaded sequentially rather than
simultaneously. Do not imply that the existing final-only attention/state tables contain
training dynamics.

## 10. Tests and acceptance criteria

Extend `tests/test_synthetic_counting_v16_2.py` and add focused dynamics tests if keeping
the file manageable requires a new test module. Verify:

1. `checkpoint_every=500` round-trips through config, CLI, run metadata, checkpoint
   metadata, notebook settings, and the planned-run summary; invalid values fail early.
2. A tiny run saves step 0, cadence steps, an exact nonaligned language-loss boundary,
   an exact nonaligned final step, and the final alias without duplicate analysis rows.
3. Arbitrary-checkpoint loading rejects config/vocab/pool/split, step, mode, and position
   mismatches before collecting metrics.
4. Fixed diagnostic manifests reproduce identical examples and fingerprints across all
   enabled variants and checkpoints.
5. Hand-computed attention tensors verify correct-k mass, top-1, diagonal dominance,
   chance normalization, category masses, needle enrichment, top-n recall, and entropy.
6. Toy hidden states verify nearest-centroid, ridge MAE/R-squared, PCA geometry,
   adjacent-direction consistency, cross-site transfer, direction cosine, and linear CKA.
7. The counterfactual renderer removes exactly the final trace pair, leaves prefix and
   prompt tokens unchanged, sets supplied trace length to `n-1`, and never enters training
   data.
8. Gold/generated/counterfactual contexts remain separate in schemas and figures; a
   malformed or unterminated generated trace receives an explicit status rather than an
   invented answer state.
9. One-model, two-model, and all-four debug artifacts produce only requested rows and
   plots and never load disabled checkpoints.
10. Interrupted analysis resumes completed variant/step partitions; `--force` changes
    only checkpoint-dynamics outputs.
11. Plotting succeeds with one checkpoint, missing generated diagnostics, arbitrary
    training length, and either mode alone, with correct phase-boundary labeling.
12. The builder-generated and checked-in notebook contain the new checkpoint cadence and
    post-training analysis block, pass all arguments, compile every code cell, and do not
    run tests or training inside the analysis block.
13. Runtime instrumentation emits deterministic, deduplicated complete/failed/cached
    rows for training intervals, periodic TF/AR evaluations, checkpoint I/O, every
    dynamics metric family, plots, and Drive synchronization; mocked timing tests verify
    aggregation without asserting machine-specific durations.
14. README and pipeline documentation name every emitted table/figure and accurately
    state the interpretation limits.
15. Existing final-only v16_2 outputs, historical config loading, and untouched v16 and
    v11-v14 tests continue to pass.

Run focused metric/unit tests first, then notebook compilation, Ruff, and
`git diff --check`. Finish with a deterministic CPU debug run using two modes, four
training steps, checkpoint cadence `2`, and language boundary `2`. Confirm checkpoints at
0, 2, and 4; run the dynamics analyzer; inspect every table schema and figure; interrupt
and resume once; and verify that final-only attention/state stages still produce their
historical outputs. Before a new main Colab run, perform a short CUDA smoke analysis that
loads checkpoints sequentially and successfully syncs all derived CSVs, figures, and the
dynamics manifest to Google Drive.

## 11. Implementation order

1. Add the 500-step cadence control, step-0 and exact-boundary checkpoint saving, CLI and
   notebook wiring, plus checkpoint-inventory tests.
2. Add the validated arbitrary-checkpoint loader, fixed diagnostic manifests, dynamics
   manifest, atomic partition writes, and resume behavior.
3. Add the shared runtime-event recorder and instrument training intervals, periodic
   evaluation, checkpoint I/O, dynamics partitions, plotting, notebook blocks, and Drive
   synchronization.
4. Implement attention detail metrics, summaries, head-role/stability tracking, and
   behavior linkage for both modes.
5. Implement gold/generated final-state probes, trace-progress probes, ordered geometry,
   cross-site transfer, counterfactual trace analysis, and representation similarity.
6. Implement table-only plotting functions, the complete dynamics figure set, and the
   runtime-breakdown figure.
7. Add the new notebook analysis block and preserve the user's existing easy-to-edit
   experiment settings while updating the builder and checked-in notebook.
8. Add unit, schema, subset, resume, timing, notebook, plotting, and legacy regression
   tests.
9. Run CPU and CUDA smoke analyses, inspect Drive synchronization, and confirm that no
   analysis path retrains or mutates a checkpoint.
10. Update README and the v16_2 pipeline document with metric definitions, commands,
   artifacts, runtime/storage expectations, and interpretation limits.

## 12. Confirmed clarification questions and user decisions

The following implementation choices were explicitly confirmed before coding:

1. **Question:** Should all metric families be implemented in one revision, while
   remaining independently switchable for incremental debugging?

   **User answer:** Yes. Implement the complete revision, with separate switches for core
   attention, core hidden states, generated-trace diagnostics, counterfactual diagnostics,
   and representation-stability/CKA metrics.

2. **Question:** Should checkpoint dynamics, generated-trace dynamics, and
   counterfactual dynamics run by default in the new notebook analysis block?

   **User answer:** Yes. Use enabled defaults while retaining independent controls to
   disable any expensive family.

3. **Question:** Should the diagnostic defaults be 20 attention examples per count, 10
   autoregressive/generated examples per count, 40 state-probe training examples per
   count, and 15 state-probe evaluation examples per count, with the 20 attention
   examples split into 10 head-selection and 10 held-out reporting examples?

   **User answer:** Yes. Use those recommended sample sizes and the disjoint 10/10 head
   selection/reporting split.

4. **Question:** Should the workflow save an initialization checkpoint and retain the
   complete sequence `0, 500, 1000, 1500, ..., 5000` for the present design?

   **User answer:** Yes. Save step 0, every 500 steps, the exact objective boundary, and
   the final step.

5. **Question:** Should the counterfactual trace intervention remove only the final
   `<n> marker` pair for examples with `n >= 2`, leaving prefix and prompt unchanged, so
   the analysis contrasts true prompt count `n` with supplied trace length `n-1`?

   **User answer:** Yes. Use the deterministic shortened-trace intervention as proposed.

6. **Additional user requirement:** Log running time for every main workflow block and
   evaluation point to make debugging and performance diagnosis easier. Implement both
   live start/done messages and persisted structured timing events as specified in
   Section 8.
