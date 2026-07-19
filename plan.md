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
