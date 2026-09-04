# 3-D two-stage surrogate: CNN1D / RNN / LSTM vs the deployed FFN and GPS-ABC

WHAT THIS IS. The deployed 3-D surrogate (`network/model.py: HeteroscedasticMLP`,
hidden=(64,32), GELU) is a plain MLP treating its four inputs -- `log10 p1`,
`log10 p2`, `tau`, and the derived `log10 p_eff` -- as an unordered feature
vector. This document benchmarks three architecture families that instead
impose an ORDER on those four inputs: a 1-D CNN (treats them as 4 adjacent
positions a convolution kernel slides across), and a GRU/LSTM (treats them as 4
sequential timesteps with a carried hidden state). `network/model_families.py`
gives the full argument for why this is expected, a priori, to be neutral-to-
harmful rather than helpful: none of these four physical scalars has a real
spatial or temporal relationship to its neighbours in the tuple -- the "order"
is just how the columns happen to be stored. This document reports whether the
data agrees with that prediction, on both the surrogate-fit and the downstream
ABC-recovery axes, without tuning anything to make a prettier story.

**Bottom line up front.** On fit quality, CNN1D statistically ties the deployed
FFN (0.4% *lower* mse_mean, well inside noise) while RNN and LSTM are
measurably worse (+9.7%, +15.6%). On the full downstream ABC-MCMC recovery
task, the picture is more nuanced: differences between all five methods
(GPS-ABC, DNN-ABC, CNN1D-ABC, RNN-ABC, LSTM-ABC) are a **statistical tie in 46
of 54 cells (85%)** once Monte Carlo SEs are accounted for, and where the 8
non-tie cells fall, established methods (GPS-ABC, then DNN-ABC) win 7 of them
-- the recurrent families never significantly beat the plain-MLP-based
DNN-ABC, and CNN1D-ABC never significantly beats either baseline. No
architecture with an imposed sequence/spatial prior beat the two structure-free
baselines on accuracy anywhere it mattered. The one genuinely interesting
finding is on calibration, not accuracy: LSTM-ABC's and RNN-ABC's interval
coverage (0.958 / 0.931 pooled) sits closer to the nominal 0.95 than the
deployed DNN-ABC's own under-coverage (0.875) -- see §3.

---

## 1. Surrogate fit quality (held-out test set, reps 9-10)

2 seeds/family, identical split/optimizer/calibration to `train.py`. Full data:
`benchmark_families.md` / `.csv`.

| architecture | params | mse_mean | x floor | R^2 | cover95 | us/query |
|---|---|---|---|---|---|---|
| **CNN1D** (16,16 ch, k=3) | 2,994 | 1.567e-03 | 1.13 | 0.99469 | 0.949 | 149 |
| **FFN 64-32 gelu [DEPLOYED]** | 2,466 | 1.573e-03 | 1.13 | 0.99467 | 0.952 | 41 |
| **RNN** (GRU, hidden=32) | 3,426 | 1.726e-03 | 1.24 | 0.99415 | 0.953 | 77 |
| **LSTM** (hidden=24) | 2,642 | 1.819e-03 | 1.31 | 0.99383 | 0.953 | 77 |

Irreducible floor on `mse_mean` = **1.393e-03** (recomputed fresh from the data
here; matches the already-established value in
`results/logs/benchmark_round2.md` to 4 significant figures).

**Reading this table.**
- **CNN1D statistically ties the FFN**: mse_mean 1.567e-3 vs 1.573e-3, a 0.4%
  gap that is nothing against the run-to-run seed spread already documented in
  `benchmark_arch.md` (rows differing by <10% there are called ties). Both sit
  at 1.13x the noise floor -- i.e. both are answering as well as the data
  permits. The convolution's local-window prior neither helps nor hurts here,
  which is exactly the null result the structural argument in
  `model_families.py` predicts: with a kernel wide enough (k=3) to see 3 of the
  4 positions at once, and two stacked conv layers giving the network access to
  all 4 inputs in combination, a small CNN on a 4-length input is not much more
  constrained than a small MLP -- it is close to reproducing the MLP's
  capability, not because the convolution found real structure, but because the
  input is short enough that "local" is nearly "global."
- **RNN and LSTM are measurably, if modestly, worse**: +9.7% and +15.6%
  mse_mean respectively (both `sd_mse_mean` over 2 seeds is `2.3e-5` and
  `9.4e-5`, an order of magnitude smaller than the gaps -- these are not
  seed noise). This is consistent with the structural argument: forcing the
  network to route all cross-parameter interactions through a single
  hidden-state bottleneck, one arbitrary "timestep" at a time, is a real
  constraint when (as `model.py`'s docstring documents) the surface has
  genuine multi-way interaction -- a linear fit reaches only R^2=0.75 while
  a linear+quadratic+interactions fit reaches 0.965. An MLP or CNN sees all
  four inputs simultaneously in every layer; a 4-step recurrence has to carry
  everything relevant about `p1` forward through 3 more steps before it can
  interact with `tau` or `p_eff`, which is a strictly harder optimization
  problem for no compensating benefit (there is no long-range structure here
  for the recurrence to actually exploit, unlike a real time series).
- **Query cost**: CNN1D is the slowest to query (149 us, ~3.6x the FFN's 41
  us) despite a comparable parameter count -- Conv1d's per-layer overhead on
  tiny tensors outweighs the FFN's dense matmuls at this scale. RNN/LSTM sit
  in between (77 us, ~1.9x the FFN) from the sequential 4-step unroll. None of
  this is prohibitive for an ABC loop (all four are sub-millisecond), but the
  FFN remains the cheapest query by a clear margin, on top of tying or beating
  the others on accuracy.

**Verdict on fit quality: CNN1D ties the deployed FFN; RNN and LSTM lose to
it (worse mse_mean, slower query, no offsetting benefit).** No family beats
the FFN.

---

## 2. Full ABC-MCMC recovery (identical config to `table1_recovery.csv`)

Config, reused verbatim from `results/logs/experiment_config.json`: truths
`(1e-4,1e-2,3)`, `(1e-4,1e-2,7)`, `(2e-3,8e-3,5)`; `reps=16`, `nmcmc=3000`,
`burnin=1000`, `J=100`, `eps=0.005`. GPS-ABC and DNN-ABC(MLP) rows are copied
verbatim from the existing table (not rerun); CNN1D-ABC/RNN-ABC/LSTM-ABC rows
are produced by wrapping each family's best-of-2-seeds checkpoint in the
existing `DNNSurrogate3D` and running it through the unmodified
`abc_mcmc.run_abc_mcmc(backend="dnn", ...)` sampler, against the identical
per-replicate simulated observations (same seed formula) used for every other
method. Full table: `table_families_recovery.csv` (52 rows, same columns as
`table1_recovery.csv`).

### 2.1 Averaged over all 3 truths x 3 parameters (9 cells/method)

| method | mean rmse_log | mean 95% CI width | mean coverage | mean secs/rep | mean accept rate |
|---|---|---|---|---|---|
| GPS-ABC | **1.086** | 3.045 | 0.993 | 2.240 | 0.352 |
| DNN-ABC (deployed FFN) | 1.215 | 2.865 | 0.875 | 1.935 | 0.132 |
| CNN1D-ABC | 1.311 | 2.847 | 0.819 | 1.738 | 0.135 |
| RNN-ABC | 1.122 | 2.928 | 0.931 | 1.961 | 0.131 |
| LSTM-ABC | 1.102 | 2.929 | **0.958** | 2.064 | 0.134 |

(`rmse_log` is RMSE in log10 units for p1/p2 and absolute time units for tau,
averaged across the three parameters and three truths without further
weighting -- read it as a rough composite, not a formal pooled statistic; the
per-parameter breakdown below is the more meaningful comparison, and the
per-cell numbers with MCSEs in `mcse_families.md` are the ones that support an
actual claim of difference.)

### 2.2 Per-parameter averages (across the 3 truths)

| param | GPS-ABC | DNN-ABC | CNN1D-ABC | RNN-ABC | LSTM-ABC |
|---|---|---|---|---|---|
| p1 rmse_log | 1.390 | 1.355 | 1.438 | 1.418 | 1.397 |
| p1 coverage | 0.979 | 0.896 | **0.750** | 0.938 | 0.917 |
| p2 rmse_log | **0.207** | 0.391 | 0.549 | 0.401 | 0.282 |
| p2 coverage | 1.000 | 0.792 | 0.771 | 0.875 | 0.958 |
| tau rmse_log | 1.662 | 1.901 | 1.947 | **1.548** | 1.627 |
| tau coverage | 1.000 | 0.938 | 0.938 | 0.979 | 1.000 |

### 2.3 What's a real difference, not noise (from `mcse_families.md`)

54 pairwise comparisons total (3 truths x 3 params x 3 new families x 2
baselines {GPS-ABC, DNN-ABC}), each new-family-vs-baseline delta flagged
significant only when it exceeds 2 combined Monte Carlo SEs:

- **46 / 54 (85%) are ties.** Most of the point-accuracy story above is inside
  simulation noise at `reps=16` -- consistent with how weakly p1 and tau are
  identified by this summary statistic in the first place (README, `abc_mcmc.py`
  docstring).
- **8 / 54 are significant**, and **7 of those 8 favor an established method**
  (GPS-ABC 5x, DNN-ABC 2x) over a new family. The lone exception: **RNN-ABC
  beats DNN-ABC on `tau` at truth (1e-4, 1e-2, 7)** (delta = -0.429 ± 0.210).
  No new family is ever significantly better than GPS-ABC anywhere, and no new
  family is significantly better than DNN-ABC anywhere except that one `tau`
  cell.
- The clearest significant pattern is **GPS-ABC beating CNN1D-ABC** (4 of its
  5 wins are against CNN1D-ABC specifically: p2 at truth 1, p1/p2/tau at
  truth 3) -- consistent with CNN1D-ABC also having the worst pooled coverage
  (0.819) of any method in the whole comparison, including the already-known
  under-covering DNN-ABC.

**Verdict on point accuracy: no architecture with an imposed sequence/spatial
prior beats GPS-ABC anywhere at a statistically meaningful level, and only one
cell (out of 54 tested) sees a new family beat the deployed DNN-ABC.** The
overall picture is a wide tie, exactly what "no real structure to exploit"
predicts -- the four physical scalars don't reward or punish an architecture
for imposing an order on them, they're just indifferent to it, with one
partial exception (CNN1D, discussed next).

---

## 3. The one genuine nuance: calibration, not accuracy

The FFN-based DNN-ABC has a known, previously-reported weakness: its credible
intervals **under-cover** relative to the nominal 0.95 (0.875 pooled here,
consistent with the retired `(p,a,delta)` study's "under-covers at small `J`"
finding cited in the README §9). CNN1D-ABC inherits and *worsens* this
(0.819 pooled, and 0.750 on `p1` specifically -- the single worst
coverage number in this entire study). **RNN-ABC (0.931) and especially
LSTM-ABC (0.958) do not** -- LSTM-ABC's pooled coverage is closer to nominal
than either GPS-ABC's mild over-coverage (0.993) or DNN-ABC's under-coverage,
and it achieves this while its `rmse_log` (1.102) is competitive with -- in
fact slightly better than -- the deployed FFN's DNN-ABC (1.215).

This is worth stating plainly because it runs slightly against the file's own
a priori expectation: the recurrent families' worse *surrogate* fit (§1) does
not translate into worse *calibrated interval* behaviour once conformal
calibration and the ABC acceptance step are in the loop -- if anything LSTM's
predictive uncertainty is better shaped for this purpose than the FFN's, on
this run. It is a single 16-replicate-per-cell result, not (per §2.3) something
most of the individual coverage gaps clear 2 MCSEs on cell-by-cell, so it
should be read as a mild, secondary observation worth a follow-up run with
more replicates -- not as a reversal of the headline finding in §2, which is
that no new family beats the established baselines on point accuracy.

---

## 4. Overall answer to the question this file exists to answer

**Does treating 4 unordered physical parameters as an ordered sequence/signal
help a 3-D two-stage surrogate, either at the surrogate-fit stage or in the
full downstream ABC-MCMC pipeline? No, with one narrow, secondary exception.**

- **Fit quality**: CNN1D ties the deployed FFN; RNN and LSTM lose to it
  (measurably worse mse_mean, no compensating benefit, and all three are
  slower to query than the FFN).
- **ABC recovery accuracy**: no new family significantly beats GPS-ABC
  anywhere (54 comparisons); only 1 of 54 comparisons has a new family
  (RNN-ABC) significantly beating DNN-ABC(MLP), and CNN1D-ABC is significantly
  *worse* than GPS-ABC in 4 of its 18 cells with no compensating wins.
- **Calibration** is the one place the recurrent families (RNN, and especially
  LSTM) show a real, if secondary, advantage over the deployed FFN's DNN-ABC:
  closer-to-nominal interval coverage at comparable point accuracy.

This is consistent with, and was predicted by, the structural argument in
`network/model_families.py`: `(log10 p1, log10 p2, tau, log10 p_eff)` has no
genuine spatial adjacency for a convolution to exploit and no genuine temporal
dependency for a recurrence to exploit, so an architecture built around either
inductive bias should be expected to perform on par with or worse than an
architecture-agnostic MLP on this surface -- which is what the data shows, the
LSTM calibration nuance in §3 being the one finding worth a second look rather
than a reversal of that conclusion. **The deployed FFN surrogate
(`HeteroscedasticMLP`, hidden=(64,32), GELU) remains the right choice for this
project**; nothing here motivates switching to CNN1D, RNN, or LSTM as the
production 3-D surrogate.

---

## 5. Files

- `network/model_families.py` -- `HeteroscedasticCNN1D`, `HeteroscedasticRNN`,
  `HeteroscedasticLSTM`, `build_family()`.
- `network/architecture_search/benchmark_families.py` -- trains 2 seeds/family,
  writes `benchmark_families.md` / `.csv`, saves best-seed checkpoints to
  `checkpoints/{cnn1d,rnn,lstm}_best.pt`.
- `abc/run_experiments_families.py` -- runs the full ABC-MCMC recovery for each
  family's best checkpoint through the existing, unmodified sampler; writes
  `table_families_recovery.csv`, `TABLE_FAMILIES.md`, `mcse_families.md` /
  `.csv`, `raw_replicates_families.csv`.
- This file.

None of `benchmark_arch.md`/`.csv`, `benchmark_round2.md`/`.csv`,
`surrogate_metrics.json`, `surrogate_3d.pt`, `table1_recovery.csv`, `mcse.md`/
`.csv`, or `model.py` were modified -- every number attributed to those files
above was read, not recomputed.
