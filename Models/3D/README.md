# 3-D two-stage surrogate — `(p1, p2, τ)`

A neural surrogate for the **two-stage mutation model**, the multi-parameter
model the JTB paper (Lu, Zhu & Wu 2023) actually uses in its Study 2. The
mutation probability is a step function of time,

```
p(t) = p1   for 0 < t ≤ τ
       p2   for τ < t ≤ tp
```

and all three of `(p1, p2, τ)` are estimated jointly from a single scalar
summary statistic.

> **This replaces an earlier 3-D study** that varied `(p, a, δ)` under a
> *constant* mutation rate. That was the wrong model — not a reparameterization
> of this one — and its `a` axis was analytically non-identifiable, making it
> effectively 2-D. See [`../../updates.md`](../../updates.md).

---

## 1. The short version

| | |
|---|---|
| Ground truth | exact cell-by-cell simulator, 2000 LHS design points × 10 reps = **20,000 rows** |
| Fixed | `Z0=1`, `a=1`, `J=100`, `tp=10` |
| Varied | `log10 p1, log10 p2 ∈ [−5, −1.3]`, `τ ∈ [0.1, 9.9]` |
| Target | `log10(S)` where `S = meanᵢ (Xᵢ/Zᵢ)^(1/4)` — the paper's two-stage statistic |
| Model | `(log10 p1, log10 p2, τ)` → Dense 64 (GELU) → Dense 32 (tanh) → (mean, log-variance) |
| Held-out fit | `mse_mean` = 3.63e-4 = **1.05× the irreducible noise floor** |
| Calibration | 95% interval coverage **0.955** after a conformal scale of 1.019 |

> **The summary statistic is the fourth root, not the square root.** The paper uses
> `√(X/Z)` for the constant-rate study (its Fig. 1 selects it) but switches to
> `⁴√(X/Z)` for the two-stage model — Sec. 2.2, "to make the response curve smooth
> and non-flat", and Algorithm 1 throughout. This package followed the square root
> until that was caught. Because the ground truth stores every per-culture
> `dᵢ = √(Xᵢ/Zᵢ)`, the fourth root is recoverable exactly as `√(dᵢ)` with no
> re-simulation; `network/train.py:summary_from_cultures` does it, and
> `abc/simulator.py:SUMMARY_ROOT` is the single switch. Adopting it cut the
> surrogate's held-out error from 1.19× the floor to 1.11×, and reselecting the
> activation under it took that to 1.05×.

## 2. Three findings that shaped the design

**(a) The noise is violently heteroscedastic — so the model has two heads.**
The within-design-point replicate sd of `log10(S)` varies **184×** across the
design and correlates **−0.82** with the target: where few mutants arise, `S` is
small *and* its scatter is huge. (Under the old square-root target those figures
were 292× and −0.74 — the fourth root compresses the spread but does not remove
it, and the correlation with the mean in fact tightens.) A single homoscedastic noise term — all a
Gaussian process offers — cannot represent that. The second head predicts an
input-dependent variance, which the ABC acceptance step then consumes directly.
(`results/figures/fig_noise.png`)

**(b) The network takes the three parameters directly.** Inputs are
`(log10 p1, log10 p2, τ)` and nothing else — no derived or hand-engineered
features sit between the parameters being inferred and the prediction. Outputs
are the predicted `log10(S)` and its log predictive variance.

**(c) Capacity is not the binding constraint — so the network is small.**
Because the held-out target is itself a 2-replicate mean, it carries
`E[σ²]/2` of sampling noise no model can predict away. That floor is
`mse_mean = 3.45e-4`. Measured against it
(`architecture_search/benchmark_round2.py`, 3 seeds):

| hidden | params | × floor | R² | µs/query |
|---|---|---|---|---|
| 256-128-64 | 42,306 | 1.01 | 0.99681 | 56 |
| 128-64 | 8,898 | 1.04 | 0.99671 | 42 |
| **64-32** ← used | **2,402** | **1.09** | **0.99656** | **40** |
| 32-16 | 690 | 1.17 | 0.99628 | 37 |
| 16-8 | 218 | 1.41 | 0.99553 | 38 |
| 8-4 | 78 | 1.81 | 0.99426 | 37 |
| linear | 8 | **140.88** | 0.55319 | 29 |

A network *is* emphatically needed — the linear control sits at **141× the
floor** with R² = 0.55 — but a 2,402-parameter one comes within 8% of a
42,306-parameter one. LayerNorm and residual depth are ties
(`benchmark_arch.py`). `64-32` is used for parsimony with a little headroom,
**not** for speed: query latency is dominated by Python/PyTorch call overhead,
so 18× fewer parameters buys only ~29% less latency.
(`results/figures/fig_capacity.png`)

> Under the old square-root target the linear control was 24.8× the floor. The
> fourth root makes the surface *harder* for a linear model by a factor of
> nearly six, so the case for a network is stronger under the paper's own
> statistic than under the one this package previously used.

## 3. Why the inference is hard (and expected to stay hard)

One scalar summary carries very uneven information about three parameters:

| | `p2` | `p1` | `τ` |
|---|---|---|---|
| `corr(log S, ·)` | **0.778** | 0.337 | 0.160 |

(Square-root target, for comparison: 0.742, 0.404, 0.135. The fourth root
sharpens the `p2` signal and slightly lifts `τ`, at the cost of `p1`.)

So `p2` should be recovered well and `p1`/`τ` poorly, with wide and possibly
multimodal marginals. **This is a property of the model, not a bug in the
sampler** — the paper reports the same and names it as the model's known
weakness. Sharpening those marginals is the stated target of the rebuild
(`updates.md` §5), so the tables report per-parameter accuracy separately rather
than a single score that would hide it.

Because of this, prefer **`rmse_log`** to `nrmse` in the tables: for a weakly
identified parameter the posterior mean sits wherever the prior puts its mass,
and natural-scale nRMSE then explodes without conveying anything.

## 3b. What the comparison found

Three surrogates are run through the same sampler
(`abc/run_experiments.py`, 16 replicates, 3000 MCMC draws, 1000 burn-in):

| method | what it is | mean `rmse_log` | mean s/fit |
|---|---|---|---|
| GPS-ABC | GP, strengthened: anisotropic RBF, log-scaled inputs, learned noise | **0.990** | 1.98 |
| GPS-ABC-ref | GP, faithful to `demoGPS_fluc_exp2.m`: isotropic, raw inputs/target | 1.116 | 1.75 |
| DNN-ABC | this package's heteroscedastic MLP | 1.139 | 1.59 |

**The headline is negative, and it is robust.** DNN-ABC does not beat the GP: it
wins 1 of 9 parameter-by-truth cells. That verdict survived three separate
corrections, each of which genuinely improved the surrogate — fixing the
mutation-time convention, adopting the paper's fourth-root statistic (held-out
error 1.19× floor → 1.11×), and reselecting the activation under it (→ 1.05×).
A surrogate that got materially better three times over did not change the
answer, which is the point: **what binds here is the model's identifiability,
not surrogate error.** `p1` and `τ` are weakly determined by one scalar summary,
so a better approximation of that summary cannot help.

Two things not to over-read from the table:

- **Coverage differences are noise at 16 replicates.** DNN-ABC's mean coverage
  is 0.972 against GPS-ABC's 1.000, but that is 15/16 versus 16/16 replicates —
  one replicate, well inside the Monte Carlo error `abc/mcse.py` reports. Mean
  interval widths are near-identical (`p1`: 3.430e-2 vs 3.456e-2). Every method
  over-covers on `τ` at 1.000, which is what a nearly-unidentified parameter
  looks like: the interval spans most of the prior box.
- **Reporting both GPs matters.** Ours is stronger than the paper's on every
  axis, and the gap is not small — on raw-scale surrogate fit the reference
  reaches R² ≈ 0.71 where an otherwise identical anisotropic kernel reaches
  ≈ 0.99. One shared length scale cannot serve inputs whose ranges differ ~200×
  (`p1, p2` span ~0.05, `τ` spans ~9.8). Quoting only the strengthened GP would
  overstate what the published baseline achieves.

## 4. Layout

```
3D/
├── paths.py                       # single source of truth for data/results locations
├── matlab/                        # the professor's reference code, byte-identical
│   ├── mut2stage_bMBP.m           #   exact two-stage simulator
│   ├── demoGPS_fluc_exp2.m        #   THE Study 2 GP reference -- read before
│   │                              #   touching abc/surrogates.py
│   └── runsimu.m                  #   driver for both simulators
│                                  # (constant-rate MATLAB lives in ../1D/matlab/)
├── data/slow_data_3D.csv          # ground truth (generated by ../../RCode/genSlowData_3D.R)
├── network/                       # THE SURROGATE
│   ├── model.py                   #   two-headed MLP + the derived feature
│   ├── train.py                   #   training + conformal calibration; load_surrogate()
│   ├── gen_architecture_svg.py    #   diagram, generated from the live config
│   └── architecture_search/
│       ├── benchmark_arch.py      #   round 1: shape, activation, normalisation, ablation
│       ├── benchmark_round2.py    #   round 2: how small can it be?
│       ├── benchmark_activations.py       # ten activations, single-activation nets
│       ├── benchmark_activation_pairs.py  # all 100 ordered pairs -- SUPERSEDED,
│       │                                  #   selected on the test split (leakage)
│       └── benchmark_activation_select.py # the one that decides: selection on
│                                          #   VALIDATION, fresh confirmation seeds
├── abc/                           # THE INFERENCE PIPELINE
│   ├── simulator.py               #   exact + fast two-stage sims; SUMMARY_ROOT
│   ├── surrogates.py              #   predict(θ)→(mean,sd): DNN, GP, reference GP
│   ├── abc_mcmc.py                #   joint MH over (log10 p1, log10 p2, τ)
│   ├── run_experiments.py         #   the result tables
│   └── mcse.py                    #   Monte Carlo SEs — which differences are real
├── tests/validate_simulator.py    #   degeneracy, convention gap, data provenance
├── figures/make_figures.py
└── results/{figures,logs,model,tables}
```

## 5. Reproducing

```bash
# ground truth (from the repo root; ~24 min on 30 cores)
Rscript RCode/genSlowData_3D.R

cd Models/3D
python tests/validate_simulator.py                       # sanity + data provenance
python network/architecture_search/benchmark_arch.py     # round 1
python network/architecture_search/benchmark_round2.py   # round 2 (capacity floor)
python network/architecture_search/benchmark_activation_select.py   # activation
python network/train.py                                  # train + calibrate
python abc/run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --no-sim
python abc/mcse.py                                       # attach Monte Carlo SEs
python figures/make_figures.py
python network/gen_architecture_svg.py
```

Drop `--no-sim` to include the exact ABC-MCMC baseline; it runs the simulator
inside every MCMC iteration and is dramatically slower.

## 6. Open items

- **Mutation-time convention — RESOLVED, and it was a live bug.** `mut2stage_bMBP.m`
  contains two: the live lines 25–26 evaluate `p(t)` at the offspring's own
  division time, the commented-out lines 23–24 at the parent's. The pipeline used
  to train its surrogates on ground truth generated under *offspring* while
  drawing observations under *parent* — different models, differing by up to
  **10.1%** in the statistic. The convention was recovered empirically (no
  generation log survives): all 60 of the most discriminating design points sit
  closer to offspring, mean |error| 0.0054 vs 0.0400 log₁₀ units, paired
  t = +11.3. That matches the live MATLAB, the R generator's default, and the
  paper's Algorithm 3. Everything now defaults to `offspring`;
  `tests/validate_simulator.py` re-checks it on every run.
  Still worth asking the professor which he *intends*, but the pipeline is
  self-consistent either way now.
- **The paper's own regime is out of reach exactly.** Study 2 uses `tp = 20`
  (~5e8 cells/culture); `tp = 10` here is ~2.2e4. Reaching it needs the fast
  two-stage simulator (`mut2stage_fast` in `abc/simulator.py`), which is written
  but not yet validated against the exact one at scale.
- **No `δ`.** The professor's two-stage MATLAB has no differential mutant growth
  rate, so this is a genuine 3-parameter study. The paper's full model is 4-D;
  `simulator.py` threads `delta` through so that extension does not need a
  rewrite.
