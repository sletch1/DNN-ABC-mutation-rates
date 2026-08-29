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
| Target | `log10(d̄)` where `d̄ = meanᵢ √(Xᵢ/Zᵢ)` |
| Model | `(log10 p1, log10 p2, τ, log10 p_eff)` → Dense 64 → Dense 32 → (mean, log-variance) |
| Held-out fit | `mse_mean` = 1.56e-3 = **1.12× the irreducible noise floor**, R² = 0.9947 |
| Calibration | 95% interval coverage **0.956** after a conformal scale of 1.018 |

## 2. Three findings that shaped the design

**(a) The noise is violently heteroscedastic — so the model has two heads.**
The within-design-point replicate sd of `log10(d̄)` varies **292×** across the
design and correlates **−0.74** with the target: where few mutants arise, `d̄` is
small *and* its scatter is huge. A single homoscedastic noise term — all a
Gaussian process offers — cannot represent that. The second head predicts an
input-dependent variance, which the ABC acceptance step then consumes directly.
(`results/figures/fig_noise.png`)

**(b) A derived feature falls out of the biology, and it is the best single
predictor.** A mutation arising at time `t` founds a clone that grows to about
`e^(a(tp−t))`, while the number of divisions available to mutate near `t` grows
like `e^(a·t)`. **The two exponentials cancel**, so every unit of time contributes
equally and the physically correct aggregate of a time-varying rate is its
*time average*:

```
p_eff = ( p1·τ + p2·(tp − τ) ) / tp
```

`log10(p_eff)` alone explains **R² = 0.874** of the design-mean variation — more
than the full three-input linear model (0.752) — and regressing the target on it
gives slope **0.59** against the 0.5 predicted by `d̄ ~ √(X/Z)`. It is supplied to
the network alongside the raw inputs. (A Yule-arrival-CDF weighting, which looks
plausible but ignores the cancellation, reaches only 0.824.)

**But size the gain honestly:** that R² is for a *linear* model. A network learns
the same structure from raw inputs anyway, and the measured ablation shows only a
**1.01–1.03×** improvement. The feature is kept because it encodes a checkable
derivation and because it is exactly what the MOM/MLE baselines estimate — not
because it is what makes the surrogate accurate.

**(c) Capacity is not the binding constraint — so the network is small.**
Because the held-out target is itself a 2-replicate mean, it carries
`E[σ²]/2` of sampling noise no model can predict away. That floor is
`mse_mean = 1.39e-3`, i.e. a **maximum achievable R² of 0.99528**. Measured
against it:

| hidden | params | × floor | R² | µs/query |
|---|---|---|---|---|
| 256-128-64 | 42,562 | 1.08 | 0.99491 | 56 |
| 128-64 | 9,026 | 1.12 | 0.99472 | 43 |
| **64-32** ← used | **2,466** | **1.13** | **0.99467** | **41** |
| 32-16 | 722 | 1.14 | 0.99460 | 40 |
| 8-4 | 86 | 1.23 | 0.99419 | 39 |
| linear | 10 | **24.77** | 0.88300 | 29 |

A network *is* needed — the linear control is 25× the floor — but a
722-parameter one matches a 42,562-parameter one to within 6%. Activation
choice, LayerNorm and residual depth were likewise ties. `64-32` is used for
parsimony with a little headroom, **not** for speed: query latency here is
dominated by Python/PyTorch call overhead, so 59× fewer parameters buys only
~29% less latency. (`results/figures/fig_capacity.png`)

## 3. Why the inference is hard (and expected to stay hard)

One scalar summary carries very uneven information about three parameters:

| | `p2` | `p1` | `τ` |
|---|---|---|---|
| `corr(log d̄, ·)` | **0.742** | 0.404 | 0.135 |

So `p2` should be recovered well and `p1`/`τ` poorly, with wide and possibly
multimodal marginals. **This is a property of the model, not a bug in the
sampler** — the paper reports the same and names it as the model's known
weakness. Sharpening those marginals is the stated target of the rebuild
(`updates.md` §5), so the tables report per-parameter accuracy separately rather
than a single score that would hide it.

Because of this, prefer **`rmse_log`** to `nrmse` in the tables: for a weakly
identified parameter the posterior mean sits wherever the prior puts its mass,
and natural-scale nRMSE then explodes without conveying anything.

## 4. Layout

```
3D/
├── paths.py                       # single source of truth for data/results locations
├── matlab/                        # the professor's reference code, byte-identical
│   ├── mut2stage_bMBP.m           #   exact two-stage simulator
│   └── runsimu.m                  #   its driver
├── data/slow_data_3D.csv          # ground truth (generated by ../../RCode/genSlowData_3D.R)
├── network/                       # THE SURROGATE
│   ├── model.py                   #   two-headed MLP + the derived feature
│   ├── train.py                   #   training + conformal calibration; load_surrogate()
│   ├── gen_architecture_svg.py    #   diagram, generated from the live config
│   └── architecture_search/
│       ├── benchmark_arch.py      #   round 1: shape, activation, normalisation, ablation
│       └── benchmark_round2.py    #   round 2: how small can it be?
├── abc/                           # THE INFERENCE PIPELINE
│   ├── simulator.py               #   exact + fast two-stage simulators
│   ├── estimators.py              #   MOM / MLE baselines, and p_eff
│   ├── surrogates.py              #   predict(θ)→(mean,sd): DNN + GP baseline
│   ├── abc_mcmc.py                #   joint MH over (log10 p1, log10 p2, τ)
│   ├── run_experiments.py         #   the result tables
│   └── mcse.py                    #   Monte Carlo SEs — which differences are real
├── figures/make_figures.py
└── results/{figures,logs,model,tables}
```

## 5. Reproducing

```bash
# ground truth (from the repo root; ~24 min on 30 cores)
Rscript RCode/genSlowData_3D.R

cd DNN_Models/3D
python network/architecture_search/benchmark_arch.py     # round 1
python network/architecture_search/benchmark_round2.py   # round 2 (capacity floor)
python network/train.py                                  # train + calibrate
python abc/run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --no-sim
python abc/mcse.py                                       # attach Monte Carlo SEs
python figures/make_figures.py
python network/gen_architecture_svg.py
```

Drop `--no-sim` to include the exact ABC-MCMC baseline; it runs the simulator
inside every MCMC iteration and is dramatically slower.

## 6. Open items

- **Which mutation-time convention is intended.** `mut2stage_bMBP.m` contains
  two: the live lines evaluate `p(t)` at the offspring's own division time, the
  commented-out lines at the parent's (the birth of the daughter). They differ by
  3–10% in `d̄`. The ground truth was generated with the **live** version;
  everything downstream defaults to `parent`, which is the paper's model, the
  only one consistent with the fast simulator, and the only one that composes
  with a mutant growth rate `δ`. **This needs confirming with the professor**,
  and `RCode/genSlowData_3D.R --mut-time parent --limit N` produces a paired
  comparison on identical design points and seeds.
- **The paper's own regime is out of reach exactly.** Study 2 uses `tp = 20`
  (~5e8 cells/culture); `tp = 10` here is ~2.2e4. Reaching it needs the fast
  two-stage simulator (`mut2stage_fast` in `abc/simulator.py`), which is written
  but not yet validated against the exact one at scale.
- **No `δ`.** The professor's two-stage MATLAB has no differential mutant growth
  rate, so this is a genuine 3-parameter study. The paper's full model is 4-D;
  `simulator.py` threads `delta` through so that extension does not need a
  rewrite.
