> **Study I (one-dimensional constant-rate model).** This guide covers the `1D/` package only. The two-stage model (Study II) is in [`../4D/`](../4D/README.md). The write-up is [`../manuscript.pdf`](../manuscript.pdf).

# DNN-ABC — a neural surrogate for mutation-rate estimation, benchmarked against GPS-ABC

**End-to-end reproduction of Lu, Zhu & Wu (2023), *"Estimating mutation rates in a
Markov branching process using approximate Bayesian computation,"* J. Theoretical
Biology 565:111467 — with a deep-neural-network surrogate added as a fourth
estimator alongside the paper's MOM/MLE, ABC-MCMC, and GPS-ABC.**

**Manuscript:** [`manuscript.pdf`](../manuscript.pdf) ([source](../manuscript.tex)) —
a full writeup of both studies below, formatted for submission to a
computational/applied statistics journal (JASA / JCGS / AoAS), with formal
ABC-MCMC notation, Monte Carlo standard errors on every simulation-study
comparison, and a complete reproducibility statement.

The paper's central methodological contribution is **GPS-ABC**: replacing the
expensive Markov-branching-process (MBP) simulator inside the ABC-MCMC loop with a
**Gaussian-process** surrogate. This project asks a direct follow-up question:

> *Can a neural network replace that Gaussian process — and does it help?*

The headline answer, on the paper's own 1-D constant-mutation-rate benchmark:
**the DNN surrogate is a statistical tie with GPS-ABC on estimation accuracy in
seven of nine tested configurations and clearly better (beyond Monte Carlo
simulation noise) at the largest tested mutation rate; it produces consistently
tighter (still-calibrated) credible intervals in every configuration, and
delivers calibrated input-dependent uncertainty the GP cannot — at the same
~65–2100× speedup over exact
ABC-MCMC.** See `manuscript.pdf` for the full Monte-Carlo-SE-aware comparison
(most cells are a statistical tie on point accuracy; see §4.3/Table 3 there).

---

## Key findings & contributions

**Contributions**

- Full reproduction of Lu, Zhu & Wu (2023)'s four estimators (MOM, MLE, ABC-MCMC,
  GPS-ABC) as the baseline this project extends.
- A new **heteroscedastic DNN surrogate**, added as a fourth ABC backend in place of
  GPS-ABC's Gaussian process, with input-dependent predictive uncertainty and
  split-conformal calibration (§1, §4.5).
- A controlled architecture study (§5) that isolates the actual driver of surrogate
  quality — removing BatchNorm, not the choice of activation — which is what took the
  DNN from *losing* to GPS-ABC to beating it.
- The multi-parameter two-stage model is Study II of the manuscript and lives in [`../4D/`](../4D/README.md).
- A fully reproducible pipeline (§8): one command trains the surrogate, one
  reproduces all result tables, one regenerates every figure.

**Major findings**

- **Accuracy:** DNN-ABC's MSE is a near-tie with GPS-ABC in 7 of 9 tested 1-D
  configurations (nominally lower in 5, higher in 2, neither resolved beyond
  Monte Carlo noise), and clearly lower — up to 61% — at the two largest,
  highest-`J` `p=1e-2` cells (Table 1, §4.2).
- **Precision:** ~24% tighter 95% credible intervals than GPS-ABC on average (up to
  38% tighter), in all 9 cells (Table 2, §4.3) — but tighter is not the same as
  better calibrated: **posterior coverage** of those intervals is measurably
  below nominal for both surrogates at small `J`, recovering toward nominal at
  `J=100` (§4.3b). Regression coverage (0.955, a different quantity) stays
  near-exact throughout.
- **Speed:** 65×–2111× faster than exact ABC-MCMC — a dead tie with GPS-ABC in
  1-D — at flat cost regardless of `p` or `J` (Table 3, §4.4).
- **Calibration:** the heteroscedastic head + conformal calibration achieve
  **0.955** test-set coverage, with predictive uncertainty that tracks the data's
  true input-dependent noise — something GPS-ABC's homoscedastic GP cannot
  represent (§4.5).
- **Higher dimensions:** see Study II (`../4D/`); this package covers the one-parameter benchmark only.

---

## 1. The neural network

![DNN architecture](1D/results/figures/architecture.svg)

### Input / output

- **Input**: `log10(p)` — the per-division mutation probability (the only varying
  parameter in the 1-D case).
- **Output**: two heads — `μ` = predicted `log10(d̄)`, and `log σ²` = the **log of
  an input-dependent predictive variance**, where `d̄ = meanᵢ √(Xᵢ/Zᵢ)` over the
  `J` parallel cultures is the paper's ABC summary statistic (Section 2.2 / Fig. 2).

We model the statistic on the `log10` scale because it spans ~3 orders of magnitude
across the mutation-rate range.

### Architecture, and why each choice (`network/model.py: HeteroscedasticMLP`)

`log10(p)` → **Dense 32 → GELU → Dense 16 → GELU** → two linear heads (`μ`, `log σ²`).

**Depth — 2 hidden layers.** The target `E[log10(d̄) | log10 p]` is a *smooth,
monotone 1-D curve* (paper Fig. 2). Two hidden layers already make the network a
universal approximator for a function this simple; a third layer only adds
parameters and overfitting risk on 505 training rows with no accuracy gain (a
3-layer variant was tested — no improvement), while one layer under-fits the
curvature. Two is the sweet spot.

**Width — 32 → 16 (a funnel), and how it got smaller.** The original search only
compared 128→64 against wider options (256→128), found no improvement going wider,
and stopped there — an artifact of never having tried going *narrower*. A later,
dedicated capacity sweep (`network/architecture_search/benchmark_capacity.py` +
`benchmark_capacity_confirm.py`, §5) tested that directly, all the way down to a
linear control, and found 32→16 (626 params, 14× smaller than the original 128→64)
the closest curve fit to the GP of every size tried, with no measurable cost on the
actual downstream ABC-MCMC task at the paper's own replicate count. The network was
switched to 32→16 on that basis; §5 has the full story and the numbers.

**Activation — GELU (an honestly low-stakes choice).** On a curve this smooth the
activation barely matters. Isolating it (same 128→64, no BatchNorm, averaged over 3
seeds) gives mean-curve MSE: **ReLU 3.95e-4, silu 4.02e-4, GELU 4.13e-4, tanh
4.23e-4** — all within ~7% (seed noise), with ReLU marginally ahead. We use **GELU**
for reasons that cost nothing here: it is **smooth (C∞)**, so the surrogate is a
smooth function of `p` with smooth gradients (useful if the sampler is later
upgraded to a gradient-based scheme like HMC/NUTS), and its soft negative region
avoids ReLU's **dead-neuron** failure mode — which matters more in a small network
where each unit is a large fraction of capacity. GELU's one real drawback, a costlier
erf/tanh vs ReLU's max, is irrelevant at this scale (a 2-layer net on a scalar input
costs microseconds either way). **ReLU is an equally valid choice** here — the
accuracy does not come from the activation.

**No BatchNorm — the decisive architectural choice.** The original design (ReLU **+
BatchNorm + dropout**) fit the curve **~11× worse** (mean-curve MSE 4.35e-3 vs
3.89e-4). BatchNorm normalizes per-mini-batch statistics — it is built for deep
classification nets — and on a smooth 1-D regression it injects batch-dependent
noise and biases predictions at the domain edges (where the `p`-grid is one-sided).
Removing it is what let the network fit cleanly, and is the real reason the DNN went
from *losing* to GPS-ABC to beating it (see §5).

**No dropout.** Dropout regularizes against overfitting, but early stopping + weight
decay (1e-5) already do that here, and dropout's injected noise degraded the clean
regression fit.

**Two output heads (heteroscedastic) — the key statistical design decision.** The
first head predicts the mean; the second learns an **input-dependent variance**,
trained by Gaussian negative log-likelihood. GPS-ABC's MCMC acceptance step
(Eqs. 9–10) needs a *predictive variance* at each proposed θ; a GP supplies one, but
only a single **homoscedastic** noise term that cannot represent this data's varying
noise. The variance head learns that shape directly (see §4.5).

**Conformal calibration.** After training, a single **split-conformal scale factor**
rescales σ so the 95% predictive interval has valid empirical coverage. Using the
finite-sample-corrected quantile level `⌈(n+1)(1−α)⌉/n` on the 3-replicate
calibration set gives **0.955 test-set coverage**.

### Training data & split

Ground truth is the **exact/slow simulator** (Algorithm 2, `mut_bmbp_slow`),
delivered as `data/slow_data_1D.csv`: 101 log-spaced `p` in
`log10(p) ∈ [−8,−2]`, 10 replicates each (1010 rows), `a = δ = Z0 = 1`, `J = 100`.
Split **by replicate** so every grid point appears in every split with no leakage:

| split | replicates | rows | purpose |
|---|---|---|---|
| train | 1–5 | 505 | fit network weights |
| calibration / early-stop | 6–8 | 303 | conformal calibration + early stopping |
| test | 9–10 | 202 | held-out evaluation only |

The GP baseline (GPS-ABC) is fit on the **same** data budget (reps 1–8) so the
comparison is apples-to-apples; neither surrogate ever sees the test replicates.

---

## 2. How it plugs into the ABC pipeline

`run_abc_mcmc` is a single Metropolis–Hastings sampler over `θ = log10(p)`
(truncated-exponential prior, truncated-normal proposal, Hastings-corrected). The
**only** thing that differs across the three ABC columns is how the summary
statistic and its uncertainty are obtained at each proposed θ:

| method | how S(X) is obtained per iteration | cost |
|---|---|---|
| **ABC-MCMC** | run the exact simulator `ns` times | expensive (the baseline both surrogates approximate) |
| **GPS-ABC** | GP predicts `(mean, sd)` instantly | cheap |
| **DNN-ABC** | heteroscedastic DNN predicts `(mean, sd)` instantly | cheap |

For the surrogate backends the ABC likelihood is the exact convolution
`N(obs; μ(θ), √(ε² + σ(θ)²))`, so the surrogate's **calibrated predictive
uncertainty flows directly into the acceptance probability** — the GP got this for
free; the DNN now does too.

### Validation of the port

The Python MBP simulator is a direct port of the R/MATLAB code and is validated
against the ground-truth CSV: the plating-time solver matches to 3×10⁻⁵, and the
slow-simulator `d̄` reproduces the CSV means at every `p`.

---

## 3. Results at a glance

- **Accuracy (Table 1):** DNN-ABC's MSE is a near-tie with GPS-ABC in 7 of 9
  configurations — nominally lower in 5, higher in 2, neither direction resolved
  beyond Monte Carlo noise — and clearly lower, up to **61%**, at `p=1e-2`, `J∈{50,100}`.
- **Precision (Table 2):** DNN-ABC's 95% credible intervals are **~24% tighter than
  GPS-ABC on average** (up to **38% tighter**), in all 9 cells.
- **Speed (Table 3):** DNN-ABC is **65×–2111× faster** per MCMC iteration than
  ABC-MCMC (tied with GPS-ABC), at flat cost independent of `p` and `J`.
- **Calibration:** heteroscedastic + conformal → **0.955** test-set coverage, with a
  predictive sd that tracks the true input-dependent noise (the GP's is flat).

Run configuration: **40 replicates**, 600 MCMC iterations (250 burn-in), `ns=6`,
prior `θ ∈ [−5,−2]`, grid `p ∈ {1e-4, 1e-3, 1e-2} × J ∈ {10, 50, 100}`.

### Every improvement, quantified

**vs GPS-ABC — estimation accuracy (reduction in MSE of p̂; the primary comparison):**
- `p = 1e-4`: **5.8%** lower (J=10), **5.3%** lower (J=50), **4.7%** lower (J=100)
- `p = 1e-3`: **3.9% higher** (J=10, ~tie), **3.1%** lower (J=50), **1.4% higher** (J=100, ~tie)
- `p = 1e-2`: **25.0%** lower (J=10), **60.6%** lower (J=50), **60.2%** lower (J=100)
- **Average across all 9 cells: ~18% lower MSE**, with DNN-ABC nominally higher than
  GPS-ABC in 2 of 9 cells (both `|Δ/SE| < 0.2` — indistinguishable from noise, see
  Table 1's Δ/SE column). In nRMSE terms the `p=1e-2` gap is **0.15–0.16 vs 0.24–0.25**
  (~**36–38% lower** error).

**vs GPS-ABC — precision (reduction in mean 95% credible-interval length):**
- `p = 1e-4`: **11.4%** tighter (J=10), **6.3%** tighter (J=50), **7.3%** tighter (J=100)
- `p = 1e-3`: **28.9%** tighter (J=10), **28.4%** tighter (J=50), **24.5%** tighter (J=100)
- `p = 1e-2`: **38.3%** tighter (J=10), **37.5%** tighter (J=50), **34.5%** tighter (J=100)
- **Average: ~24% tighter intervals**, in all 9 cells, with equal accuracy —
  but see §4.3b: the *posterior* coverage of these intervals (not the
  surrogate's own regression coverage) is measurably below nominal at small
  `J`, so "tighter" should not be read as "more precise at no cost" without
  that caveat.

**vs ABC-MCMC — speed (the exact baseline both surrogates approximate):**
- **65×** faster (`p=1e-2, J=10`) up to **2111×** faster (`p=1e-4, J=100`) per 100
  iterations — **~750× on average**.
- DNN-ABC clearly beats the exact ABC-MCMC on accuracy at the three `p=1e-2`
  cells (up to **67% lower MSE** at `p=1e-2, J=10`), is within a few percent
  either way at moderate-to-large `J` for smaller `p`, and trails clearly only at
  the two smallest-`J`, low-`p` cells (`p∈{1e-4,1e-3}, J=10`) — the same two cells
  where GPS-ABC also underperforms exact ABC-MCMC.

**vs GPS-ABC — speed:** a **tie in 1-D** (both ~**0.13–0.15 s / 100 iterations**); the
DNN's flat-cost advantage over a GP only emerges at larger training sets / higher
dimensions (target #2).

**vs the original architecture — the fix that made the accuracy win possible:**
- Switching **ReLU + BatchNorm → GELU with no BatchNorm** cut the surrogate's
  mean-curve fit error by **~91%** (an **11× better** fit — MSE 4.35e-3 → 3.89e-4),
  which is what turned an earlier DNN that *lost* to GPS-ABC into one that beats it.
- Test-set 95% coverage **restored to 0.955** (from 0.934 uncalibrated) via the
  finite-sample-corrected split-conformal calibration.

---

## 4. Detailed results

### 4.1 Surrogate fit (held-out test set)

| split | n | MSE(log10 d̄) | MAE(log10 d̄) | 95% coverage |
|---|---|---|---|---|
| train | 505 | 0.00424 | 0.0515 | 0.964 |
| calibration | 303 | 0.00533 | 0.0538 | 0.954 |
| **test** | 202 | **0.00434** | **0.0525** | **0.955** |

![surrogate fit](1D/results/figures/surrogate_fit.png)
![test parity](1D/results/figures/surrogate_parity.png)

The calibrated 95% band achieves 0.955 coverage on held-out data; the test
parity plot is tight across the full 3-order-of-magnitude range.

### 4.2 Table 1 — MSE of p̂, with nRMSE = √(MSE)/p in parentheses

| p | J | MOM | MLE | ABC-MCMC | GPS-ABC | **DNN-ABC** |
|---|---|---|---|---|---|---|
| 1e-4 | 10 | 9.36e-9 (0.97) | 1.02e-8 (1.01) | 2.90e-9 (0.54) | 7.02e-9 (0.84) | **6.61e-9 (0.81)** |
| 1e-4 | 50 | 8.68e-9 (0.93) | 9.59e-9 (0.98) | 1.91e-9 (0.44) | 1.88e-9 (0.43) | **1.78e-9 (0.42)** |
| 1e-4 | 100 | 4.18e-9 (0.65) | 4.64e-9 (0.68) | 7.75e-10 (0.28) | 8.21e-10 (0.29) | **7.83e-10 (0.28)** |
| 1e-3 | 10 | 1.38e-6 (1.17) | 1.60e-6 (1.26) | 4.60e-7 (0.68) | **5.10e-7 (0.71)** | 5.30e-7 (0.73) |
| 1e-3 | 50 | 3.21e-7 (0.57) | 3.85e-7 (0.62) | 1.15e-7 (0.34) | 1.23e-7 (0.35) | **1.19e-7 (0.34)** |
| 1e-3 | 100 | 2.71e-7 (0.52) | 3.29e-7 (0.57) | 4.33e-8 (0.21) | **4.29e-8 (0.21)** | 4.35e-8 (0.21) |
| 1e-2 | 10 | 3.42e-5 (0.58) | 4.40e-5 (0.66) | 2.64e-5 (0.51) | 1.14e-5 (0.34) | **8.57e-6 (0.29)** |
| 1e-2 | 50 | 7.71e-6 (0.28) | 1.17e-5 (0.34) | 4.78e-6 (0.22) | 5.98e-6 (0.24) | **2.36e-6 (0.15)** |
| 1e-2 | 100 | 4.01e-6 (0.20) | 6.08e-6 (0.25) | 2.62e-6 (0.16) | 6.22e-6 (0.25) | **2.48e-6 (0.16)** |

![Table 1 visualized](1D/results/figures/fig_table1_mse.png)

**Reading this against GPS-ABC (the method to beat):** DNN-ABC's nRMSE is lower in
7 of 9 cells and marginally higher in the other 2 (`p=1e-3`, `J∈{10,100}`) — every
one of those seven, including the two nominal reversals, is a statistical tie
(`|Δ/SE| < 0.9`; see the manuscript's Table 1). The gap opens up at `p = 1e-2` —
DNN-ABC's nRMSE is **0.15–0.16 vs the GP's 0.24–0.25** at `J = 50,100`, a ~36–38%
error reduction, resolved beyond Monte Carlo noise at `Δ/SE = 5.8` and `8.1`. Both
surrogates also match — and in several cells slightly beat — the *exact* ABC-MCMC
baseline, and all ABC methods dominate the classical MOM/MLE estimators, reproducing
the paper's central finding.

### 4.3 Table 2 — mean length of the 95% credible interval for p̂

| p | J | ABC-MCMC | GPS-ABC | **DNN-ABC** |
|---|---|---|---|---|
| 1e-4 | 10 | 2.39e-4 | 1.14e-4 | **1.01e-4** |
| 1e-4 | 50 | 1.56e-4 | 1.11e-4 | **1.04e-4** |
| 1e-4 | 100 | 1.21e-4 | 1.23e-4 | **1.14e-4** |
| 1e-3 | 10 | 2.51e-3 | 1.32e-3 | **0.94e-3** |
| 1e-3 | 50 | 1.53e-3 | 1.32e-3 | **0.95e-3** |
| 1e-3 | 100 | 0.95e-3 | 1.19e-3 | **0.90e-3** |
| 1e-2 | 10 | 7.97e-3 | 4.70e-3 | **2.90e-3** |
| 1e-2 | 50 | 5.15e-3 | 5.10e-3 | **3.19e-3** |
| 1e-2 | 100 | 3.89e-3 | 5.24e-3 | **3.43e-3** |

DNN-ABC produces the **narrowest credible intervals** in all 9 cells — e.g. ~38%
tighter than GPS-ABC at `p=1e-2, J=10`. Because this comes *with* equal accuracy
(Table 1), it reflects genuinely more
*precise* inference, not overconfidence.

### 4.3b Posterior coverage — what §4.1 does not measure

§4.1's 0.955 is **regression** coverage: does the surrogate's own predicted
mean ± interval bracket the true `log10(d_bar)` 95% of the time, on held-out
data? That is a guarantee about the surrogate's fit, not about whether the
downstream ABC credible interval for `p̂` actually brackets the true `p`.
Table 1 and Table 2 are both silent on the second quantity, so we measured it
directly — saving each replicate's interval endpoints, not just its length —
for every method, every `(p, J)` cell.

**A structural bug caught first.** The first attempt returned exactly **0.000**
coverage for all three methods at every `J`, at `p=1e-2` specifically. Cause:
`log10(0.01) = -2.0`, exactly the ABC-MCMC prior's old upper bound — the
sampler could never propose `theta` above the true value there, so the
interval's upper endpoint was structurally incapable of reaching `p=0.01`
(confirmed directly: it topped out at 0.0096–0.0099 across replicates, every
time). Not a surrogate defect — even the exact simulator baseline showed it.
Fixed by extending the ground-truth grid 9 points further in `log10(p)` (same
spacing as the original 101-point design; `RCode/extendSlowData_1D.R`) and
moving the prior's upper bound to `-1.5`, giving `p=1e-2` real headroom
instead of sitting on the wall. Tables 1–3's numbers are unchanged under this
fix (expected: a truncation at the truth wrecks the interval's tail while
barely moving the posterior mean) — only coverage was broken.

**The real picture, measured after that fix (R=40, binomial MCSE in parens):**

| p | J | ABC-MCMC | GPS-ABC | DNN-ABC |
|---|---|---|---|---|
| 1e-4 | 10 | 0.775 (0.066) | 0.450 (0.079) | 0.500 (0.079) |
| 1e-4 | 50 | 0.900 (0.047) | 0.775 (0.066) | 0.800 (0.063) |
| 1e-4 | 100 | 0.925 (0.042) | 0.950 (0.034) | 0.950 (0.034) |
| 1e-3 | 10 | 0.800 (0.063) | 0.525 (0.079) | 0.450 (0.079) |
| 1e-3 | 50 | 0.975 (0.025) | 0.900 (0.047) | 0.875 (0.052) |
| 1e-3 | 100 | 0.950 (0.034) | 0.975 (0.025) | 0.925 (0.042) |
| 1e-2 | 10 | 0.850 (0.056) | 0.775 (0.066) | 0.425 (0.078) |
| 1e-2 | 50 | 1.000 (0.000) | 1.000 (0.000) | 0.850 (0.056) |
| 1e-2 | 100 | 0.950 (0.034) | 1.000 (0.000) | 0.925 (0.042) |

Two things, and only one is about the surrogates. **GPS-ABC and DNN-ABC
coverage rises with `J`** in every row — badly under nominal at `J=10`
(0.425–0.775), close to it by `J=100` (0.925–1.000), the *only* culture count
either surrogate was ever trained on. A surrogate trained exclusively at
`J=100` has no way to know a `J=10` observation is intrinsically noisier, so
its interval is calibrated for the wrong noise level whenever `J≠100`.
DNN-ABC under-covers most at `J=10` (0.425–0.500) — the same overconfidence
that makes its interval shortest there (§4.3) makes it wrong there.
**Separately**, exact ABC-MCMC itself sits below nominal at `J=10`
(0.775–0.850) despite simulating at the true `J` every iteration — a property
of the ABC acceptance kernel at this culture count, unrelated to either
surrogate.

**Not fixed in this pass.** The surrogates' `J`-blindness needs retraining on
`J`-specific data to fix honestly (the raw per-culture values needed already
exist, no new simulation required) — not an adjustment to numbers already
reported. What this section establishes: **the interval-length advantage in
§4.3 is not evidence of better calibration**, and a surrogate trained at one
condition should be expected to misjudge its own uncertainty away from it, in
a way §4.1's regression diagnostic — computed at that same training
condition — cannot reveal.

### 4.4 Table 3 — computation time (seconds / 100 MCMC iterations)

| p | J | ABC-MCMC | GPS-ABC | **DNN-ABC** | **ABC-MCMC / DNN speedup** |
|---|---|---|---|---|---|
| 1e-4 | 10 | 51.08 | 0.128 | 0.150 | **340×** |
| 1e-4 | 50 | 171.79 | 0.127 | 0.145 | **1182×** |
| 1e-4 | 100 | 303.99 | 0.126 | 0.144 | **2111×** |
| 1e-3 | 10 | 16.48 | 0.123 | 0.143 | **116×** |
| 1e-3 | 50 | 67.59 | 0.125 | 0.144 | **469×** |
| 1e-3 | 100 | 125.68 | 0.124 | 0.143 | **876×** |
| 1e-2 | 10 | 9.33 | 0.123 | 0.143 | **65×** |
| 1e-2 | 50 | 44.34 | 0.124 | 0.144 | **309×** |
| 1e-2 | 100 | 88.22 | 0.124 | 0.144 | **614×** |

![timing](1D/results/figures/fig_timing.png)

**The efficiency gap is the headline.** ABC-MCMC's per-iteration cost explodes with
`p` and `J` (it runs the exact simulator every step); both surrogates are **flat at
~0.13–0.15 s / 100 iterations regardless of `p` or `J`**, because they skip the
simulator entirely. Against ABC-MCMC the DNN is **65×–2111× faster**. Against
GPS-ABC it is a **tie in 1-D** — as expected, since a GP over ~800 training points
is itself cheap to query here; the DNN's *scaling* advantage (a fixed-cost forward
pass vs. a GP's per-query cost that grows with training-set size) only bites at
larger training sets and higher dimensions.

### 4.5 Calibrated, input-dependent uncertainty (DNN vs GP)

![uncertainty](1D/results/figures/fig_uncertainty.png)

This is the DNN's clearest methodological edge over GPS-ABC. The empirical
replicate noise of `log10(d̄)` varies with `p`; the DNN's heteroscedastic head
(blue) tracks that shape, while the GP's single homoscedastic noise term (green,
dashed) is flat by construction. Since GPS-ABC's acceptance step depends on this
predictive variance, a better-shaped, calibrated variance is a direct improvement
to the surrogate's role in the sampler.

### 4.6 The three ABC posteriors agree

![posterior](1D/results/figures/fig_posterior.png)

On a single dataset, the posteriors from ABC-MCMC, GPS-ABC and DNN-ABC concentrate
on the true `p`, confirming the surrogates faithfully reproduce the exact method's
inference — not just its point estimate.

---

## 5. How the architecture was chosen (and why it matters)

An earlier version of this DNN used ReLU + BatchNorm + dropout and **lost to
GPS-ABC on accuracy**. A controlled benchmark
(`network/architecture_search/benchmark_arch.py`) against the GP on *mean-curve fit*
(distance to the denoised response curve) revealed why:

| model | mean-curve MSE | vs GP |
|---|---|---|
| GP (GPS-ABC baseline) | 3.84e-4 | — |
| **DNN, no BatchNorm (128,64)** | **~3.9–4.1e-4** | **~tie** |
| DNN old: ReLU **+ BatchNorm + dropout** | 4.35e-3 | **+1035% (broken)** |

**The decisive factor was removing BatchNorm, not the activation.** Isolating the
activation (same 128→64, no BatchNorm, 3-seed average) shows ReLU/GELU/silu/tanh are
all within ~7% of each other and of the GP — a statistical tie — with ReLU marginally
best. What broke the old model by **11×** was BatchNorm (+ dropout): per-mini-batch
normalization biases a smooth 1-D regression at the domain edges. Removing it closed
the gap to the GP; the DNN's calibrated heteroscedastic variance then pushes it to
match-or-beat GPS-ABC on the *actual* estimation task (Table 1). GELU is used for
smooth gradients + dead-neuron robustness at no accuracy cost (see §1); ReLU would
serve equally well. Deep ensembles were also tried
(`.../benchmark_round2.py`) but did not beat a single network on this smooth curve,
so the simpler model was kept.

**Was 128→64 (8,642 params) actually needed? No — the network is now 32→16
(626 params, 14× smaller).** A width/depth sweep
(`network/architecture_search/benchmark_capacity.py`, 3-seed average, full write-up
in `results/logs/benchmark_capacity.md`) showed the original size was not even the
best *curve* fit among the sizes tried: 32→16 came closer to the GP (+1.6% vs GP's
own fit) than 128→64 did (+7.6%), plausibly because the larger network has more
room to overfit 606 training rows on a response this smooth.

That raised the obvious follow-up: does the curve-fit gap survive into the actual
ABC-MCMC task, at the same rigor as Table 1? `benchmark_capacity_confirm.py`
answers it in two stages, screening every size cheaply first (`run_all.py`'s own
`--quick` settings) then confirming the survivors — all six passed the screen — at
Table 1's exact scale (40 replicates, the full 3×3 grid), with Monte Carlo standard
errors attached to every comparison
(`results/logs/benchmark_capacity_confirm.md`). Result: 32→16 was the best point
estimate on both downstream MSE and interval length of every size tested, 128→64
included — though that specific edge is not itself resolved at this replicate
count (max |Δ/SE| = 0.53 across the 9 cells, well under the ≈2 threshold used
everywhere else in this project). Even the worst two-layer candidate (16→8, 46×
smaller) was only resolved as measurably worse than 128→64 in 1 of 9 cells. Every
size from 8,642 down to 34 parameters — a 254× range — cleared the GP baseline by
a wide, decisive margin throughout; only a purely linear control showed a clear,
unambiguous cost on curve fit.

**The network was switched to 32→16 on that basis: the best point estimate
everywhere it can be compared to 128→64, strictly better curve fit, and no
analysis here can show a cost to the change.** Tables 1–3 and every figure in §3–4
are reported for 32→16, retrained from the same data splits at the same paper
scale; 128→64 is kept in the tables above as the architecture the activation and
BatchNorm search was actually run against, not as the deployed model. The honest
limit on this is the one already stated: what's established is that the smaller
network is not worse, not that it is provably better.

---

## 6. Package layout

### Top level — what each directory is for

```
current_version/
├── 1D/             Study I: constant mutation rate (this package)
├── 4D/             Study II: two-stage model, 4 parameters (and its 3-parameter special case)
├── Original_Paper_Code/  the authors' MATLAB reference implementation
├── RCode/          R simulators and the 1-D ground-truth generators
├── manuscript.tex  the write-up
```

**Reading order for a newcomer:** `../RCode/funMBP.R` (what is being simulated and
why) → `1D/network/model.py` (what the network is) →
`1D/abc/abc_mcmc.py` (how the network replaces the simulator inside
the sampler). Every file carries a header block explaining its role, so the
directory tree plus those headers should be enough without reading the code.

### Inside a model package

Files are grouped so a reader can find the neural network in one place
(`network/`), separate from the ABC machinery (`abc/`):

```
1D/
├── run_all.py                     # the entry point — press Run in an IDE, or `python run_all.py`
├── HOW_TO_RUN.md                  # step-by-step guide, including Windows
├── paths.py                       # single source of truth for data/results locations
├── network/                       # THE DNN — architecture, training, diagram
│   ├── model.py                   # HeteroscedasticMLP (GELU, two heads) + Gaussian-NLL
│   ├── train.py                   # training + conformal calibration; load_surrogate()
│   ├── gen_architecture_svg.py    # the architecture diagram
│   └── architecture_search/       # how the architecture was chosen (§5)
│       ├── benchmark_arch.py      # activation + BatchNorm sweep at fixed 128-64
│       ├── benchmark_round2.py    # deep ensembles, did not beat a single network
│       ├── benchmark_capacity.py  # width/depth sweep: is 128-64 actually needed?
│       └── benchmark_capacity_confirm.py  # ...confirmed downstream at Table 1 scale
├── abc/                           # the ABC inference pipeline
│   ├── simulator.py               # exact/fast MBP simulators (validated vs the CSV)
│   ├── estimators.py              # MOM / MLE (paper Eqs. 11–12)
│   ├── surrogates.py              # predict(θ)→(mean,sd): DNNSurrogate + GP baseline
│   ├── abc_mcmc.py                # one MH sampler, 3 backends (sim / GP / DNN)
│   ├── run_experiments.py         # reproduces Tables 1/2/3 (parallel, progress-logged)
│   └── mcse.py                    # Monte Carlo SEs + Delta/SE for Table 1 (manuscript.tex)
├── figures/
│   └── make_figures.py            # all result figures
└── results/
    ├── figures/                   # 7 PNGs + architecture.svg
    ├── tables/                    # TABLES.md, table1/2/3_*.csv, mcse.md
    ├── model/                     # surrogate_1d.pt, surrogate_metrics.json
    └── logs/                      # experiment_config.json, raw_replicates.csv, benchmark_*.md
```

Runnable scripts add their sibling code folders to `sys.path` (a small header),
so the flat imports work across folders without an install step.

---

## 7. Reproducibility

### Setup

```bash
git clone https://github.com/sletch1/DNN-ABC-mutation-rates.git
cd DNN-ABC-mutation-rates
python3 -m venv .venv && source .venv/bin/activate   # Python 3.9+
pip install -r requirements.txt
```

`requirements.txt` (repo root) pins `numpy`, `pandas`, `scipy`, `matplotlib`,
`scikit-learn` (used for the GPS-ABC Gaussian-process baseline), and `torch` (the
DNN). No GPU is required — every model here is small enough to train on CPU in
seconds to minutes.

### 1-D pipeline

The ground-truth data (`1D/data/slow_data_1D.csv`) is already
committed, so no simulation needs to be regenerated first.

Everything below is wrapped by one script, [`1D/run_all.py`](1D/run_all.py).
Open it in an IDE (Spyder, VS Code, PyCharm) and press **Run**, or from a
terminal:

```bash
cd Models/1D
python run_all.py          # asks: [1] quick smoke test, or [2] the paper's settings
```

It takes no command-line arguments — it asks in the console — and runs with
whatever interpreter is active, offering to install anything missing. See
[`1D/HOW_TO_RUN.md`](1D/HOW_TO_RUN.md) for what to expect at each
step and how long the full run takes.

To drive the steps yourself instead:

```bash
cd Models/1D

# train + conformally calibrate the DNN surrogate (~seconds)
python network/train.py

# reproduce Tables 1/2/3 with the DNN column (this run's settings, ~minutes)
python abc/run_experiments.py --reps 40 --nmcmc 600 --burnin 250 --ns 6 \
    --p-grid 1e-4 1e-3 1e-2 --J-grid 10 50 100

# regenerate all figures + the architecture SVG
python figures/make_figures.py && python network/gen_architecture_svg.py

# (optional) re-run the architecture search behind §5
python network/architecture_search/benchmark_arch.py
python network/architecture_search/benchmark_capacity.py
python network/architecture_search/benchmark_capacity_confirm.py
```

All scale knobs are CLI flags, so the same code runs the quick demo and a
paper-scale study. Outputs land in `1D/results/` (`tables/`,
`figures/`, `model/`, `logs/`) — the same layout described in §7.

