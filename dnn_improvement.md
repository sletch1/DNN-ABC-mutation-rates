# NN Surrogate: Where It Can Improve on GPS-ABC

Reference paper: Lu, Zhu & Wu (2023), *"Estimating mutation rates in a Markov branching
process using approximate Bayesian computation,"* J. Theoretical Biology 565:111467.
(`MutationRates_ABC_paper.pdf` in this folder.)

## Background

The paper proposes **GPS-ABC**: instead of calling the expensive Markov branching
process (MBP) simulator inside every ABC-MCMC iteration, train a Gaussian Process (GP)
regression model on a small set of simulator outputs, then substitute the GP's
prediction for the simulator during MCMC sampling.

Two simulators exist in the codebase (`RCode/funMBP.R`):

- **Algorithm 2 / `mut_bMBP_slow`** — the exact, literal cell-by-cell simulation.
  Slow (0.07s to 781s per call, per the paper), used as ground truth.
- **Algorithm 4 / `mut_bMBP_fast`** — a fast approximate simulator using known
  distributional shortcuts (Yule process results). Used for cheap bulk data generation.

**MCMC vs. simulator — these are not the same thing.** ABC-MCMC is the *inference
algorithm* (a Metropolis–Hastings sampling loop). The simulator (slow or fast) is
what that loop calls at every iteration to generate comparison data. GPS-ABC keeps
the same MCMC loop but swaps the simulator call for a pre-trained GP prediction.

Our project trains a **neural network** as the surrogate instead of a GP, and extends
the parameter space beyond the paper's own scope (adding division rate `a` and mutant
relative growth rate `delta` as inputs, not just mutation probability `p`).

## Evidence from the paper

### Speed (Table 3, p.8) — seconds per 100 MCMC iterations

| p | J | ABC-MCMC | GPS-ABC (excl. training) | GPS-ABC (incl. training) |
|---|---|---|---|---|
| 10⁻⁴ | 10 | 246.05s | 2.75s | 67.76s |
| 10⁻⁴ | 50 | 767.96s | 1.56s | 200.66s |
| 10⁻⁴ | 100 | 1323.73s | 1.37s | 344.02s |
| 10⁻³ | 100 | 675.42s | 1.68s | 177.86s |
| 10⁻² | 100 | 380.13s | 2.57s | 101.83s |

Once trained, GPS-ABC is ~100-1000x faster per iteration than plain ABC-MCMC, because
it skips the simulator call entirely. Even including one-time training cost, GPS-ABC
still wins in almost every row.

### Accuracy (Table 1, p.7) — MSE of mutation rate estimate

On the **fast-simulator-trained** table, GPS-ABC is consistently ≤ ABC-MCMC's MSE.

On the **slow-simulator-trained** table (the more relevant one for us — this is the
expensive ground truth we're generating on stat86), the picture is mixed:

| p | J | ABC-MCMC MSE | GPS-ABC MSE |
|---|---|---|---|
| 10⁻⁴ | 10 | 2.91×10⁻⁹ | 3.65×10⁻⁹ (worse) |
| 10⁻⁴ | 50 | 1.39×10⁻⁹ | 3.07×10⁻⁹ (worse) |
| 10⁻³ | 10 | 2.50×10⁻⁷ | 3.96×10⁻⁷ (worse) |

GPS-ABC sometimes *loses* to plain ABC-MCMC when trained on the slow simulator's data.
The likely cause: a GP fit cost scales `O(n³)`, so the paper could only afford ~50
training points for the 1D case (and ~200 for their 4D piecewise case, via Latin
Hypercube design). Small training sets → weaker GP fit → this accuracy gap.

## Key targets for the NN surrogate

Ranked by strength of evidence / expected payoff.

### 1. Training-set scale (strongest target)
The paper was capped at ~50-200 training points by the GP's cubic fit cost. Our
`slow_data_1D.csv` (1,010 rows) and `slow_data_3D.csv` (10,000 rows) are 20-50x larger.
An NN has no cubic wall on training-set size. This directly attacks the accuracy gap
in Table 1's slow-simulator rows above — testable by rerunning the same MSE comparison
against ABC-MCMC once the NN is trained.

### 2. Inference-time cost as training data grows
Table 3's "excl. training" numbers were measured at the paper's small training-set
sizes. A GP's per-prediction cost grows with training-set size `n` (needs the full
kernel matrix at query time); an NN's forward pass is a fixed cost independent of `n`.
As training data scales up (which we want, per #1), a GP's per-query MCMC cost creeps
up while the NN's stays flat — this is the fair, at-scale version of Table 3 to beat.

### 3. Genuine dimensionality the paper never covered
Our 3D grid (p, a, delta jointly varying) is new — the paper only ever did 1D (p alone)
or a separate 4D piecewise-mutation case (p1, p2, tau, delta), never a joint p/a/delta
surrogate for the constant-rate model. A working, accurate surrogate here is a
contribution independent of speed. It's also where a GP would struggle most — the
paper explicitly cites "curse of dimensionality" for rejection-based ABC and needed
Latin-Hypercube tricks just to make their own GP tractable at 4D.

### 4. Correctness of the ground truth itself
Two bugs were found and fixed in `mut_bMBP_slow` (see `RCode/funMBP.R`) before the 3D
run started:
- `delta` was never used in the exact simulator at all (matches the paper's own
  Algorithm 2, which has no `delta` input — but `genSlowData_3D.R`'s attempt to extend
  to 3D silently inherited this, so `delta` had zero effect on outcomes).
- The `a` parameter's rate convention was inverted relative to how `a` is used
  everywhere else (the `tp`-solving equation, and `mut_bMBP_fast`'s `exp(-a*tp)`).
  Invisible until now because every existing script hardcoded `a=1`.

Both are fixed and empirically verified (growth now matches `exp(a*tp)` theory;
mutant fraction now scales monotonically with `delta`). This means our 3D ground
truth is more correct than a naive extension of the paper's own code would have been
— worth stating explicitly in any writeup, and worth a quick sanity check with the
professors since it's a real extension beyond the published Algorithm 2.

### 5. Uncertainty quantification — a risk to manage, not necessarily an "improvement"
GPS-ABC's MCMC acceptance step (Eqs. 9-10 in the paper) relies on the GP's *native*
predictive variance to control decision error at each iteration. An NN doesn't get
this for free — `RCode/trainNN.R` already reaches for a conformal-prediction interval
as a substitute. Before claiming parity with GPS-ABC, confirm that conformal CI is
actually wired into an equivalent decision-error-controlled MCMC step — otherwise the
surrogate may be fast and accurate on point predictions but weaker on the calibrated
uncertainty that made GPS-ABC's MCMC mixing reliable in the first place.

## Recommended framing

The strongest, most defensible claim is **not** "NN beats GP on the paper's own 1D
table" (diminishing returns — GPS-ABC is already near-instant there). It's:

> Same MSE-vs-ABC-MCMC comparison as Table 1, same seconds/100-iterations comparison
> as Table 3 — but trained on 20-50x more data than a GP could afford, at flat
> (not growing) inference cost, and extended to a 3-parameter regime the paper's GP
> approach never attempted.
