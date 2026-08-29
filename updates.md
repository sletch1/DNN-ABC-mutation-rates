# Project Update: 3-D Model Mismatch and Path Forward

**Date:** 2026-08-17
**Status:** Blocking — the 3-D study must be rebuilt before the manuscript can proceed
**Reference:** Lu, R., Zhu, H., Wu, X. (2023). *Estimating mutation rates in a Markov branching process using approximate Bayesian computation.* Journal of Theoretical Biology 565:111467.

> **Bottom line**
> 1. We built a surrogate for $(p, a, \delta)$. The paper's multi-parameter model is $(p_1, p_2, \tau, \delta)$ — a two-stage mutation rate. Different models, not different parameterizations.
> 2. Independently, the third axis of our own model ($a$) is analytically non-identifiable. Our 3-D study is both the wrong model *and* effectively 2-D.
> 3. Both problems resolve the same way: adopt the paper's two-stage model. Doing so also gives us a **stronger** contribution than the one we currently claim.

---

## 1. The problem

Our 3-D extension estimates $(p, a, \delta)$: constant mutation probability, wild-type division rate, mutant relative growth rate. The JTB paper's multi-parameter model is $(p_1, p_2, \tau, \delta)$: a **piecewise-constant mutation rate** with differential growth. We built a surrogate for a model the paper does not use, and skipped the model that is the paper's actual contribution.

Separately — and this would matter even if the paper *had* used $(p,a,\delta)$ — the $a$ axis of our design carries exactly zero information (§3).

---

## 2. What the paper does

### 2.1 The two-stage model

The mutation rate is a step function of time:

$$p(t) = p_1 \mathbf{1}_{\{0 < t \le \tau\}} + p_2 \mathbf{1}_{\{\tau < t \le t_p\}}$$

Non-mutant lifetimes are $\exp(a)$, mutant lifetimes $\exp(\delta a)$. All four parameters $(p_1, p_2, \tau, \delta)$ are estimated **jointly**.

The motivation is experimental: mutagenesis of *E. coli* under sub-inhibitory antibiotic stress (Thi et al., 2011) includes a cell-recovery step before plating, producing two genuinely distinct mutation regimes. That is the sense in which the model "makes more sense in practice."

**The division rate $a$ is never a parameter in the paper** — fixed at 1 in Figures 1–2, both simulation studies, and the real-data analysis.

### 2.2 Side by side

| | JTB paper (Study 2) | Our repo (`DNN_Prototypes/3D/`) |
|---|---|---|
| Mutation rate | Piecewise constant, $p_1 \to p_2$ at $\tau$ | Constant $p$ |
| Free parameters | $p_1, p_2, \tau, \delta$ (4, all inferred) | $p, a, \delta$ (3; only $p$ inferred) |
| Division rate $a$ | Fixed at 1 | Varied over $[0.5,2]$ — **no effect** (§3) |
| Summary statistic | $\sqrt[4]{X/Z}$ | $\sqrt{X/Z}$ |
| Simulator | Algorithm 3 (exact) / Algorithm 4 extended | Algorithm 2 / 4, single-$p$ |
| Design | Latin hypercube, $S_0 = 1{,}500$ | Full factorial $10\times10\times10$, 10 reps |
| Plating time $t_p$ | Fixed at 20 | Solved per $(p,a)$ so $E[\text{mutants}]=20$ |
| Estimator | GPS-ABC only | MOM, MLE, ABC-MCMC, GP-ABC, DNN-ABC |

### 2.3 The code was never uploaded

Exhaustive search of `MatlabCode/`, `RCode/`, and `DNN_Prototypes/`: **no occurrence of `p1`, `p2`, or `tau` anywhere.** Every simulator is single-$p$ — `mut_bMBP_rev.m`, `funMBP.R`, `simulator.py` all share the signature `(Z0, a, delta, p, tp)`.

Algorithm 3 has no implementation. Algorithm 4 as uploaded is single-$p$; the paper's footnote 5 says it "can be easily extended," but the extension is not in the code. This confirms the professor's recollection.

---

## 3. Second problem: $a$ is a null direction

**Analytically.** Our plating time solves $Z_0(e^{at} - e^{at(1-2p)}) = c = 20$. Substituting $s = at$ gives $e^{s} - e^{s(1-2p)} = c$ — a function of $s$ and $p$ only. So $a \cdot t_p$ depends on $p$ alone. Everything downstream depends on $a$ only through $a t_p$ and $a(t_p - t_m)$:

- $Z \sim \text{geo}(e^{-a t_p})$
- arrival times: $a\,t_m = \log\!\left(u(e^{a t_p}-1)+1\right)$
- clone sizes $\sim \text{geo}(e^{-\delta a(t_p - t_m)})$

The joint law of $(Z, X)$, and hence $\bar d$, is **exactly invariant to $a$**.

**Empirically** (against `DNN_Prototypes/3D/data/slow_data_3D.csv`, 9,998 rows):

| Check | Result |
|---|---|
| $a \cdot t_p$ across the 10 $a$ values, per $p$ | constant to $10^{-5}$ (= `uniroot` tolerance) |
| Var of $\bar d$ across $a$ at fixed $(p,\delta)$ ÷ replicate noise | **1.05** — pure Monte Carlo noise |
| corr(residual after removing $(p,\delta)$, $a$) | **0.005** |
| corr(residual after removing $(p,a)$, $\delta$) | 0.584 — for contrast, real signal |
| `results/tables/region_error.csv` | monotone in $\delta$, no pattern in $a$ |

**Consequence.** The 3-D surrogate is a 2-D $(p,\delta)$ surface with a dummy third input. Every "3-D" claim in the manuscript needs re-examination.

---

## 4. Discrepancies to resolve

| # | Discrepancy | Severity | Resolution |
|---|---|---|---|
| D1 | Model is $(p,a,\delta)$, not $(p_1,p_2,\tau,\delta)$ | **Blocking** | Rebuild around the two-stage model |
| D2 | $a$ analytically non-identifiable in our design | **Blocking** | Drop $a$ as a parameter; fix $a=1$ |
| D3 | Summary statistic $\sqrt{X/Z}$; paper uses $\sqrt[4]{X/Z}$ in 4-D | High | Switch for 4-D; keep $\sqrt{\cdot}$ for 1-D |
| D4 | Full factorial grid; paper uses Latin hypercube | High | Switch to LHS (also required for 4-D feasibility) |
| D5 | **Paper contradicts itself on design size.** §2.2: "we only need 200 design points for the four-dimensional case." §3.2: $S_0 = 1{,}500$. We cite 200 at `manuscript.tex:250-253`. | High | Cite 1,500 — what was actually run. Confirm (Q4) |
| D6 | We treat $\delta$ as a known covariate; the paper infers it, and it is their **best**-estimated parameter (nRMSE 0.06–0.08) | Medium | Infer $\delta$ jointly |
| D7 | $t_p$ solved per-$(p,a)$ here; fixed at 20 in Study 2 | Medium | Needs a decision (Q2) |
| D8 | `manuscript.tex:682-685` claims "$a$ rescales the entire time axis" as evidence of real interactions | Medium | False — the rescaling cancels by construction |
| D9 | No external validation target | Medium | Reproduce paper Table 4 (two fully-specified cases) |

---

## 5. Why the two-stage model is the better target

The paper reports that **$p_1$ and $\tau$ are poorly identified**, and says so plainly: "the posterior distributions of $p_1$ and $\tau$ exhibit multimodal shape. This is often caused by lack of identifiability of the parameters."

Study 2, Case 1 (truth: $p_1 = 10^{-9}$, $p_2 = 10^{-8}$, $\tau = 14$, $\delta = 0.8$):

| | $p_1$ | $p_2$ | $\tau$ | $\delta$ |
|---|---|---|---|---|
| Posterior mean | $1.06\times10^{-9}$ | $7.82\times10^{-9}$ | 11.66 | 0.87 |
| Posterior mode | $3.40\times10^{-10}$ | $6.86\times10^{-9}$, $1.43\times10^{-8}$ | 0.14, 16.38 | 0.61 |
| nRMSE | 0.12 | 0.31 | 0.20 | 0.08 |
| 95% CI | $[1.33\times10^{-11},\, 7.80\times10^{-8}]$ | $[9.91\times10^{-10},\, 2.50\times10^{-8}]$ | $[4.79,\, 17.75]$ | $[0.52,\, 1.76]$ |

The $p_1$ interval spans essentially the whole prior domain $[10^{-11}, 10^{-7}]$. The posterior means look respectable only because the prior is centered near truth; the modes are far off. The *joint* 95% HPD set does achieve 0.95 coverage, so the failure is in the marginals, not the posterior as a whole.

There is an intrinsic reason: mutations arising before $\tau$ are exponentially rare, because few cells exist that early, so $p_1$'s contribution is swamped by $p_2$'s. Consistent with this, Case 2 (larger $p_1$–$p_2$ gap) estimates better than Case 1.

**The opening.** The paper's §4(d) names the limitation itself: "when there are too many parameters to estimate, it may still suffer from the curse of dimensionality. The high dimension of the parameter space brings challenges in training the GP surrogate." A DNN surrogate trained on $10^4$–$10^5$ LHS points instead of 1,500 answers a limitation the authors flagged, on a problem with a known failure mode and published numbers to beat.

This also sharpens the scaling argument we already make: a $1{,}500^3$ Cholesky is ~420× the work of $200^3$, so the GP cost we compete against is far worse than the manuscript currently claims.

---

## 6. What we can do now

Unblocked, in dependency order:

1. **Algorithm 3** (exact two-stage) in `funMBP.R` and `simulator.py`. Small change: draw the offspring indicator from `Bern(p1)` if the parent's accumulated lifetime $T \le \tau$, `Bern(p2)` if $T > \tau$, 1 if the parent is already a mutant.
2. **Two-stage Algorithm 4** (fast). Arrival times already come from $F(t) = (e^{at}-1)/(e^{at_p}-1)$; split the mutation count at $\tau$ by drawing $M_1, M_2$ from the two CDF pieces. This is the simulator Study 2 actually uses.
3. **Summary statistic → $\sqrt[4]{X/Z}$** for the 4-D study; keep $\sqrt{X/Z}$ in 1-D.
4. **Grid → Latin hypercube** over $[-11,-7]^2 \times [0.1, 19.9] \times [\log_{10}0.5, \log_{10}2]$.
5. **Generate ground truth with the fast simulator** — the long pole, start as soon as (2) is verified. Exact simulation at $p \sim 10^{-9}$, $t_p = 20$ is infeasible; we already exhausted 62 GB at small $p$, and the paper uses Algorithm 4 throughout Study 2 for the same reason.
6. **Reproduce paper Table 4**, both cases — external validation the current 3-D study has no equivalent of.
7. **Add $p_1$/$\tau$ marginal identifiability as an explicit outcome measure** alongside MSE: posterior modality, CI width relative to prior width, marginal coverage.
8. **Fix the manuscript**: `manuscript.tex:250-253` (200 → 1,500); rewrite `sec:sim3D` once the new study exists; re-examine the "two correctness issues" paragraph at `manuscript.tex:656-667`, since the claim about an inverted $a$ convention is moot if $a$ has no effect either way.

**Reusable:** residual-MLP architecture, heteroscedastic heads, calibration machinery, ABC-MCMC driver, GP-scaling benchmark, timing harness.
**Discarded:** the 3-D ground-truth CSV and the trained $(p,a,\delta)$ surrogate.

---

## 7. What I need from the professor

**Q1 — Does the original two-stage code still exist?** Re-deriving Algorithm 3 is straightforward, but matching Study 2's exact configuration (refinement schedule, LHS seeding, proposal tuning) is far easier with the original. Even a partial version helps.

**Q2 — $t_p$ fixed at 20, or solved per-parameter?** Study 2 fixes it; our code solves it so $E[\text{mutants}] = 20$. Different experiments, not comparable. I would default to fixing $t_p = 20$ to match Table 4.

**Q3 — $\tau$ in absolute time, or as a fraction of $t_p$?** §3.2 notes the relative parameterization as an alternative but uses absolute. Relative would likely improve identifiability — exactly the weakness we would target — but changes the prior and breaks direct comparability with Table 4. Worth it, or stay comparable?

**Q4 — Confirm the 200 vs. 1,500 discrepancy (D5).** Was the 200 an early estimate the experiments superseded? I plan to cite 1,500.

**Q5 — Is sharpening the $p_1$/$\tau$ marginals the right target?** My read is yes (§5). If he thinks the multimodality is intrinsic to the model rather than an artifact of a 1,500-point GP, we should know before committing — though a documented negative result would still be worth having.

**Q6 — Manuscript scope.** (a) Replace the 3-D study with the 4-D two-stage study; (b) keep both, reframing $(p,a,\delta)$ with the $a$-invariance as a validation result rather than a study; (c) something else. I lean (a), but his call.

**Suggested first meeting:** Q1–Q3 (unblocks implementation), then Q5–Q6 (sets scope). Q4 is a two-minute confirmation and could go by email beforehand.

---

## 8. Manuscript impact

| Location | Issue | Action |
|---|---|---|
| `manuscript.tex:250-253` | Cites "about 200 points" for the 4-D GP design | Change to 1,500 |
| `manuscript.tex:627+` (`sec:sim3D`) | Entire 3-D study | Rewrite around the two-stage model |
| `manuscript.tex:631-632` | "genuine three-parameter regime $(p,a,\delta)$" | Not genuine — effectively 2-D |
| `manuscript.tex:656-667` | "Two correctness issues," incl. inverted $a$ convention | Re-examine; $a$ has no effect either way |
| `manuscript.tex:682-685` | "$a$ rescales the entire time axis" | Rescaling cancels; claim is false |
| `manuscript.tex:772-780` | Region-error analysis over the $(a,\delta)$ box | Error is $\delta$-driven only |
| 3-D tables and figures | Computed on $(p,a,\delta)$ ground truth | Regenerate |

The 1-D study (`DNN_Prototypes/1D/`) is unaffected — it matches the paper's Study 1 and its constant-rate assumptions hold.

---

## 9. Reference: Study 2 configuration

| Setting | Value |
|---|---|
| True values | $p_1 = 10^{-9}$, $\tau = 14$, $\delta = 0.8$; Case 1: $p_2 = 10^{-8}$; Case 2: $p_2 = 5\times10^{-8}$ |
| $t_p$ / $J$ / $Z_0$ / $a$ | 20 (fixed) / 100 / 1 / 1 (fixed) |
| Simulator | Algorithm 4 (fast) |
| Estimator | GPS-ABC only (ABC-MCMC too expensive at this scale) |
| Parameter space | $[-11,-7] \times [-11,-7] \times [0.1, 19.9] \times [\log_{10}0.5, \log_{10}2]$ |
| Priors | $\theta_1, \theta_2 \sim TN(\log_{10}\hat p_{MOM}, 20, [-11,-7])$; $\tau \sim TN(10, 20, [0.1, 19.9])$; $\log_{10}\delta \sim TN(0, 20, [\log_{10}0.5, \log_{10}2])$ |
| Design | $S_0 = 1{,}500$ LHS; $\Delta = 10$ added when decision error exceeds $\xi = 0.3$ |
| Tolerance | $\epsilon = 10^{-8}$ |
| Proposal sds | $(0.8,\, 3.5,\, 3.5,\, 4)$, tuned for 20–40% acceptance |
| MCMC | 20,000 samples, 5,000 burn-in, 100 replications |

**Real data (§3.3)** additionally compares M1 (constant rate, equal growth), M2 (constant rate, differential growth), and M3 (two-stage, differential growth) across 13 *M. tuberculosis* strains, 25 cultures each. Roughly half prefer M3; only two prefer M1 — evidence the two-stage model is not merely a theoretical exercise, and a second validation target.
