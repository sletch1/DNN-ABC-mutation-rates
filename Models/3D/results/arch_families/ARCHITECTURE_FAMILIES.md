# 3-D two-stage surrogate: CNN1D / RNN / LSTM vs the deployed FFN

**Numbers in this file are not maintained by hand.** They are read from the two
generated files beside it, which `benchmark_families.py` and
`run_experiments_families.py` rewrite on every run:

- `benchmark_families.md` — surrogate fit quality per family
- `TABLE_FAMILIES.md`, `mcse_families.md` — downstream ABC recovery and its
  Monte Carlo standard errors

An earlier version of this document restated those numbers in prose. It drifted:
when the summary statistic changed to the paper's fourth root, the generated
tables moved and the restated copies did not, leaving a comparison between two
different targets. Prefer the generated files; this one exists for the argument,
not the arithmetic.

---

## The question

The deployed surrogate (`network/model.py: HeteroscedasticMLP`, hidden=(64,32),
GELU → tanh) is a plain MLP treating its three inputs — `log10 p1`, `log10 p2`,
`tau` — as an unordered feature vector. Does imposing *structure* on those
inputs help? Three families that do so are tested: a 1-D CNN (treats the inputs
as adjacent positions a kernel slides across) and a GRU/LSTM (treats them as
sequential timesteps with a carried hidden state).

## The a priori answer, which the measurement then confirms

It should not help, and `network/model_families.py` gives the full argument.
`(log10 p1, log10 p2, tau)` has no spatial or temporal relationship between
neighbouring entries — the "order" is an arbitrary storage convention, not a
property of the physics. A convolution's weight sharing and a recurrence's
sequential bottleneck are both *constraints*; they pay off only when the
constraint matches real structure in the data. Here there is none to match, so
the expected outcome is neutral-to-harmful rather than helpful.

There is also a concrete optimisation cost for the recurrent families: an MLP
sees all three inputs simultaneously in every layer, whereas a 3-step recurrence
must route everything relevant about `p1` through two further hidden-state
updates before it can interact with `tau`.

## What was measured

**Surrogate fit** (`benchmark_families.md`): CNN1D ties the deployed FFN — a few
per cent lower MSE, both close to the irreducible noise floor, and inside the
seed-to-seed spread already documented in `benchmark_round2.md`. It costs
roughly 3.6× the query latency for that tie. LSTM and RNN are both measurably
worse.

**Downstream ABC** (`TABLE_FAMILIES.md` + `mcse_families.md`): comparing each new
family against GPS-ABC and DNN-ABC at the 2-MCSE level over all
3 families × 3 truths × 3 parameters × 2 baselines = 54 comparisons,
**49 are statistical ties**. Every one of the five resolved comparisons favours
GPS-ABC; none favours a new family anywhere. LSTM-ABC is tied in all eighteen of
its own comparisons.

Interval calibration adds nothing either way: pooled coverage sits between 0.95
and 0.98 for the four neural families against a nominal 0.95, a spread of about
three percentage points, which at sixteen replicates is at most one cell per
parameter. Every method over-covers on `tau` at exactly 1.000, which reflects
that parameter's weak identifiability rather than any property of an
architecture. An earlier version of this document read a calibration story into
that column; the corrected numbers do not support one.

## Conclusion

Neither a spatial prior (CNN1D) nor a temporal one (RNN, LSTM) on three
physically unordered parameters improves downstream inference over the
architecture-agnostic MLP. Two of the three cost real fit quality; the third
buys a statistical tie at several times the query cost. This is the outcome the
structural argument predicted, and it is reported as a negative result.

It is also consistent with the study's broader finding: on this surface the
surrogate is not the binding constraint, so changing its architecture — like
improving its accuracy — does not move the answer. What binds is the
identifiability of `(p1, p2, tau)` from a single scalar summary.
