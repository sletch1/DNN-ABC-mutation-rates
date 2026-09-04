# Radev, Mertens, Voss, Ardizzone & Köthe (2020) — BayesFlow

**Citation:** Radev, S.T., Mertens, U.K., Voss, A., Ardizzone, L. & Köthe, U.
(2020). BayesFlow: Learning Complex Stochastic Models with Invertible Neural
Networks. arXiv:[2003.06281](https://arxiv.org/abs/2003.06281).

## What it does

Proposes globally amortized Bayesian inference: a **summary network** learns a
fixed-size embedding of arbitrary-size observed data, feeding a **conditional
invertible neural network** (normalizing flow) that maps that embedding to the
posterior over model parameters. Applied to intractable models from population
dynamics, cognitive science, epidemiology, and ecology. Trained entirely from
simulation, no real data needed during training — inference on new observed
datasets is then a single forward pass (no per-dataset optimization loop),
which is what "globally amortized" means as opposed to case-based methods like
ABC-MCMC or GPS-ABC that re-run inference from scratch for every dataset.

## The passage this whole architecture comparison is built on (§2.4, "Summary Network," quoted directly from the PDF)

> "The architecture of the summary network should be aligned with the
> probabilistic symmetry of the observed data. An obvious choice for time
> series-data is an LSTM-network, since recurrent networks can naturally deal
> with long sequences of variable size. Another choice might be a 1D fully
> convolutional network, which has already been applied in the context of
> likelihood-free inference. A different architecture is needed when dealing
> with i.i.d. samples of variable size. Such data are often referred to as
> exchangeable, or permutation invariant... [encoded via] a permutation
> invariant function through an equivariant non-linear transformation followed
> by a pooling operator."

This is an explicit statement of the design principle: **architecture choice
should be dictated by the symmetry/structure of the raw data**, not chosen a
la carte. Three cases are named — sequential/time-series (LSTM or 1D-CNN),
i.i.d./exchangeable (permutation-invariant pooling network) — with a plain
feedforward network implicit as the remaining default when neither symmetry
holds.

## Relevance to our project

Our surrogate's input, `(log10 p1, log10 p2, tau, [log10 p_eff])`, is neither
of Radev et al.'s two named cases: it is not a time series (there's no
temporal ordering to `p1` vs. `p2` vs. `tau`; swapping their order in an
input vector changes nothing about the physical model, but *would* change an
RNN/LSTM's or CNN's output, since those architectures are not permutation- or
reordering-invariant), and the four values are not i.i.d. exchangeable samples
either (they are four different physical quantities, not four draws from a
common distribution). By BayesFlow's own stated principle, none of CNN, RNN,
or LSTM is the architecture this input's symmetry calls for — a plain
feedforward network is. This is the direct citation for treating our
CNN/RNN/LSTM comparison (in the 3-D study) as a deliberately structure-mismatched
control rather than a genuine candidate for improvement, and for not building
them at all in the 1-D study, where the input is a single scalar and the
mismatch becomes outright degeneracy (see `../DNN_Models/3D/results/arch_families/`
for the empirical result once the comparison finishes running).
