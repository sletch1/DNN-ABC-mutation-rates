# Åkesson, Singh, Wrede & Hellander (2021) — CNNs as Summary Statistics for ABC

**Citation:** Åkesson, M., Singh, P., Wrede, F. & Hellander, A. (2021).
Convolutional Neural Networks as Summary Statistics for Approximate Bayesian
Computation. *IEEE/ACM Transactions on Computational Biology and
Bioinformatics*. DOI: 10.1109/TCBB.2021.3108695.
(arXiv preprint 2020: [2001.11760](https://arxiv.org/abs/2001.11760))

## What it does

Proposes CNNs as a *learned* summary-statistic extractor for ABC in systems
biology (stochastic gene regulatory network models), replacing the manual
selection of summary statistics from a large hand-built candidate pool.
Evaluated against other deep-network architectures and partially-exchangeable
network designs on benchmark problems ranging from small-scale to a
high-dimensional genetic oscillator model.

## Architecture comparison (what the paper itself tests)

Per the search abstract/summary: the CNN is benchmarked against "state-of-the-art
deep neural network and partially exchangeable network architectures." The
convolutional design **outperforms the alternatives specifically on
large-scale, high-dimensional stochastic biochemical reaction network
problems**; on small-scale problems, all three architecture families perform
comparably (i.e., architecture stops mattering once the problem is easy enough
relative to the networks' capacity — directly analogous to what we found for
our own 3-D architecture search once every candidate sat within ~15% of the
irreducible noise floor, see `../DNN_Models/3D/results/logs/benchmark_arch.md`).

## Relevance to our project

Two things carry over directly:

1. Like Flagel et al., the CNN's advantage here is tied to the *raw
   trajectory/reaction-network data* having structure worth convolving over —
   again, not our situation (a length-3/4 tuple of unordered scalars).
2. The "architecture stops mattering on easy problems" finding is an almost
   exact precedent for our own 3-D architecture-search result: once a
   surface is within ~10-15% of its own irreducible noise floor, every
   reasonable architecture family lands in the same place, and the deciding
   factor becomes something other than raw fit quality (there: computational
   cost; here: whether the architecture's assumptions match the data at all).
