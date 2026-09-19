# Flagel, Brandvain & Schrider (2019) — The Unreasonable Effectiveness of CNNs in Population Genetic Inference

**Citation:** Flagel, L., Brandvain, Y. & Schrider, D.R. (2019). The
Unreasonable Effectiveness of Convolutional Neural Networks in Population
Genetic Inference. *Molecular Biology and Evolution* 36(2): 220–238.
DOI: [10.1093/molbev/msy224](https://doi.org/10.1093/molbev/msy224)

## What it does

Represents DNA sequence alignments **as images** (rows = individuals/haplotypes,
columns = segregating sites) and runs 1D and 2D CNNs directly over that image to
infer population-genetic parameters and classify evolutionary scenarios —
including the **population mutation rate θ = 4Nμ**, population recombination
rate ρ = 4Nr, demographic parameters, introgression, and selective sweeps.

## Architecture

Stacked Conv → pooling layers, a secondary branch feeding in positional
information about segregating sites, then fully-connected layers to the output
heads. Kernel size/depth varied per task.

## Key numbers

| Task | CNN | Baseline method | Baseline result |
|---|---|---|---|
| Introgression detection (accuracy) | 88.5% (CI 87.7–89.2) | FILET | 82.5% (CI 81.7–83.4) |
| Recombination rate (R², RMSE) | 0.86, 0.011 | LDhat | 0.77, 0.016 |
| Selective sweep detection (accuracy) | 60.6% (CI 58.8–62.3) | S/HIC | 58.5% (CI 56.7–60.2) |
| Autotetraploid recombination (R², RMSE) | 0.83, 0.012 | — (no existing likelihood method) | — |

## Relevance to our project — the contrast case

This is the paper that makes CNN look good in a directly related domain
(mutation/recombination-rate estimation), and it's important to be precise
about *why*: the CNN's input here is a genotype **matrix** — rows of
individuals, columns of sites — with real, exploitable 2-D structure (nearby
sites are correlated by linkage, nearby individuals by relatedness). That is
exactly the kind of structured input a convolutional kernel is built for.

Our surrogate's input is the opposite case: a length-3/4 vector of physically
distinct scalars `(log10 p1, log10 p2, tau, [log10 p_eff])` with no spatial
adjacency for a kernel to exploit — p2 is not "near" tau in any sense a
convolution can use. So Flagel et al. is cited not as "CNNs work for
mutation-rate problems, therefore ours should too," but as the calibration
case showing what a *genuine* structural match looks like, which our problem
does not have. Comparing our result against theirs is exactly what motivates
treating our own CNN/RNN/LSTM result honestly rather than assuming it should
work here just because it worked there.
