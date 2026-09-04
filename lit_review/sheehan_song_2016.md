# Sheehan & Song (2016) — Deep Learning for Population Genetic Inference

**Citation:** Sheehan, S. & Song, Y.S. (2016). Deep Learning for Population
Genetic Inference. *PLOS Computational Biology* 12(3): e1004845.
DOI: [10.1371/journal.pcbi.1004845](https://doi.org/10.1371/journal.pcbi.1004845)

## What it does

Likelihood-free framework using a **feedforward neural network** (3 hidden
layers, 25/25/10 nodes; deeper 6-layer variants also tested) to jointly infer
population size history and natural selection from genomic summary statistics.
Applied to 197 African *Drosophila melanogaster* genomes. Trained with
autoencoder pretraining, which mattered a lot: random init gave 74.8%
misclassification vs. 6.1% with autoencoder init.

## Architecture

Plain FFN on hand-computed summary statistics of the genomic data — logistic
activations on hidden layers, linear output head for the continuous
(population-size) targets, softmax head for the discrete (selection-class)
target. No CNN, no RNN — this predates that shift in the field (see
`flagel_brandvain_schrider_2019.md` for the CNN follow-on three years later).

## Key numbers (their Table/comparison, reconstructed from the paper)

| Metric | Deep Learning (FFN) | ABCtoolbox |
|---|---|---|
| N1 (recent pop size) accuracy | Superior | Inferior |
| N2, N3 accuracy | Comparable | Comparable |
| Handling correlated summary stats | Robust | Struggles |
| Test-time runtime | "Instantaneous" | Weeks |

Selection-class classification accuracy: 93.8% (6.2% misclassification) on
simulated validation data.

## Relevance to our project

This is the closest historical analogue to what we're doing: a plain
heteroscedastic-free FFN trained on simulator output to replace an
ABC-style inference step, in the same broad domain (population/evolutionary
genetics parameter estimation) our mutation-rate model sits in. It's the
paper that establishes FFN-on-tabular-summary-statistics as the *default*,
not CNN/RNN — those only enter the picture once the raw data itself has
spatial or sequential structure the network can exploit (see
`radev_et_al_bayesflow_2020.md` for the explicit principle, and
`flagel_brandvain_schrider_2019.md` for the case where that structure exists
and CNN earns its place).
