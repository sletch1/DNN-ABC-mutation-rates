# Jiang, Wu, Zheng & Wong (2017) — Learning Summary Statistic for ABC via Deep Neural Network

**Citation:** Jiang, B., Wu, T.Y., Zheng, C. & Wong, W.H. (2017). Learning
Summary Statistic for Approximate Bayesian Computation via Deep Neural
Network. *Statistica Sinica* 27: 1595–1618.
DOI: [10.5705/ss.202015.0340](https://doi.org/10.5705/ss.202015.0340)
(arXiv preprint 2015: [1510.02175](https://arxiv.org/abs/1510.02175))

## What it does

One of the earliest papers to replace hand-crafted ABC summary statistics with
a plain feedforward network trained directly on simulated `(data, theta)`
pairs. Key result: under squared-error training, the network's output
converges to (approximately) the **posterior mean** of theta given the data —
so the DNN isn't just a feature extractor, it's already an approximate
point-estimator for the ABC target. Demonstrated on the Ising model and a
moving-average time-series model; competitive with or better than
theoretically-derived summary statistics, with minimal per-model tuning.

## Architecture

Plain FFN. No CNN, no RNN — architecture choice isn't the point of this paper;
the point is that a generic FFN can replace statistician-derived summary
statistics at all.

## Relevance to our project

This is close to the statistical justification for why our own surrogate's
mean head, trained by Gaussian NLL, is a sound way to plug a neural network
into an ABC acceptance step — it's the direct intellectual predecessor of
"replace the ABC summary/likelihood machinery with a trained network" that
GPS-ABC's Gaussian-process version and our FFN/CNN/RNN/LSTM comparison both
build on. Like Sheehan & Song, this is FFN-on-tabular-input, reinforcing that
FFN — not CNN/RNN — is the default architecture family in the ABC
literature whenever the raw simulator output (or its reduction) doesn't carry
spatial or sequential structure.
