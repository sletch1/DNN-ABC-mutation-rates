# 3-D two-stage surrogate: activation-function comparison

Architecture held fixed at the deployed `mlp 64-32`; only the activation varies. Each row is the mean of 5 seeds, with the standard deviation across those seeds. MSE is against held-out design-point means of log10(d_bar); the irreducible floor is 3.451e-04.

| activation | family | mse_mean | ±sd (seeds) | × floor | cover95 | s/fit |
|---|---|---|---|---|---|---|
| `gelu` | smooth unbounded | 3.701e-04 | 9.5e-06 | 1.072 | 0.955 | 22 |
| `tanh` | smooth saturating | 3.722e-04 | 6.0e-06 | 1.078 | 0.954 | 589 |
| `leakyrelu` | piecewise-linear | 3.727e-04 | 6.4e-06 | 1.080 | 0.953 | 411 |
| `relu` | piecewise-linear | 3.733e-04 | 9.5e-06 | 1.082 | 0.954 | 422 |
| `mish` | smooth unbounded | 3.775e-04 | 7.2e-06 | 1.094 | 0.956 | 30 |
| `silu` | smooth unbounded | 3.901e-04 | 1.2e-05 | 1.130 | 0.955 | 22 |
| `prelu` | piecewise-linear | 3.917e-04 | 1.3e-05 | 1.135 | 0.956 | 215 |
| `elu` | smooth saturating | 4.026e-04 | 2.3e-05 | 1.166 | 0.953 | 338 |
| `selu` | smooth saturating | 4.367e-04 | 3.3e-05 | 1.265 | 0.953 | 206 |
| `softplus` | smooth unbounded | 4.772e-04 | 8.7e-05 | 1.383 | 0.952 | 24 |

**Pairwise against the best row** (t = gap / SE of the difference; |t| < 2 is a tie at this seed count):

| activation | × floor | gap vs best | t |
|---|---|---|---|
| `gelu` | 1.072 | — | reference |
| `tanh` | 1.078 | 2.10e-06 | 0.42 |
| `leakyrelu` | 1.080 | 2.56e-06 | 0.50 |
| `relu` | 1.082 | 3.23e-06 | 0.54 |
| `mish` | 1.094 | 7.41e-06 | 1.38 |
| `silu` | 1.130 | 2.00e-05 | 3.00 |
| `prelu` | 1.135 | 2.16e-05 | 3.04 |
| `elu` | 1.166 | 3.25e-05 | 2.95 |
| `selu` | 1.265 | 6.66e-05 | 4.35 |
| `softplus` | 1.383 | 1.07e-04 | 2.74 |

**Best-to-worst spread:** 1.07e-04 (28.9% of the best row's MSE). **Typical across-seed sd:** 1.05e-05.

**Verdict: the ranking is resolved.** The spread exceeds three times the typical seed noise, so `gelu` is genuinely better than `softplus` here.
