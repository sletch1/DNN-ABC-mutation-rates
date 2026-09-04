# 3-D two-stage surrogate: new architecture families vs the deployed FFN

Data: `slow_data_3D.csv`, split by replicate (train 1-5 / val 6-8 / test 9-10), identical to `train.py` / `benchmark_arch.py`. New families averaged over 2 seeds; the FFN row is **not retrained** here -- it is the literal already-deployed/already-reported number (`results/logs/benchmark_round2.md`, `results/model/surrogate_metrics.json`).

Irreducible floor on `mse_mean` = **1.393e-03** (max achievable R^2 = 0.99528).

| architecture | params | mse_mean | x floor | R^2 | cover95 | us/query |
|---|---|---|---|---|---|---|
| CNN1D (16,16 ch, k=3) | 2994 | 1.567e-03 | 1.13 | 0.99469 | 0.949 | 149 |
| FFN 64-32 gelu  [DEPLOYED] | 2466 | 1.573e-03 | 1.13 | 0.99467 | 0.952 | 41 |
| RNN (GRU, hidden=32) | 3426 | 1.726e-03 | 1.24 | 0.99415 | 0.953 | 77 |
| LSTM (hidden=24) | 2642 | 1.819e-03 | 1.31 | 0.99383 | 0.953 | 77 |

**Best by mse_mean: `CNN1D (16,16 ch, k=3)`** (1.567e-03 = 1.13x floor, R^2 = 0.99469, cover95 = 0.949).

- `CNN1D (16,16 ch, k=3)` vs deployed FFN: mse_mean -0.4% (0.4% better), both within 1.13x of the noise floor.
- `RNN (GRU, hidden=32)` vs deployed FFN: mse_mean +9.7% (9.7% worse), both within 1.24x of the noise floor.
- `LSTM (hidden=24)` vs deployed FFN: mse_mean +15.6% (15.6% worse), both within 1.31x of the noise floor.

See `network/model_families.py` for why none of the three new families were expected a priori to beat the plain MLP here: the input has no real spatial/temporal structure for a convolution or recurrence to exploit, only an arbitrary storage order.

