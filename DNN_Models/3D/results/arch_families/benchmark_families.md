# 3-D two-stage surrogate: new architecture families vs the deployed FFN

Data: `slow_data_3D.csv`, split by replicate (train 1-5 / val 6-8 / test 9-10), identical to `train.py` / `benchmark_arch.py`. New families averaged over 2 seeds; the FFN row is **not retrained** here -- it is the literal already-deployed/already-reported number (`results/logs/benchmark_round2.md`, `results/model/surrogate_metrics.json`).

Irreducible floor on `mse_mean` = **3.451e-04** (max achievable R^2 = 0.99683).

| architecture | params | mse_mean | x floor | R^2 | cover95 | us/query |
|---|---|---|---|---|---|---|
| CNN1D (16,16 ch, k=3) | 2482 | 3.589e-04 | 1.04 | 0.99670 | 0.949 | 143 |
| FFN 64-32  [DEPLOYED] | 2402 | 3.749e-04 | 1.09 | 0.99656 | 0.952 | 40 |
| LSTM (hidden=24) | 2642 | 4.221e-04 | 1.22 | 0.99612 | 0.950 | 69 |
| RNN (GRU, hidden=32) | 3426 | 4.520e-04 | 1.31 | 0.99585 | 0.949 | 66 |

**Best by mse_mean: `CNN1D (16,16 ch, k=3)`** (3.589e-04 = 1.04x floor, R^2 = 0.99670, cover95 = 0.949).

- `CNN1D (16,16 ch, k=3)` vs deployed FFN: mse_mean -4.3% (4.3% better), both within 1.09x of the noise floor.
- `LSTM (hidden=24)` vs deployed FFN: mse_mean +12.6% (12.6% worse), both within 1.22x of the noise floor.
- `RNN (GRU, hidden=32)` vs deployed FFN: mse_mean +20.6% (20.6% worse), both within 1.31x of the noise floor.

See `network/model_families.py` for why none of the three new families were expected a priori to beat the plain MLP here: the input has no real spatial/temporal structure for a convolution or recurrence to exploit, only an arbitrary storage order.

