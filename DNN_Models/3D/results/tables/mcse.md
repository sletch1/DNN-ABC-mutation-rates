# Monte Carlo standard errors

`rmse_log` with its MCSE. Two methods differ meaningfully only when the gap exceeds about 2 combined MCSEs; anything smaller is simulation noise, not evidence.

| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |
|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | p1 | GPS-ABC | 1.939 ± 0.036 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p2 | GPS-ABC | 0.191 ± 0.067 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | GPS-ABC | 2.711 ± 0.139 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | DNN-ABC | 1.919 ± 0.070 | 0.75 ± 0.11 |
| (1e-04, 1e-02, 3) | 100 | p2 | DNN-ABC | 0.443 ± 0.119 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 3) | 100 | tau | DNN-ABC | 2.883 ± 0.240 | 0.81 ± 0.10 |
| (1e-04, 1e-02, 7) | 100 | p1 | GPS-ABC | 1.706 ± 0.045 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | GPS-ABC | 0.321 ± 0.035 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | GPS-ABC | 1.674 ± 0.152 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | DNN-ABC | 1.668 ± 0.071 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | p2 | DNN-ABC | 0.471 ± 0.084 | 0.56 ± 0.12 |
| (1e-04, 1e-02, 7) | 100 | tau | DNN-ABC | 1.720 ± 0.150 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | GPS-ABC | 0.524 ± 0.032 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | GPS-ABC | 0.109 ± 0.017 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | GPS-ABC | 0.602 ± 0.074 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | DNN-ABC | 0.477 ± 0.050 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | DNN-ABC | 0.259 ± 0.085 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | tau | DNN-ABC | 1.098 ± 0.147 | 1.00 ± 0.00 |

## Method comparisons (rmse_log)

- (1e-04,1e-02,3) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = +0.019 ± 0.079 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.251 ± 0.137 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.173 ± 0.278 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = +0.038 ± 0.084 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.150 ± 0.092 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.046 ± 0.213 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = +0.047 ± 0.059 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.150 ± 0.087 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.497 ± 0.165 -> **GPS-ABC better**
