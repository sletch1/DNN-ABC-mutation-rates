# Monte Carlo standard errors

`rmse_log` with its MCSE. Two methods differ meaningfully only when the gap exceeds about 2 combined MCSEs; anything smaller is simulation noise, not evidence.

| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |
|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | p1 | GPS-ABC | 1.846 ± 0.038 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p2 | GPS-ABC | 0.083 ± 0.011 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | GPS-ABC | 2.259 ± 0.177 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | GPS-ABC-ref | 1.852 ± 0.015 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p2 | GPS-ABC-ref | 0.156 ± 0.038 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | GPS-ABC-ref | 3.082 ± 0.102 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | DNN-ABC | 1.865 ± 0.055 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p2 | DNN-ABC | 0.132 ± 0.016 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | DNN-ABC | 2.550 ± 0.126 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | GPS-ABC | 1.838 ± 0.030 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | GPS-ABC | 0.247 ± 0.026 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | GPS-ABC | 1.305 ± 0.126 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | GPS-ABC-ref | 1.653 ± 0.018 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | GPS-ABC-ref | 0.399 ± 0.031 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | GPS-ABC-ref | 1.243 ± 0.062 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | DNN-ABC | 1.789 ± 0.055 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 7) | 100 | p2 | DNN-ABC | 0.269 ± 0.041 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | DNN-ABC | 1.597 ± 0.185 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | GPS-ABC | 0.490 ± 0.039 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | GPS-ABC | 0.138 ± 0.034 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | GPS-ABC | 0.701 ± 0.087 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | GPS-ABC-ref | 0.422 ± 0.019 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | GPS-ABC-ref | 0.167 ± 0.023 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | GPS-ABC-ref | 1.071 ± 0.064 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | DNN-ABC | 0.568 ± 0.053 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | p2 | DNN-ABC | 0.372 ± 0.137 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | tau | DNN-ABC | 1.112 ± 0.185 | 1.00 ± 0.00 |

## Method comparisons (rmse_log)

- (1e-04,1e-02,3) J=100 `p1`: GPS-ABC vs GPS-ABC-ref: delta = -0.006 ± 0.041 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = -0.018 ± 0.067 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: GPS-ABC-ref vs DNN-ABC: delta = -0.013 ± 0.057 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: GPS-ABC vs GPS-ABC-ref: delta = -0.073 ± 0.040 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.050 ± 0.020 -> **GPS-ABC better**
- (1e-04,1e-02,3) J=100 `p2`: GPS-ABC-ref vs DNN-ABC: delta = +0.024 ± 0.042 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: GPS-ABC vs GPS-ABC-ref: delta = -0.823 ± 0.204 -> **GPS-ABC better**
- (1e-04,1e-02,3) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.291 ± 0.217 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: GPS-ABC-ref vs DNN-ABC: delta = +0.532 ± 0.162 -> **DNN-ABC better**
- (1e-04,1e-02,7) J=100 `p1`: GPS-ABC vs GPS-ABC-ref: delta = +0.185 ± 0.035 -> **GPS-ABC-ref better**
- (1e-04,1e-02,7) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = +0.049 ± 0.063 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: GPS-ABC-ref vs DNN-ABC: delta = -0.136 ± 0.058 -> **GPS-ABC-ref better**
- (1e-04,1e-02,7) J=100 `p2`: GPS-ABC vs GPS-ABC-ref: delta = -0.152 ± 0.040 -> **GPS-ABC better**
- (1e-04,1e-02,7) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.022 ± 0.048 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: GPS-ABC-ref vs DNN-ABC: delta = +0.130 ± 0.051 -> **DNN-ABC better**
- (1e-04,1e-02,7) J=100 `tau`: GPS-ABC vs GPS-ABC-ref: delta = +0.061 ± 0.141 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.292 ± 0.224 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: GPS-ABC-ref vs DNN-ABC: delta = -0.353 ± 0.195 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: GPS-ABC vs GPS-ABC-ref: delta = +0.068 ± 0.044 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: GPS-ABC vs DNN-ABC: delta = -0.078 ± 0.066 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: GPS-ABC-ref vs DNN-ABC: delta = -0.146 ± 0.056 -> **GPS-ABC-ref better**
- (2e-03,8e-03,5) J=100 `p2`: GPS-ABC vs GPS-ABC-ref: delta = -0.029 ± 0.041 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: GPS-ABC vs DNN-ABC: delta = -0.235 ± 0.141 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: GPS-ABC-ref vs DNN-ABC: delta = -0.206 ± 0.139 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: GPS-ABC vs GPS-ABC-ref: delta = -0.370 ± 0.108 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `tau`: GPS-ABC vs DNN-ABC: delta = -0.411 ± 0.204 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `tau`: GPS-ABC-ref vs DNN-ABC: delta = -0.041 ± 0.196 -> **TIE**
