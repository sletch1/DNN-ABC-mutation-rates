# Monte Carlo standard errors -- new architecture families

`rmse_log` with its MCSE, computed identically to `abc/mcse.py`. Two methods differ meaningfully only when the gap exceeds about 2 combined MCSEs; anything smaller is simulation noise, not evidence.

| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |
|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | p1 | GPS-ABC | 1.939 ± 0.036 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p2 | GPS-ABC | 0.191 ± 0.067 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | GPS-ABC | 2.711 ± 0.139 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | DNN-ABC | 1.919 ± 0.070 | 0.75 ± 0.11 |
| (1e-04, 1e-02, 3) | 100 | p2 | DNN-ABC | 0.443 ± 0.119 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 3) | 100 | tau | DNN-ABC | 2.883 ± 0.240 | 0.81 ± 0.10 |
| (1e-04, 1e-02, 3) | 100 | p1 | CNN1D-ABC | 1.941 ± 0.073 | 0.62 ± 0.12 |
| (1e-04, 1e-02, 3) | 100 | p2 | CNN1D-ABC | 0.638 ± 0.145 | 0.75 ± 0.11 |
| (1e-04, 1e-02, 3) | 100 | tau | CNN1D-ABC | 3.061 ± 0.228 | 0.81 ± 0.10 |
| (1e-04, 1e-02, 3) | 100 | p1 | RNN-ABC | 1.919 ± 0.062 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 3) | 100 | p2 | RNN-ABC | 0.391 ± 0.114 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 3) | 100 | tau | RNN-ABC | 2.524 ± 0.203 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p1 | LSTM-ABC | 1.871 ± 0.069 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 3) | 100 | p2 | LSTM-ABC | 0.355 ± 0.108 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | tau | LSTM-ABC | 2.442 ± 0.212 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | GPS-ABC | 1.706 ± 0.045 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | GPS-ABC | 0.321 ± 0.035 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | GPS-ABC | 1.674 ± 0.152 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | DNN-ABC | 1.668 ± 0.071 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | p2 | DNN-ABC | 0.471 ± 0.084 | 0.56 ± 0.12 |
| (1e-04, 1e-02, 7) | 100 | tau | DNN-ABC | 1.720 ± 0.150 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | CNN1D-ABC | 1.724 ± 0.050 | 0.75 ± 0.11 |
| (1e-04, 1e-02, 7) | 100 | p2 | CNN1D-ABC | 0.437 ± 0.075 | 0.69 ± 0.12 |
| (1e-04, 1e-02, 7) | 100 | tau | CNN1D-ABC | 1.830 ± 0.168 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | RNN-ABC | 1.714 ± 0.053 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | p2 | RNN-ABC | 0.519 ± 0.127 | 0.75 ± 0.11 |
| (1e-04, 1e-02, 7) | 100 | tau | RNN-ABC | 1.291 ± 0.148 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | LSTM-ABC | 1.736 ± 0.054 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 7) | 100 | p2 | LSTM-ABC | 0.320 ± 0.043 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | tau | LSTM-ABC | 1.638 ± 0.175 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | GPS-ABC | 0.524 ± 0.032 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | GPS-ABC | 0.109 ± 0.017 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | GPS-ABC | 0.602 ± 0.074 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | DNN-ABC | 0.477 ± 0.050 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | DNN-ABC | 0.259 ± 0.085 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | tau | DNN-ABC | 1.098 ± 0.147 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | CNN1D-ABC | 0.650 ± 0.049 | 0.88 ± 0.08 |
| (2e-03, 8e-03, 5) | 100 | p2 | CNN1D-ABC | 0.571 ± 0.196 | 0.88 ± 0.08 |
| (2e-03, 8e-03, 5) | 100 | tau | CNN1D-ABC | 0.951 ± 0.146 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | RNN-ABC | 0.619 ± 0.045 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | RNN-ABC | 0.292 ± 0.063 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | RNN-ABC | 0.828 ± 0.108 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | LSTM-ABC | 0.583 ± 0.041 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | LSTM-ABC | 0.170 ± 0.029 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | LSTM-ABC | 0.799 ± 0.176 | 1.00 ± 0.00 |

## Method comparisons (rmse_log): each new family vs GPS-ABC and DNN-ABC(MLP)

- (1e-04,1e-02,3) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = +0.002 ± 0.081 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = +0.022 ± 0.101 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = -0.020 ± 0.072 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = -0.001 ± 0.094 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = -0.068 ± 0.078 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = -0.048 ± 0.098 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = +0.447 ± 0.160 -> **GPS-ABC better**
- (1e-04,1e-02,3) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = +0.195 ± 0.188 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.199 ± 0.133 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = -0.052 ± 0.165 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = +0.164 ± 0.127 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = -0.087 ± 0.160 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = +0.351 ± 0.267 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = +0.178 ± 0.332 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = -0.186 ± 0.246 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.359 ± 0.314 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = -0.268 ± 0.253 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.441 ± 0.320 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = +0.018 ± 0.068 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = +0.056 ± 0.087 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = +0.009 ± 0.070 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = +0.046 ± 0.088 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = +0.030 ± 0.071 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = +0.068 ± 0.089 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = +0.116 ± 0.083 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = -0.034 ± 0.113 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.198 ± 0.132 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = +0.048 ± 0.153 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = -0.001 ± 0.056 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = -0.151 ± 0.095 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = +0.156 ± 0.227 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = +0.110 ± 0.225 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = -0.383 ± 0.212 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.429 ± 0.210 -> **RNN-ABC better**
- (1e-04,1e-02,7) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = -0.035 ± 0.232 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.082 ± 0.230 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = +0.125 ± 0.058 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = +0.172 ± 0.070 -> **DNN-ABC better**
- (2e-03,8e-03,5) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = +0.095 ± 0.055 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = +0.142 ± 0.068 -> **DNN-ABC better**
- (2e-03,8e-03,5) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = +0.059 ± 0.051 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = +0.106 ± 0.065 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = +0.462 ± 0.197 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = +0.312 ± 0.214 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.184 ± 0.065 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = +0.034 ± 0.106 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = +0.061 ± 0.033 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = -0.089 ± 0.090 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = +0.349 ± 0.163 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = -0.148 ± 0.207 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = +0.226 ± 0.131 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.270 ± 0.182 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = +0.197 ± 0.191 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.300 ± 0.229 -> **TIE**
