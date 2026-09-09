# Monte Carlo standard errors -- new architecture families

`rmse_log` with its MCSE, computed identically to `abc/mcse.py`. Two methods differ meaningfully only when the gap exceeds about 2 combined MCSEs; anything smaller is simulation noise, not evidence.

| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |
|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | p1 | GPS-ABC | 1.846 ± 0.038 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p2 | GPS-ABC | 0.083 ± 0.011 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | GPS-ABC | 2.259 ± 0.177 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | DNN-ABC | 1.865 ± 0.055 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p2 | DNN-ABC | 0.132 ± 0.016 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | DNN-ABC | 2.550 ± 0.126 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | CNN1D-ABC | 1.795 ± 0.068 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p2 | CNN1D-ABC | 0.202 ± 0.052 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | CNN1D-ABC | 2.248 ± 0.199 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | RNN-ABC | 1.856 ± 0.044 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p2 | RNN-ABC | 0.127 ± 0.019 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | RNN-ABC | 2.471 ± 0.201 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | p1 | LSTM-ABC | 1.776 ± 0.067 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 3) | 100 | p2 | LSTM-ABC | 0.114 ± 0.014 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 3) | 100 | tau | LSTM-ABC | 2.423 ± 0.214 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | GPS-ABC | 1.838 ± 0.030 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | GPS-ABC | 0.247 ± 0.026 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | GPS-ABC | 1.305 ± 0.126 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | DNN-ABC | 1.789 ± 0.055 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 7) | 100 | p2 | DNN-ABC | 0.269 ± 0.041 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | tau | DNN-ABC | 1.597 ± 0.185 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | CNN1D-ABC | 1.715 ± 0.062 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | p2 | CNN1D-ABC | 0.240 ± 0.026 | 0.94 ± 0.06 |
| (1e-04, 1e-02, 7) | 100 | tau | CNN1D-ABC | 1.650 ± 0.201 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | RNN-ABC | 1.771 ± 0.062 | 0.81 ± 0.10 |
| (1e-04, 1e-02, 7) | 100 | p2 | RNN-ABC | 0.638 ± 0.182 | 0.81 ± 0.10 |
| (1e-04, 1e-02, 7) | 100 | tau | RNN-ABC | 1.566 ± 0.197 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p1 | LSTM-ABC | 1.831 ± 0.038 | 1.00 ± 0.00 |
| (1e-04, 1e-02, 7) | 100 | p2 | LSTM-ABC | 0.326 ± 0.061 | 0.88 ± 0.08 |
| (1e-04, 1e-02, 7) | 100 | tau | LSTM-ABC | 1.520 ± 0.143 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | GPS-ABC | 0.490 ± 0.039 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | GPS-ABC | 0.138 ± 0.034 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | GPS-ABC | 0.701 ± 0.087 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | DNN-ABC | 0.568 ± 0.053 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | p2 | DNN-ABC | 0.372 ± 0.137 | 0.94 ± 0.06 |
| (2e-03, 8e-03, 5) | 100 | tau | DNN-ABC | 1.112 ± 0.185 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | CNN1D-ABC | 0.609 ± 0.041 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | CNN1D-ABC | 0.167 ± 0.047 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | CNN1D-ABC | 1.166 ± 0.168 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | RNN-ABC | 0.505 ± 0.050 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | RNN-ABC | 0.171 ± 0.021 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | RNN-ABC | 1.062 ± 0.128 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p1 | LSTM-ABC | 0.561 ± 0.046 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | p2 | LSTM-ABC | 0.187 ± 0.025 | 1.00 ± 0.00 |
| (2e-03, 8e-03, 5) | 100 | tau | LSTM-ABC | 0.950 ± 0.137 | 1.00 ± 0.00 |

## Method comparisons (rmse_log): each new family vs GPS-ABC and DNN-ABC(MLP)

- (1e-04,1e-02,3) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = -0.052 ± 0.078 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = -0.070 ± 0.088 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = +0.010 ± 0.058 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = -0.009 ± 0.071 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = -0.070 ± 0.077 -> **TIE**
- (1e-04,1e-02,3) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = -0.089 ± 0.087 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = +0.119 ± 0.053 -> **GPS-ABC better**
- (1e-04,1e-02,3) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = +0.070 ± 0.054 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.044 ± 0.022 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = -0.005 ± 0.025 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = +0.031 ± 0.017 -> **TIE**
- (1e-04,1e-02,3) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = -0.018 ± 0.021 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = -0.011 ± 0.266 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = -0.302 ± 0.235 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = +0.212 ± 0.268 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.079 ± 0.237 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = +0.164 ± 0.277 -> **TIE**
- (1e-04,1e-02,3) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.127 ± 0.248 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = -0.123 ± 0.069 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = -0.074 ± 0.083 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = -0.067 ± 0.069 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = -0.018 ± 0.083 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = -0.007 ± 0.048 -> **TIE**
- (1e-04,1e-02,7) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = +0.042 ± 0.067 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = -0.007 ± 0.037 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = -0.029 ± 0.048 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.391 ± 0.184 -> **GPS-ABC better**
- (1e-04,1e-02,7) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = +0.369 ± 0.186 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = +0.079 ± 0.066 -> **TIE**
- (1e-04,1e-02,7) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = +0.057 ± 0.073 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = +0.345 ± 0.237 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = +0.053 ± 0.273 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = +0.262 ± 0.234 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.031 ± 0.270 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = +0.216 ± 0.191 -> **TIE**
- (1e-04,1e-02,7) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.076 ± 0.233 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: CNN1D-ABC vs GPS-ABC: delta = +0.120 ± 0.057 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `p1`: CNN1D-ABC vs DNN-ABC: delta = +0.041 ± 0.067 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: RNN-ABC vs GPS-ABC: delta = +0.016 ± 0.064 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: RNN-ABC vs DNN-ABC: delta = -0.063 ± 0.073 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: LSTM-ABC vs GPS-ABC: delta = +0.072 ± 0.061 -> **TIE**
- (2e-03,8e-03,5) J=100 `p1`: LSTM-ABC vs DNN-ABC: delta = -0.007 ± 0.071 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: CNN1D-ABC vs GPS-ABC: delta = +0.029 ± 0.058 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: CNN1D-ABC vs DNN-ABC: delta = -0.205 ± 0.145 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: RNN-ABC vs GPS-ABC: delta = +0.033 ± 0.040 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: RNN-ABC vs DNN-ABC: delta = -0.201 ± 0.139 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: LSTM-ABC vs GPS-ABC: delta = +0.049 ± 0.042 -> **TIE**
- (2e-03,8e-03,5) J=100 `p2`: LSTM-ABC vs DNN-ABC: delta = -0.185 ± 0.139 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: CNN1D-ABC vs GPS-ABC: delta = +0.465 ± 0.189 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `tau`: CNN1D-ABC vs DNN-ABC: delta = +0.054 ± 0.250 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: RNN-ABC vs GPS-ABC: delta = +0.361 ± 0.154 -> **GPS-ABC better**
- (2e-03,8e-03,5) J=100 `tau`: RNN-ABC vs DNN-ABC: delta = -0.050 ± 0.225 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: LSTM-ABC vs GPS-ABC: delta = +0.249 ± 0.162 -> **TIE**
- (2e-03,8e-03,5) J=100 `tau`: LSTM-ABC vs DNN-ABC: delta = -0.162 ± 0.230 -> **TIE**
