# 3-D two-stage model: parameter recovery

Config: `{"reps": 16, "nmcmc": 3000, "burnin": 1000, "ns": 4, "eps": 0.005, "gp_budget": 300, "J_grid": [100], "truths": [[0.0001, 0.01, 3.0], [0.0001, 0.01, 7.0], [0.002, 0.008, 5.0]], "mut_time": "offspring", "with_sim": false, "workers": 12}`

`rmse_log` is RMSE in log10 units for p1/p2 (so 1.0 = off by an order of magnitude on average) and in absolute time units for tau. **Prefer it to `nrmse`**: where a parameter is weakly identified the posterior mean sits wherever the prior puts its mass, and natural-scale nRMSE then explodes without conveying anything.

| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |
|---|---|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | p1 | 1.846 | 75.438 | 3.545e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | p2 | 0.083 | 0.212 | 3.596e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | tau | 2.259 | 0.753 | 9.102e+00 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC-ref | p1 | 1.852 | 71.357 | 3.753e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC-ref | p2 | 0.156 | 0.273 | 3.638e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC-ref | tau | 3.082 | 1.027 | 8.953e+00 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | p1 | 1.865 | 86.594 | 3.677e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | p2 | 0.132 | 0.340 | 3.682e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | tau | 2.550 | 0.850 | 8.918e+00 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | p1 | 1.838 | 71.897 | 3.439e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | p2 | 0.247 | 0.413 | 2.907e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | tau | 1.305 | 0.186 | 9.117e+00 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC-ref | p1 | 1.653 | 45.038 | 3.186e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC-ref | p2 | 0.399 | 0.583 | 2.436e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC-ref | tau | 1.243 | 0.178 | 9.108e+00 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | p1 | 1.789 | 73.211 | 3.251e-02 | 0.88 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | p2 | 0.269 | 0.432 | 2.629e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | tau | 1.597 | 0.228 | 8.581e+00 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | p1 | 0.490 | 2.400 | 3.383e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | p2 | 0.138 | 0.286 | 3.145e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | tau | 0.701 | 0.140 | 9.001e+00 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC-ref | p1 | 0.422 | 1.700 | 3.480e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC-ref | p2 | 0.167 | 0.305 | 3.091e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC-ref | tau | 1.071 | 0.214 | 8.979e+00 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | p1 | 0.568 | 3.345 | 3.362e-02 | 0.94 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | p2 | 0.372 | 0.427 | 2.898e-02 | 0.94 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | tau | 1.112 | 0.222 | 8.421e+00 | 1.00 |
