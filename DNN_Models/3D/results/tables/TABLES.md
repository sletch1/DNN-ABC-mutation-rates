# 3-D two-stage model: parameter recovery

Config: `{"reps": 16, "nmcmc": 3000, "burnin": 1000, "ns": 4, "eps": 0.005, "gp_budget": 300, "J_grid": [100], "truths": [[0.0001, 0.01, 3.0], [0.0001, 0.01, 7.0], [0.002, 0.008, 5.0]], "mut_time": "parent", "with_sim": false, "workers": 12}`

`rmse_log` is RMSE in log10 units for p1/p2 (so 1.0 = off by an order of magnitude on average) and in absolute time units for tau. **Prefer it to `nrmse`**: where a parameter is weakly identified the posterior mean sits wherever the prior puts its mass, and natural-scale nRMSE then explodes without conveying anything. MOM/MLE are constant-rate baselines scored against `p_eff`, the time-average rate they actually estimate -- they cannot identify p1, p2 or tau individually.

| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |
|---|---|---|---|---|---|---|---|
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | p1 | 1.939 | 94.301 | 3.690e-02 | 0.94 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | p2 | 0.191 | 0.290 | 3.544e-02 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | GPS-ABC | tau | 2.711 | 0.904 | 8.940e+00 | 1.00 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | p1 | 1.919 | 106.655 | 3.654e-02 | 0.75 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | p2 | 0.443 | 0.492 | 2.718e-02 | 0.88 |
| (1e-04, 1e-02, 3) | 100 | DNN-ABC | tau | 2.883 | 0.961 | 8.164e+00 | 0.81 |
| (1e-04, 1e-02, 3) | 100 | MOM | p_eff | 0.013 | 0.029 | - | - |
| (1e-04, 1e-02, 3) | 100 | MLE | p_eff | 0.024 | 0.056 | - | - |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | p1 | 1.706 | 56.633 | 2.976e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | p2 | 0.321 | 0.492 | 2.294e-02 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | GPS-ABC | tau | 1.674 | 0.239 | 9.150e+00 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | p1 | 1.668 | 62.361 | 2.807e-02 | 0.94 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | p2 | 0.471 | 0.547 | 1.773e-02 | 0.56 |
| (1e-04, 1e-02, 7) | 100 | DNN-ABC | tau | 1.720 | 0.246 | 8.754e+00 | 1.00 |
| (1e-04, 1e-02, 7) | 100 | MOM | p_eff | 0.006 | 0.014 | - | - |
| (1e-04, 1e-02, 7) | 100 | MLE | p_eff | 0.018 | 0.043 | - | - |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | p1 | 0.524 | 2.556 | 3.423e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | p2 | 0.109 | 0.277 | 3.136e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | GPS-ABC | tau | 0.602 | 0.120 | 9.120e+00 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | p1 | 0.477 | 2.498 | 3.094e-02 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | p2 | 0.259 | 0.497 | 3.024e-02 | 0.94 |
| (2e-03, 8e-03, 5) | 100 | DNN-ABC | tau | 1.098 | 0.220 | 8.697e+00 | 1.00 |
| (2e-03, 8e-03, 5) | 100 | MOM | p_eff | 0.035 | 0.082 | - | - |
| (2e-03, 8e-03, 5) | 100 | MLE | p_eff | 0.038 | 0.096 | - | - |
