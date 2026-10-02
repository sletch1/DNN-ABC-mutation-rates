# M3: ABC tolerance (eps) sweep, Study II

Grid: [0.0025, 0.005, 0.01, 0.02, 0.04]. Same 48 replicates (3 truths x 16 reps) as the main sweep, `obs` reused unchanged; GPS-ABC, GPS-ABC-ref, DNN-ABC only (NPE has no acceptance kernel and no eps).

| eps | method | param | coverage | mean 95% CI width |
|---|---|---|---|---|
| 0.0025 | GPS-ABC | p1 | 0.979 | 3.586e-02 |
| 0.0025 | GPS-ABC | p2 | 1.000 | 3.269e-02 |
| 0.0025 | GPS-ABC | tau | 1.000 | 9.125e+00 |
| 0.0025 | GPS-ABC-ref | p1 | 1.000 | 3.364e-02 |
| 0.0025 | GPS-ABC-ref | p2 | 1.000 | 3.030e-02 |
| 0.0025 | GPS-ABC-ref | tau | 1.000 | 9.069e+00 |
| 0.0025 | DNN-ABC | p1 | 0.875 | 3.327e-02 |
| 0.0025 | DNN-ABC | p2 | 0.875 | 2.872e-02 |
| 0.0025 | DNN-ABC | tau | 0.979 | 8.540e+00 |
| 0.005 | GPS-ABC | p1 | 1.000 | 3.723e-02 |
| 0.005 | GPS-ABC | p2 | 1.000 | 3.310e-02 |
| 0.005 | GPS-ABC | tau | 1.000 | 9.125e+00 |
| 0.005 | GPS-ABC-ref | p1 | 1.000 | 3.472e-02 |
| 0.005 | GPS-ABC-ref | p2 | 1.000 | 2.945e-02 |
| 0.005 | GPS-ABC-ref | tau | 1.000 | 9.088e+00 |
| 0.005 | DNN-ABC | p1 | 0.938 | 3.330e-02 |
| 0.005 | DNN-ABC | p2 | 0.938 | 2.960e-02 |
| 0.005 | DNN-ABC | tau | 0.979 | 8.625e+00 |
| 0.01 | GPS-ABC | p1 | 1.000 | 3.505e-02 |
| 0.01 | GPS-ABC | p2 | 1.000 | 3.243e-02 |
| 0.01 | GPS-ABC | tau | 1.000 | 9.105e+00 |
| 0.01 | GPS-ABC-ref | p1 | 1.000 | 3.441e-02 |
| 0.01 | GPS-ABC-ref | p2 | 1.000 | 2.937e-02 |
| 0.01 | GPS-ABC-ref | tau | 1.000 | 9.025e+00 |
| 0.01 | DNN-ABC | p1 | 0.979 | 3.546e-02 |
| 0.01 | DNN-ABC | p2 | 0.958 | 2.937e-02 |
| 0.01 | DNN-ABC | tau | 0.979 | 8.675e+00 |
| 0.02 | GPS-ABC | p1 | 0.979 | 3.510e-02 |
| 0.02 | GPS-ABC | p2 | 1.000 | 3.230e-02 |
| 0.02 | GPS-ABC | tau | 1.000 | 9.072e+00 |
| 0.02 | GPS-ABC-ref | p1 | 1.000 | 3.455e-02 |
| 0.02 | GPS-ABC-ref | p2 | 1.000 | 2.886e-02 |
| 0.02 | GPS-ABC-ref | tau | 1.000 | 9.062e+00 |
| 0.02 | DNN-ABC | p1 | 0.958 | 3.689e-02 |
| 0.02 | DNN-ABC | p2 | 1.000 | 3.091e-02 |
| 0.02 | DNN-ABC | tau | 0.979 | 8.904e+00 |
| 0.04 | GPS-ABC | p1 | 1.000 | 3.754e-02 |
| 0.04 | GPS-ABC | p2 | 1.000 | 3.483e-02 |
| 0.04 | GPS-ABC | tau | 1.000 | 9.162e+00 |
| 0.04 | GPS-ABC-ref | p1 | 1.000 | 3.494e-02 |
| 0.04 | GPS-ABC-ref | p2 | 1.000 | 2.935e-02 |
| 0.04 | GPS-ABC-ref | tau | 1.000 | 9.097e+00 |
| 0.04 | DNN-ABC | p1 | 1.000 | 3.759e-02 |
| 0.04 | DNN-ABC | p2 | 1.000 | 3.406e-02 |
| 0.04 | DNN-ABC | tau | 1.000 | 9.101e+00 |
