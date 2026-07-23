# ABC 95% credible-interval coverage (target 0.95)

## Overall (all cells pooled)
| method | coverage | n |
|---|---|---|
| ABC-MCMC | 0.927 | 384 | OK
| GPS-ABC | 0.992 | 384 | OK
| DNN-ABC | 0.865 | 384 | LOW

## Per cell
| p | a | delta | J | ABC-MCMC | GPS-ABC | DNN-ABC |
|---|---|---|---|---|---|---|
| 1e-03 | 1.0 | 0.5 | 50 | 0.97 | 1.00 | 0.84 |
| 1e-03 | 1.0 | 0.5 | 100 | 0.91 | 1.00 | 0.91 |
| 1e-03 | 1.0 | 1.0 | 50 | 0.88 | 0.94 | 0.69 |
| 1e-03 | 1.0 | 1.0 | 100 | 0.97 | 1.00 | 0.94 |
| 1e-03 | 1.5 | 1.5 | 50 | 0.78 | 0.97 | 0.62 |
| 1e-03 | 1.5 | 1.5 | 100 | 0.94 | 1.00 | 0.94 |
| 1e-02 | 1.0 | 0.5 | 50 | 0.94 | 1.00 | 0.91 |
| 1e-02 | 1.0 | 0.5 | 100 | 1.00 | 1.00 | 0.97 |
| 1e-02 | 1.0 | 1.0 | 50 | 0.97 | 1.00 | 0.91 |
| 1e-02 | 1.0 | 1.0 | 100 | 0.94 | 1.00 | 0.91 |
| 1e-02 | 1.5 | 1.5 | 50 | 0.88 | 1.00 | 0.78 |
| 1e-02 | 1.5 | 1.5 | 100 | 0.97 | 1.00 | 0.97 |

Reading: coverage near 0.95 means the intervals are honest. If DNN-ABC holds ~0.95 while giving shorter intervals than GPS-ABC (Table 2), its extra precision is real, not overconfidence.
