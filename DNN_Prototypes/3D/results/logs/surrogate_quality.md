# Surrogate quality (held-out test set)

## 1. Calibration reliability
| nominal | empirical | status |
|---|---|---|
| 0.50 | 0.499 | OK |
| 0.80 | 0.825 | OK |
| 0.90 | 0.913 | OK |
| 0.95 | 0.960 | OK |
| 0.99 | 0.989 | OK |

## 2. Per-region error over (a, delta)
- overall test MSE(log) = 0.01415
- best region:  a=1.50, delta=0.50 -> MSE 0.00094
- worst region: a=0.50, delta=2.00 -> MSE 0.11182
- worst/best ratio = 119.5x  (uniformity of fit)

## 3. Residual diagnostics (standardized)
- mean(z) = -0.022 (target 0), std(z) = 0.985 (target 1)
