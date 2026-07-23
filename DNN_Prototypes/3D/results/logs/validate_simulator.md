# Simulator port validation

## 1. Plating time tp (all 100 (p,a) grid points)
- max |tp_py - tp_csv| = 3.02e-05, mean = 6.91e-06
- **PASS** (threshold 1e-4)

## 2. Summary-stat mean log10(d_bar): Python slow-sim vs CSV
| p | a | delta | CSV mean | Python mean | abs diff |
|---|---|---|---|---|---|
| 1e-02 | 1.17 | 1.83 | -0.460 | -0.435 | 0.025 |
| 1e-02 | 1.00 | 1.50 | -0.506 | -0.507 | 0.002 |
| 2e-03 | 1.33 | 1.17 | -0.904 | -0.903 | 0.002 |
| 2e-03 | 1.33 | 0.50 | -1.144 | -1.147 | 0.003 |
| 1e-02 | 1.33 | 1.00 | -0.660 | -0.651 | 0.009 |
| 1e-02 | 0.67 | 1.33 | -0.549 | -0.555 | 0.006 |
| 2e-03 | 1.00 | 2.00 | -0.651 | -0.639 | 0.012 |
| 1e-02 | 1.50 | 1.17 | -0.598 | -0.595 | 0.003 |
- max abs diff = 0.025  -> **PASS** (distributional, threshold 0.15)

## 3. Estimator unit check (p_true=5e-3, 3000 cultures)
- MOM = 5.148e-03, MLE = 5.629e-03, truth = 5.000e-03
- **PASS** (within 50% of truth)

## Overall: ALL CHECKS PASSED