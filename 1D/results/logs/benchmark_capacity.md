# Capacity benchmark: how small can the 1-D surrogate be?

GP mean-curve MSE = 3.83586e-04. Curve MSE is against the denoised response curve, averaged over 3 seeds. Downstream MSE/CI length are DNN-ABC on the full p x J grid, 20 reps/cell, one representative seed per candidate (see script docstring); the exact-simulator ABC-MCMC baseline is not re-run per candidate since it does not depend on the surrogate.

| Hidden layers | Parameters | Curve MSE (3-seed mean +/- sd) | vs GP | Downstream MSE | Downstream 95% CI length |
|---|---|---|---|---|---|
| GP (GPS-ABC baseline) | -- | 3.836e-04 | --- | 2.292e-06 | 2.101e-03 |
| 128-64 [deployed] | 8642 | 4.129e-04 +/- 2.4e-05 | +7.6% | 9.694e-07 | 1.300e-03 |
| 64-32 | 2274 | 3.957e-04 +/- 6.5e-06 | +3.2% | 1.043e-06 | 1.314e-03 |
| 32-16 | 626 | 3.896e-04 +/- 9.7e-06 | +1.6% | 1.236e-06 | 1.382e-03 |
| 16-8 | 186 | 4.508e-04 +/- 2.3e-05 | +17.5% | 9.018e-07 | 1.203e-03 |
| 16 (1 layer) | 66 | 4.460e-04 +/- 2.2e-05 | +16.3% | 1.207e-06 | 1.388e-03 |
| 8 (1 layer) | 34 | 4.983e-04 +/- 8.4e-05 | +29.9% | 1.310e-06 | 1.400e-03 |
| Linear (control) | 4 | 4.672e-04 +/- 6.0e-06 | +21.8% | 1.940e-06 | 1.800e-03 |
