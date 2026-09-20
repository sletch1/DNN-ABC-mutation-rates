# 3-D two-stage surrogate: architecture search

Data: `slow_data_3D.csv`, split by replicate (train 1-5 / val 6-8 / test 9-10). Each row averaged over 3 seeds.

`mse_mean` is the MSE of the predicted mean against the held-out **design-point mean** of log10(d_bar) -- the fitted surface, with replicate noise averaged out. `cover95` should sit near 0.95.

| architecture | params | mse_mean | ±sd | mse_obs | NLL | cover95 | s/fit |
|---|---|---|---|---|---|---|---|
| mlp 256-128-64 gelu | 42306 | 3.475e-04 | 4.8e-06 | 6.908e-04 | -3.888 | 0.950 | 21 |
| resmlp w128 x2 silu | 67842 | 3.481e-04 | 1.4e-06 | 6.914e-04 | -3.888 | 0.950 | 351 |
| mlp 256-128-64 gelu +LayerNorm | 43202 | 3.506e-04 | 2.8e-06 | 6.939e-04 | -3.886 | 0.948 | 34 |
| resmlp w128 x3 silu | 101378 | 3.525e-04 | 4.2e-06 | 6.958e-04 | -3.888 | 0.950 | 55 |
| mlp 256-128-64 tanh | 42306 | 3.545e-04 | 1.8e-06 | 6.979e-04 | -3.880 | 0.949 | 27 |
| mlp 256-128-64 relu | 42306 | 3.566e-04 | 2.7e-06 | 6.999e-04 | -3.876 | 0.941 | 16 |
| mlp 128-64 gelu            [1-D shape] | 8898 | 3.582e-04 | 1.3e-06 | 7.015e-04 | -3.881 | 0.951 | 16 |
| mlp 128-128-64 silu | 25410 | 3.588e-04 | 2.0e-06 | 7.021e-04 | -3.877 | 0.950 | 26 |
| mlp 64-32 gelu             [small] | 2402 | 3.749e-04 | 4.8e-06 | 7.182e-04 | -3.866 | 0.952 | 20 |

**Irreducible floor on `mse_mean` = 3.451e-04** (the test target is a 2-replicate mean, so it carries E[sigma^2]/2 of sampling noise no model can predict away). Max achievable R^2 = 0.99683.

**Best by mse_mean: `mlp 256-128-64 gelu`** (mse_mean 3.475e-04 = **1.01x the floor**, R^2 = 0.99681, coverage 0.950).

**Read this table with care.** Best and worst differ by only 1.08x while the best is 1.01x the floor, so the architectures are effectively tied: this surface is easy relative to the replicate noise, and capacity is not the binding constraint. The design question is therefore how SMALL a model still reaches the floor (see benchmark_round2.py), because the surrogate's value is query speed inside the MCMC loop.

