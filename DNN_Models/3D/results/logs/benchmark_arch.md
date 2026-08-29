# 3-D two-stage surrogate: architecture search

Data: `slow_data_3D.csv`, split by replicate (train 1-5 / val 6-8 / test 9-10). Each row averaged over 2 seeds.

`mse_mean` is the MSE of the predicted mean against the held-out **design-point mean** of log10(d_bar) -- the fitted surface, with replicate noise averaged out. `cover95` should sit near 0.95.

| architecture | derived feat | params | mse_mean | ±sd | mse_obs | NLL | cover95 | s/fit |
|---|---|---|---|---|---|---|---|---|
| resmlp w128 x2 silu | yes | 67970 | 1.497e-03 | 1.3e-07 | 2.893e-03 | -3.230 | 0.951 | 32 |
| resmlp w128 x3 silu        [old 3-D] | yes | 101506 | 1.497e-03 | 5.5e-06 | 2.893e-03 | -3.229 | 0.951 | 53 |
| mlp 256-128-64 gelu | yes | 42562 | 1.500e-03 | 4.8e-06 | 2.896e-03 | -3.229 | 0.949 | 21 |
| mlp 256-128-64 relu | yes | 42562 | 1.502e-03 | 2.8e-06 | 2.898e-03 | -3.207 | 0.943 | 25 |
| mlp 256-128-64 gelu +LayerNorm | yes | 43458 | 1.503e-03 | 8.1e-06 | 2.899e-03 | -3.227 | 0.948 | 31 |
| mlp 128-128-64 silu | yes | 25538 | 1.509e-03 | 3.8e-06 | 2.905e-03 | -3.229 | 0.951 | 60 |
| resmlp w128 x3 silu   NO derived feat | NO | 101378 | 1.513e-03 | 5.1e-06 | 2.909e-03 | -3.226 | 0.949 | 51 |
| mlp 256-128-64 tanh | yes | 42562 | 1.531e-03 | 2.4e-06 | 2.927e-03 | -3.226 | 0.951 | 22 |
| mlp 256-128-64 gelu   NO derived feat | NO | 42306 | 1.549e-03 | 2.8e-06 | 2.945e-03 | -3.224 | 0.950 | 22 |
| mlp 128-64 gelu            [1-D shape] | yes | 9026 | 1.556e-03 | 3.3e-06 | 2.952e-03 | -3.228 | 0.953 | 15 |
| mlp 64-32 gelu             [small] | yes | 2466 | 1.573e-03 | 7.7e-06 | 2.969e-03 | -3.223 | 0.952 | 12 |

**Irreducible floor on `mse_mean` = 1.393e-03** (the test target is a 2-replicate mean, so it carries E[sigma^2]/2 of sampling noise no model can predict away). Max achievable R^2 = 0.99528.

**Best by mse_mean: `resmlp w128 x2 silu`** (mse_mean 1.497e-03 = **1.08x the floor**, R^2 = 0.99492, coverage 0.951).

**Read this table with care.** Best and worst differ by only 1.05x while the best is 1.08x the floor, so the architectures are effectively tied: this surface is easy relative to the replicate noise, and capacity is not the binding constraint. The design question is therefore how SMALL a model still reaches the floor (see benchmark_round2.py), because the surrogate's value is query speed inside the MCMC loop.

- Derived feature on `mlp 256-128-64 gelu`: mse_mean 1.549e-03 (without) -> 1.500e-03 (with), a 1.03x change.
- Derived feature on `resmlp w128 x3 silu`: mse_mean 1.513e-03 (without) -> 1.497e-03 (with), a 1.01x change.
