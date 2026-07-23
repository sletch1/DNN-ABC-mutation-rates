# 3-D architecture benchmark: residual MLP vs GP on mean-surface fit

Irreducible replicate noise var(log10 d_bar) ~ 0.0130 (per-point MSE floor). Denoised surface = mean over 10 reps at each of 1000 grid points.

| model | mean-surface MSE | vs GP(300) |
|---|---|---|
| ResMLP relu+LN (w128, 3blk) | 1.37141e-03 | +22% better |
| ResMLP silu+LN big (w256, 4blk) | 1.44363e-03 | +18% better |
| ResMLP silu, NO LayerNorm (w128, 3blk) | 1.44618e-03 | +18% better |
| ResMLP gelu+LN (w128, 3blk) | 1.45196e-03 | +18% better |
| ResMLP silu+LN (w128, 3blk) [chosen] | 1.46330e-03 | +17% better |
| ResMLP silu+LN small (w64, 2blk) | 1.46711e-03 | +17% better |
| GP (budget 1000, fit 8.2s) | 1.60691e-03 | +9% better |
| GP (budget 300, fit 1.2s) | 1.76652e-03 | — (baseline) |
