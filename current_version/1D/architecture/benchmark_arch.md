# Architecture benchmark: DNN vs GP on mean-curve fit

Irreducible replicate noise var(log10 d_bar) ~ 0.0047 (per-point MSE floor).

| model | mean-curve MSE | vs GP |
|---|---|---|
| GP (GPS-ABC baseline) | 3.83586e-04 | — |
| DNN D: gelu (128,64) | 3.88835e-04 | +1% worse |
| DNN ENSEMBLE x5: silu (128,128,64) | 3.96024e-04 | +3% worse |
| DNN ENSEMBLE x5: tanh (256,128) | 4.17774e-04 | +9% worse |
| DNN C: silu (128,128,64) | 4.39986e-04 | +15% worse |
| DNN B: tanh (64,64,32) | 4.43472e-04 | +16% worse |
| DNN E: tanh (256,128) | 4.50342e-04 | +17% worse |
| DNN A: relu+BN+drop (64,64,32) [old] | 4.35238e-03 | +1035% worse |
