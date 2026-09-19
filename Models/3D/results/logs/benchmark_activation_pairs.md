# 3-D two-stage surrogate: mixed activation pairs

Architecture fixed at the deployed `mlp 64-32`; the two hidden layers may use different activations. All 100 ordered pairs were screened on 2 seed(s); the top 8 plus reference points were then re-fit on 5 **fresh** seeds that played no part in selection. Only the stage-2 numbers below are results; stage-1 was selection only. Irreducible floor 1.393e-03.

## Stage 2 (fresh seeds -- the reportable numbers)

| layer 1 (64) | layer 2 (32) | mse_mean | ±sd (seeds) | × floor | cover95 |
|---|---|---|---|---|---|
| `relu` | `relu` | 1.632e-03 | 3.7e-05 | 1.172 | 0.953 |
| `tanh` | `tanh` | 1.639e-03 | 3.7e-05 | 1.177 | 0.954 |
| `leakyrelu` | `relu` | 1.645e-03 | 4.5e-05 | 1.181 | 0.954 |
| `tanh` | `relu` | 1.647e-03 | 7.2e-05 | 1.183 | 0.955 |
| `relu` | `gelu` | 1.648e-03 | 7.7e-05 | 1.184 | 0.954 |
| `prelu` | `tanh` | 1.654e-03 | 5.2e-05 | 1.187 | 0.954 |
| `mish` | `gelu` | 1.656e-03 | 3.1e-05 | 1.189 | 0.956 |
| `gelu` | `gelu` | 1.697e-03 | 6.3e-05 | 1.218 | 0.954 |

**Stage-2 spread (best to worst):** 6.47e-05 (4.0% of the best). **Typical across-seed sd:** 4.82e-05.

**Selection optimism:** finalists were on average +3.84e-05 worse on fresh seeds than on the seeds that selected them -- direct evidence that the stage-1 ranking was substantially seed noise.

**Verdict: NOT resolved.** On seeds that played no part in selection, the finalists are within three across-seed standard deviations of one another. Mixing activations across layers buys nothing measurable on this surface, and neither does the choice of activation itself. The deployed activation should therefore be justified on structural grounds -- smoothness in the inputs, and no dead units in a small network -- not on a leaderboard position.

## Stage 1 (screening only -- NOT results)

| layer 1 | layer 2 | mse (screen) |
|---|---|---|
| `mish` | `gelu` | 1.611e-03 |
| `tanh` | `tanh` | 1.611e-03 |
| `leakyrelu` | `relu` | 1.612e-03 |
| `relu` | `relu` | 1.614e-03 |
| `relu` | `gelu` | 1.615e-03 |
| `prelu` | `tanh` | 1.616e-03 |
| `tanh` | `relu` | 1.616e-03 |
| `gelu` | `gelu` | 1.616e-03 |
| `relu` | `silu` | 1.623e-03 |
| `leakyrelu` | `leakyrelu` | 1.630e-03 |
| `selu` | `leakyrelu` | 1.631e-03 |
| `relu` | `tanh` | 1.631e-03 |
| `selu` | `relu` | 1.632e-03 |
| `mish` | `tanh` | 1.633e-03 |
| `elu` | `relu` | 1.634e-03 |

_100 pairs screened; showing the top 15._
