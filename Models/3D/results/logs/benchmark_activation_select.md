# 3-D two-stage surrogate: activation selection (no test-set leakage)

All 100 ordered activation pairs for the deployed `mlp 64-32`, screened on 2 seed(s) and confirmed on 15 **fresh** seeds. **Selection is on the validation split**; the test column is shown for reference only and played no part in the choice.

Winner: **`gelu` -> `tanh`**  (val MSE 2.980e-04, test MSE 3.666e-04 = 1.062x the irreducible test floor 3.451e-04).

| layer 1 (64) | layer 2 (32) | val MSE | ±sd | t vs best | test MSE (reference) | cover95 |
|---|---|---|---|---|---|---|
| `gelu` | `tanh` | 2.980e-04 | 1.3e-05 | 0.00 | 3.666e-04 | 0.950 |
| `mish` | `tanh` | 3.006e-04 | 1.4e-05 | 0.54 | 3.715e-04 | 0.950 |
| `silu` | `tanh` | 3.021e-04 | 1.4e-05 | 0.86 | 3.747e-04 | 0.950 |
| `leakyrelu` | `leakyrelu` | 3.059e-04 | 8.4e-06 | 2.00 | 3.710e-04 | 0.950 |
| `prelu` | `relu` | 3.071e-04 | 1.2e-05 | 2.05 | 3.719e-04 | 0.950 |
| `mish` | `mish` | 3.120e-04 | 1.1e-05 | 3.19 | 3.729e-04 | 0.950 |
| `elu` | `gelu` | 3.137e-04 | 8.5e-06 | 3.99 | 3.771e-04 | 0.950 |
| `silu` | `mish` | 3.164e-04 | 1.4e-05 | 3.84 | 3.785e-04 | 0.950 |
| `gelu` | `mish` | 3.172e-04 | 1.7e-05 | 3.51 | 3.746e-04 | 0.950 |
| `mish` | `silu` | 3.205e-04 | 1.8e-05 | 3.94 | 3.801e-04 | 0.950 |

**Verdict: NOT resolved.** 3 of the 10 finalists sit within 2 standard errors of the winner on the selection split, so the winner is the best point estimate but is NOT statistically distinguishable from the rest of that group. What the comparison does resolve is the bottom of the field, not the top. The deployed pair is therefore justified on the best point estimate together with the structural argument -- smooth in the inputs, no dead units in a small network -- rather than on a leaderboard position that the data do not support.
