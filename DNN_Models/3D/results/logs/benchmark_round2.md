# 3-D two-stage surrogate: capacity floor (round 2)

Irreducible floor on `mse_mean` = **1.393e-03** (max achievable R^2 = 0.99528). Each row is the mean of 2 seeds, all with the derived `log10(p_eff)` feature.

| hidden | params | mse_mean | x floor | R^2 | cover95 | us/query |
|---|---|---|---|---|---|---|
| 256-128-64 | 42562 | 1.500e-03 | 1.08 | 0.99491 | 0.949 | 56 |
| 128-64 | 9026 | 1.556e-03 | 1.12 | 0.99472 | 0.953 | 43 |
| 64-32 | 2466 | 1.573e-03 | 1.13 | 0.99467 | 0.952 | 41 |
| 32-16 | 722 | 1.593e-03 | 1.14 | 0.99460 | 0.949 | 40 |
| 16-8 | 234 | 1.693e-03 | 1.22 | 0.99426 | 0.952 | 40 |
| 8-4 | 86 | 1.714e-03 | 1.23 | 0.99419 | 0.952 | 39 |
| linear | 10 | 3.450e-02 | 24.77 | 0.88300 | 0.971 | 29 |

**Smallest network within 10% of the best: `32-16` (722 parameters, 1.14x floor, R^2 = 0.99460, 40 us/query).**

The linear row is the control: if it were competitive, no network would be justified at all. Any row whose `x floor` is near 1.0 is answering as well as the data permits, so among those the choice is purely about query cost.

