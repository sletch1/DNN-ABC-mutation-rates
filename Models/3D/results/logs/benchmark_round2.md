# 3-D two-stage surrogate: capacity floor (round 2)

Irreducible floor on `mse_mean` = **3.451e-04** (max achievable R^2 = 0.99683). Each row is the mean of 3 seeds, all on the three raw parameters (no derived feature).

| hidden | params | mse_mean | x floor | R^2 | cover95 | us/query |
|---|---|---|---|---|---|---|
| 256-128-64 | 42306 | 3.475e-04 | 1.01 | 0.99681 | 0.950 | 56 |
| 128-64 | 8898 | 3.582e-04 | 1.04 | 0.99671 | 0.951 | 42 |
| 64-32 | 2402 | 3.749e-04 | 1.09 | 0.99656 | 0.952 | 40 |
| 32-16 | 690 | 4.051e-04 | 1.17 | 0.99628 | 0.949 | 37 |
| 16-8 | 218 | 4.864e-04 | 1.41 | 0.99553 | 0.948 | 38 |
| 8-4 | 78 | 6.242e-04 | 1.81 | 0.99426 | 0.952 | 37 |
| linear | 8 | 4.862e-02 | 140.88 | 0.55319 | 0.952 | 29 |

**Smallest network within 10% of the best: `64-32` (2402 parameters, 1.09x floor, R^2 = 0.99656, 40 us/query).**

The linear row is the control: if it were competitive, no network would be justified at all. Any row whose `x floor` is near 1.0 is answering as well as the data permits, so among those the choice is purely about query cost.

