# 3-D architecture benchmark, round 2: deep ensemble vs single network

| model | mean-surface MSE |
|---|---|
| single ResMLP (mean of 5 seeds) | 1.45888e-03 +/- 1.2e-05 |
| single ResMLP (best seed) | 1.43688e-03 |
| deep ensemble x5 | 1.44174e-03 (+1.2% vs single mean) |

**Verdict:** ensemble's 1.2% gain is below the 5% bar for its 5x cost -> keep the single network.
