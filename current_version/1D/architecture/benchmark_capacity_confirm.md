# Capacity confirm: downstream accuracy/precision at Table-1 scale

Quick screen (run_all.py --quick settings) followed by a full paper-scale confirm (40 reps, full p x J grid) for every candidate. GP baseline: MSE=2.7019e-06, 95% CI=2.1371e-03.

| Architecture | Parameters | Downstream MSE | Downstream 95% CI | Resolved vs. deployed | max \|Delta/SE\| |
|---|---|---|---|---|---|
| 32-16 | 626 | 1.4331e-06 | 1.3001e-03 | 0/9 | 0.53 |
| 128-64 [deployed] | 8642 | 1.5142e-06 | 1.3081e-03 | --/9 | 0.00 |
| 64-32 | 2274 | 1.5653e-06 | 1.3767e-03 | 0/9 | 0.53 |
| 16 (1 layer) | 66 | 1.5790e-06 | 1.3750e-03 | 0/9 | 0.74 |
| 8 (1 layer) | 34 | 1.6684e-06 | 1.3929e-03 | 0/9 | 1.49 |
| 16-8 | 186 | 1.8098e-06 | 1.5283e-03 | 1/9 | 2.43 |
