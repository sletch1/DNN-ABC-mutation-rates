# 3-D capacity confirm: downstream accuracy at Table-1 scale

Quick screen (reps=2) followed by a full paper-scale confirm (reps=16, Table 1's exact settings) for every candidate that survives it. GP baseline mean rmse_log = 0.9897.

| Architecture | Parameters | Mean rmse_log (all params/truths) | Resolved vs. deployed | max \|Delta/SE\| |
|---|---|---|---|---|
| 32-16 | 690 | 1.0667 | 0/9 | 1.63 |
| 128-64 | 8898 | 1.1048 | 0/9 | 1.16 |
| 16-8 | 218 | 1.1326 | 0/9 | 1.06 |
| 64-32 [deployed] | 2402 | 1.1418 | --/-- | 0.00 |
| 8-4 | 78 | 1.1488 | 0/9 | 1.32 |
| 256-128-64 | 42306 | 1.2083 | 0/9 | 1.55 |
