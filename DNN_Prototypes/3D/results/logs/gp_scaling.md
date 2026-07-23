# GP-vs-DNN scaling (targets #1 and #2)

| model | budget | surface MSE | fit (s) | query (us/pt) |
|---|---|---|---|---|
| GP | 50 | 6.4304e-03 | 1.0 | 1.9 |
| GP | 100 | 2.6880e-03 | 0.1 | 3.2 |
| GP | 200 | 2.0110e-03 | 1.0 | 7.1 |
| GP | 300 | 1.7564e-03 | 1.3 | 12.3 |
| GP | 500 | 1.7526e-03 | 4.1 | 26.8 |
| GP | 1000 | 1.6263e-03 | 23.1 | 89.2 |
| GP | 1000 | 1.6263e-03 | 23.2 | 88.8 |
| DNN | 5000 | 1.4624e-03 | - | 17.9 |

- DNN surface MSE 1.4624e-03 beats the best GP (budget 1000, 1.6263e-03) by 10%.
- GP fit time grows ~cubically with budget; DNN query time (17.9us/pt) is independent of training-set size.
