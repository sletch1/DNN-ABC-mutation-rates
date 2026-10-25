Original vs Code Discrepenacies:

Range on t (Tau):
 - Paper uses [0.1, 19.9]
 - Simulation2_server.m (code) uses [4,18]

Wrong-decision Threshold
 - Paper uses e = 0.3
 - Code uses e - 0.35

 Parameter order
  - Paper is (p1, p2, t, delta)
  - Code is (delta, p1, p2, t)

Prior on t (tau):
  - Paper: TN(10, 20, [0.1, 19.9])
  - [4,18]

All construction defaults to paper mechanism