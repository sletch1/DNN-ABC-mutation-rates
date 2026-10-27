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

Further differences and defects found while reproducing the study
-----------------------------------------------------------------

Step widths of the proposal
 - Paper gives s = (0.8, 3.5, 3.5, 4) without naming the order; the code's order is (log10 delta, log10 p1, log10 p2, tau).
   Only that reading is sensible (0.8 on the 0.6-wide log-delta box would be absurd for p1), so we read it that way.

Meaning of p
 - Paper's Algorithm 4 (fast) draws round(p*z) mutation events; Algorithm 2/3 (exact) lets each daughter mutate with probability p.
   At the same p the fast simulator gives about 2.3-2.5x fewer mutants (measured at tp = 9; tests/test_sim.py).
   We use Algorithm 4 as published, because the paper's Study 2 does; p is therefore "in the sense of Algorithm 4".

Released code that does not run as shipped
 - Original_Paper_Code/ABC_mu2.m (three-parameter GPS-ABC): uses an undefined variable (dt_range) and calls get_theta2 with
   the wrong arguments. The four-parameter ABC_mu2a.m runs and is what we use.
 - Original_Paper_Code/mut_bMBP_rev.m draws the second-stage cell count by summing geornd over a vector of about exp(tau)
   entries; at the paper's tau <= 19.9 that array is gigabytes per worker and MATLAB runs out of memory. We use a distribution-identical
   negative-binomial draw (4D/gps_abc/patched/mut_bMBP_rev.m). The original is untouched.

Training data
 - The training set the authors ship (training/init_2a.mat) has tau in [4, 18], matching simulation2_server.m, not the paper's [0.1, 19.9].
   We follow the paper; our generator is tested against the shipped data at their design points (4D/tests/validate_vs_published.py).
