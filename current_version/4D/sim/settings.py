"""Settings follow the PAPER (Lu, Zhu, Wu 2023, JTB 565:111467) wherever paper and code
disagree (rule set by the user; see ../paper_code_discrepancy.md). Everything the paper does not
state comes from simulation2_server.m / simulation1_server.m."""
import numpy as np
from scipy.optimize import brentq

A = 1.0
Z0 = 1
ROOT = 4            # obs_X = sqrt(sqrt(X/Z)) in every published script

# Study 2 (3-param, delta fixed at 1) and 4-param (adds delta)
DT_RANGE = [np.log10(0.5), np.log10(2.0)]
P_RANGE_2 = [[-11, -7], [-11, -7], [0.1, 19.9]]  # log10 p1, log10 p2, tau  (paper; code used tau in [4,18])
TP_2 = 20.0
BOX_3D = P_RANGE_2
BOX_4D = P_RANGE_2 + [DT_RANGE]                  # PAPER order theta=(log10 p1, log10 p2, tau, log10 delta); the code uses delta first
SPEC_2 = dict(n_train=1500, n_refine=10, ksi=0.3, eps=1e-8, M=100, N=20000,
              num_rep=5, J=100, theta_sigma=20.0,
              # Paper: s=(0.8,3.5,3.5,4) without naming the order. Only the code's order (delta,p1,p2,tau)
              # gives sensible widths (0.8 on the 0.6-wide log-delta box would be absurd for p1), so the
              # numbers are assigned in that order and re-sorted to the paper's theta order here.
              trans_step=[3.5, 3.5, 4.0, 0.8],
              theta_old=[-9, -8, 14, np.log10(0.8)],
              # paper priors: TN(log10 MOM,20,[-11,-7]) x2, TN(10,20,[0.1,19.9]), TN(0,20,[log10 .5,log10 2])
              prior_mu_fixed=[None, None, 10.0, 0.0],   # None -> log10(p_hat_MOM) of the dataset
              truths={1: dict(p1=1e-9, p2=1e-8, tau=14.0, delta=0.8),   # simulation2.m case 1 (0.01e-7, 0.1e-7)
                      2: dict(p1=1e-9, p2=5e-8, tau=14.0, delta=0.8)})  # case 2 (0.01e-7, 0.5e-7)

# Study 1: p in {1e-4..1e-8}; range & step depend on p (simulation1_server.m)
S1_RANGE = {8: ([-10, -6], 0.35), 7: ([-9, -5], 0.30), 6: ([-8, -4], 0.30),
            5: ([-6, -4], 0.25), 4: ([-5, -3], 0.20)}
SPEC_1 = dict(n_train=1500, n_refine=10, ksi=0.25, eps=0.0, M=100, N=30000,
              num_rep=10, theta_sigma=20.0, c=100.0, Js=(10, 50, 100))


def tp_study1(p, a=A, Z0=Z0, c=100.0):
    """simulation1.m: root of Z0*(e^{at}-e^{at(1-2p)}) - c, fzero start 20."""
    f = lambda t: Z0 * (np.exp(a * t) - np.exp(a * t * (1 - 2 * p))) - c
    return brentq(f, 1.0, 60.0, xtol=1e-12)
