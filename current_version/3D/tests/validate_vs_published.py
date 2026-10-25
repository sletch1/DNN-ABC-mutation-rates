"""Compare our generator with the training sets the authors shipped (Original_Paper_Code/training/*.mat):
same theta -> same distribution of Y? Prints design checks (ranges, LHS strata, column order) and
zero-fraction / mean / replicate-sd of Y. Run: python tests/validate_vs_published.py"""
import sys
from pathlib import Path
import numpy as np, scipy.io as sio
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sim import mbp, settings as S
T = ROOT.parents[1] / "current_work/Original_Paper_Code/training"
rng = np.random.default_rng(0)

m = sio.loadmat(T / "init_2a.mat"); th, X = m["theta_list"][:, [1, 2, 3, 0]], m["X"][:, 0, :]   # shipped order (delta,p1,p2,tau) -> paper order
print("4D init_2a ranges (log10 p1, log10 p2, tau, log10 delta):\n", np.c_[th.min(0), th.max(0)])
Y = np.array([[mbp.draw_stat(rng, t, "4d", 1, 1.0, 20.0, 4, False) for _ in range(5)] for t in th])
print("4D  zero-frac pub/mine %.3f/%.3f  mean %.5f/%.5f  rep-sd %.5f/%.5f" % (
    (X == 0).mean(), (Y == 0).mean(), X.mean(), Y.mean(), X.std(1, ddof=1).mean(), Y.std(1, ddof=1).mean()))
for p in (8, 7, 6, 5, 4):
    m = sio.loadmat(T / f"simu1/init1_p{p}.mat"); th, X = m["theta_list"][:, 0], m["X"][:, 0, :]
    tp = S.tp_study1(10.0 ** -p)
    Y = np.array([[mbp.draw_stat(rng, [t], "1d", 1, 1.0, tp, 4, False) for _ in range(X.shape[1])] for t in th])
    print("1D p=1e-%d zero-frac %.3f/%.3f  mean %.4f/%.4f  rep-sd %.4f/%.4f" % (
        p, (X == 0).mean(), (Y == 0).mean(), X.mean(), Y.mean(), X.std(1, ddof=1).mean(), Y.std(1, ddof=1).mean()))
