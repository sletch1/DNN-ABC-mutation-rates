"""3-D figures. (1) training design coloured by the feature m(theta); (2) posterior draws in
(log10 p1, log10 p2, tau) with the truth marked; (3) optional surrogate slices.
  python figures/fig3d.py design --data data/slow_data_3D.csv --out results/fig_design3d.png
  python figures/fig3d.py post --abc results/abc_3d_G.npz --obs data/obs_3d_case1.npz --case 1 --out results/fig_post3d.png
"""
import argparse, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from sim import settings as S

LAB = ["log10 p1", "log10 p2", "tau"]


def design(a):
    from datagen import io as dio
    th, Y, _ = dio.load(a.data); m = Y.mean(1)
    th = th[:, :3]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*th.T, c=m, s=6, cmap="viridis"); fig.colorbar(sc, label="mean of (X/Z)^(1/4)", shrink=0.6)
    ax.set_xlabel(LAB[0]); ax.set_ylabel(LAB[1]); ax.set_zlabel(LAB[2]); ax.view_init(22, -55)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")


def post(a):
    s = np.load(a.abc)["samples"]; t = S.SPEC_2["truths"][a.case]
    s = s.reshape(-1, s.shape[-1])[:, :3]; s = s[len(s) // 4:]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*s.T, s=2, alpha=0.2, c="tab:blue")
    ax.scatter([np.log10(t["p1"])], [np.log10(t["p2"])], [t["tau"]], c="red", s=80, marker="*", label="truth")
    ax.set_xlabel(LAB[0]); ax.set_ylabel(LAB[1]); ax.set_zlabel(LAB[2]); ax.legend(); ax.view_init(22, -55)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
d = sp.add_parser("design"); d.add_argument("--data"); d.add_argument("--out"); d.set_defaults(f=design)
p = sp.add_parser("post"); p.add_argument("--abc"); p.add_argument("--obs"); p.add_argument("--case", type=int, default=1)
p.add_argument("--out"); p.set_defaults(f=post)
a = ap.parse_args(); a.f(a)
