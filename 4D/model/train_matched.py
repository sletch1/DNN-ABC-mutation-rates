"""Matched-budget control: the chosen architecture trained on only N design points (default 1,500: the original GPS-ABC design size),
everything else identical (same validation set for early stopping and conformal calibration, same test designs). Separates the
effect of having more training data from the effect of the surrogate itself.
    python model/train_matched.py --data data/slow_data_4D.csv --arch res_128x3_gelu --n 1500 --out results/surrogate_4d_n1500.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arch", required=True)
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--corr", action="store_true", help="model the correlation of the two features"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X, zero, y, des, idx = A.load(a.data); floor = A.noise_floor(y, zero, des, idx)
    keep = np.unique(des[idx["train"]])[:a.n]                       # first n training designs (design ids are in LHS order)
    idx = dict(idx); idx["train"] = idx["train"][np.isin(des[idx["train"]], keep)]
    arch = dict(A.ARCHS[a.arch], corr=a.corr)
    net, sc, secs, ep, _ = A.fit(arch, X, zero, y, idx, seed=a.seed)
    ev = A.evaluate(net, sc, X, zero, y, des, idx, floor); D = X.shape[1]
    box = [[-11, -7], [-11, -7], [0.1, 19.9]] + ([[float(np.log10(.5)), float(np.log10(2))]] if D == 4 else [])
    torch.save(dict(state=net.state_dict(), arch=arch, arch_name=a.arch, scalers=sc, in_dim=D,
                    conformal_scale=[ev["conformal_scale_f1"], ev["conformal_scale_f2"]], box=box), a.out)
    json.dump(dict(arch=a.arch, n_train_designs=int(len(keep)), n_train_rows=int(len(idx["train"])), metrics=ev, floor=floor), open(a.out.replace(".pt", "_metrics.json"), "w"), indent=1)
    print(f"{a.arch}{' +rho' if a.corr else ''} trained on {len(keep)} designs ({len(idx['train'])} rows): test x_floor {ev['test_f1_x_floor']:.3f} | {ev['test_f2_x_floor']:.3f}, joint NLL {ev['test_joint_nll']:.4f}, cov {ev['test_f1_cov95_conf']:.3f}, {secs:.0f}s")
