"""One member of a bootstrap ensemble of surrogates, for the epistemic-uncertainty test (status item 8.3).

The chosen architecture is trained on a bootstrap resample of the TRAINING DESIGN POINTS (designs drawn with replacement, all their replicate
rows kept), with a different seed per member. The validation designs (early stopping and conformal calibration) and the test designs are the
same as for the deployed surrogate, so members differ only through the training sample and the initialisation, which is the variation in the
network's own parameters that a single trained network ignores.
    for m in 0 1 2 ... 9; do python model/train_ensemble.py --data data/slow_data_4D.csv --arch res_128x3_gelu --member $m --out results/ensemble/member_$m.pt; done
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arch", required=True)
    ap.add_argument("--member", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    X, zero, y, des, idx = A.load(a.data); floor = A.noise_floor(y, zero, des, idx)
    rng = np.random.default_rng(10_000 + a.member)
    tr_designs = np.unique(des[idx["train"]]); boot = rng.choice(tr_designs, size=len(tr_designs), replace=True)
    rows = np.concatenate([idx["train"][des[idx["train"]] == d] for d in boot])        # duplicates kept (bootstrap multiplicity)
    idx = dict(idx); idx["train"] = rows
    arch = dict(A.ARCHS[a.arch], corr=False)
    net, sc, secs, ep, _ = A.fit(arch, X, zero, y, idx, seed=100 + a.member)
    ev = A.evaluate(net, sc, X, zero, y, des, idx, floor); D = X.shape[1]
    box = [[-11, -7], [-11, -7], [0.1, 19.9]] + ([[float(np.log10(.5)), float(np.log10(2))]] if D == 4 else [])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), arch=arch, arch_name=a.arch, scalers=sc, in_dim=D,
                    conformal_scale=[ev["conformal_scale_f1"], ev["conformal_scale_f2"]], box=box), a.out)
    json.dump(dict(arch=a.arch, member=a.member, n_unique_designs=int(len(np.unique(boot))), n_rows=int(len(rows)), seconds=secs, epochs=ep, metrics=ev, floor=floor),
              open(a.out.replace(".pt", "_metrics.json"), "w"), indent=1)
    print(f"member {a.member}: {len(np.unique(boot))} unique designs, {len(rows)} rows, {secs:.0f} s, {ep} epochs, test x_floor {ev['test_f1_x_floor']:.3f} | {ev['test_f2_x_floor']:.3f}", flush=True)
