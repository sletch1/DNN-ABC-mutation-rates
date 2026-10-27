"""Joint test NLL of the two ABC features for saved surrogates, on the same held-out design points (rho = 0 for models without a
correlation head).   python model/eval_joint.py --data data/slow_data_4D.csv results/surrogate_4d.pt results/surrogate_4d_rho.pt ..."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A
from nets import HurdleNet

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("models", nargs="+"); a = ap.parse_args()
    X, zero, y, des, idx = A.load(a.data); ii = idx["test"]; nz = zero[ii] < .5
    for f in a.models:
        c = torch.load(f, weights_only=False); net = HurdleNet(in_dim=c["in_dim"], **c["arch"]); net.load_state_dict(c["state"]); net.eval()
        out = A.predict(net, c["scalers"], X[ii]); mu, sd = out[1][nz], out[2][nz]; rho = out[3][nz] if len(out) > 3 else np.zeros(nz.sum())
        z = (y[ii][nz] - mu) / sd
        nll = np.mean(np.log(2 * np.pi) + np.log(sd).sum(1) + 0.5 * np.log1p(-rho ** 2) + 0.5 * (z[:, 0] ** 2 - 2 * rho * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / (1 - rho ** 2))
        print(f"{Path(f).name:34s} test joint NLL {nll:8.4f}   (corr head: {len(out) > 3}; mean |rho| {np.abs(rho).mean():.2f})")
