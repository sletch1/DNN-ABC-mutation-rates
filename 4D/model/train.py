"""DNN surrogate for the ABC features. Mirrors what the GP in the repo models:
at each theta, mean and sd (over replicate single-culture draws) of Y=(X/Z)^(1/4).
Network: theta (standardised) -> (m(theta), log s(theta)); trained on the per-design-point
mean and sd with MSE. Splits by DESIGN POINT (no leakage).

  python model/train.py --data data/slow_data_4D.csv --out results/surrogate_4d.pt
"""
import argparse
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen import io as dio
from sim import settings as S


class Net(nn.Module):
    def __init__(self, d, hidden=(128, 128, 64)):
        super().__init__()
        L, p = [], d
        for h in hidden:
            L += [nn.Linear(p, h), nn.GELU()]; p = h
        self.body = nn.Sequential(*L); self.head = nn.Linear(p, 2)

    def forward(self, x):
        return self.head(self.body(x))


class Surrogate:
    def __init__(self, net, lo, hi, tmu, tsd):
        self.net, self.lo, self.hi, self.tmu, self.tsd = net.eval(), lo, hi, tmu, tsd

    @torch.no_grad()
    def predict(self, theta):
        x = torch.tensor((np.atleast_2d(theta) - self.lo) / (self.hi - self.lo) * 2 - 1, dtype=torch.float32)
        o = self.net(x).numpy() * self.tsd + self.tmu
        return o[:, 0], np.exp(o[:, 1])           # mean(Y), sd(Y)


def load(path):
    c = torch.load(path, weights_only=False)
    net = Net(len(c["lo"])); net.load_state_dict(c["state"])
    return Surrogate(net, c["lo"], c["hi"], c["tmu"], c["tsd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    th, Y, _ = dio.load(a.data)
    box = np.array({1: [[-11.0, -7.0]], 3: S.BOX_3D, 4: S.BOX_4D}[th.shape[1]], float)
    lo, hi = box[:, 0], box[:, 1]
    T = np.column_stack([Y.mean(1), np.log(Y.std(1, ddof=1) + 1e-6)])
    tmu, tsd = T.mean(0), T.std(0)
    X = torch.tensor((th - lo) / (hi - lo) * 2 - 1, dtype=torch.float32)
    Tt = torch.tensor((T - tmu) / tsd, dtype=torch.float32)
    perm = rng.permutation(len(X)); nv = len(X) // 10
    va, tr = perm[:nv], perm[nv:]
    net = Net(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, bs = 1e9, None
    for ep in range(a.epochs):
        net.train(); idx = tr[rng.permutation(len(tr))]
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]; opt.zero_grad()
            nn.functional.mse_loss(net(X[b]), Tt[b]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(X[va]), Tt[va]).item()
        if v < best: best, bs = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(bs)
    with torch.no_grad():
        r2 = 1 - ((net(X[va]) - Tt[va]) ** 2).sum(0) / ((Tt[va] - Tt[va].mean(0)) ** 2).sum(0)
    print(f"best val MSE (standardised) {best:.4f}; R2 mean-head {r2[0]:.3f}, logsd-head {r2[1]:.3f}")
    torch.save(dict(state=net.state_dict(), lo=lo, hi=hi, tmu=tmu, tsd=tsd), a.out)


if __name__ == "__main__":
    main()
