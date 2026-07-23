"""Head-to-head: can a DNN beat the GP on mean-curve fit for the 1-D response?

Metric = MEAN-CURVE MSE: how close each model's prediction is to the *denoised*
response curve (empirical mean of log10(d_bar) over all 10 replicates at each of
the 101 p grid points). This is the right target because per-test-point MSE is
dominated by irreducible replicate noise (std ~0.10) -- both models sit at that
floor, so it cannot discriminate mean-function fit quality.

Models only train on reps 1-6 (same as the deployed surrogate); the "truth" is
the best available estimate of the mean curve (all 10 reps). Lower is better.

Run: python benchmark_arch.py   (writes results/benchmark_arch.md)
"""
import sys
import warnings
from pathlib import Path

# --- make sibling code folders + paths.py importable (package uses flat imports) ---
_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import HeteroscedasticMLP, Standardizer, gaussian_nll
from surrogates import DNNSurrogate, fit_gp_surrogate
from paths import DATA, LOG_DIR

warnings.filterwarnings("ignore")
TRAIN_REPS = {1, 2, 3, 4, 5, 6}


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).unsqueeze(1)


def load():
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin(TRAIN_REPS)]
    x_tr = np.log10(tr["p"].to_numpy()); y_tr = np.log10(tr["d_bar"].to_numpy())
    va = df[df["rep"].isin({7, 8})]
    x_va = np.log10(va["p"].to_numpy()); y_va = np.log10(va["d_bar"].to_numpy())
    # denoised mean curve from ALL reps
    g = df.groupby("p")["d_bar"].apply(lambda s: np.mean(np.log10(s)))
    x_true = np.log10(g.index.to_numpy()); y_true = g.to_numpy()
    return (x_tr, y_tr), (x_va, y_va), (x_true, y_true)


def train_one(cfg, x_tr, y_tr, x_va, y_va, seed=0, epochs=700, warmup=60, patience=60):
    torch.manual_seed(seed); np.random.seed(seed)
    xs = Standardizer().fit(_t(x_tr)); ys = Standardizer().fit(_t(y_tr))
    xt, yt = xs.transform(_t(x_tr)), ys.transform(_t(y_tr))
    xv, yv = xs.transform(_t(x_va)), ys.transform(_t(y_va))
    model = HeteroscedasticMLP(**cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True)
    best, best_state, since = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            m, lv = model(xb)
            loss = torch.nn.functional.mse_loss(m, yb) if ep < warmup else gaussian_nll(m, lv, yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mv, lvv = model(xv); vl = gaussian_nll(mv, lvv, yv).item()
        sched.step(vl)
        if ep >= warmup and vl < best - 1e-5:
            best, since, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        elif ep >= warmup:
            since += 1
            if since >= patience:
                break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return DNNSurrogate(model, xs, ys, sd_scale=1.0)


def curve_mse(pred_fn, x_true, y_true):
    m, _ = pred_fn(x_true)
    return float(np.mean((m - y_true) ** 2))


def main():
    (x_tr, y_tr), (x_va, y_va), (x_true, y_true) = load()
    noise = float(np.mean(pd.read_csv(DATA).groupby("p")["d_bar"]
                          .apply(lambda s: np.var(np.log10(s), ddof=1))))
    print(f"train n={len(x_tr)}  grid pts={len(x_true)}  irreducible noise var(log)~{noise:.4f}\n")

    rows = []
    # GP baseline (the target to beat)
    gp = fit_gp_surrogate(x_tr, y_tr, budget=None)
    rows.append(("GP (GPS-ABC baseline)", curve_mse(gp.predict, x_true, y_true)))

    configs = {
        "DNN A: relu+BN+drop (64,64,32) [old]":
            dict(hidden_dims=(64, 64, 32), activation="relu", use_bn=True, dropout=0.1),
        "DNN B: tanh (64,64,32)":
            dict(hidden_dims=(64, 64, 32), activation="tanh", use_bn=False, dropout=0.0),
        "DNN C: silu (128,128,64)":
            dict(hidden_dims=(128, 128, 64), activation="silu", use_bn=False, dropout=0.0),
        "DNN D: gelu (128,64)":
            dict(hidden_dims=(128, 64), activation="gelu", use_bn=False, dropout=0.0),
        "DNN E: tanh (256,128)":
            dict(hidden_dims=(256, 128), activation="tanh", use_bn=False, dropout=0.0),
    }
    trained = {}
    for name, cfg in configs.items():
        s = train_one(cfg, x_tr, y_tr, x_va, y_va, seed=0)
        trained[name] = (s, cfg)
        rows.append((name, curve_mse(s.predict, x_true, y_true)))

    # Deep ensembles (average the mean heads) of the two best smooth configs
    for label, cfg in [("silu (128,128,64)", configs["DNN C: silu (128,128,64)"]),
                       ("tanh (256,128)", configs["DNN E: tanh (256,128)"])]:
        members = [train_one(cfg, x_tr, y_tr, x_va, y_va, seed=s) for s in range(5)]
        def ens_pred(x, members=members):
            ms = np.stack([m.predict(x)[0] for m in members])
            sds = np.stack([m.predict(x)[1] for m in members])
            mbar = ms.mean(0)
            # predictive var = mean(var) + var(means)  (deep-ensemble rule)
            sd = np.sqrt((sds ** 2).mean(0) + ms.var(0))
            return mbar, sd
        rows.append((f"DNN ENSEMBLE x5: {label}", curve_mse(ens_pred, x_true, y_true)))

    rows.sort(key=lambda r: r[1])
    gp_mse = [v for n, v in rows if n.startswith("GP")][0]
    lines = ["| model | mean-curve MSE | vs GP |", "|---|---|---|"]
    for name, mse in rows:
        rel = "—" if name.startswith("GP") else (f"{(1 - mse/gp_mse)*100:+.0f}% better"
                                                 if mse < gp_mse else f"{(mse/gp_mse - 1)*100:+.0f}% worse")
        lines.append(f"| {name} | {mse:.5e} | {rel} |")
    table = "\n".join(lines)
    print(table)
    (LOG_DIR / "benchmark_arch.md").write_text(
        f"# Architecture benchmark: DNN vs GP on mean-curve fit\n\n"
        f"Irreducible replicate noise var(log10 d_bar) ~ {noise:.4f} "
        f"(per-point MSE floor).\n\n{table}\n")
    print("\nwritten -> results/benchmark_arch.md")


if __name__ == "__main__":
    main()
