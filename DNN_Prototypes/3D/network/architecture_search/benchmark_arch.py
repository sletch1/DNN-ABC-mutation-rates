"""Head-to-head on the 3-D response surface: which architecture fits best, and
how does it compare to a budget-limited GP (the GPS-ABC baseline)?

Metric = MEAN-SURFACE MSE: distance of each model's prediction to the *denoised*
surface (empirical mean of log10(d_bar) over all 10 replicates at each of the
1000 (p, a, delta) grid points). Per-test-point MSE is dominated by irreducible
replicate noise; the denoised surface isolates mean-function fit quality.

All neural models train on reps 1-5 and early-stop on reps 6-8 (exactly the
deployed surrogate's budget). The GP rows show the core dnn_improvement.md story:
a GP capped at 300 space-filling points (its practical O(n^3) ceiling) vs the DNN
that learns the surface from all ~5000 rows. Lower is better.

Run: python benchmark_arch.py   (writes results/logs/benchmark_arch.md)
"""
import sys
import time
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

from model import HeteroscedasticResMLP, Standardizer, gaussian_nll
from surrogates import DNNSurrogate3D, fit_gp_surrogate_3d
from paths import DATA, LOG_DIR

warnings.filterwarnings("ignore")
TRAIN_REPS, VAL_REPS = {1, 2, 3, 4, 5}, {6, 7, 8}


def _feats(df):
    return np.column_stack([np.log10(df["p"].to_numpy()),
                            df["a"].to_numpy(), df["delta"].to_numpy()]).astype(np.float32)


def load():
    df = pd.read_csv(DATA)
    tr = df[df["rep"].isin(TRAIN_REPS)]
    va = df[df["rep"].isin(VAL_REPS)]
    X_tr, y_tr = _feats(tr), np.log10(tr["d_bar"].to_numpy()).astype(np.float32)
    X_va, y_va = _feats(va), np.log10(va["d_bar"].to_numpy()).astype(np.float32)
    # denoised mean surface from ALL reps, one point per (p, a, delta)
    g = df.groupby(["p", "a", "delta"])["d_bar"].apply(lambda s: np.mean(np.log10(s)))
    keys = np.array(list(g.index))
    X_true = np.column_stack([np.log10(keys[:, 0]), keys[:, 1], keys[:, 2]]).astype(np.float32)
    y_true = g.to_numpy()
    return (X_tr, y_tr), (X_va, y_va), (X_true, y_true)


def _tx(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _ty(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32).unsqueeze(1)


def train_one(cfg, X_tr, y_tr, X_va, y_va, seed=0, epochs=900, warmup=80, patience=60):
    torch.manual_seed(seed); np.random.seed(seed)
    xs = Standardizer().fit(_tx(X_tr)); ys = Standardizer().fit(_ty(y_tr))
    xt, yt = xs.transform(_tx(X_tr)), ys.transform(_ty(y_tr))
    xv, yv = xs.transform(_tx(X_va)), ys.transform(_ty(y_va))
    model = HeteroscedasticResMLP(in_dim=3, **cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)
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
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return DNNSurrogate3D(model, xs, ys, sd_scale=1.0)


def surface_mse(pred_fn, X_true, y_true):
    m, _ = pred_fn(X_true)
    return float(np.mean((m - y_true) ** 2))


def main():
    (X_tr, y_tr), (X_va, y_va), (X_true, y_true) = load()
    noise = float(pd.read_csv(DATA).groupby(["p", "a", "delta"])["d_bar"]
                  .apply(lambda s: np.var(np.log10(s), ddof=1)).mean())
    print(f"train n={len(X_tr)}  surface pts={len(X_true)}  "
          f"irreducible noise var(log)~{noise:.4f}\n")

    rows = []

    # GP baselines at two budgets: the deployed GPS-ABC budget (300) and a larger,
    # much slower one (1000) to expose the O(n^3) cost/accuracy tradeoff.
    for budget in (300, 1000):
        t0 = time.time()
        gp = fit_gp_surrogate_3d(X_tr, y_tr, budget=budget)
        dt = time.time() - t0
        rows.append((f"GP (budget {gp.budget}, fit {dt:.1f}s)",
                     surface_mse(gp.predict, X_true, y_true)))
        print(f"  GP budget {gp.budget} fit in {dt:.1f}s")

    configs = {
        "ResMLP silu+LN (w128, 3blk) [chosen]":
            dict(width=128, n_blocks=3, activation="silu", use_ln=True, dropout=0.0),
        "ResMLP silu, NO LayerNorm (w128, 3blk)":
            dict(width=128, n_blocks=3, activation="silu", use_ln=False, dropout=0.0),
        "ResMLP gelu+LN (w128, 3blk)":
            dict(width=128, n_blocks=3, activation="gelu", use_ln=True, dropout=0.0),
        "ResMLP silu+LN small (w64, 2blk)":
            dict(width=64, n_blocks=2, activation="silu", use_ln=True, dropout=0.0),
        "ResMLP silu+LN big (w256, 4blk)":
            dict(width=256, n_blocks=4, activation="silu", use_ln=True, dropout=0.0),
        "ResMLP relu+LN (w128, 3blk)":
            dict(width=128, n_blocks=3, activation="relu", use_ln=True, dropout=0.0),
    }
    for name, cfg in configs.items():
        t0 = time.time()
        s = train_one(cfg, X_tr, y_tr, X_va, y_va, seed=0)
        rows.append((name, surface_mse(s.predict, X_true, y_true)))
        print(f"  {name}: {time.time()-t0:.1f}s")

    rows.sort(key=lambda r: r[1])
    gp300 = [v for n, v in rows if n.startswith("GP (budget 3")]
    gp_ref = gp300[0] if gp300 else [v for n, v in rows if n.startswith("GP")][0]
    lines = ["| model | mean-surface MSE | vs GP(300) |", "|---|---|---|"]
    for name, mse in rows:
        if name.startswith("GP (budget 3"):
            rel = "— (baseline)"
        else:
            rel = (f"{(1 - mse/gp_ref)*100:+.0f}% better" if mse < gp_ref
                   else f"{(mse/gp_ref - 1)*100:+.0f}% worse")
        lines.append(f"| {name} | {mse:.5e} | {rel} |")
    table = "\n".join(lines)
    print("\n" + table)
    (LOG_DIR / "benchmark_arch.md").write_text(
        f"# 3-D architecture benchmark: residual MLP vs GP on mean-surface fit\n\n"
        f"Irreducible replicate noise var(log10 d_bar) ~ {noise:.4f} "
        f"(per-point MSE floor). Denoised surface = mean over 10 reps at each of "
        f"{len(X_true)} grid points.\n\n{table}\n")
    print("\nwritten -> results/logs/benchmark_arch.md")


if __name__ == "__main__":
    main()
