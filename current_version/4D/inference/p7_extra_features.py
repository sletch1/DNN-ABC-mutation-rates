"""Problem 7: are p1 and tau unidentified because of the summary (dbar, s), or because of the model?
No new simulation: slow_data_4D.csv stores all J=100 per-culture values d_1..d_100 of every data set. We train NPE twice on the
SAME rows and split: (a) x = (log10 dbar, log10 s) as in the main comparison, (b) x = (a) + extras (fraction of cultures with no
mutants, quantiles q50/q75/q90, max, skewness). Both are scored by SBC on held-out DESIGNS (rows are uniform prior draws), with
posterior contraction 1 - E[Var_post]/Var_prior for each parameter. Either outcome is reportable.
    python inference/p7_extra_features.py --data data/slow_data_4D.csv --out results/p7_extra_features.json [--K 1000]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import chi2, skew
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch_search as A
from npe import BOX

NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]
FLOOR = 1e-8

def extras(D):
    z = (D == 0).mean(1)
    q = np.quantile(D, [.5, .75, .9], axis=1).T
    mx = D.max(1)
    sd = D.std(1, ddof=1)
    sk = np.where(sd > 0, skew(D, axis=1, bias=False), 0.0)
    lg = lambda v: np.log10(np.maximum(v, FLOOR))
    return np.column_stack([z, lg(q), lg(mx), sk]).astype(np.float32)

def fit(th, x, box, seed, est="nsf"):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    lo, hi = zip(*box)
    prior = BoxUniform(torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32))
    torch.manual_seed(seed)
    inf = NPE(prior=prior, density_estimator=est, show_progress_bars=False)
    inf.append_simulations(torch.tensor(th), torch.tensor(x))
    de = inf.train(max_num_epochs=300, training_batch_size=512)
    return inf.build_posterior(de)

def sbc(post, th, x, L=1000):
    R, V, LQ = [], [], []
    for i in range(len(th)):
        xi = torch.tensor(x[i:i + 1])
        s = post.sample((L,), x=xi, show_progress_bars=False).numpy()
        R.append((s < th[i]).sum(0)); V.append(s.var(0))
        LQ.append(float(post.log_prob(torch.tensor(th[i:i + 1]), x=xi)))
    return np.array(R), np.array(V), np.mean(LQ), L

def summarise(R, V, L, box):
    out = {}
    for i, (lo, hi) in enumerate(box):
        h = np.histogram(R[:, i], bins=np.linspace(-.5, L + .5, 11))[0]; e = len(R) / 10
        out[NAMES[i]] = dict(contraction=float(1 - V[:, i].mean() / ((hi - lo) ** 2 / 12)),
                             chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                             cov95=float(np.mean((R[:, i] >= L * .025) & (R[:, i] <= L * .975))))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=1000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    X, zero, y, des, idx = A.load(a.data); box = BOX[X.shape[1]]
    D = pd.read_csv(a.data, usecols=lambda c: c.startswith("d_") and c != "d_bar").to_numpy(np.float64)
    E = extras(D); keep = zero < .5
    tr = np.concatenate([idx["train"], idx["val"]]); tr = tr[keep[tr]]
    te = idx["test"]; te = te[keep[te]]
    te = np.random.default_rng(0).choice(te, min(a.K, len(te)), replace=False)
    # standardise extras on the training rows (the flow expects O(1) inputs)
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9; En = ((E - mu) / sd).astype(np.float32)
    feats = {"(dbar, s)": y, "(dbar, s) + extras": np.column_stack([y, En]).astype(np.float32)}
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "extras": ["frac_zero_cultures", "log10 q50", "log10 q75", "log10 q90", "log10 max", "skewness"], "seeds": a.seeds}
    for name, F in feats.items():
        per = []
        for sd_ in range(a.seeds):
            t0 = time.time(); post = fit(X[tr], F[tr], box, sd_)
            R, V, lq, L = sbc(post, X[te], F[te]); s = summarise(R, V, L, box); s["heldout_logq"] = lq; s["train_s"] = time.time() - t0
            per.append(s); print(name, "seed", sd_, json.dumps({k: (round(v["contraction"], 3) if isinstance(v, dict) else round(v, 2)) for k, v in s.items()}), flush=True)
        res[name] = per
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
