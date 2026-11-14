"""Head-to-head test of amortised neural posterior estimators on the 4-parameter two-stage model.

Every candidate sees exactly the same rows (d_bar > 0), the same two features x = (log10 d_bar, log10 s), the same design-point split and
the same uniform prior on the paper's box, and is scored on the same held-out designs:
  * held-out mean log q(theta | x) over unseen design points, renormalised to the prior box (the selection criterion);
  * simulation-based calibration on the same held-out designs: coverage of central 95% intervals and a chi-square test of rank uniformity;
  * posterior contraction 1 - E[Var_post]/Var_prior (sharpness);
  * training time and sampling time per data set.
Candidates (see the manuscript for why each was chosen):
  nsf, maf, mdn     sbi conditional density estimators (neural spline flow, masked autoregressive flow, mixture density network)
  fmpe              flow-matching posterior estimation (Wildberger et al., 2023), sbi
  npse              neural posterior score estimation, a conditional diffusion model (Sharrock et al., 2022), sbi
  bf_affine, bf_spline   BayesFlow coupling flows (Radev et al., 2020)
    python inference/npe_candidates.py CANDIDATE --threads 2        (from 4D/; writes results/npe_candidates/CANDIDATE.json)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model")); sys.path.insert(0, str(HERE))
import arch_search as A
from npe import BOX
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def summarise(R, V, L, box):
    out = {}
    for i, (lo, hi) in enumerate(box):
        h = np.histogram(R[:, i], bins=np.linspace(-.5, L + .5, 11))[0]; e = len(R) / 10
        out[NAMES[i]] = dict(cov95=float(np.mean((R[:, i] >= L * .025) & (R[:, i] <= L * .975))), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                             contraction=float(1 - V[:, i].mean() / ((hi - lo) ** 2 / 12)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("candidate"); ap.add_argument("--data", default=str(HERE.parent / "data" / "slow_data_4D.csv"))
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--n_lp", type=int, default=300, help="held-out designs scored for log q")
    ap.add_argument("--n_sbc", type=int, default=300, help="held-out designs used for SBC"); ap.add_argument("--L", type=int, default=500); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="training seed (the held-out designs are the same for every seed)")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "npe_candidates")); a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads); import torch; torch.set_num_threads(a.threads)
    X, zero, y, des, idx = A.load(a.data); keep = zero < .5; box = BOX[4]; lo, hi = zip(*box)
    tr = np.concatenate([idx["train"], idx["val"]]); tr = tr[keep[tr]]
    te = idx["test"]; te = te[keep[te]]; te = np.random.default_rng(0).permutation(te)[:max(a.n_lp, a.n_sbc)]
    c = a.candidate; t0 = time.time()
    if c.startswith("bf_"):
        import bflow
        cfg = dict(depth=8, transform="affine", width=256) if c == "bf_affine" else dict(depth=6, transform="spline", width=128)
        post = bflow.fit(X[tr], y[tr], box, seed=a.seed, patience=50, **cfg)
        draw = lambda x_, n, s: post.sample(n, x_, seed=s); logq = lambda th, x_: post.log_prob(th, x_)
    else:
        from sbi.inference import NPE, FMPE, NPSE
        from sbi.utils import BoxUniform
        prior = BoxUniform(torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)); torch.manual_seed(a.seed)
        inf = {"nsf": lambda: NPE(prior=prior, density_estimator="nsf", show_progress_bars=False), "maf": lambda: NPE(prior=prior, density_estimator="maf", show_progress_bars=False),
               "mdn": lambda: NPE(prior=prior, density_estimator="mdn", show_progress_bars=False), "fmpe": lambda: FMPE(prior=prior, show_progress_bars=False),
               "npse": lambda: NPSE(prior=prior, show_progress_bars=False)}[c]()
        inf.append_simulations(torch.tensor(X[tr], dtype=torch.float32), torch.tensor(y[tr], dtype=torch.float32))
        p = inf.build_posterior(inf.train(max_num_epochs=a.epochs, training_batch_size=512))
        draw = lambda x_, n, s: (torch.manual_seed(s + 1000 * a.seed), p.sample((n,), x=torch.tensor(np.asarray(x_, np.float32).reshape(1, -1)), show_progress_bars=False))[1].numpy()
        logq = lambda th, x_: np.array([float(p.log_prob(torch.tensor(t[None], dtype=torch.float32), x=torch.tensor(xx[None], dtype=torch.float32))) for t, xx in zip(th, x_)])
    train_s = time.time() - t0; print(c, "trained", round(train_s), "s", flush=True)
    t1 = time.time(); lp = float(np.mean(logq(X[te[:a.n_lp]], y[te[:a.n_lp]]))); lp_s = time.time() - t1; print(c, "heldout logq", round(lp, 3), flush=True)
    R, V = [], []; t1 = time.time()
    for k, i in enumerate(te[:a.n_sbc]):
        s = draw(y[i:i + 1], a.L, k); R.append((s < X[i]).sum(0)); V.append(s.var(0))
    sample_s = (time.time() - t1) / a.n_sbc
    res = dict(candidate=c, seed=a.seed, train_s=train_s, heldout_logq=lp, n_lp=int(a.n_lp), n_sbc=int(a.n_sbc), L=a.L, sample_s_per_dataset_L=sample_s, logq_s=lp_s, sbc=summarise(np.array(R), np.array(V), a.L, box))
    Path(a.out).mkdir(parents=True, exist_ok=True); json.dump(res, open(Path(a.out) / (f"{c}.json" if a.seed == 0 else f"{c}_seed{a.seed}.json"), "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
