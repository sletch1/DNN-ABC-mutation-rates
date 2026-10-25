"""Amortised neural posterior estimation (NPE) baseline for the two-stage model (sbi, single round).

WHY NPE IS IN THIS STUDY. In one dimension a surrogate-plus-MCMC pipeline is a reasonable way to reuse simulator
output; in the higher-dimensional two-stage model the same simulator output can instead be used to learn the
posterior itself. NPE is the natural comparator because (i) it consumes exactly the training simulations DNN-ABC
does (same CSV, same two summary features, same split), (ii) it removes the two components DNN-ABC adds on top of
the surrogate -- the MCMC chain and the surrogate's predictive-variance model -- so it tests whether those are
needed, and (iii) once trained it returns i.i.d. posterior draws for any new dataset in milliseconds. It cannot be
used to beat the information in the summary statistic, which makes it a useful ceiling for what any method using
(d_bar, s) can recover.

Training set: rows with d_bar > 0 only. NPE fits q(theta | x) by maximising E log q(theta | x) over joint draws; the
population minimiser is p(theta | x) for every x in the support of the training x, whatever the marginal of x, so
discarding the d_bar = 0 rows (where log d_bar is undefined) leaves the posterior at any observed x with d_bar > 0
unchanged. All observed datasets here have d_bar > 0.
Prior: uniform on the paper's box (the paper's TN(.,20,box) prior is flat to within 2% across a box of width <= 20 sd).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import arch_search as A

BOX = {3: [[-11, -7], [-11, -7], [0.1, 19.9]], 4: [[-11, -7], [-11, -7], [0.1, 19.9], [float(np.log10(.5)), float(np.log10(2))]]}


def data(path):
    X, zero, y, des, idx = A.load(path)
    keep = zero < .5
    return X, y, keep, idx


def train(path, estimator="nsf", seed=0, max_epochs=300):
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    X, y, keep, idx = data(path); D = X.shape[1]; lo, hi = zip(*BOX[D])
    prior = BoxUniform(torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32))
    tr = np.concatenate([idx["train"], idx["val"]]); tr = tr[keep[tr]]
    th = torch.tensor(X[tr], dtype=torch.float32); xs = torch.tensor(y[tr], dtype=torch.float32)
    torch.manual_seed(seed); t0 = time.time()
    inf = NPE(prior=prior, density_estimator=estimator, show_progress_bars=False)
    inf.append_simulations(th, xs)
    de = inf.train(max_num_epochs=max_epochs, training_batch_size=512)
    post = inf.build_posterior(de)
    te = idx["test"]; te = te[keep[te]]
    with torch.no_grad():
        lp = float(torch.stack([post.log_prob(torch.tensor(X[i:i + 1], dtype=torch.float32), x=torch.tensor(y[i:i + 1], dtype=torch.float32))
                                for i in te[:1000]]).mean())
    return post, time.time() - t0, lp


def sample(post, obs_x, n=20000, seed=0):
    x = torch.tensor([[np.log10(max(np.mean(obs_x), 1e-300)), np.log10(max(np.std(obs_x, ddof=1), 1e-300))]], dtype=torch.float32)
    torch.manual_seed(seed)
    return post.sample((n,), x=x, show_progress_bars=False).numpy()


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--obs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--estimators", default="nsf,mdn"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args(); best = None
    for est in a.estimators.split(","):   # pick the density estimator on held-out-design log-prob, never on the observed sets
        post, secs, lp = train(a.data, est); print(est, "train s", round(secs), "heldout mean log q(theta|x)", round(lp, 3), flush=True)
        if best is None or lp > best[3]: best = (est, post, secs, lp)
    est, post, secs, lp = best; O = dict(np.load(a.obs)); t0 = time.time()
    pickle.dump(post, open(a.out.replace(".npz", ".pkl"), "wb"))
    S = np.array([sample(post, O["obs"][i], a.draws, seed=i) for i in range(a.n)], dtype=np.float32)
    np.savez_compressed(a.out, samples=S, estimator=est, train_s=secs, heldout_logq=lp, sample_s=(time.time() - t0) / a.n)
    print("done", est, "train", round(secs), "s; sampling", round((time.time() - t0) / a.n, 3), "s/dataset")


if __name__ == "__main__":
    main()
