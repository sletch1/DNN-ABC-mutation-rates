"""BayesFlow baseline (Radev et al., 2020): amortised posterior estimation with a conditional invertible network, for the two-stage model.

WHY THIS FILE EXISTS. The paper's NPE baseline is the sbi neural spline flow (inference/npe.py); this BayesFlow implementation of the same estimator was tried as an alternative (see bayesflow_comparison.py) and scored lower on held-out log-probability, so it is not used for the results.
WHY AN NPE BASELINE AT ALL. In one dimension a surrogate-plus-MCMC pipeline is a reasonable way to reuse simulator output; in the
higher-dimensional two-stage model the same simulator output can instead be used to learn the posterior itself. BayesFlow is the natural
comparator because (i) it consumes exactly the training simulations DNN-ABC does (same CSV, same two summary features, same split),
(ii) it removes the two components DNN-ABC adds on top of the surrogate -- the MCMC chain and the surrogate's predictive-variance model --
so it tests whether those are needed, and (iii) once trained it returns i.i.d. posterior draws for any new dataset in milliseconds. With the
two fixed features (d_bar, s) as the condition (no learned summary network), it cannot beat the information in the summary statistic, which
makes it a useful ceiling for what any method using (d_bar, s) can recover.

Network: a conditional coupling flow (affine coupling blocks, as in the BayesFlow paper) from theta to a standard-normal latent, conditioned
on x = (log10 d_bar, log10 s); trained by maximum likelihood on simulated (theta, x) pairs, which is the loss of Radev et al. (2020, eq. 14).

Training set: rows with d_bar > 0 only. The estimator fits q(theta | x) by maximising E log q(theta | x) over joint draws; the population
minimiser is p(theta | x) for every x in the support of the training x, whatever the marginal of x, so discarding the d_bar = 0 rows (where
log d_bar is undefined) leaves the posterior at any observed x with d_bar > 0 unchanged. All observed datasets here have d_bar > 0.
Prior: uniform on the paper's box (the paper's TN(.,20,box) prior is flat to within 2% across a box of width <= 20 sd). Draws that fall
outside the box are rejected, so the posterior has the prior's support.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import arch_search as A

BOX = {3: [[-11, -7], [-11, -7], [0.1, 19.9]], 4: [[-11, -7], [-11, -7], [0.1, 19.9], [float(np.log10(.5)), float(np.log10(2))]]}


def data(path):
    X, zero, y, des, idx = A.load(path)
    keep = zero < .5
    return X, y, keep, idx


def _approximator(depth, seed, transform="affine", width=128):
    import keras, bayesflow as bf
    keras.utils.set_random_seed(seed)
    net = bf.networks.CouplingFlow(depth=depth, transform=transform, subnet="mlp", subnet_kwargs=dict(widths=(width, width), dropout=0.0))
    adapter = bf.Adapter().convert_dtype("float64", "float32").concatenate(["theta"], into="inference_variables").concatenate(["x"], into="inference_conditions")
    return bf.approximators.ContinuousApproximator(inference_network=net, adapter=adapter, standardize="inference_variables")


class Posterior:
    """Wrapper that gives the BayesFlow approximator the small interface the rest of the study uses: sample((n,), x=...) and log_prob."""
    def __init__(self, approx, box):
        self.approx = approx; self.lo, self.hi = np.array(box, dtype=np.float32).T

    def sample(self, n, x, seed=0):
        """n i.i.d. draws given one feature vector x (shape (1, 2) or (2,)); draws outside the prior box are rejected."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1); n = n[0] if isinstance(n, (tuple, list)) else n
        out, have, k = [], 0, 0
        while have < n:
            m = int(1.3 * (n - have)) + 64
            s = self.approx.sample(num_samples=m, conditions={"x": x}, seed=int(seed) * 100 + k)["theta"].reshape(m, -1)
            s = s[np.all((s >= self.lo) & (s <= self.hi), axis=1)]; out.append(s); have += len(s); k += 1
        return np.concatenate(out)[:n].astype(np.float32)

    def log_prob(self, theta, x, norm=True, n_acc=1000):
        """log q(theta | x) for matching rows of theta (N, D) and x (N, 2). With norm=True the density is renormalised to the prior box (divided by
        the flow's probability mass inside the box, estimated from n_acc draws), which is the density of the truncated posterior that sample() draws
        from and what sbi's DirectPosterior reports, so the held-out log-probabilities of the two estimators are comparable."""
        theta = np.asarray(theta, dtype=np.float32); x = np.asarray(x, dtype=np.float32)
        lq = np.asarray(self.approx.log_prob(data={"theta": theta, "x": x})).reshape(-1)
        if not norm: return lq
        acc = np.array([np.all((s >= self.lo) & (s <= self.hi), axis=1).mean() for s in
                        (self.approx.sample(num_samples=n_acc, conditions={"x": xi[None]}, seed=i)["theta"].reshape(n_acc, -1) for i, xi in enumerate(x))])
        return lq - np.log(np.maximum(acc, 1.0 / n_acc))

    def save(self, path): self.approx.save(path)


def load(path, box):
    import keras, bayesflow  # noqa: F401  (importing bayesflow registers its classes for keras)
    return Posterior(keras.saving.load_model(path), box)


def train(path, depth=6, seed=0, max_epochs=300, patience=20, batch_size=512, transform="affine", width=128):
    import keras
    X, y, keep, idx = data(path); D = X.shape[1]; box = BOX[D]
    tr, va, te = (idx[k][keep[idx[k]]] for k in ("train", "val", "test"))
    appx = _approximator(depth, seed, transform, width); appx.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    f = lambda ii: dict(theta=X[ii].astype(np.float32), x=y[ii].astype(np.float32))
    t0 = time.time()
    import bayesflow as bf
    ds = bf.OfflineDataset(f(tr), batch_size=batch_size, adapter=appx.adapter); vd = bf.OfflineDataset(f(va), batch_size=batch_size, adapter=appx.adapter)
    appx.fit(dataset=ds, validation_data=vd, epochs=max_epochs, verbose=0,
             callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)])
    post = Posterior(appx, box)
    lp = float(np.mean(post.log_prob(X[te[:1000]], y[te[:1000]])))       # mean held-out log q(theta | x) over unseen design points
    return post, time.time() - t0, lp


def fit(th, x, box, seed=0, depth=6, max_epochs=300, patience=20, batch_size=512, val_frac=0.1, transform="affine", width=128):
    """Train on arbitrary (theta, x) arrays (used for the variants with other features); a random val_frac of rows is held out for early stopping."""
    import keras, bayesflow as bf
    pm = np.random.default_rng(seed).permutation(len(th)); th = np.asarray(th, dtype=np.float32)[pm]; x = np.asarray(x, dtype=np.float32)[pm]; n = len(th); nv = int(val_frac * n)
    appx = _approximator(depth, seed, transform, width); appx.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    ds = bf.OfflineDataset(dict(theta=th[:n - nv], x=x[:n - nv]), batch_size=batch_size, adapter=appx.adapter)
    vd = bf.OfflineDataset(dict(theta=th[n - nv:], x=x[n - nv:]), batch_size=batch_size, adapter=appx.adapter)
    appx.fit(dataset=ds, validation_data=vd, epochs=max_epochs, verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)])
    return Posterior(appx, box)


def sample(post, obs_x, n=20000, seed=0):
    x = np.array([[np.log10(max(np.mean(obs_x), 1e-300)), np.log10(max(np.std(obs_x, ddof=1), 1e-300))]], dtype=np.float32)
    return post.sample(n, x, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--obs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", default=None, help="a saved posterior (.keras) from an earlier run on the SAME training data: skip training, only sample")
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args(); O = dict(np.load(a.obs))
    if a.reuse:      # the amortised posterior does not depend on the observed data: reuse it for another case
        post = load(a.reuse, BOX[4]); z = np.load(a.reuse.replace(".keras", ".npz")); t0 = time.time()
        S = np.array([sample(post, O["obs"][i], a.draws, seed=i) for i in range(a.n)], dtype=np.float32)
        np.savez_compressed(a.out, samples=S, train_s=float(z["train_s"]), heldout_logq=float(z["heldout_logq"]), sample_s=(time.time() - t0) / a.n)
        post.save(a.out.replace(".npz", ".keras")); print("reused", a.reuse); return
    post, secs, lp = train(a.data, depth=a.depth, max_epochs=a.epochs); print("train s", round(secs), "heldout mean log q(theta|x)", round(lp, 3), flush=True)
    post.save(a.out.replace(".npz", ".keras")); t0 = time.time()
    S = np.array([sample(post, O["obs"][i], a.draws, seed=i) for i in range(a.n)], dtype=np.float32)
    np.savez_compressed(a.out, samples=S, train_s=secs, heldout_logq=lp, sample_s=(time.time() - t0) / a.n)
    print("done: train", round(secs), "s; sampling", round((time.time() - t0) / a.n, 3), "s/dataset")


if __name__ == "__main__":
    main()
