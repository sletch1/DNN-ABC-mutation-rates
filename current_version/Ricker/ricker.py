"""Second zero-inflated benchmark: the stochastic Ricker map with extinction and Poisson sampling (Wood 2010, Nature 466:1102).
    N_{t+1} = r N_t exp(-N_t + e_t), e_t ~ N(0, sigma^2);  N_t set to 0 if it falls below 1e-6 (extinction);  y_t ~ Poisson(phi N_t).
theta = (log r, sigma, phi) uniform on [2.0, 3.8] x [0.1, 0.8] x [3, 20] (about 24% of prior draws give ybar = 0); T = 100 observations after a 50-step burn-in.
Features (like the mutation model): ybar = mean(y), s = sd(y), z = fraction of zero counts. ybar = 0 (extinct or never seen) is an
atom, so the surrogate is a hurdle model: P(ybar=0 | theta) and a Gaussian on (log10 ybar, log10 s, arcsine-sqrt z) otherwise.
Methods compared: synthetic-likelihood MCMC (the exact-simulator reference, simulator in the loop), DNN-ABC (hurdle network in the
same sampler), NPE trained on ALL rows with a zero indicator. All scored by simulation-based calibration (K prior draws).
    python ricker.py gen|train|sbc|time ...  (see main)
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
BOX = np.array([[2.0, 3.8], [0.1, 0.8], [3.0, 20.0]]); NAMES = ["log r", "sigma", "phi"]; T, BURN = 100, 50


def simulate(theta, rng, n=1):
    """n independent series for one theta, vectorised over series; returns y (n, T)."""
    lr, sg, ph = theta; r = np.exp(lr); N = np.ones(n); y = np.empty((n, T))
    for t in range(BURN + T):
        N = r * N * np.exp(-N + sg * rng.standard_normal(n)); N = np.where(N < 1e-6, 0.0, N)
        if t >= BURN: y[:, t - BURN] = rng.poisson(ph * N)
    return y


def features(y):
    m = y.mean(-1); s = y.std(-1, ddof=1); z = (y == 0).mean(-1)
    return m, s, z


def tf(m, s, z):   # transformed features for rows with ybar > 0 and s > 0
    return np.stack([np.log10(m), np.log10(np.maximum(s, 1e-3)), np.arcsin(np.sqrt(z))], -1)


def _gen1(args):
    from dyncores import wait_for_core
    if args[0] % 200 == 0: wait_for_core(jitter=2, poll=5)
    return features(simulate(args[1], np.random.default_rng(100 + args[0]), 1)[0])


def gen(a):
    rng = np.random.default_rng(1); th = BOX[:, 0] + rng.random((a.n, 3)) * (BOX[:, 1] - BOX[:, 0])
    with Pool(a.cores) as p: Y = np.array(p.map(_gen1, [(i, t) for i, t in enumerate(th)], chunksize=200))
    np.savez(OUT / "train.npz", theta=th, m=Y[:, 0], s=Y[:, 1], z=Y[:, 2]); print("gen", a.n, "zero fraction", float((Y[:, 0] == 0).mean()))


class Net:
    """numpy forward pass of the trained hurdle MLP (3 -> hidden -> logit p0, 3 means, 3 log-variances)."""
    def __init__(self, W): self.W, self.sd_scale = W["layers"], W["sd_scale"]; self.xm, self.xs, self.ym, self.ys = W["xm"], W["xs"], W["ym"], W["ys"]

    def predict(self, theta):
        h = (np.atleast_2d(theta) - self.xm) / self.xs
        for k, (Wt, b) in enumerate(self.W[:-1]): h = Wt @ h.T if False else h @ Wt + b; h = h * 0.5 * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
        o = h @ self.W[-1][0] + self.W[-1][1]
        p0 = 1 / (1 + np.exp(-o[:, 0])); mu = o[:, 1:4] * self.ys + self.ym; sd = np.exp(0.5 * o[:, 4:7]) * self.ys * self.sd_scale
        return p0, mu, sd


def train(a):
    import torch; torch.manual_seed(0); torch.set_num_threads(a.threads)
    D = np.load(OUT / "train.npz"); th, m, s, z = D["theta"], D["m"], D["s"], D["z"]; zero = (m == 0) | (s == 0); n = len(th)
    Y = np.where(zero[:, None], 0.0, tf(np.where(zero, 1, m), np.where(zero, 1, s), z)); ntr, nva = int(.7 * n), int(.15 * n)
    xm, xs = BOX.mean(1), BOX.std(1) * 0 + (BOX[:, 1] - BOX[:, 0]) / np.sqrt(12); nz = ~zero[:ntr]; ym, ys = Y[:ntr][nz].mean(0), Y[:ntr][nz].std(0)
    X = torch.tensor((th - xm) / xs, dtype=torch.float32); Yn = torch.tensor((Y - ym) / ys, dtype=torch.float32); Z = torch.tensor(zero, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 7))
    opt = torch.optim.Adam(net.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    def loss(i):
        o = net(X[i]); bce = torch.nn.functional.binary_cross_entropy_with_logits(o[:, 0], Z[i]); k = Z[i] < .5
        nll = (0.5 * o[:, 4:7] + 0.5 * (Yn[i] - o[:, 1:4]) ** 2 * torch.exp(-o[:, 4:7]))[k].sum(1).mean() if k.any() else 0.0
        return bce + nll
    best, bs = 1e9, None
    for ep in range(a.epochs):
        perm = torch.randperm(ntr)
        for j in range(0, ntr, 256):
            opt.zero_grad(); l = loss(perm[j:j + 256]); l.backward(); opt.step()
        sched.step()
        with torch.no_grad(): v = float(loss(torch.arange(ntr, ntr + nva)))
        if v < best: best, bs = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(bs)
    with torch.no_grad():     # split-conformal scale on the calibration split: 95% of |resid|/sd within 1.96 per feature
        i = torch.arange(ntr + nva, n); o = net(X[i]); k = (Z[i] < .5).numpy(); r = ((Yn[i] - o[:, 1:4]) * torch.exp(-0.5 * o[:, 4:7])).numpy()[k]
        c = np.quantile(np.abs(r), .95, axis=0) / 1.96; test_mse = float(((Yn[i] - o[:, 1:4]).numpy()[k] ** 2).mean())
    sd = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}; L = [(sd[f"{j}.weight"].T, sd[f"{j}.bias"]) for j in (0, 2, 4, 6)]
    np.savez(OUT / "surrogate.npz", **{f"W{j}": L[j][0] for j in range(4)}, **{f"b{j}": L[j][1] for j in range(4)}, xm=xm, xs=xs, ym=ym, ys=ys, sd_scale=c)
    print("trained: best val loss", best, "conformal scales", c, "test std-resid mse", test_mse)


def load_net():
    S = np.load(OUT / "surrogate.npz"); return Net(dict(layers=[(S[f"W{j}"], S[f"b{j}"]) for j in range(4)], sd_scale=S["sd_scale"], xm=S["xm"], xs=S["xs"], ym=S["ym"], ys=S["ys"]))


def ll_dnn(net, th, f):
    p0, mu, sd = net.predict(th); p0 = float(np.clip(p0[0], 1e-12, 1 - 1e-12))
    if f[0] <= 0 or f[1] <= 0: return np.log(p0)
    u = (tf(*f) - mu[0]) / sd[0]; return np.log1p(-p0) - np.log(sd[0]).sum() - 0.5 * (u ** 2).sum()


def ll_sl(rng, th, f, n_s=100):
    y = simulate(th, rng, n_s); m, s, z = features(y); ok = (m > 0) & (s > 0); p0 = (n_s - ok.sum() + .5) / (n_s + 1)
    if f[0] <= 0 or f[1] <= 0: return np.log(p0)
    if ok.sum() < 10: return -1e10
    F = tf(m[ok], s[ok], z[ok]); C = np.cov(F.T) + 1e-9 * np.eye(3); d = tf(*f) - F.mean(0)
    return float(np.log1p(-p0) - 0.5 * np.linalg.slogdet(2 * np.pi * C)[1] - 0.5 * d @ np.linalg.solve(C, d))


def mcmc(ll, f, rng, n_iter, adapt, start):
    lo, hi = BOX[:, 0], BOX[:, 1]; w = hi - lo; L = np.linalg.cholesky(np.diag((0.05 * w) ** 2)); cur = np.array(start, float); lc = ll(cur); out = np.empty((n_iter, 3)); acc = 0
    for i in range(n_iter):
        if 500 <= i < adapt and i % 250 == 0:
            try: L = np.linalg.cholesky(np.cov(out[max(0, i - 2000):i].T) * 2.38 ** 2 / 3 + np.diag((0.005 * w) ** 2))
            except np.linalg.LinAlgError: pass
        can = cur + L @ rng.standard_normal(3)
        if np.all(can >= lo) and np.all(can <= hi):
            lcn = ll(can)
            if np.log(rng.random()) < lcn - lc: cur, lc = can, lcn; acc += i >= adapt
        out[i] = cur
    return out, acc / max(1, n_iter - adapt)


G = {}
def _job(k):
    from dyncores import wait_for_core; wait_for_core(jitter=5); rng = np.random.default_rng(777 + k)
    th = BOX[:, 0] + rng.random(3) * (BOX[:, 1] - BOX[:, 0]); y = simulate(th, rng, 1)[0]; f = tuple(float(v) for v in features(y)); out = dict(theta=th, f=f)
    t0 = time.time(); ch, acc = mcmc(lambda t: ll_dnn(G["net"], t, f), f, rng, G["iters"], G["adapt"], BOX.mean(1)); out["dnn"] = ch[G["adapt"]::G["thin"]]; out["dnn_s"] = time.time() - t0
    if G["sl"]:
        t0 = time.time(); ch, acc = mcmc(lambda t: ll_sl(rng, t, f, 100), f, rng, G["sl_iters"], G["sl_iters"] // 3, BOX.mean(1)); out["sl"] = ch[G["sl_iters"] // 3::5]; out["sl_s"] = time.time() - t0
    return out


def sbc(a):
    G.update(net=load_net(), iters=a.iters, adapt=a.iters // 3, thin=20, sl=a.sl, sl_iters=a.sl_iters)
    with Pool(a.cores) as p: res = p.map(_job, range(a.K), chunksize=1)
    from scipy.stats import chi2; summ = {}
    for meth in ["dnn"] + (["sl"] if a.sl else []):
        R = np.array([(r[meth] < r["theta"]).sum(0) for r in res]); Lr = len(res[0][meth]); V = np.array([r[meth].var(0) for r in res]); nz = np.array([r["f"][0] > 0 for r in res])
        for tag, msk in (("all", np.ones(len(res), bool)), ("ybar>0", nz)):
            o = {}
            for i, nm in enumerate(NAMES):
                r = R[msk, i]; h = np.histogram(r, bins=np.linspace(-.5, Lr + .5, 11))[0]; e = len(r) / 10; c95 = float(np.mean((r >= Lr * .025) & (r <= Lr * .975)))
                o[nm] = dict(cov95=c95, z=float((c95 - .95) / np.sqrt(.95 * .05 / len(r))), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                             contraction=float(1 - V[msk, i].mean() / ((BOX[i, 1] - BOX[i, 0]) ** 2 / 12)))
            summ[f"{meth} ({tag})"] = o
        summ[f"{meth}_sec_per_dataset"] = float(np.mean([r[f"{meth}_s"] for r in res]))
    summ["zero_fraction"] = float(1 - np.mean([r["f"][0] > 0 for r in res])); summ["K"] = len(res)
    np.savez_compressed(OUT / "sbc_raw.npz", theta=np.array([r["theta"] for r in res]), f=np.array([r["f"] for r in res]), **{m: np.array([r[m] for r in res]) for m in ["dnn"] + (["sl"] if a.sl else [])})
    json.dump(summ, open(OUT / "sbc_ricker.json", "w"), indent=1); print(json.dumps(summ, indent=1))


def npe(a):
    """NPE trained on ALL rows (zero indicator + features 0 where undefined) -- the fair comparator -- and scored by SBC on fresh prior draws."""
    import torch; from sbi.inference import NPE; from sbi.utils import BoxUniform; torch.set_num_threads(a.threads); torch.manual_seed(0)
    D = np.load(OUT / "train.npz"); th, m, s, z = D["theta"], D["m"], D["s"], D["z"]; zero = (m == 0) | (s == 0)
    F = np.column_stack([zero, np.where(zero[:, None], 0, tf(np.where(zero, 1, m), np.where(zero, 1, s), z))]).astype(np.float32)
    prior = BoxUniform(torch.tensor(BOX[:, 0], dtype=torch.float32), torch.tensor(BOX[:, 1], dtype=torch.float32)); t0 = time.time()
    inf = NPE(prior=prior, density_estimator="nsf", show_progress_bars=False); inf.append_simulations(torch.tensor(th, dtype=torch.float32), torch.tensor(F))
    post = inf.build_posterior(inf.train(max_num_epochs=300, training_batch_size=256)); tr = time.time() - t0
    from scipy.stats import chi2; R, V, nzs = [], [], []
    for k in range(a.K):
        rng = np.random.default_rng(777 + k); t = BOX[:, 0] + rng.random(3) * (BOX[:, 1] - BOX[:, 0]); y = simulate(t, rng, 1)[0]; mm, ss, zz = features(y); zr = (mm == 0) or (ss == 0)
        f = np.array([[zr, 0, 0, 0] if zr else [0, *tf(mm, ss, zz)]], np.float32); sm = post.sample((1000,), x=torch.tensor(f), show_progress_bars=False).numpy()
        R.append((sm < t).sum(0)); V.append(sm.var(0)); nzs.append(not zr)
    R, V, nzs = np.array(R), np.array(V), np.array(nzs); summ = {"train_s": tr}
    for tag, msk in (("all", np.ones(len(R), bool)), ("ybar>0", nzs)):
        o = {}
        for i, nm in enumerate(NAMES):
            r = R[msk, i]; h = np.histogram(r, bins=np.linspace(-.5, 1000.5, 11))[0]; e = len(r) / 10; c95 = float(np.mean((r >= 25) & (r <= 975)))
            o[nm] = dict(cov95=c95, z=float((c95 - .95) / np.sqrt(.95 * .05 / len(r))), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)), contraction=float(1 - V[msk, i].mean() / ((BOX[i, 1] - BOX[i, 0]) ** 2 / 12)))
        summ[f"npe_z ({tag})"] = o
    json.dump(summ, open(OUT / "npe_ricker.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["gen", "train", "sbc", "npe"]); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--K", type=int, default=500); ap.add_argument("--iters", type=int, default=35000); ap.add_argument("--cores", type=int, default=24); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sl", action="store_true"); ap.add_argument("--sl_iters", type=int, default=6000); a = ap.parse_args(); {"gen": gen, "train": train, "sbc": sbc, "npe": npe}[a.stage](a)
