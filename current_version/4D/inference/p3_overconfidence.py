"""Problem 3: why are the surrogate-based samplers overconfident, and can it be repaired?  (4-parameter model, K = 500)

Stages (run in this order; every stage writes to results/p3/ and later stages read it):
  data     K prior draws and one simulated data set each (same seeds as sbc.py), cached so that every variant is scored on the SAME data.
  feature  feature-level checks on held-out simulator output: marginal PITs, joint (2-D) Mahalanobis QQ against chi2_2, residual
           skewness/kurtosis, correlation, joint 95% conformal-ellipse coverage; fits the Student-t df on the validation split.
  selfsbc  SELF-CONSISTENCY SBC: features are drawn from the surrogate's own predictive density, so any miscalibration left is the
           SAMPLER's (calibrated here + miscalibrated on simulator data => surrogate misspecification).
  variant  SBC on the cached simulator data for: base, student-t predictive, kernel width eps in {...}.
  recal    recalibrate credible levels from SBC ranks (cross-fitted: fit the level map on one half, evaluate on the other).
    python inference/p3_overconfidence.py STAGE --surrogate results/surrogate_4d.pt [--K 500]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm, t as tdist, skew, kurtosis, kstest
from scipy.special import gammaln
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
from sim import settings as S
import dnn_abc, sbc as SBC
from surrogate import HurdleSurrogate
from dyncores import wait_for_core, wide_pool_size
OUT = HERE.parent / "results" / "p3"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]
G = {}


class Variant:
    """Surrogate wrapper exposing loglik(theta, feat): Gaussian (nu=None) or Student-t predictive, plus kernel width eps."""
    def __init__(self, sur, nu=None, eps=0.0):
        self.sur, self.nu, self.eps, self.dim, self.box = sur, nu, eps, sur.dim, sur.box

    def loglik(self, theta, f):
        p0, mu, sd = self.sur.predict(theta); p0 = float(np.clip(p0[0], 1e-12, 1 - 1e-12))
        if f[0] <= 0: return np.log(p0)
        s = np.sqrt(sd[0] ** 2 + self.eps ** 2); u = (np.log10(np.maximum(np.asarray(f, float), 1e-300)) - mu[0]) / s
        if self.nu is None: ll = -0.5 * np.log(2 * np.pi) * 2 - np.log(s).sum() - 0.5 * (u ** 2).sum()
        else:
            n = self.nu; ll = (2 * (gammaln((n + 1) / 2) - gammaln(n / 2) - 0.5 * np.log(n * np.pi)) - np.log(s).sum()
                               - (n + 1) / 2 * np.log1p(u ** 2 / n).sum())
        return np.log1p(-p0) + float(ll)


def fake_obs(m, s):    # two points with sample mean m and sd s, so run_chain's (mean, sd(ddof=1)) recovers the feature pair
    return np.array([m + s / np.sqrt(2), m - s / np.sqrt(2)])


def _init(sur_path, nu, eps, n_iter, scale, thin=20):
    import torch; torch.set_num_threads(1)
    G.update(sur=HurdleSurrogate(sur_path), nu=nu, eps=eps, n=n_iter, scale=scale, thin=thin)


def _gen(k):
    wait_for_core(); rng = np.random.default_rng(31337 + k)
    box = np.array(S.BOX_4D); th = box[:, 0] + rng.random(4) * (box[:, 1] - box[:, 0])
    obs, mom = SBC.dataset(rng, th, 4); return th, obs, mom


def _chain(args):
    wait_for_core(); k, feat, mom = args; v = Variant(G["sur"], G["nu"], G["eps"]); rng = np.random.default_rng(31337 + k)
    ch, acc = dnn_abc.run_chain(v, fake_obs(*feat) if feat[0] > 0 else np.array([0.0, 0.0]), mom, G["n"], rng, step_scale=G["scale"])
    return ch[15000::G["thin"]], acc, ess(ch[15000:])


def ess(x):
    """effective sample size per parameter: initial-positive-sequence autocorrelation estimate on the post-burn-in chain."""
    x = np.asarray(x, float); n = len(x); out = []
    for j in range(x.shape[1]):
        v = x[:, j] - x[:, j].mean(); f = np.fft.rfft(v, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n] / (v @ v)
        t = 1.0
        for k in range(1, n - 1, 2):
            pr = ac[k] + ac[k + 1]
            if pr < 0: break
            t += 2 * pr
        out.append(n / t)
    return np.array(out)


def pool(a, nu=None, eps=0.0, iters=None, thin=20):
    return Pool(wide_pool_size(28), initializer=_init, initargs=(a.surrogate, nu, eps, iters or a.iters, a.step_scale, thin))


def score(TH, R, L, mask=None, label=""):
    m = np.ones(len(TH), bool) if mask is None else mask; out = {}
    for i, nm in enumerate(NAMES):
        r = R[m, i]; h = np.histogram(r, bins=np.linspace(-.5, L + .5, 11))[0]; e = len(r) / 10
        c95 = float(np.mean((r >= L * .025) & (r <= L * .975))); se = np.sqrt(.95 * .05 / len(r))
        out[nm] = dict(cov95=c95, z=float((c95 - .95) / se), chi2_p=float(1 - chi2.cdf(((h - e) ** 2 / e).sum(), 9)),
                       cov50=float(np.mean((r >= L * .25) & (r <= L * .75))), cov90=float(np.mean((r >= L * .05) & (r <= L * .95))))
    out["pass"] = bool(all(abs(out[n]["z"]) <= 2 and out[n]["chi2_p"] > .05 for n in NAMES)); out["n"] = int(m.sum()); return out


def stage_data(a):
    with Pool(wide_pool_size(28)) as p: res = p.map(_gen, range(a.K), chunksize=2)
    feat = np.array([[o.mean(), o.std(ddof=1)] for _, o, _ in res])
    np.savez_compressed(OUT / "data.npz", theta=np.array([r[0] for r in res]), feat=feat, mom=np.array([r[2] for r in res])); print("data done", feat.shape)


def stage_feature(a):
    import arch_search as A
    sur = HurdleSurrogate(a.surrogate); X, zero, y, des, idx = A.load(a.data)
    def resid(ii):
        ii = ii[zero[ii] < .5]; p0, mu, sd = sur.predict(X[ii]); return (y[ii] - mu) / sd, p0, ii
    rv, _, _ = resid(idx["val"]); rt, p0t, it = resid(idx["test"])
    res = {"n_test_nonzero": int(len(it))}
    for f, nm in enumerate(["log10 dbar", "log10 s"]):
        u = norm.cdf(rt[:, f]); res[nm] = dict(mean=float(rt[:, f].mean()), sd=float(rt[:, f].std()), skew=float(skew(rt[:, f])), excess_kurt=float(kurtosis(rt[:, f])),
                                               pit_ks_p=float(kstest(u, "uniform").pvalue), cov95=float(np.mean(np.abs(rt[:, f]) < 1.96)))
    d2 = (rt ** 2).sum(1); qs = [.5, .9, .95, .99]
    res["joint"] = dict(corr=float(np.corrcoef(rt.T)[0, 1]), mahalanobis_cov={str(q): float(np.mean(d2 < chi2.ppf(q, 2))) for q in qs},
                        mahalanobis_mean=float(d2.mean()), note="independent-Gaussian predictive; chi2_2 mean is 2")
    # Student-t df by maximum likelihood on the validation residuals (shared df, independent features)
    best = max([3, 4, 5, 6, 8, 10, 15, 20, 30, 50], key=lambda n: tdist.logpdf(rv, n).sum()); res["student_t_df"] = best
    res["val_loglik_gauss"] = float(norm.logpdf(rv).sum()); res["val_loglik_t"] = float(tdist.logpdf(rv, best).sum())
    # conditional check: is the residual spread constant in the zero-probability / in theta?  decile table of mean squared residual by p0
    qd = np.quantile(p0t, np.linspace(0, 1, 6)); b = np.clip(np.digitize(p0t, qd[1:-1]), 0, 4)
    res["msr_by_p0_quintile"] = [float((rt[b == j] ** 2).mean()) for j in range(5)]
    json.dump(res, open(OUT / "feature_checks.json", "w"), indent=1); print(json.dumps(res, indent=1))


def run_variant(a, tag, nu=None, eps=0.0, selfc=False, iters=None, thin=20, K=None):
    D = np.load(OUT / "data.npz"); TH, F, MOM = D["theta"][:K], D["feat"][:K], D["mom"][:K]
    sur = HurdleSurrogate(a.surrogate)
    if selfc:        # features drawn from the surrogate's own predictive density at the prior draw
        rng = np.random.default_rng(99); F = np.empty_like(F)
        for k in range(len(TH)):
            p0, mu, sd = sur.predict(TH[k])
            F[k] = [0.0, 0.0] if rng.random() < p0[0] else 10 ** (mu[0] + sd[0] * rng.standard_normal(2))
    with pool(a, nu, eps, iters, thin) as p: res = p.map(_chain, [(k, F[k], MOM[k]) for k in range(len(TH))], chunksize=1)
    R = np.array([SBC.ranks(TH[k], res[k][0]) for k in range(len(TH))]); L = len(res[0][0]); nz = F[:, 0] > 0
    np.savez_compressed(OUT / f"sbc_{tag}.npz", ranks=R, L=L, theta=TH, nonzero=nz, acc=np.array([r[1] for r in res]), var=np.array([r[0].var(0) for r in res]), ess=np.array([r[2] for r in res]))
    sc = {"all": score(TH, R, L, label=tag), "dbar>0": score(TH, R, L, nz)}
    box = np.array(S.BOX_4D); V = np.array([r[0].var(0) for r in res])[nz]
    sc["ess_median"] = {n: float(np.median([r[2][i] for r in res])) for i, n in enumerate(NAMES)}; sc["acc_median"] = float(np.median([r[1] for r in res]))
    sc["contraction"] = {n: float(1 - V[:, i].mean() / ((box[i, 1] - box[i, 0]) ** 2 / 12)) for i, n in enumerate(NAMES)}
    json.dump(sc, open(OUT / f"sbc_{tag}.json", "w"), indent=1); print(tag, json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else None) for k, v in sc.items()}))
    return sc


def stage_recal(a, tag="base"):
    Z = np.load(OUT / f"sbc_{tag}.npz"); R, L, nz = Z["ranks"][Z["nonzero"]], int(Z["L"]), None
    u = (R + .5) / (L + 1); rng = np.random.default_rng(0); h = rng.permutation(len(u)) % 2; out = {}
    for i, nm in enumerate(NAMES):
        res = []
        for fit, ev in ((0, 1), (1, 0)):
            w = np.abs(u[h == fit, i] - .5) * 2           # central-interval level at which the truth enters
            lvl = np.quantile(w, .95)                      # nominal level whose empirical coverage is 0.95 on the fit half
            res.append(dict(adjusted_level=float(lvl), cov_before=float(np.mean(np.abs(u[h == ev, i] - .5) * 2 <= .95)),
                            cov_after=float(np.mean(np.abs(u[h == ev, i] - .5) * 2 <= lvl))))
        out[nm] = dict(adjusted_level=float(np.mean([r["adjusted_level"] for r in res])), cov_before=float(np.mean([r["cov_before"] for r in res])),
                       cov_after_cross_fitted=float(np.mean([r["cov_after"] for r in res])))
    json.dump(out, open(OUT / "recal.json", "w"), indent=1); print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["data", "feature", "selfsbc", "variant", "recal", "mixing", "all"])
    ap.add_argument("--surrogate", default="results/surrogate_4d.pt"); ap.add_argument("--data", default="data/slow_data_4D.csv")
    ap.add_argument("--K", type=int, default=500); ap.add_argument("--iters", type=int, default=35000); ap.add_argument("--step-scale", type=float, default=0.05)
    ap.add_argument("--eps", type=float, nargs="*", default=[0.02, 0.05, 0.1]); a = ap.parse_args()
    if a.stage in ("data", "all"): stage_data(a)
    if a.stage in ("feature", "all"): stage_feature(a)
    if a.stage in ("selfsbc", "all"): run_variant(a, "selfconsistent", selfc=True)
    if a.stage in ("variant", "all"):
        run_variant(a, "base"); nu = json.load(open(OUT / "feature_checks.json"))["student_t_df"]; run_variant(a, f"student_t{nu}", nu=nu)
        for e in a.eps: run_variant(a, f"eps{e}", eps=e)
    if a.stage == "mixing":      # is the overconfidence the sampler's? 5x longer chains, thinned to the same 1,000 draws
        run_variant(a, "selfconsistent_long", selfc=True, iters=175000, thin=160, K=a.K); run_variant(a, "base_long", iters=175000, thin=160, K=a.K)
    if a.stage in ("recal", "all"): stage_recal(a)


if __name__ == "__main__":
    main()
