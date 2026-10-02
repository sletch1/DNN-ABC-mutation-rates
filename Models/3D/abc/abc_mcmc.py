"""ABC-MCMC for the 3-D two-stage mutation model. Random-walk Metropolis-
Hastings over the full theta = (log10 p1, log10 p2, tau), jointly estimated
from one scalar summary statistic (paper's Study 2) -- unlike the 1-D study,
where a single scalar was inferred with other inputs known. `d_bar` responds
strongly to p2 (corr 0.74), weakly to p1 (0.40), barely to tau (0.14), so
expect a well-identified p2 and broad/multimodal p1, tau marginals.

Three backends, same sampler: "sim" runs the real simulator `ns` times per
proposal and scores obs with a synthetic-likelihood Gaussian (Wood 2010);
"dnn"/"gp" use the surrogate's (mean, sd) directly via the exact convolution
`N(obs; mean, sqrt(eps^2+sd^2))`, no Monte Carlo sampling needed.

Priors are per-coordinate ("uniform"/"normal"/"expon") because the paper uses
different families per study: Study 2 (this module) uses truncated normals
nearly flat over the box, so "uniform" is a faithful default; Study 1 (1-D)
uses a truncated exponential on log10(p), meaningless on tau. See _log_prior.

Proposal: component-wise random walk, per-component step sizes (log10 units
vs. absolute time are different scales), truncated and Hastings-corrected.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, truncnorm

from simulator import fluc_exp_2stage, summary_stat

# (log10 p1, log10 p2, tau) box; matches the ground-truth design in
# RCode/genSlowData_3D.R, so the surrogate is never queried out of range.
DEFAULT_BOX = ((-5.0, -1.3), (-5.0, -1.3), (0.1, 9.9))
DEFAULT_STEPS = (0.45, 0.25, 1.30)   # tuned for ~25-40% acceptance; see run_experiments.py


def _tn_sample(mu, s, lo, hi, rng):
    """One draw from N(mu, s^2) truncated to [lo, hi]."""
    return float(truncnorm.rvs((lo - mu) / s, (hi - mu) / s, loc=mu, scale=s,
                               random_state=rng))


def _tn_logZ(mu, s, lo, hi):
    """log of the truncated-normal normalising constant, for the Hastings ratio."""
    return np.log(max(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s), 1e-300))


def _log_prior(theta, box, kind, centres, sds, lam=2.0):
    """Independent log prior on the box, per coordinate. `kind` is one name
    for all three coordinates, or a per-coordinate sequence (the paper uses
    different families for the rates vs. the transition time):

      "uniform": flat on [lo, hi].
      "normal" : truncated normal, centred on `centres` with sd `sds`. Study 2's
                 actual prior (TN(log10 p_hat_MOM, 20, box) for p1/p2, TN(10, 20,
                 [0.1,19.9]) for tau) is nearly flat at sd=20 over a width-4 box,
                 so "uniform" is used as the faithful default instead.
      "expon"  : truncated exponential, rate `lam`, favouring small values --
                 Study 1's (1-D) prior on theta=log10(p); meaningless on tau,
                 hence per-coordinate rather than a single shared kind.
    """
    kinds = [kind] * len(box) if isinstance(kind, str) else list(kind)
    lp = 0.0
    for v, (lo, hi), c, s, k in zip(theta, box, centres, sds, kinds):
        if not (lo <= v <= hi):
            return -np.inf
        if k == "normal":
            lp += norm.logpdf(v, loc=c, scale=s) - _tn_logZ(c, s, lo, hi)
        elif k == "expon":
            # log[ lam*exp(-lam(v-lo)) / (1-exp(-lam(hi-lo))) ]; `lam` kept (it
            # cancels in the acceptance ratio) so this is a real log-density.
            lp += (np.log(lam) - lam * (v - lo)
                   - np.log1p(-np.exp(-lam * (hi - lo))))
        elif k != "uniform":
            raise ValueError(f"unknown prior kind {k!r}")
    return lp


def run_abc_mcmc(obs, backend, n_mcmc=2000, theta_init=None, steps=DEFAULT_STEPS,
                 box=DEFAULT_BOX, rng=None, eps=0.005,
                 prior="uniform", prior_centres=None, prior_sds=(2.0, 2.0, 4.0),
                 prior_lambda=2.0, sim_kwargs=None, ns=1, surrogate=None):
    """Return (samples, accept_rate). `samples` is (n_mcmc, 3) in theta coordinates.

    theta = (log10 p1, log10 p2, tau). `obs` is the observed d_bar on the RAW
    scale; it is converted to log10 internally, which is the scale the surrogates
    are trained and calibrated on.
    """
    if rng is None:
        rng = np.random.default_rng()
    box = tuple(box)
    steps = np.asarray(steps, dtype=float)
    centres = prior_centres if prior_centres is not None else [
        0.5 * (lo + hi) for lo, hi in box]

    LOG_FLOOR = -6.0
    obs_log = np.log10(max(obs, 10.0 ** LOG_FLOOR))

    if backend == "sim":
        sk = sim_kwargs

        def log_like(th):
            # Synthetic-likelihood ABC: mean/var from ns sims, score obs under
            # N(mean, eps^2+var) -- same form the surrogates use, paid for by simulation.
            vals = np.empty(ns)
            for k in range(ns):
                Z, X = fluc_exp_2stage(sk["Z0"], sk["a"], 10.0 ** th[0], 10.0 ** th[1],
                                       th[2], sk["tp"], sk["J"], rng,
                                       use_slow=sk.get("use_slow", True),
                                       mut_time=sk.get("mut_time", "parent"))
                vals[k] = summary_stat(Z, X)
            v = np.log10(np.maximum(vals, 10.0 ** LOG_FLOOR))
            m = float(np.mean(v))
            var = float(np.var(v, ddof=1)) if ns > 1 else 0.0
            return norm.logpdf(obs_log, loc=m, scale=np.sqrt(eps ** 2 + var))
    elif backend in ("dnn", "gp"):
        # Target follows the surrogate's declared scale (DNN/strengthened GP: log10;
        # reference GP: raw, matching ABC_fluc_exp1.m) -- scoring the wrong scale
        # silently produces a wrong posterior.
        target = obs_log if getattr(surrogate, "scale", "log10") == "log10" else float(obs)

        def log_like(th):
            mean, sd = surrogate.predict([th[0], th[1], th[2]])
            return norm.logpdf(target, loc=mean, scale=np.sqrt(eps ** 2 + sd ** 2))
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if theta_init is None:
        theta_init = np.array([0.5 * (lo + hi) for lo, hi in box])
    theta_init = np.clip(np.asarray(theta_init, dtype=float),
                         [b[0] + 1e-9 for b in box], [b[1] - 1e-9 for b in box])

    samples = np.empty((n_mcmc, 3))
    samples[0] = theta_init
    ll = log_like(theta_init)
    lp = _log_prior(theta_init, box, prior, centres, prior_sds, prior_lambda)
    n_accept = 0

    for i in range(1, n_mcmc):
        cur = samples[i - 1]
        # Hastings term doesn't cancel (truncation mass differs per point), accumulated per coordinate.
        can = np.empty(3)
        log_q = 0.0
        for k in range(3):
            lo, hi = box[k]
            can[k] = _tn_sample(cur[k], steps[k], lo, hi, rng)
            log_q += _tn_logZ(cur[k], steps[k], lo, hi) - _tn_logZ(can[k], steps[k], lo, hi)
        ll_can = log_like(can)
        lp_can = _log_prior(can, box, prior, centres, prior_sds, prior_lambda)
        if np.log(rng.random()) < min(0.0, (ll_can - ll) + (lp_can - lp) + log_q):
            samples[i] = can
            ll, lp = ll_can, lp_can
            n_accept += 1
        else:
            samples[i] = cur
    return samples, n_accept / (n_mcmc - 1)


def summarize(samples, burn_in, cred=0.95):
    """Posterior summaries per parameter from post-burn-in draws. p1/p2 are
    reported on the natural (not log) scale, matching the paper's tables;
    tau is already natural. Returns a dict of dicts keyed by parameter name.
    """
    post = samples[burn_in:]
    lo_q, hi_q = (1 - cred) / 2, 1 - (1 - cred) / 2
    out = {}
    for k, name in enumerate(("p1", "p2", "tau")):
        v = 10.0 ** post[:, k] if name in ("p1", "p2") else post[:, k]
        out[name] = dict(mean=float(np.mean(v)), median=float(np.median(v)),
                         ci_lo=float(np.quantile(v, lo_q)),
                         ci_hi=float(np.quantile(v, hi_q)),
                         ci_len=float(np.quantile(v, hi_q) - np.quantile(v, lo_q)))
    return out


def ess(x):
    """Effective sample size via the initial-positive-sequence rule. Reported
    alongside acceptance because a joint chain can accept healthily while
    still mixing badly in poorly-informed directions -- expected for p1/tau.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    s, k = 0.0, 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        s += pair
        k += 2
    return float(n / (1 + 2 * s)) if s > 0 else float(n)
