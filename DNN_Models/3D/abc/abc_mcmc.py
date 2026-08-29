"""ABC-MCMC for the 3-D two-stage mutation model.

Random-walk Metropolis-Hastings over the FULL parameter vector

    theta = ( log10 p1 , log10 p2 , tau )

This is the substantive difference from the 1-D pipeline, and from the retired
(p, a, delta) study. There, a single scalar was inferred while the other inputs
were supplied as known covariates. Here all three parameters are estimated
jointly from one scalar summary statistic, which is what the paper's Study 2
does and what makes the problem hard: `d_bar` responds strongly to p2
(corr 0.74 on the ground truth), weakly to p1 (0.40), and barely to tau (0.14).
Expect a well-identified p2 and broad, possibly multimodal, p1 and tau
marginals -- the paper reports exactly that, and sharpening those marginals is
the stated target of the rebuild (see ../../updates.md section 5).

Three backends share one sampler; only the way the summary statistic and its
uncertainty are obtained differs:

- backend="sim" : exact ABC-MCMC. Run the two-stage simulator `ns` times at the
  proposed theta and score the observation with a synthetic-likelihood Gaussian
  (Wood 2010) of width sqrt(eps^2 + var_sim). This is the expensive baseline the
  surrogates approximate.
- backend="dnn" / "gp" : the surrogate returns (mean, sd) instantly and the ABC
  likelihood is the exact convolution
      p(obs | theta) = N( obs ; mean, sqrt(eps^2 + sd^2) ),
  so the surrogate's calibrated uncertainty widens the likelihood exactly where
  it is unsure, with no Monte-Carlo noise.

PRIORS. Independent truncated normals on each component, matching the paper's
Study 2 setup: theta1, theta2 ~ TN(centre, sd, [lo, hi]) on the log10 scale, and
tau ~ TN(tp/2, sd, [tau_lo, tau_hi]). Passing `prior="uniform"` switches to a
flat prior on the same box, which is the honest default when no MOM-style
pilot estimate is available to centre on.

PROPOSAL. Component-wise random walk with per-component step sizes, since the
three coordinates have very different natural scales (log10 units vs. absolute
time). Steps are truncated to the box and Hastings-corrected accordingly.
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


def _log_prior(theta, box, kind, centres, sds):
    """Independent truncated-normal (or uniform) log prior on the box."""
    lp = 0.0
    for v, (lo, hi), c, s in zip(theta, box, centres, sds):
        if not (lo <= v <= hi):
            return -np.inf
        if kind == "normal":
            lp += norm.logpdf(v, loc=c, scale=s) - _tn_logZ(c, s, lo, hi)
    return lp


def run_abc_mcmc(obs, backend, n_mcmc=2000, theta_init=None, steps=DEFAULT_STEPS,
                 box=DEFAULT_BOX, rng=None, eps=0.005,
                 prior="uniform", prior_centres=None, prior_sds=(2.0, 2.0, 4.0),
                 sim_kwargs=None, ns=1, surrogate=None):
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
            # Synthetic-likelihood ABC: estimate the statistic's mean and
            # variance from `ns` fresh simulations at this theta, then score the
            # observation under a Gaussian with that mean and eps^2 + variance --
            # the same form the surrogate backends use, paid for by brute force.
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
        def log_like(th):
            mean, sd = surrogate.predict([th[0], th[1], th[2]])
            return norm.logpdf(obs_log, loc=mean, scale=np.sqrt(eps ** 2 + sd ** 2))
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if theta_init is None:
        theta_init = np.array([0.5 * (lo + hi) for lo, hi in box])
    theta_init = np.clip(np.asarray(theta_init, dtype=float),
                         [b[0] + 1e-9 for b in box], [b[1] - 1e-9 for b in box])

    samples = np.empty((n_mcmc, 3))
    samples[0] = theta_init
    ll = log_like(theta_init)
    lp = _log_prior(theta_init, box, prior, centres, prior_sds)
    n_accept = 0

    for i in range(1, n_mcmc):
        cur = samples[i - 1]
        # component-wise truncated random walk; the Hastings term does not cancel
        # because the truncation mass differs between the current and candidate
        # points, so it is accumulated per coordinate.
        can = np.empty(3)
        log_q = 0.0
        for k in range(3):
            lo, hi = box[k]
            can[k] = _tn_sample(cur[k], steps[k], lo, hi, rng)
            log_q += _tn_logZ(cur[k], steps[k], lo, hi) - _tn_logZ(can[k], steps[k], lo, hi)
        ll_can = log_like(can)
        lp_can = _log_prior(can, box, prior, centres, prior_sds)
        if np.log(rng.random()) < min(0.0, (ll_can - ll) + (lp_can - lp) + log_q):
            samples[i] = can
            ll, lp = ll_can, lp_can
            n_accept += 1
        else:
            samples[i] = cur
    return samples, n_accept / (n_mcmc - 1)


def summarize(samples, burn_in, cred=0.95):
    """Posterior summaries per parameter from the post-burn-in draws.

    p1 and p2 are summarised on the natural (not log) scale so the numbers are
    comparable with the paper's tables; tau is already on its natural scale.
    Returns a dict of dicts keyed by parameter name.
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
    """Effective sample size of a 1-D chain via the initial-positive-sequence rule.

    Reported alongside acceptance because a joint 3-parameter chain on a weakly
    identified surface can accept healthily while still mixing badly in the
    poorly-informed directions -- which is precisely the failure mode expected
    for p1 and tau here.
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
