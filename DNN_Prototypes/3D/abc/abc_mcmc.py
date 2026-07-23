"""ABC-MCMC for the 3-D constant-rate fluctuation model.

Metropolis-Hastings sampler over theta = log10(p), the mutation rate -- the
quantity of scientific interest -- with the division rate `a` and mutant relative
growth `delta` supplied as KNOWN covariates. This is the honest 3-D analog of the
paper's 1-D inference: a single scalar summary d_bar cannot jointly identify all
three of (p, a, delta), so we infer p while (a, delta) index the regime. What is
new vs. the 1-D pipeline is that the surrogate is queried at arbitrary (a, delta)
-- something the 1-D surrogate (hard-wired to a=delta=1) fundamentally cannot do.

Three backends share one sampler:

- backend="sim" : ABC-MCMC. Run the (slow) exact simulator at (10**theta, a, delta)
  each iteration, form the summary statistic, score with a synthetic-likelihood
  Gaussian (Wood 2010) of width sqrt(eps^2 + var_sim).
- backend="dnn" / "gp" : surrogate ABC. The surrogate returns (mean, sd) for the
  summary statistic at (theta, a, delta); the exact convolution
      p(obs | theta) = N(obs; mean, sqrt(eps^2 + sd^2))
  lets the surrogate's predictive uncertainty widen the likelihood where it is
  unsure (target #5), with no Monte-Carlo noise.

Prior: truncated shifted exponential on `prior_range` (rate `lam`).
Proposal: truncated normal, sd `s`, bounds `prior_range`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from simulator import fluc_exp, summary_stat


def _trunc_norm_sample(mu, s, lo, hi, rng):
    a, b = (lo - mu) / s, (hi - mu) / s
    from scipy.stats import truncnorm
    return float(truncnorm.rvs(a, b, loc=mu, scale=s, random_state=rng))


def _trunc_norm_lognorm(mu, s, lo, hi):
    """log of the truncated-normal normalizing constant Z(mu) = Phi(b)-Phi(a)."""
    return np.log(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s))


def _log_prior(theta, lam, lo, hi):
    """Truncated shifted exponential: density prop. exp(-lam (theta - lo)) on [lo, hi]."""
    if theta < lo or theta > hi:
        return -np.inf
    log_norm = np.log((1 - np.exp(-lam * (hi - lo))) / lam)
    return -lam * (theta - lo) - log_norm


def run_abc_mcmc(obs, backend, a_known=1.0, delta_known=1.0,
                 n_mcmc=1000, theta_init=None, s=0.15,
                 rng=None, prior_range=(-6.0, -2.0), lam=2.0, eps=0.005,
                 # sim backend:
                 sim_kwargs=None, ns=1,
                 # surrogate backend:
                 surrogate=None):
    """Return (samples, accept_rate). `samples` are theta = log10(p) draws.

    a_known, delta_known are the true covariate values for this dataset; the
    surrogate is queried at [theta, a_known, delta_known] each iteration.
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = prior_range

    # ABC likelihood evaluated on the log10(d_bar) scale for ALL backends (the scale
    # the surrogates are trained and calibrated on). The observed statistic matches.
    LOG_FLOOR = -6.0
    obs_log = np.log10(max(obs, 10.0 ** LOG_FLOOR))

    def _log10_floor(v):
        return np.log10(np.maximum(v, 10.0 ** LOG_FLOOR))

    if backend == "sim":
        sk = sim_kwargs

        def summary_at(theta):
            vals = np.empty(ns)
            for k in range(ns):
                Z, X = fluc_exp(sk["Z0"], a_known, delta_known, 10.0 ** theta,
                                sk["tp_fn"](theta), sk["J"], rng, use_slow=sk["use_slow"])
                vals[k] = summary_stat(Z, X)
            return _log10_floor(vals)

        def log_like(theta):
            sims = summary_at(theta)
            m = float(np.mean(sims))
            v = float(np.var(sims, ddof=1)) if ns > 1 else 0.0
            return norm.logpdf(obs_log, loc=m, scale=np.sqrt(eps ** 2 + v))
    elif backend in ("dnn", "gp"):
        def log_like(theta):
            mean, sd = surrogate.predict([theta, a_known, delta_known])
            scale = np.sqrt(eps ** 2 + sd ** 2)
            return norm.logpdf(obs_log, loc=mean, scale=scale)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if theta_init is None:
        theta_init = 0.5 * (lo + hi)
    samples = np.empty(n_mcmc)
    samples[0] = theta_init
    ll_cur = log_like(theta_init)
    lp_cur = _log_prior(theta_init, lam, lo, hi)
    n_accept = 0

    for i in range(1, n_mcmc):
        theta = samples[i - 1]
        theta_can = _trunc_norm_sample(theta, s, lo, hi, rng)
        log_q = _trunc_norm_lognorm(theta, s, lo, hi) - _trunc_norm_lognorm(theta_can, s, lo, hi)
        ll_can = log_like(theta_can)
        lp_can = _log_prior(theta_can, lam, lo, hi)
        log_alpha = min(0.0, (ll_can - ll_cur) + (lp_can - lp_cur) + log_q)
        if np.log(rng.random()) < log_alpha:
            samples[i] = theta_can
            ll_cur, lp_cur = ll_can, lp_can
            n_accept += 1
        else:
            samples[i] = theta
    return samples, n_accept / (n_mcmc - 1)


def point_and_interval(samples, burn_in, cred=0.95):
    """Posterior mean of p = mean(10**theta) and its 95% credible interval."""
    post = samples[burn_in:]
    p_post = 10.0 ** post
    p_hat = float(np.mean(p_post))
    lo_q = (1 - cred) / 2
    hi_q = 1 - lo_q
    ci_lower = float(np.quantile(p_post, lo_q))
    ci_upper = float(np.quantile(p_post, hi_q))
    return p_hat, ci_lower, ci_upper, ci_upper - ci_lower
