"""ABC-MCMC for the constant-mutation-rate fluctuation model.

Metropolis-Hastings sampler over theta = log10(p), ported from
NN_ABC/MatlabCode/ABC_fluc_exp1_rev.m. Three backends share one sampler:

- backend="sim" : ABC-MCMC. At each iteration, run the (slow) simulator at
  theta and theta_can, form the summary statistic, and score it against the
  observed statistic with a Gaussian ABC kernel of width eps (Eq. 2/3).
- backend="dnn" / "gp" : surrogate ABC (our DNN-ABC / the paper's GPS-ABC).
  The surrogate returns (mean, sd) for the summary statistic. Instead of
  Monte-Carlo drawing sim ~ N(mean, sd) then scoring with N(obs; sim, eps)
  (MATLAB lines 57-63), we use the exact convolution:
      p(obs | theta) = N(obs; mean(theta), sqrt(eps^2 + sd(theta)^2)),
  so the surrogate's *predictive uncertainty* sd(theta) widens the likelihood
  exactly where the surrogate is unsure -- this is the calibrated-uncertainty
  wiring the GP got for free (target #5 in dnn_improvement.md), with no MC noise.

Prior: truncated shifted exponential on `range` with rate `lam` (paper: rate 2).
Proposal: truncated normal, sd `s`, bounds `range` (paper: s=0.15).
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
    """Truncated shifted exponential: density ∝ exp(-lam (theta - lo)) on [lo, hi]."""
    if theta < lo or theta > hi:
        return -np.inf
    log_norm = np.log((1 - np.exp(-lam * (hi - lo))) / lam)
    return -lam * (theta - lo) - log_norm


def run_abc_mcmc(obs, backend, n_mcmc=1000, theta_init=None, s=0.15,
                 rng=None, prior_range=(-8.0, -2.0), lam=2.0, eps=0.005,
                 # sim backend:
                 sim_kwargs=None, ns=1,
                 # surrogate backend:
                 surrogate=None):
    """Return (samples, accept_rate). `samples` are theta = log10(p) draws."""
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = prior_range

    # The ABC likelihood is evaluated on the log10(d_bar) scale for ALL backends,
    # because that is the scale the surrogates are trained and calibrated on
    # (homoscedastic, symmetric). The observed statistic is transformed to match.
    LOG_FLOOR = -6.0  # log10 floor for the (rare) all-extinct summary of 0
    obs_log = np.log10(max(obs, 10.0 ** LOG_FLOOR))

    def _log10_floor(v):
        return np.log10(np.maximum(v, 10.0 ** LOG_FLOOR))

    if backend == "sim":
        sk = sim_kwargs
        sim_method = sk.get("method", "synthetic")

        def summary_at(theta):
            vals = np.empty(ns)
            for k in range(ns):
                Z, X = fluc_exp(sk["Z0"], sk["a"], sk["delta"], 10.0 ** theta,
                                sk["tp_fn"](theta), sk["J"], rng, use_slow=sk["use_slow"])
                vals[k] = summary_stat(Z, X)
            return _log10_floor(vals)  # ns simulated summary stats on log10 scale

        if sim_method == "kernel":
            # Faithful to the paper (Eq. 3): average the Gaussian ABC kernel over
            # ns fresh simulator replicates. Exact but needs large ns to mix.
            def log_like(theta):
                sims = summary_at(theta)
                return np.log(np.mean(norm.pdf(obs_log, loc=sims, scale=eps)) + 1e-300)
        else:
            # Synthetic-likelihood ABC (Wood 2010): estimate the summary stat's
            # mean and variance from ns pilot sims, score obs against
            # N(mean_sim, sqrt(eps^2 + var_sim)). Same mean+variance likelihood
            # form the surrogates use -- here paid for by brute-force simulation,
            # there predicted instantly. Mixes smoothly at modest ns.
            def log_like(theta):
                sims = summary_at(theta)
                m = float(np.mean(sims))
                v = float(np.var(sims, ddof=1)) if ns > 1 else 0.0
                return norm.logpdf(obs_log, loc=m, scale=np.sqrt(eps ** 2 + v))
    elif backend in ("dnn", "gp"):
        def log_like(theta):
            mean, sd = surrogate.predict(theta)  # already log10(d_bar) scale
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
        # Hastings correction for the asymmetric truncated-normal proposal
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
    """Posterior point estimate and credible interval of p (not log p).

    Returns (p_hat, ci_lower, ci_upper, ci_length) where p_hat is the posterior
    mean of p = mean(10**theta) over post-burn-in samples.
    """
    post = samples[burn_in:]
    p_post = 10.0 ** post
    p_hat = float(np.mean(p_post))
    lo_q = (1 - cred) / 2
    hi_q = 1 - lo_q
    ci_lower = float(np.quantile(p_post, lo_q))
    ci_upper = float(np.quantile(p_post, hi_q))
    return p_hat, ci_lower, ci_upper, ci_upper - ci_lower
