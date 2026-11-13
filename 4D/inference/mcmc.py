"""ABC-MCMC with the published sampler structure (get_theta2a.m proposal: component-wise
truncated-normal random walk with the Hastings correction of the repo's tnorm ratios;
priors TN(theta_mu, theta_sigma=20) on the box; features Feat=(mean(obs), sd(obs)) over
the J=100 observed cultures, as ABC_mu2a.m).

Likelihood estimators for a proposal theta:
  "G"     : run G fresh simulations of J cultures each at theta; feature vector
            s_g=(mean,sd) of Y; L_hat = (1/G) sum_g prod_f K((s_obs_f - s_g,f)/h_f),
            Gaussian kernel. NO normality assumption on the data -- only a kernel of width h
            (the ABC tolerance). Pseudo-marginal MCMC (L_hat is unbiased for the
            kernel-smoothed likelihood).
  "gauss" : surrogate (m, s) -> mean of J cultures ~ N(m, s^2/J), sd feature ~ N(s, s^2/(2(J-1))),
            the closed-form normal approximation the previous version used.
"""
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import logsumexp
from sim import mbp


def _tn(mu, s, lo, hi, rng):
    return float(truncnorm.rvs((lo - mu) / s, (hi - mu) / s, loc=mu, scale=s, random_state=rng))


def _lz(mu, s, lo, hi):
    return np.log(max(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s), 1e-300))


def features(y):
    return np.array([y.mean(), y.std(ddof=1)])


def ll_G(theta, obs_feat, h, G, J, kind, tp, rng, exact=False):
    f = np.empty((G, 2))
    for g in range(G):
        y = np.array([mbp.draw_stat(rng, theta, kind, 1, 1.0, tp, 4, exact) for _ in range(J)])
        f[g] = features(y)
    z = -0.5 * (((obs_feat - f) / h) ** 2).sum(1)
    return logsumexp(z) - np.log(G)


def ll_gauss(theta, obs_feat, h, J, surrogate):
    m, s = surrogate.predict(theta)
    m, s = m[0], s[0]
    sd_m = np.sqrt(s ** 2 / J + h[0] ** 2)
    sd_s = np.sqrt(s ** 2 / (2 * (J - 1)) + h[1] ** 2)
    return norm.logpdf(obs_feat[0], m, sd_m) + norm.logpdf(obs_feat[1], s, sd_s)


def run(obs_y, box, steps, theta0, n_iter, rng, mode="G", h=(0.01, 0.01), G=20, kind="4d",
        tp=20.0, surrogate=None, theta_mu=None, theta_sigma=20.0, exact=False):
    box = np.asarray(box, float); d = len(box); J = len(obs_y)
    obs_feat = features(np.asarray(obs_y)); h = np.asarray(h, float)
    theta_mu = (box.mean(1) if theta_mu is None else theta_mu)

    def lp(t):
        return sum(norm.logpdf(t[i], theta_mu[i], theta_sigma) - _lz(theta_mu[i], theta_sigma, *box[i])
                   for i in range(d))

    def ll(t):
        return (ll_G(t, obs_feat, h, G, J, kind, tp, rng, exact) if mode == "G"
                else ll_gauss(t, obs_feat, h, J, surrogate))

    cur = np.asarray(theta0, float); lcur, pcur = ll(cur), lp(cur)
    out = np.empty((n_iter, d)); acc = 0
    for i in range(n_iter):
        can = np.array([_tn(cur[k], steps[k], *box[k], rng) for k in range(d)])
        lq = sum(_lz(cur[k], steps[k], *box[k]) - _lz(can[k], steps[k], *box[k]) for k in range(d))
        lc, pc = ll(can), lp(can)
        if np.log(rng.random()) < lc - lcur + pc - pcur + lq:
            cur, lcur, pcur = can, lc, pc; acc += 1
        out[i] = cur
    return out, acc / n_iter
