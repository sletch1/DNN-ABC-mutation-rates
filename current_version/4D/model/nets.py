"""Zero-aware (hurdle) heteroscedastic surrogate for the 3-D two-stage ABC statistic.

GOAL (inherited from the previous version of this project): replace the Gaussian-process
surrogate inside the ABC-MCMC loop. At each proposed theta = (log10 p1, log10 p2, tau) the
sampler needs the predictive distribution of the J=100-culture summary d_bar = mean_i (X_i/Z_i)^(1/4),
cheaply, with an HONEST spread (the acceptance step consumes it). The old network predicted
(mean, variance) of log10(d_bar). Under the paper's prior box (p down to 1e-11) about 30% of
datasets have d_bar == 0 exactly (no mutants in any of the 100 cultures), where log10 is undefined,
so the output here is a hurdle model:

    P(d_bar = 0 | theta)          -> logistic head
    (log10 d_bar, log10 s) | d_bar > 0 -> two Gaussian heads (mean, log-variance each), heteroscedastic,

where s = sd over the J cultures: the same two features (mean and sd of the observed statistic) the
GPS-ABC sampler of Lu et al. compares (Feat = [mean(obs_X), std(obs_X)]).
Likelihood of an observation: d_bar = 0 -> P0 ; otherwise (1-P0) * N(log10 d_bar; mu1, sd1) * N(log10 s; mu2, sd2)
(the two features are treated as independent given theta, as in GPS-ABC).
"""
import torch
import torch.nn as nn

ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU, "softplus": nn.Softplus}


class _Res(nn.Module):
    def __init__(self, w, act):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(w, w), ACT[act](), nn.Linear(w, w))
        self.a = ACT[act]()

    def forward(self, x):
        return self.a(x + self.f(x))


class HurdleNet(nn.Module):
    """kind='mlp': hidden=(w1,w2,..), activation = name or per-layer list.
    kind='res' : hidden=(w, n_blocks)  pre-widened residual MLP."""

    def __init__(self, in_dim=3, kind="mlp", hidden=(64, 32), activation="gelu",
                 min_logvar=-12.0, max_logvar=4.0, corr=False):
        super().__init__()
        if kind == "mlp":
            acts = [activation] * len(hidden) if isinstance(activation, str) else list(activation)
            L, p = [], in_dim
            for h, a in zip(hidden, acts):
                L += [nn.Linear(p, h), ACT[a]()]; p = h
            self.body = nn.Sequential(*L)
        elif kind == "res":
            w, nb = hidden
            self.body = nn.Sequential(nn.Linear(in_dim, w), ACT[activation](), *[_Res(w, activation) for _ in range(nb)])
            p = w
        else:
            raise ValueError(kind)
        self.corr = corr
        self.head = nn.Linear(p, 6 if corr else 5)   # logit(P0), mu1, mu2, logvar1, logvar2 [, atanh(rho)]
        self.lo, self.hi = min_logvar, max_logvar

    def forward(self, x):
        o = self.head(self.body(x))
        lv = self.hi - nn.functional.softplus(self.hi - o[:, 3:5])
        lv = self.lo + nn.functional.softplus(lv - self.lo)
        if self.corr:        # correlation of the two features given theta (the GPS-ABC likelihood assumes it is zero)
            return o[:, 0], o[:, 1:3], lv, 0.99 * torch.tanh(o[:, 5])
        return o[:, 0], o[:, 1:3], lv


def hurdle_loss(logit, mu, lv, zero, y, rho=None, w_cls=1.0):
    """BCE on all rows + Gaussian NLL (summed over the two features) on the nonzero rows (y standardised, (n,2))."""
    bce = nn.functional.binary_cross_entropy_with_logits(logit, zero)
    nz = zero < 0.5
    if not nz.any():
        nll = logit.sum() * 0
    elif rho is None:
        nll = (0.5 * (lv[nz] + (y[nz] - mu[nz]) ** 2 * torch.exp(-lv[nz]))).sum(1).mean()
    else:                    # bivariate normal with correlation rho
        z = (y[nz] - mu[nz]) * torch.exp(-0.5 * lv[nz]); r = rho[nz]
        nll = (0.5 * (lv[nz].sum(1) + torch.log1p(-r ** 2) + (z[:, 0] ** 2 - 2 * r * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / (1 - r ** 2))).mean()
    return w_cls * bce + nll, bce, nll
