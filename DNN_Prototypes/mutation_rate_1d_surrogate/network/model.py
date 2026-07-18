"""Heteroscedastic MLP surrogate for the ABC summary statistic.

Maps  log10(p)  ->  ( mean of log10(d_bar),  log predictive variance ).

Why heteroscedastic: GPS-ABC's MCMC acceptance step (Eqs. 9-10 in Lu, Zhu &
Wu 2023) needs a *predictive variance* at each proposed theta -- the GP
supplies this natively. The data's noise is strongly input-dependent (the
residual std of d_bar grows ~100x from p=1e-8 to p=1e-2), which a GP with one
homoscedastic noise term cannot represent. A network with a second output head
for log-variance, trained by Gaussian negative log-likelihood, learns that
input-dependent noise directly -- a genuine improvement over the GP's fixed
noise, and exactly the calibrated uncertainty the acceptance step relies on.

The mean head still trains toward the same target as a plain regressor, so
point-prediction accuracy is preserved; the variance head is what's new.
"""

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dims=(128, 128, 64), dropout=0.0,
                 activation="silu", use_bn=False,
                 min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        act = _ACT[activation]
        dims = [in_dim] + list(hidden_dims)
        trunk = []
        for i in range(len(dims) - 1):
            trunk.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                trunk.append(nn.BatchNorm1d(dims[i + 1]))
            trunk.append(act())
            if dropout > 0:
                trunk.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk)
        self.mean_head = nn.Linear(dims[-1], 1)
        self.logvar_head = nn.Linear(dims[-1], 1)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x):
        h = self.trunk(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # soft-clamp log-variance to a sane range for numerical stability
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


def gaussian_nll(mean, logvar, target):
    """Negative log-likelihood of target under N(mean, exp(logvar))."""
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + inv_var * (target - mean) ** 2).mean()


class Standardizer:
    """Z-score using statistics fit once on the training split only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor):
        self.mean_ = x.mean()
        self.std_ = x.std()
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / self.std_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_ + self.mean_

    def inverse_std(self, s: torch.Tensor) -> torch.Tensor:
        """Map a standardized-scale std back to the original scale (multiply by sigma)."""
        return s * self.std_

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self
