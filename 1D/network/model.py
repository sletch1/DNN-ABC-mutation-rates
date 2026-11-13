"""Heteroscedastic MLP surrogate: log10(p) -> (mean, log-variance) of
log10(d_bar). Two output heads because the ABC acceptance step (Eqs. 9-10,
Lu/Zhu/Wu 2023) needs a predictive variance per theta, and this data's noise
is strongly input-dependent (residual sd grows ~100x from p=1e-8 to 1e-2) --
a single homoscedastic value, what a GP gives natively, can't capture that.
"""

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class HeteroscedasticMLP(nn.Module):
    """Stacked linear layers with a nonlinearity between them, ending in two
    output heads: one for the mean, one for the log-variance.
    """

    def __init__(self, in_dim=1, hidden_dims=(128, 128, 64), dropout=0.0,
                 activation="silu", use_bn=False,
                 min_logvar=-12.0, max_logvar=4.0):
        # use_bn off by default: fit this smooth 1-D curve ~11x worse (README §5).
        # min/max_logvar: soft bounds so the variance head can't run off to +-inf.
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
        """Predict `(mean, logvar)` for a batch of inputs `x`."""
        h = self.trunk(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # Soft-clamp to [min_logvar, max_logvar]; softplus keeps it differentiable
        # everywhere (a hard clamp would zero the gradient past the bounds).
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


def gaussian_nll(mean, logvar, target):
    """Negative Normal log-likelihood (training loss); minimizing it is joint
    MLE of mean and variance. Constant 0.5*log(2*pi) dropped, doesn't move the minimum."""
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + inv_var * (target - mean) ** 2).mean()


class Standardizer:
    """Z-scores fit on the TRAINING split only, reused unchanged elsewhere
    (refitting on calibration/test would leak those splits' information)."""

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
        """Same as `inverse`, for a standard deviation: scale only, no recentering."""
        return s * self.std_

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self
