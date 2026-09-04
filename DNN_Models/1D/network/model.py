"""Heteroscedastic MLP surrogate for the ABC summary statistic.

Maps  log10(p)  ->  ( mean of log10(d_bar),  log predictive variance ).

This is nonlinear heteroscedastic regression: E[Y|X] and Var(Y|X) are fit
jointly by maximum likelihood under a Normal working model (`gaussian_nll`
below is the negative Normal log-likelihood), solved numerically since
neither has a closed form. The "network" is just a composition of linear
maps separated by a fixed nonlinearity -- closest classical analogue is a
basis-expansion regression, except the basis functions are learned.

Why the variance head matters: the ABC acceptance step (Eqs. 9-10 in Lu, Zhu
& Wu 2023) consumes a predictive variance at each proposed theta. A GP
supplies one natively but only a single homoscedastic value, while this
data's noise is strongly input-dependent (residual sd of d_bar grows ~100x
from p=1e-8 to p=1e-2). The second output head learns that shape directly.
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
        """
        Args:
            in_dim: number of input features (1 here: log10(p)).
            hidden_dims: width of each hidden layer, in order.
            dropout: dropout probability; 0.0 disables it.
            activation: one of the keys in `_ACT` above.
            use_bn: insert BatchNorm after each linear layer. Off by
                default -- it fit this smooth 1-D curve ~11x worse (README §5).
            min_logvar, max_logvar: soft bounds on the predicted log-variance,
                so the variance head can't run off to +-infinity chasing
                outliers.
        """
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
        # Soft-clamp log-variance into [min_logvar, max_logvar]. A hard clamp
        # would have zero gradient past the bounds, so the optimizer could get
        # stuck there; softplus keeps it differentiable everywhere.
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


def gaussian_nll(mean, logvar, target):
    """Negative Normal log-likelihood, averaged over the batch -- the training
    loss. Minimizing it is joint MLE of the mean and the variance. The constant
    0.5*log(2*pi) is dropped since it doesn't move the minimum.
    """
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + inv_var * (target - mean) ** 2).mean()


class Standardizer:
    """Z-scores using statistics fit on the TRAINING split only, then reused
    unchanged elsewhere (refitting on calibration/test would leak information
    about those splits). Same reason you'd scale predictors before ridge/lasso:
    it keeps the optimization well conditioned.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor):
        """Store the mean/sd of `x`. Call once, on training data only."""
        self.mean_ = x.mean()
        self.std_ = x.std()
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Raw scale -> standardized scale."""
        return (x - self.mean_) / self.std_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized scale -> raw scale."""
        return x * self.std_ + self.mean_

    def inverse_std(self, s: torch.Tensor) -> torch.Tensor:
        """Same, for a standard deviation: scale only, no recentering."""
        return s * self.std_

    def state_dict(self):
        """The fitted mean/sd, for saving alongside the model checkpoint."""
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        """Restore mean/sd from `state_dict` output."""
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self
