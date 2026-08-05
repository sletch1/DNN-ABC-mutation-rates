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

# String -> layer-class lookup, so the activation used by the network can be
# set from a plain config string (e.g. "silu") instead of importing and
# passing an nn.Module subclass around.
_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class HeteroscedasticMLP(nn.Module):
    """Fully-connected network with two output heads: a mean and a
    log-variance, trained jointly so the model reports its own predictive
    uncertainty rather than just a point estimate (see the module docstring
    above for why that matters here).

    Structure: a shared "trunk" of Linear -> [BatchNorm] -> activation ->
    [Dropout] blocks, one per entry in `hidden_dims`, feeding into two
    separate final Linear layers (the "heads"), one for the mean and one
    for the log-variance. Subclassing `nn.Module` and calling `super().__init__()`
    is boilerplate PyTorch expects for anything with learnable parameters;
    it's what lets `.parameters()`, `.to(device)`, saving/loading, etc. all
    work automatically for every layer registered as an attribute below.
    """

    def __init__(self, in_dim=1, hidden_dims=(128, 128, 64), dropout=0.0,
                 activation="silu", use_bn=False,
                 min_logvar=-12.0, max_logvar=4.0):
        """
        Args:
            in_dim: number of input features (1 here: log10(p)).
            hidden_dims: width of each trunk layer, in order, e.g.
                (128, 128, 64) builds three hidden layers of those sizes.
            dropout: dropout probability applied after each trunk layer;
                0.0 disables it (no layer is added at all).
            activation: one of the keys in `_ACT` above.
            use_bn: whether to insert BatchNorm1d after each Linear layer.
                Off by default here — see the top-level README's discussion
                of why BatchNorm hurt this particular smooth 1-D regression.
            min_logvar, max_logvar: soft bounds on the predicted
                log-variance (see `forward`), to keep training numerically
                stable — without them the network can drive the variance
                head to +-infinity chasing a handful of outlier points.
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
        # nn.Sequential just chains these layers so a single call runs all
        # of them in order; it's equivalent to writing x = layer1(x);
        # x = layer2(x); ... by hand.
        self.trunk = nn.Sequential(*trunk)
        self.mean_head = nn.Linear(dims[-1], 1)
        self.logvar_head = nn.Linear(dims[-1], 1)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x):
        """Run the network on a batch of inputs `x` (shape: [batch, in_dim])
        and return `(mean, logvar)`, each of shape [batch, 1].

        PyTorch calls this automatically when you write `model(x)` — you
        don't call `.forward(x)` directly elsewhere in this codebase.
        """
        h = self.trunk(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # Soft-clamp log-variance into [min_logvar, max_logvar]. A hard
        # clamp (torch.clamp) would have zero gradient outside the bounds,
        # so if the network ever predicted a value past the limit it could
        # get permanently stuck there with no training signal to pull it
        # back. Softplus(z) = log(1 + exp(z)) is a smooth approximation to
        # max(z, 0) that stays differentiable everywhere, so this
        # double-softplus squeezes the value toward the bounds while still
        # giving the optimizer a gradient to work with near the edges.
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


def gaussian_nll(mean, logvar, target):
    """Negative log-likelihood of `target` under N(mean, exp(logvar)),
    averaged over the batch. This is the training loss: minimizing it is
    equivalent to maximum-likelihood estimation of both the mean and the
    variance jointly, which is what lets the variance head learn real,
    input-dependent uncertainty instead of just being an unused extra
    output. Dropping the constant term (0.5*log(2*pi)) doesn't change
    where the minimum is, so it's omitted here.
    """
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + inv_var * (target - mean) ** 2).mean()


class Standardizer:
    """Z-score (subtract mean, divide by std) using statistics fit once on
    the training split only, then reused unchanged on the calibration/test
    splits and at inference time.

    This is the usual reason to standardize before training a neural net:
    without it, features/targets on very different numeric scales make the
    loss surface badly conditioned for gradient descent (some weights need
    much bigger updates than others). It's the same idea as scaling
    predictors before ridge/lasso, just applied to a target here.
    Deliberately *not* refit on calibration/test data — using their
    statistics would leak information about those splits into a
    transformation applied before the model ever sees them.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor):
        """Compute and store the mean/std of `x`. Call this once, on the
        training data only."""
        self.mean_ = x.mean()
        self.std_ = x.std()
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stored z-score transform to `x` (e.g. raw targets ->
        standardized targets for training)."""
        return (x - self.mean_) / self.std_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Undo `transform`: map standardized-scale values back to the
        original units (e.g. a model prediction -> log10(d_bar) scale)."""
        return x * self.std_ + self.mean_

    def inverse_std(self, s: torch.Tensor) -> torch.Tensor:
        """Map a standardized-scale standard deviation back to the
        original scale. Note this multiplies by sigma only (no mean
        shift) — a std is a spread, not a location, so re-centering
        doesn't apply the way it does in `inverse`."""
        return s * self.std_

    def state_dict(self):
        """Package the fitted mean/std so they can be saved alongside the
        model checkpoint (mirrors the naming of `nn.Module.state_dict`,
        though this class doesn't inherit from nn.Module)."""
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        """Restore mean/std from a dict produced by `state_dict`, e.g.
        when loading a saved model for inference on new data."""
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self
