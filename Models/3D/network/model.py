"""Heteroscedastic surrogate for the 3-D two-stage ABC summary statistic:
(log10 p1, log10 p2, tau) -> (mean, log predictive variance) of log10(d_bar).

Trained once on pre-computed simulator output so ABC-MCMC can query it
instantly instead of resimulating; needs both a mean AND an honest predictive
variance since the ABC acceptance step consumes one at every proposal
(Lu-Zhu-Wu 2023 Eqs. 9-10). Two-headed (mean, log-variance) design because
the noise is violently heteroscedastic (within-design-point sd varies 292x
across the design, correlates -0.74 with the target) -- a GP's single
homoscedastic noise term can't represent that. No BatchNorm (found ~11x
worse on this smooth regression in the 1-D study); soft-clamped log-variance
for NLL stability without killing the gradient at the bounds.
"""

import torch
import torch.nn as nn

# Grouped by whether the function is smooth (keeps the surrogate differentiable
# in its inputs) and whether it has a live negative region (inputs are
# standardized, so ~half of pre-activations are negative).
_ACT = {
    "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU,  # piecewise-linear
    "tanh": nn.Tanh, "elu": nn.ELU, "selu": nn.SELU,  # smooth, saturating
    "softplus": nn.Softplus, "gelu": nn.GELU, "silu": nn.SiLU, "mish": nn.Mish,  # smooth, unbounded
}

FEATURES_RAW = ["log10p1", "log10p2", "tau"]


def _act_per_layer(activation, n_layers):
    """Normalise `activation` to one name per hidden layer: accepts a single
    name ("gelu") or a per-layer sequence (("gelu", "tanh")), the latter for
    benchmark_activation_pairs.py's mixed-activation comparison."""
    if isinstance(activation, str):
        names = [activation] * n_layers
    else:
        names = list(activation)
        if len(names) != n_layers:
            raise ValueError(
                f"got {len(names)} activations for {n_layers} hidden layers")
    unknown = [a for a in names if a not in _ACT]
    if unknown:
        raise KeyError(f"unknown activation(s) {unknown}; choose from {sorted(_ACT)}")
    return names


class HeteroscedasticMLP(nn.Module):
    """Plain funnel MLP with mean and log-variance heads. `hidden`: widths
    of each hidden layer; `activation`: one name or a per-layer sequence."""

    def __init__(self, in_dim=3, hidden=(128, 64), activation="gelu",
                 dropout=0.0, use_ln=False, min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        acts = _act_per_layer(activation, len(hidden))
        layers, prev = [], in_dim
        for h, a in zip(hidden, acts):
            layers.append(nn.Linear(prev, h))
            if use_ln:
                layers.append(nn.LayerNorm(h))
            layers.append(_ACT[a]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x):
        h = self.body(x)
        logvar = self.logvar_head(h)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return self.mean_head(h), logvar


class _ResBlock(nn.Module):
    """Pre-activation residual block: x -> x + MLP(x), width preserved. The
    skip connection means a block only learns a correction to its input, so
    depth can increase without the network getting harder to train."""

    def __init__(self, width, activation="silu", use_ln=True, dropout=0.0):
        super().__init__()
        act = _ACT[activation]
        layers = []
        for _ in range(2):
            if use_ln:
                layers.append(nn.LayerNorm(width))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(width, width))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class HeteroscedasticResMLP(nn.Module):
    """Residual variant, for when extra depth is wanted on the interaction surface."""

    def __init__(self, in_dim=3, width=128, n_blocks=3, activation="silu",
                 use_ln=True, dropout=0.0, min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        act = _ACT[activation]
        self.input_proj = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(
            *[_ResBlock(width, activation, use_ln, dropout) for _ in range(n_blocks)])
        self.out_act = act()
        self.mean_head = nn.Linear(width, 1)
        self.logvar_head = nn.Linear(width, 1)
        self.min_logvar, self.max_logvar = min_logvar, max_logvar

    def forward(self, x):
        h = self.out_act(self.blocks(self.input_proj(x)))
        logvar = self.logvar_head(h)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return self.mean_head(h), logvar


def build(kind="mlp", **kw):
    """Factory used by the architecture search and by train.py."""
    return {"mlp": HeteroscedasticMLP, "resmlp": HeteroscedasticResMLP}[kind](**kw)


def gaussian_nll(mean, logvar, target):
    """Negative log-likelihood of target under N(mean, exp(logvar))."""
    return 0.5 * (logvar + torch.exp(-logvar) * (target - mean) ** 2).mean()


class Standardizer:
    """Per-feature z-score, fit once on the training split only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor):
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse(self, x):
        return x * self.std_ + self.mean_

    def inverse_std(self, s):
        return s * self.std_.squeeze()

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        self.mean_, self.std_ = d["mean"], d["std"]
        return self
