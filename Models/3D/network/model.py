"""Heteroscedastic surrogate for the 3-D two-stage ABC summary statistic.

Maps  (log10 p1, log10 p2, tau)  ->  ( mean of log10(d_bar), log predictive variance ).

WHAT THIS REPLACES, AND WHY IT EXISTS
-------------------------------------
Estimating the mutation parameters normally means running the cell-level
simulator thousands of times inside an MCMC loop. A surrogate is trained once on
pre-computed simulator output and then answers those queries instantly, which is
what makes ABC affordable. The surrogate must supply BOTH a predicted summary
statistic and an honest uncertainty on it, because the ABC acceptance step needs
a predictive variance at every proposed parameter (Lu-Zhu-Wu 2023, Eqs. 9-10).

WHAT THE DATA SAYS (measured on data/slow_data_3D.csv; see
architecture_search/benchmark_arch.py for the runs behind each claim)
--------------------------------------------------------------------
1. The target spans 2.5 decades, so it is modelled on the log10 scale.

2. The replicate noise is violently heteroscedastic: the within-design-point sd
   of log10(d_bar) varies 292x across the design, and correlates -0.74 with the
   target itself. Where few mutants arise, d_bar is small AND its relative
   scatter is huge. A single homoscedastic noise term -- all a Gaussian process
   offers -- cannot represent that. Hence the two-headed (mean, log-variance)
   design trained by Gaussian NLL.

3. The surface is genuinely non-linear but not wild: a plain linear fit in
   (log10 p1, log10 p2, tau) reaches R^2 = 0.75, adding quadratic terms and
   interactions reaches 0.965. So real curvature and interaction exist, but the
   function is smooth -- this wants a modest network, not a deep one.

INPUTS
------
The network takes exactly the three model parameters, (log10 p1, log10 p2,
tau), and outputs (mean of log10 d_bar, log predictive variance). No derived
features: the surrogate is a direct function of the parameters being inferred.

ARCHITECTURE NOTES
------------------
- **No BatchNorm.** The 1-D study found it ~11x worse on a smooth regression: it
  injects mini-batch-dependent noise and biases predictions at the domain edges.
  That lesson carries over; LayerNorm is offered instead for deeper variants.
- **Smooth activations** (GELU/SiLU) keep the surrogate differentiable in its
  inputs, which matters if the sampler is later upgraded to a gradient-based
  scheme, and avoids ReLU's dead-unit failure mode in a small network.
- **Soft-clamped log-variance** keeps the NLL numerically stable without killing
  the gradient at the bounds (softplus rather than a hard clamp).
"""

import torch
import torch.nn as nn

# Activation registry. Grouped by the property that actually matters for this
# surface: whether the function is smooth (the target E[log10 d_bar | theta] is
# a smooth function of the parameters, and a smooth surrogate stays
# differentiable in its inputs if the sampler is later made gradient-based),
# and whether it keeps a live negative region (inputs are standardized, so
# roughly half of all pre-activations are negative).
_ACT = {
    # piecewise-linear, dead or half-dead negative region
    "relu": nn.ReLU,               # zeroes all negatives; units can die permanently
    "leakyrelu": nn.LeakyReLU,     # small fixed negative slope, no dead units
    "prelu": nn.PReLU,             # same, but the slope is learned
    # smooth and saturating
    "tanh": nn.Tanh,               # bounded, C^inf, classic for smooth regression
    "elu": nn.ELU,                 # exponential negative arm, mean activation ~0
    "selu": nn.SELU,               # ELU scaled for self-normalisation
    # smooth and unbounded
    "softplus": nn.Softplus,       # C^inf approximation to ReLU
    "gelu": nn.GELU,               # smooth, self-gating (incumbent)
    "silu": nn.SiLU,               # Swish: smooth, self-gated, non-monotone
    "mish": nn.Mish,               # smoother self-gated variant of SiLU
}

FEATURES_RAW = ["log10p1", "log10p2", "tau"]


def _act_per_layer(activation, n_layers):
    """Normalise `activation` to one name per hidden layer.

    Accepts a single name applied to every layer ("gelu"), or a sequence giving
    one name per layer (("gelu", "tanh")), which is what the mixed-activation
    comparison in architecture_search/benchmark_activation_pairs.py needs.
    """
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
    """Plain funnel MLP with mean and log-variance heads.

    Args:
        in_dim: number of inputs (3: log10 p1, log10 p2, tau).
        hidden: widths of the hidden layers, e.g. (128, 64).
        activation: one name for every layer, or a sequence with one per layer.
        dropout/use_ln: layer internals.
    """

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
    """Pre-activation residual block: x -> x + MLP(x), width preserved.

    Pre-activation (normalise/activate BEFORE each Linear) is the standard fix
    from the ResNet literature for keeping gradients well behaved in a deep
    stack. The skip connection means a block only has to learn a correction to
    its input, so depth can be increased without the network becoming harder to
    train.
    """

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
