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

THE DERIVED FEATURE (the main design decision here)
---------------------------------------------------
A mutation arising at time t founds a clone that grows to roughly e^(a(tp-t)) by
the plating time, while the number of division events available to mutate near
time t grows like e^(a*t). Those two exponentials cancel, so every unit of time
contributes equally to the final mutant count and the physically correct
aggregate of a time-varying mutation rate is its TIME AVERAGE:

    p_eff = ( p1 * tau + p2 * (tp - tau) ) / tp

Empirically log10(p_eff) alone explains R^2 = 0.874 of the design-mean variation
-- more than the full three-input linear model -- and regressing the target on it
gives a slope of 0.59 against the value 0.5 predicted by d_bar ~ sqrt(X/Z). (A
Yule-arrival-CDF weighting, which looks plausible but ignores the cancellation
above, reaches only 0.824.)

So the network is fed FOUR inputs: the three raw parameters plus log10(p_eff).
The raw inputs are kept because p_eff alone caps out around R^2 = 0.88 -- it
compresses a 3-D surface onto one coordinate and cannot express the residual
tau- and p1-dependent shape.

HONEST SIZING OF THE GAIN. The R^2 = 0.874 above is for a LINEAR model, where
supplying the right coordinate matters a great deal. A neural network can learn
that structure from the raw inputs by itself, and the measured ablation
(results/logs/benchmark_arch.md, identical specs with and without the column)
shows only a 1.01x-1.03x improvement in held-out mse_mean. The feature is
therefore kept for reasons other than accuracy: it encodes a derivation that is
checkable (the fitted slope of 0.59 against a predicted 0.5), it is the quantity
the constant-rate MOM/MLE baselines actually estimate -- so the same coordinate
makes those baselines interpretable -- and it costs four floating-point
operations. It is NOT what makes the surrogate accurate. `use_derived=False`
disables it.

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

import numpy as np
import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}

# Fixed experimental constants of the ground-truth design; needed to build the
# derived feature. These match RCode/genSlowData_3D.R.
TP_DEFAULT = 10.0

FEATURES_RAW = ["log10p1", "log10p2", "tau"]
FEATURES_ALL = FEATURES_RAW + ["log10p_eff"]


def add_derived(X, tp: float = TP_DEFAULT):
    """Append log10(p_eff) to an (N, 3) array of [log10 p1, log10 p2, tau].

    p_eff is the time-average of the two-stage mutation rate over [0, tp]; see
    the module docstring for the derivation. Returns an (N, 4) array.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    p1, p2, tau = 10.0 ** X[:, 0], 10.0 ** X[:, 1], np.clip(X[:, 2], 0.0, tp)
    p_eff = (p1 * tau + p2 * (tp - tau)) / tp
    return np.c_[X, np.log10(np.maximum(p_eff, 1e-300))].astype(np.float32)


class HeteroscedasticMLP(nn.Module):
    """Plain funnel MLP with mean and log-variance heads.

    Args:
        in_dim: number of inputs (4 with the derived feature, 3 without).
        hidden: widths of the hidden layers, e.g. (128, 64).
        activation/dropout/use_ln: layer internals.
    """

    def __init__(self, in_dim=4, hidden=(128, 64), activation="gelu",
                 dropout=0.0, use_ln=False, min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        act = _ACT[activation]
        layers, prev = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_ln:
                layers.append(nn.LayerNorm(h))
            layers.append(act())
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

    def __init__(self, in_dim=4, width=128, n_blocks=3, activation="silu",
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
