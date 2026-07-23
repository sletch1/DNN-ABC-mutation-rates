"""Heteroscedastic *residual* MLP surrogate for the 3-D ABC summary statistic.

Maps  (log10(p), a, delta)  ->  ( mean of log10(d_bar),  log predictive variance ).

Why a different architecture than the 1-D model
-----------------------------------------------
The 1-D surrogate (network/model.py in ../1D) is a shallow 2-layer funnel MLP:
the target there is a smooth *monotone 1-D curve*, so two layers already suffice
and anything deeper only overfits. The 3-D target is a genuinely harder object --
a response *surface* over (log10 p, a, delta) with real interactions (delta only
matters once mutants exist, which p controls; a rescales the whole time axis).
Fitting it well wants more depth, but a plain deep MLP on ~5000 rows trains poorly
(vanishing gradients, edge bias). So the 3-D model is a **pre-activation residual
MLP**:

  input(3) -> Linear -> [ LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU ->
                          Linear  (+ skip) ] x n_blocks -> SiLU -> two heads.

Design choices and their reasons:
- **Residual blocks** let gradients flow through the identity path, so a deep
  (here 3-block / 7-layer) network optimizes cleanly on a modest dataset -- the
  extra depth buys capacity for the interaction structure the 1-D net never needed.
- **LayerNorm, NOT BatchNorm.** The 1-D study found BatchNorm catastrophic (~11x
  worse) on a smooth regression because it injects *mini-batch-dependent* noise and
  biases the domain edges. LayerNorm normalizes per-sample across features, so it
  carries none of that batch dependence while still stabilizing the deep residual
  stack -- the safe way to get normalization's training benefit here.
- **SiLU (swish)** activation: smooth (C-inf, good gradients for any later
  gradient-based sampler) and non-monotone, which helps represent the surface's
  curvature; on 3-D data the smooth activations beat ReLU in the arch search.
- **Two heads (heteroscedastic).** Kept from the 1-D design and *more* important in
  3-D: the replicate noise of log10(d_bar) varies across the whole (p, a, delta)
  box, and GPS-ABC's acceptance step (Eqs. 9-10, Lu-Zhu-Wu 2023) needs a predictive
  variance at each proposed theta. A GP supplies only one homoscedastic term; the
  variance head learns the input-dependent shape directly (target #5).

The mean head still trains toward the same target as a plain regressor (an MSE
warm-up precedes the Gaussian-NLL phase), so point accuracy is preserved; the
variance head is what supplies calibrated uncertainty.
"""

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class _ResBlock(nn.Module):
    """Pre-activation residual block: x -> x + MLP(x), width preserved."""

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
    """Residual MLP with mean and log-variance heads for 3-D inputs.

    Args:
        in_dim:    number of inputs (3: log10 p, a, delta).
        width:     residual-block width.
        n_blocks:  number of residual blocks (depth ~ 2*n_blocks + 2 linear layers).
        activation/use_ln/dropout: block internals.
    """

    def __init__(self, in_dim=3, width=128, n_blocks=3, dropout=0.0,
                 activation="silu", use_ln=True,
                 min_logvar=-12.0, max_logvar=4.0):
        super().__init__()
        act = _ACT[activation]
        self.input_proj = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(
            *[_ResBlock(width, activation, use_ln, dropout) for _ in range(n_blocks)])
        self.out_act = act()
        self.mean_head = nn.Linear(width, 1)
        self.logvar_head = nn.Linear(width, 1)
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.out_act(h)
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
    """Per-feature z-score, fit once on the training split only.

    Generalizes the 1-D package's scalar Standardizer to multi-column inputs:
    statistics are taken along dim 0, so a (N, d) input gets one (mean, std) per
    column and a (N, 1) target behaves exactly like the 1-D case.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: torch.Tensor):
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_) / self.std_

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std_ + self.mean_

    def inverse_std(self, s: torch.Tensor) -> torch.Tensor:
        """Map a standardized-scale std back to the original (single-column) scale."""
        return s * self.std_.squeeze()

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d):
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self
