"""Amortized neural posterior estimation (NPE) baseline (M1).

Uses `sbi`'s single-round NPE-C (`sbi.inference.NPE`) with a mixture-density-
network conditional density estimator -- the standard sbi choice for a
low-dimensional posterior; a normalizing flow (the package default) collapses
to a Gaussian in a 1-D output space and emits a warning to that effect, which
an MDN does not.

Trained once, exactly like the GP and DNN surrogates, on the same rows
(replicates 1-8, resampled per J the same way `train.py`'s `resample_dbar_J`
does for the other two surrogates -- see its docstring). Unlike GPS-ABC and
DNN-ABC, NPE is not plugged into ABC-MCMC at all: it maps an observed summary
directly to posterior samples with no acceptance step and no per-dataset
simulation, which is the whole point of comparing it here (Section
"Related work" -- amortized inference vs. a fresh MCMC run per observation).

The prior used both to train NPE and to build its posterior is a Uniform
over the training grid's full range in log10(p) ([-8, -1.46] -- see
`train.load_splits`'s DEFAULT_DATA), not the narrower range ABC-MCMC's prior
is truncated to for the exact simulator's sake (Section 3.2 /
`run_experiments.PRIOR_RANGE`). All three tested truths (p in
{1e-4, 1e-3, 1e-2}) fall well inside both ranges, so this choice doesn't
affect the comparison; it keeps NPE's posterior internally consistent with
what it was actually trained on rather than needing an importance-reweighted
truncation at evaluation time.

`point_and_interval` (abc_mcmc.py) is reused unchanged for NPE's posterior
samples (with `burn_in=0`, since there is no chain), so NPE's results slot
into the same aggregation and table-formatting code as every ABC-MCMC-based
method without a parallel implementation to keep in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import torch

# The grid's actual extent (train.py's DATA / load_splits), used as NPE's
# prior. Kept as a module constant rather than re-derived from the CSV on
# every call, since it's fixed by the committed ground-truth file.
NPE_PRIOR_RANGE = (-8.0, -1.46)

LOG_FLOOR = -6.0  # matches abc_mcmc.py's floor for the (rare) all-extinct summary of 0


def train_npe_by_J(data_path, J_grid, prior_range=NPE_PRIOR_RANGE, k_resamples=5,
                    seed=0, density_estimator="mdn"):
    """Train one amortized NPE per J, on the same replicate-1-8 rows (resampled
    to that J for J != 100) the DNN and GP surrogates use. Returns
    {J: DirectPosterior}, picklable, so `_init_worker` can pass it into each
    worker exactly like `gp_by_J` already is.
    """
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    from train import load_splits

    lo, hi = prior_range
    prior = BoxUniform(low=torch.tensor([lo]), high=torch.tensor([hi]))

    posterior_by_J = {}
    for J in J_grid:
        (x_tr, y_tr), (x_va, y_va), _ = load_splits(
            str(data_path), J=(None if J == 100 else J), k_resamples=k_resamples, seed=seed)
        theta = torch.tensor(np.concatenate([x_tr, x_va]), dtype=torch.float32).unsqueeze(1)
        x = torch.tensor(np.concatenate([y_tr, y_va]), dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(seed)
        inference = NPE(prior=prior, density_estimator=density_estimator,
                         show_progress_bars=False)
        inference.append_simulations(theta, x)
        density_est = inference.train()
        posterior_by_J[J] = inference.build_posterior(density_est)
    return posterior_by_J


def npe_point_and_interval(posterior, obs, n_samples=2000, rng_seed=0, cred=0.95):
    """Sample the amortized posterior at one new observation `obs` (raw
    d_bar, not logged) and return (p_hat, ci_lo, ci_hi, ci_len) on the same
    convention as `abc_mcmc.point_and_interval`.
    """
    from abc_mcmc import point_and_interval

    obs_log = np.log10(max(obs, 10.0 ** LOG_FLOOR))
    x_obs = torch.tensor([[obs_log]], dtype=torch.float32)
    torch.manual_seed(rng_seed)
    samples = posterior.sample((n_samples,), x=x_obs, show_progress_bars=False)
    theta_samples = samples.squeeze(1).numpy()
    return point_and_interval(theta_samples, burn_in=0, cred=cred)
