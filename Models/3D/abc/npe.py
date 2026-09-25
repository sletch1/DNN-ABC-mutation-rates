"""Amortized neural posterior estimation (NPE) baseline (M1), 3-D two-stage model.

Same design as the 1-D study's `npe.py`: `sbi`'s single-round NPE-C
(`sbi.inference.NPE`) with a mixture-density-network density estimator, for
consistency between the two studies rather than because the 3-D posterior
specifically needs it (a normalizing flow is not degenerate in 3 output
dimensions the way it is in 1-D -- Section "Related work" -- but there is no
reason to use a different estimator family in the one study than the other).

Trained once on the same rows DNN-ABC trains on -- every replicate except
`TEST_REPS`, all 16,000 rows, J=100 -- conditioned on the same summary
statistic (`train.summary_from_cultures`, the paper's fourth root) every
other method in this study is scored against. The prior is `DEFAULT_BOX`
(abc_mcmc.py), the same box GPS-ABC/DNN-ABC's ABC-MCMC sampler proposes
within and that the ground-truth Latin hypercube design was drawn over, so
NPE's training assumption matches what it was actually trained on.

Not plugged into ABC-MCMC, for the same reason as the 1-D study: no
acceptance step, no chain, direct posterior samples at a new observation.
`abc_mcmc.summarize` (burn_in=0, since there is no chain to burn in) is
reused unchanged on NPE's posterior samples, so NPE's results share the
exact aggregation code every other method in this study already uses instead
of a parallel implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch

NPE_PRIOR_BOX = ((-5.0, -1.3), (-5.0, -1.3), (0.1, 9.9))  # matches abc_mcmc.DEFAULT_BOX


def train_npe(data_path, prior_box=NPE_PRIOR_BOX, seed=0, density_estimator="mdn"):
    """Train one amortized NPE on every non-test replicate (TEST_REPS held
    out, matching DNN-ABC's own training budget -- not GPS-ABC's smaller
    GP_BUDGET-capped design, which is a GP-specific capacity limitation, not
    the simulation budget available). Returns a picklable DirectPosterior.
    """
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    from train import TEST_REPS, summary_from_cultures

    lo, hi = zip(*prior_box)
    prior = BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                       high=torch.tensor(hi, dtype=torch.float32))

    df = pd.read_csv(data_path)
    tr = df[~df["rep"].isin(TEST_REPS)]
    theta = torch.tensor(
        np.column_stack([np.log10(tr.p1), np.log10(tr.p2), tr.tau]), dtype=torch.float32)
    x = torch.tensor(np.log10(summary_from_cultures(tr)), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(seed)
    inference = NPE(prior=prior, density_estimator=density_estimator, show_progress_bars=False)
    inference.append_simulations(theta, x)
    density_est = inference.train()
    return inference.build_posterior(density_est)


def npe_summary(posterior, obs, n_samples=2000, rng_seed=0, cred=0.95):
    """Sample the amortized posterior at one new observation `obs` (raw
    summary statistic, not logged) and return the same per-parameter summary
    dict `abc_mcmc.summarize` returns for an MCMC chain, plus `ess_p2` and
    `ess_tau` computed the same way `abc_mcmc.ess` computes them for a chain
    -- expected to run close to `n_samples` itself here, since posterior
    samples are i.i.d. rather than autocorrelated draws, unlike the MCMC
    backends' chains.
    """
    from abc_mcmc import summarize, ess

    obs_log = np.log10(max(obs, 1e-12))
    x_obs = torch.tensor([[obs_log]], dtype=torch.float32)
    torch.manual_seed(rng_seed)
    samples = posterior.sample((n_samples,), x=x_obs, show_progress_bars=False).numpy()
    out = summarize(samples, burn_in=0, cred=cred)
    out["ess_p2"] = ess(samples[:, 1])
    out["ess_tau"] = ess(samples[:, 2])
    return out
