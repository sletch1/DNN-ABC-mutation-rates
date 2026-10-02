"""Amortized neural posterior estimation (NPE) baseline, 3-D two-stage model.
`sbi`'s single-round NPE-C with a mixture-density-network estimator (same
family as the 1-D study's npe.py, for consistency -- a normalizing flow isn't
degenerate in 3-D the way it is in 1-D, but there's no reason to differ).

Trained once on the same 16,000 rows DNN-ABC trains on (every replicate
except TEST_REPS), same summary statistic, same prior box as the ABC-MCMC
sampler. Not plugged into ABC-MCMC: no chain, no acceptance step, direct
posterior samples at a new observation. Reuses `abc_mcmc.summarize`
(burn_in=0) so NPE's results share the same aggregation code as every other method.
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
    out) -- DNN-ABC's training budget, not GPS-ABC's smaller GP-capacity-
    limited one. Returns a picklable DirectPosterior.
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
    """Sample the posterior at one new observation `obs` (raw, not logged),
    return the same per-parameter summary dict `abc_mcmc.summarize` returns
    for an MCMC chain, plus ESS (expected near `n_samples` itself, since
    these draws are i.i.d. rather than autocorrelated like an MCMC chain).
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
