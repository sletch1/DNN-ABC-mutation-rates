"""M3: does the ABC tolerance eps drive Study II's over-coverage, or is it
the model's identifiability? Table 8 reports over-coverage at eps=0.005 but
never actually varies eps to test the explanation -- this reruns the
GP/DNN-backed samplers (NPE excluded: no acceptance kernel, no eps) across
an eps grid and checks whether coverage tracks eps or stays pinned near 1.0.

Cheap: GPS-ABC/DNN-ABC never call the simulator, so a 3000-iter chain costs
~2s once the surrogate is fit (once, shared across replicates). Reuses `obs`
from raw_replicates.csv read-only -- safe to run alongside add_exact.py.

Usage: python eps_sweep.py --workers 3
Writes results/logs/eps_sweep_raw.csv and results/tables/eps_sweep.md.
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from abc_mcmc import run_abc_mcmc, summarize, DEFAULT_BOX, DEFAULT_STEPS
from surrogates import fit_gp_surrogate_3d, fit_gp_surrogate_3d_reference
from train import load_surrogate, TEST_REPS, summary_from_cultures
from paths import DATA, LOG_DIR, TABLE_DIR, MODEL_DIR

EPS_GRID = [0.0025, 0.005, 0.01, 0.02, 0.04]
_G = {}


def _init_worker(ckpt, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
    dnn = load_surrogate(ckpt)
    df = pd.read_csv(DATA)
    tr = df[~df["rep"].isin(TEST_REPS)]
    X = np.column_stack([np.log10(tr.p1), np.log10(tr.p2), tr.tau])
    S = summary_from_cultures(tr)
    gp = fit_gp_surrogate_3d(X, np.log10(S), budget=cfg["gp_budget"])
    gp_ref = fit_gp_surrogate_3d_reference(tr.p1, tr.p2, tr.tau, S, budget=cfg["gp_budget"])
    _G.update(dnn=dnn, gp=gp, gp_ref=gp_ref, cfg=cfg)


def _one_task(task):
    i, obs, p1_true, p2_true, tau_true, eps = task
    cfg = _G["cfg"]
    truth = {"p1": p1_true, "p2": p2_true, "tau": tau_true}
    out = {"row": i, "eps": eps}
    backends = [("GPS-ABC", dict(backend="gp", surrogate=_G["gp"])),
                ("GPS-ABC-ref", dict(backend="gp", surrogate=_G["gp_ref"])),
                ("DNN-ABC", dict(backend="dnn", surrogate=_G["dnn"]))]
    for name, kw in backends:
        rng = np.random.default_rng(30_000 * i + int(eps * 1e6))
        s, acc = run_abc_mcmc(obs, n_mcmc=cfg["nmcmc"], steps=DEFAULT_STEPS,
                              box=DEFAULT_BOX, eps=eps, rng=rng, **kw)
        post = summarize(s, cfg["burnin"])
        out[f"{name}_acc"] = acc
        for k in ("p1", "p2", "tau"):
            out[f"{name}_{k}"] = post[k]["mean"]
            out[f"{name}_{k}_cilen"] = post[k]["ci_len"]
            out[f"{name}_{k}_cov"] = int(post[k]["ci_lo"] <= truth[k] <= post[k]["ci_hi"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    raw_path = LOG_DIR / "raw_replicates.csv"
    cfg_path = LOG_DIR / "experiment_config.json"
    df = pd.read_csv(raw_path)  # read-only: never written back
    base_cfg = json.loads(cfg_path.read_text())
    cfg = dict(nmcmc=base_cfg["nmcmc"], burnin=base_cfg["burnin"], gp_budget=base_cfg["gp_budget"])

    tasks = [(i, row.obs, row.p1_true, row.p2_true, row.tau_true, eps)
             for i, row in enumerate(df.itertuples(index=False)) for eps in EPS_GRID]
    print(f"{len(tasks)} tasks ({len(df)} replicates x {len(EPS_GRID)} eps values) "
          f"on {args.workers} workers", flush=True)

    ckpt = MODEL_DIR / "surrogate_3d.pt"
    results = []
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(str(ckpt), cfg)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one_task, tasks), 1):
            results.append(r)
            if n % 40 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  [{n}/{len(tasks)}] elapsed {el/60:.1f}m  "
                      f"eta {el/n*(len(tasks)-n)/60:.1f}m", flush=True)

    out = pd.DataFrame(results)
    out_path = LOG_DIR / "eps_sweep_raw.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    lines = ["# M3: ABC tolerance (eps) sweep, Study II\n",
             f"Grid: {EPS_GRID}. Same 48 replicates (3 truths x 16 reps) as "
             f"the main sweep, `obs` reused unchanged; GPS-ABC, GPS-ABC-ref, "
             f"DNN-ABC only (NPE has no acceptance kernel and no eps).\n",
             "| eps | method | param | coverage | mean 95% CI width |",
             "|---|---|---|---|---|"]
    for eps in EPS_GRID:
        sub = out[out.eps == eps]
        for m in ("GPS-ABC", "GPS-ABC-ref", "DNN-ABC"):
            for k in ("p1", "p2", "tau"):
                cov = sub[f"{m}_{k}_cov"].mean()
                cil = sub[f"{m}_{k}_cilen"].mean()
                lines.append(f"| {eps} | {m} | {k} | {cov:.3f} | {cil:.3e} |")
    (TABLE_DIR / "eps_sweep.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_DIR}/eps_sweep.md")


if __name__ == "__main__":
    main()
