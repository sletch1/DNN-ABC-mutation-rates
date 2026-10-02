"""Add the exact-simulator ABC-MCMC baseline to an already-completed 3-D
accuracy sweep, without rerunning GPS-ABC/GPS-ABC-ref/DNN-ABC/NPE. Every
method in table1_recovery.csv was validated against the known ground-truth
parameter only, never against the "expensive truth" (exact ABC-MCMC) Study I
uses throughout -- this closes that gap by reusing `obs` already stored in
raw_replicates.csv and running backend="sim" for every row, as
run_experiments.py would have with with_sim=True.

Cost: ~24 min/replicate at this study's settings (n_mcmc=3000, ns=4);
parallelized across replicates with a Pool like run_experiments.py.

Usage: python add_exact.py --workers 10

Reads and overwrites raw_replicates.csv (adding ABC-MCMC columns), flips
experiment_config.json's with_sim to true, regenerates table1_recovery.csv/TABLES.md.
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

from abc_mcmc import run_abc_mcmc, summarize, ess, DEFAULT_BOX, DEFAULT_STEPS
from run_experiments import aggregate, A, TP, Z0
from paths import LOG_DIR, TABLE_DIR

_CFG = {}


def _init_worker(cfg):
    import warnings
    warnings.filterwarnings("ignore")
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
    _CFG["cfg"] = cfg


def _one_row(task):
    i, obs, p1_true, p2_true, tau_true, J = task
    cfg = _CFG["cfg"]
    seed = 20_000 * i + 3  # fixed, simple, independent of add_npe.py's seed stream
    rng = np.random.default_rng(seed)
    sim_kwargs = dict(Z0=Z0, a=A, tp=TP, J=int(J), use_slow=True)
    t0 = time.time()
    s, acc = run_abc_mcmc(obs, backend="sim", n_mcmc=cfg["nmcmc"], steps=DEFAULT_STEPS,
                          box=DEFAULT_BOX, eps=cfg["eps"], rng=rng,
                          sim_kwargs=sim_kwargs, ns=cfg["ns"])
    secs = time.time() - t0
    post = summarize(s, cfg["burnin"])
    truth = {"p1": p1_true, "p2": p2_true, "tau": tau_true}
    out = {"row": i, "ABC-MCMC_secs": secs, "ABC-MCMC_acc": acc}
    for k in ("p1", "p2", "tau"):
        out[f"ABC-MCMC_{k}"] = post[k]["mean"]
        out[f"ABC-MCMC_{k}_cilen"] = post[k]["ci_len"]
        out[f"ABC-MCMC_{k}_cov"] = int(post[k]["ci_lo"] <= truth[k] <= post[k]["ci_hi"])
    out["ABC-MCMC_ess_p2"] = ess(s[cfg["burnin"]:, 1])
    out["ABC-MCMC_ess_tau"] = ess(s[cfg["burnin"]:, 2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=max(1, (__import__("os").cpu_count() or 2) - 2))
    args = ap.parse_args()

    raw_path = LOG_DIR / "raw_replicates.csv"
    cfg_path = LOG_DIR / "experiment_config.json"
    df = pd.read_csv(raw_path)
    cfg = json.loads(cfg_path.read_text())

    tasks = [(i, row.obs, row.p1_true, row.p2_true, row.tau_true, row.J)
             for i, row in enumerate(df.itertuples(index=False))]
    print(f"{len(tasks)} replicates on {args.workers} workers "
          f"(~24 min/replicate at n_mcmc={cfg['nmcmc']}, ns={cfg['ns']} "
          f"-> est. {len(tasks)*24/args.workers:.0f} min wall time)", flush=True)

    results = [None] * len(tasks)
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for n, r in enumerate(pool.imap_unordered(_one_row, tasks), 1):
            results[r["row"]] = r
            if n % 4 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  [{n}/{len(tasks)}] elapsed {el/60:.1f}m  "
                      f"eta {el/n*(len(tasks)-n)/60:.1f}m", flush=True)

    for col in ("ABC-MCMC_secs", "ABC-MCMC_acc", "ABC-MCMC_p1", "ABC-MCMC_p1_cilen",
                "ABC-MCMC_p1_cov", "ABC-MCMC_p2", "ABC-MCMC_p2_cilen", "ABC-MCMC_p2_cov",
                "ABC-MCMC_tau", "ABC-MCMC_tau_cilen", "ABC-MCMC_tau_cov",
                "ABC-MCMC_ess_p2", "ABC-MCMC_ess_tau"):
        df[col] = [r[col] for r in results]

    df.to_csv(raw_path, index=False)
    print(f"wrote {raw_path}")

    cfg["with_sim"] = True
    cfg_path.write_text(json.dumps(cfg, indent=2))

    tab = aggregate(df, cfg)
    tab.to_csv(TABLE_DIR / "table1_recovery.csv", index=False)
    print(f"wrote {TABLE_DIR}/table1_recovery.csv")

    lines = ["# 3-D two-stage model: parameter recovery\n",
             f"Config: `{json.dumps(cfg)}`\n",
             "`rmse_log` is RMSE in log10 units for p1/p2 (so 1.0 = off by an order of "
             "magnitude on average) and in absolute time units for tau. **Prefer it to "
             "`nrmse`**: where a parameter is weakly identified the posterior mean sits "
             "wherever the prior puts its mass, and natural-scale nRMSE then explodes "
             "without conveying anything.\n",
             "| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| ({r['p1']:.0e}, {r['p2']:.0e}, {r['tau']:.0f}) | {int(r['J'])} | "
                     f"{r['method']} | {r['param']} | {r['rmse_log']:.3f} | {r['nrmse']:.3f} | "
                     + (f"{r['ci_len']:.3e} | " if np.isfinite(r['ci_len']) else "- | ")
                     + (f"{r['coverage']:.2f} |" if np.isfinite(r['coverage']) else "- |"))
    (TABLE_DIR / "TABLES.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_DIR}/TABLES.md")


if __name__ == "__main__":
    main()
