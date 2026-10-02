"""Full ABC-MCMC recovery evaluation for the three new surrogate families
(CNN1D, RNN, LSTM), through the exact same sampler as GPS-ABC/DNN-ABC(MLP)
in run_experiments.py. This is a thin script because `run_abc_mcmc(...,
backend="dnn", surrogate=...)` only calls `surrogate.predict(theta) ->
(mean, sd)` -- wrapping any model_families.py checkpoint (all return
(mean, logvar)) in a DNNSurrogate3D runs the identical sampler/acceptance/
prior/proposal as the deployed MLP, nothing reimplemented.

GPS-ABC and DNN-ABC(MLP) are NOT rerun -- reused verbatim from
table1_recovery.csv/raw_replicates.csv, so MCSEs against the new families are
apples-to-apples without burning time or risking drift. Config (truths,
reps=16, nmcmc=3000, burnin=1000, eps=0.005, J=100, mut_time) is loaded
verbatim from experiment_config.json, not restated, so the two scripts can't
silently disagree. Every replicate uses the identical per-task seed formula
from run_experiments.py, so every family and GPS-ABC/DNN-ABC(MLP) score
against the same simulated observations per (truth, rep) cell.

Outputs (results/arch_families/, nothing existing touched):
  raw_replicates_families.csv   per-replicate summaries, new families only
  table_families_recovery.csv   same columns as table1_recovery.csv, GPS-ABC/DNN-ABC/MOM/MLE reused + new families computed
  mcse_families.md / .csv       MCSEs vs. GPS-ABC and DNN-ABC(MLP), same style as results/tables/mcse.md

Usage: python run_experiments_families.py
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]           # .../Models/3D
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import torch

from simulator import fluc_exp_2stage, summary_stat
from abc_mcmc import run_abc_mcmc, summarize, ess, DEFAULT_BOX, DEFAULT_STEPS
from surrogates import DNNSurrogate3D
from model import Standardizer
from model_families import build_family
from mcse import mcse_rmse, mcse_prop
from paths import RESULTS, TABLE_DIR, LOG_DIR

ARCH_FAMILIES_DIR = RESULTS / "arch_families"
CKPT_DIR = ARCH_FAMILIES_DIR / "checkpoints"
ARCH_FAMILIES_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_LABELS = {"cnn1d": "CNN1D-ABC", "rnn": "RNN-ABC", "lstm": "LSTM-ABC"}
A, TP, Z0 = 1.0, 10.0, 1
_G = {}


def load_family_surrogate(ckpt_path):
    """Rebuild a DNNSurrogate3D around a saved CNN1D/RNN/LSTM checkpoint.
    Parallel to train.py's load_surrogate, but dispatches through
    model_families.build_family instead of model.build.
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = build_family(ckpt["kind"], in_dim=3, **ckpt["spec"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return DNNSurrogate3D(model,
                          Standardizer().load_state_dict(ckpt["x_scaler"]),
                          Standardizer().load_state_dict(ckpt["y_scaler"]),
                          sd_scale=ckpt["sd_scale"],
                          raw_inputs=True)


def _init_worker(ckpts, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    # One torch thread per process: default multi-threaded matmul on tiny
    # tensors caused massive contention (>600 CPU-min observed for a few
    # seconds of real work) when every worker also spun up its own threads.
    torch.set_num_threads(1)
    surrs = {kind: load_family_surrogate(path) for kind, path in ckpts.items()}
    _G.update(surrs=surrs, cfg=cfg)


def _one_replicate_family(task):
    kind, p1, p2, tau, J, rep = task
    cfg = _G["cfg"]
    method = FAMILY_LABELS[kind]
    seed = abs(hash((round(p1, 12), round(p2, 12), round(tau, 3), J, rep))) % (2 ** 31)  # identical formula to run_experiments.py
    rng = np.random.default_rng(seed)

    Zv, Xv = fluc_exp_2stage(Z0, A, p1, p2, tau, TP, J, rng, use_slow=True,
                             mut_time=cfg["mut_time"])
    obs = summary_stat(Zv, Xv)
    truth = dict(p1=p1, p2=p2, tau=tau)

    t0 = time.time()
    s, acc = run_abc_mcmc(obs, backend="dnn", surrogate=_G["surrs"][kind],
                          n_mcmc=cfg["nmcmc"], steps=DEFAULT_STEPS, box=DEFAULT_BOX,
                          eps=cfg["eps"],
                          rng=np.random.default_rng(rng.integers(2 ** 63 - 1)))
    secs = time.time() - t0
    post = summarize(s, cfg["burnin"])

    out = {"family": kind, "method": method, "p1_true": p1, "p2_true": p2,
           "tau_true": tau, "J": J, "rep": rep, "obs": obs, "secs": secs, "acc": acc,
           "ess_p2": ess(s[cfg["burnin"]:, 1]), "ess_tau": ess(s[cfg["burnin"]:, 2])}
    for k in ("p1", "p2", "tau"):
        out[k] = post[k]["mean"]
        out[f"{k}_cilen"] = post[k]["ci_len"]
        out[f"{k}_cov"] = int(post[k]["ci_lo"] <= truth[k] <= post[k]["ci_hi"])
    return out


def aggregate_families(raw):
    """Per-parameter rmse_log/nRMSE/CI-width/coverage per (truth, J, family),
    in the SAME shape as `run_experiments.aggregate`'s output rows."""
    rows = []
    for (p1, p2, tau, J, method), sub in raw.groupby(
            ["p1_true", "p2_true", "tau_true", "J", "method"]):
        for k, tv in (("p1", p1), ("p2", p2), ("tau", tau)):
            est = sub[k].to_numpy(float)
            est = est[np.isfinite(est)]
            if k in ("p1", "p2"):
                rmse_log = float(np.sqrt(np.mean(
                    (np.log10(np.maximum(est, 1e-300)) - np.log10(tv)) ** 2))) if len(est) else np.nan
            else:
                rmse_log = float(np.sqrt(np.mean((est - tv) ** 2))) if len(est) else np.nan
            rows.append({"p1": p1, "p2": p2, "tau": tau, "J": J, "method": method, "param": k,
                        "rmse_log": rmse_log,
                        "nrmse": float(np.sqrt(np.mean((est - tv) ** 2)) / tv) if len(est) else np.nan,
                        "ci_len": float(sub[f"{k}_cilen"].mean()),
                        "coverage": float(sub[f"{k}_cov"].mean()),
                        "secs": float(sub["secs"].mean()), "acc": float(sub["acc"].mean())})
    return pd.DataFrame(rows)


def main():
    cfg_all = json.loads((LOG_DIR / "experiment_config.json").read_text())
    truths = [tuple(t) for t in cfg_all["truths"]]
    cfg = dict(reps=cfg_all["reps"], nmcmc=cfg_all["nmcmc"], burnin=cfg_all["burnin"],
              eps=cfg_all["eps"], mut_time=cfg_all["mut_time"])
    print(f"config (from experiment_config.json): {cfg}, truths={truths}, "
          f"J={cfg_all['J_grid']}")

    ckpts = {}
    for kind in ("cnn1d", "rnn", "lstm"):
        p = CKPT_DIR / f"{kind}_best.pt"
        if not p.exists():
            raise FileNotFoundError(f"missing checkpoint {p} -- run "
                                    "network/architecture_search/benchmark_families.py first")
        ckpts[kind] = str(p)

    J = cfg_all["J_grid"][0]
    tasks = [(kind, p1, p2, tau, J, r) for kind in ckpts
             for (p1, p2, tau) in truths for r in range(cfg["reps"])]
    workers = max(1, (__import__("os").cpu_count() or 2) - 2)
    print(f"{len(tasks)} tasks on {workers} workers "
          f"({len(ckpts)} families x {len(truths)} truths x {cfg['reps']} reps)")

    rows, t0 = [], time.time()
    with Pool(workers, initializer=_init_worker, initargs=(ckpts, cfg)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one_replicate_family, tasks), 1):
            rows.append(r)
            if i % 24 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  [{i}/{len(tasks)}] elapsed {el/60:.1f}m  "
                      f"eta {el/i*(len(tasks)-i)/60:.1f}m", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(ARCH_FAMILIES_DIR / "raw_replicates_families.csv", index=False)
    tab_new = aggregate_families(raw)

    tab_existing = pd.read_csv(TABLE_DIR / "table1_recovery.csv")  # reused, never recomputed
    tab = pd.concat([tab_existing, tab_new], ignore_index=True, sort=False)
    tab.to_csv(ARCH_FAMILIES_DIR / "table_families_recovery.csv", index=False)

    lines = ["# 3-D two-stage model: parameter recovery -- new architecture families\n",
             f"Config (identical to `results/logs/experiment_config.json`): "
             f"`{json.dumps(cfg)}`, truths=`{truths}`, J={J}.\n",
             "GPS-ABC / DNN-ABC(MLP) rows are copied verbatim from "
             "`results/tables/table1_recovery.csv` (NOT rerun). CNN1D-ABC / RNN-ABC / "
             "LSTM-ABC rows are computed here, through the identical sampler "
             "(`abc_mcmc.run_abc_mcmc`, `backend=\"dnn\"`) with each family's best "
             "checkpoint from `benchmark_families.py`, against the SAME per-replicate "
             "simulated observations (identical seed formula).\n",
             "| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.sort_values(["p1", "p2", "tau", "method", "param"]).iterrows():
        lines.append(f"| ({r['p1']:.0e}, {r['p2']:.0e}, {r['tau']:.0f}) | {int(r['J'])} | "
                     f"{r['method']} | {r['param']} | {r['rmse_log']:.3f} | {r['nrmse']:.3f} | "
                     + (f"{r['ci_len']:.3e} | " if pd.notna(r['ci_len']) else "- | ")
                     + (f"{r['coverage']:.2f} |" if pd.notna(r['coverage']) else "- |"))
    (ARCH_FAMILIES_DIR / "TABLE_FAMILIES.md").write_text("\n".join(lines) + "\n")

    # --- Monte Carlo SEs, same style as abc/mcse.py, new families vs GPS-ABC/DNN-ABC ---
    raw_existing = pd.read_csv(LOG_DIR / "raw_replicates.csv")
    all_methods = ["GPS-ABC", "DNN-ABC", "CNN1D-ABC", "RNN-ABC", "LSTM-ABC"]
    mrows = []
    for (p1, p2, tau, J), sub_new in raw.groupby(["p1_true", "p2_true", "tau_true", "J"]):
        sub_old = raw_existing[(raw_existing.p1_true == p1) & (raw_existing.p2_true == p2) &
                               (raw_existing.tau_true == tau) & (raw_existing.J == J)]
        for m in all_methods:
            for k, tv in (("p1", p1), ("p2", p2), ("tau", tau)):
                if m in ("GPS-ABC", "DNN-ABC"):
                    est = sub_old[f"{m}_{k}"].to_numpy(float)
                    cov = sub_old[f"{m}_{k}_cov"].to_numpy(float)
                else:
                    kind = {v: kk for kk, v in FAMILY_LABELS.items()}[m]
                    s2 = sub_new[sub_new.family == kind]
                    est = s2[k].to_numpy(float)
                    cov = s2[f"{k}_cov"].to_numpy(float)
                err = (np.log10(np.maximum(est, 1e-300)) - np.log10(tv)) if k in ("p1", "p2") else (est - tv)
                mrows.append(dict(p1=p1, p2=p2, tau=tau, J=J, method=m, param=k, n=len(est),
                                  rmse_log=float(np.sqrt(np.nanmean(err ** 2))) if len(err) else np.nan,
                                  rmse_log_mcse=mcse_rmse(err),
                                  coverage=float(np.nanmean(cov)) if len(cov) else np.nan,
                                  coverage_mcse=mcse_prop(float(np.nanmean(cov)), len(cov)) if len(cov) else np.nan))
    mdf = pd.DataFrame(mrows)

    mlines = ["# Monte Carlo standard errors -- new architecture families\n",
              "`rmse_log` with its MCSE, computed identically to `abc/mcse.py`. Two methods "
              "differ meaningfully only when the gap exceeds about 2 combined MCSEs; anything "
              "smaller is simulation noise, not evidence.\n",
              "| truth (p1,p2,tau) | J | param | method | rmse_log ± MCSE | coverage ± MCSE |",
              "|---|---|---|---|---|---|"]
    for _, r in mdf.iterrows():
        mlines.append(f"| ({r.p1:.0e}, {r.p2:.0e}, {r.tau:.0f}) | {int(r.J)} | {r['param']} | "
                      f"{r['method']} | {r.rmse_log:.3f} ± {r.rmse_log_mcse:.3f} | "
                      f"{r.coverage:.2f} ± {r.coverage_mcse:.2f} |")

    mlines.append("\n## Method comparisons (rmse_log): each new family vs GPS-ABC and DNN-ABC(MLP)\n")
    for (p1, p2, tau, J, k), g in mdf.groupby(["p1", "p2", "tau", "J", "param"]):
        g = g.set_index("method")
        for new_m in ("CNN1D-ABC", "RNN-ABC", "LSTM-ABC"):
            if new_m not in g.index:
                continue
            for base_m in ("GPS-ABC", "DNN-ABC"):
                if base_m not in g.index:
                    continue
                d = g.loc[new_m, "rmse_log"] - g.loc[base_m, "rmse_log"]
                se = np.hypot(g.loc[new_m, "rmse_log_mcse"], g.loc[base_m, "rmse_log_mcse"])
                verdict = "TIE" if not np.isfinite(se) or abs(d) < 2 * se else \
                    (f"{new_m} better" if d < 0 else f"{base_m} better")
                mlines.append(f"- ({p1:.0e},{p2:.0e},{tau:.0f}) J={int(J)} `{k}`: "
                              f"{new_m} vs {base_m}: delta = {d:+.3f} ± {se:.3f} -> **{verdict}**")

    (ARCH_FAMILIES_DIR / "mcse_families.md").write_text("\n".join(mlines) + "\n")
    mdf.to_csv(ARCH_FAMILIES_DIR / "mcse_families.csv", index=False)

    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
    print(f"written -> {ARCH_FAMILIES_DIR / 'table_families_recovery.csv'}")
    print(f"written -> {ARCH_FAMILIES_DIR / 'mcse_families.md'}")


if __name__ == "__main__":
    main()
