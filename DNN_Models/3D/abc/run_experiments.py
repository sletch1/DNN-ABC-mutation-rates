"""Reproduce the result tables for the 3-D two-stage model, with a DNN-ABC column.

DESIGN. For each true parameter triple (p1, p2, tau) and each replicate:
  1. simulate an observed fluctuation experiment with the exact two-stage
     simulator (J cultures) and reduce it to obs = mean_i sqrt(X_i / Z_i);
  2. estimate the parameters with each method:
       MOM / MLE  - constant-rate baselines. They cannot identify (p1, p2, tau)
                    individually; they are scored against p_eff, the time-average
                    rate they actually estimate (see estimators.py).
       ABC-MCMC   - exact simulator inside the MCMC loop (the expensive truth).
       GPS-ABC    - GP surrogate capped at a small space-filling budget.
       DNN-ABC    - this project's heteroscedastic MLP, trained on all rows.
  3. aggregate over replicates.

WHAT TO EXPECT, and why the tables are shaped this way. On this model a single
scalar summary carries very uneven information about the three parameters: on
the ground truth, corr(log d_bar, log p2) = 0.74 but only 0.40 for p1 and 0.14
for tau. So p2 should be recovered well and p1/tau poorly, with wide and
possibly multimodal marginals. That is not a bug in the sampler -- the paper
reports the same behaviour and names it as the model's known weakness -- so the
tables report per-parameter nRMSE and interval width separately rather than a
single aggregate score that would hide it. `ess` is reported alongside
acceptance because a chain can accept healthily and still barely move in the
weakly identified directions.

Usage:
    python run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --ns 4 --workers 8
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network", _ROOT / "network" / "architecture_search",
           _ROOT / "abc", _ROOT / "figures"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd

from simulator import fluc_exp_2stage, summary_stat
from estimators import estimate_mom, estimate_mle, p_eff
from abc_mcmc import run_abc_mcmc, summarize, ess, DEFAULT_BOX, DEFAULT_STEPS
from surrogates import fit_gp_surrogate_3d
from train import load_surrogate, run as train_run, TEST_REPS
from paths import DATA, RESULTS, TABLE_DIR, MODEL_DIR, LOG_DIR

# Truth triples: a low/high p2 pair crossed with an early/late switch, chosen
# inside the ground-truth design box so no surrogate is queried out of range.
TRUTHS = [
    (1e-4, 1e-2, 3.0),   # big jump, early switch
    (1e-4, 1e-2, 7.0),   # big jump, late switch
    (2e-3, 8e-3, 5.0),   # small jump, mid switch
]
GP_BUDGET = 300
A, TP, Z0 = 1.0, 10.0, 1
_G = {}


def _init_worker(ckpt, data_path, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    dnn = load_surrogate(ckpt)
    df = pd.read_csv(data_path)
    tr = df[~df["rep"].isin(TEST_REPS)]
    X = np.column_stack([np.log10(tr.p1), np.log10(tr.p2), tr.tau])
    gp = fit_gp_surrogate_3d(X, np.log10(tr.d_bar.to_numpy()), budget=cfg["gp_budget"])
    _G.update(dnn=dnn, gp=gp, cfg=cfg)


def _one_replicate(task):
    p1, p2, tau, J, rep = task
    cfg = _G["cfg"]
    seed = abs(hash((round(p1, 12), round(p2, 12), round(tau, 3), J, rep))) % (2 ** 31)
    rng = np.random.default_rng(seed)

    Zv, Xv = fluc_exp_2stage(Z0, A, p1, p2, tau, TP, J, rng, use_slow=True,
                             mut_time=cfg["mut_time"])
    obs = summary_stat(Zv, Xv)
    truth = dict(p1=p1, p2=p2, tau=tau, p_eff=float(p_eff(p1, p2, tau, TP)))

    out = {"p1_true": p1, "p2_true": p2, "tau_true": tau, "J": J, "rep": rep,
           "obs": obs, "p_eff_true": truth["p_eff"],
           "MOM": estimate_mom(Zv, Xv), "MLE": estimate_mle(Zv, Xv)}

    sim_kwargs = dict(Z0=Z0, a=A, tp=TP, J=J, use_slow=True, mut_time=cfg["mut_time"])
    backends = [("GPS-ABC", dict(backend="gp", surrogate=_G["gp"])),
                ("DNN-ABC", dict(backend="dnn", surrogate=_G["dnn"]))]
    if cfg["with_sim"]:
        backends.insert(0, ("ABC-MCMC", dict(backend="sim", sim_kwargs=sim_kwargs,
                                             ns=cfg["ns"])))
    for name, kw in backends:
        t0 = time.time()
        s, acc = run_abc_mcmc(obs, n_mcmc=cfg["nmcmc"], steps=DEFAULT_STEPS,
                              box=DEFAULT_BOX, eps=cfg["eps"],
                              rng=np.random.default_rng(rng.integers(2 ** 63 - 1)),
                              **kw)
        post = summarize(s, cfg["burnin"])
        out[f"{name}_secs"] = time.time() - t0
        out[f"{name}_acc"] = acc
        for k in ("p1", "p2", "tau"):
            out[f"{name}_{k}"] = post[k]["mean"]
            out[f"{name}_{k}_cilen"] = post[k]["ci_len"]
            out[f"{name}_{k}_cov"] = int(post[k]["ci_lo"] <= truth[k] <= post[k]["ci_hi"])
        out[f"{name}_ess_p2"] = ess(s[cfg["burnin"]:, 1])
        out[f"{name}_ess_tau"] = ess(s[cfg["burnin"]:, 2])
    return out


def aggregate(df, cfg):
    """Per-parameter nRMSE, mean CI width and empirical coverage, per method."""
    methods = (["ABC-MCMC"] if cfg["with_sim"] else []) + ["GPS-ABC", "DNN-ABC"]
    rows = []
    for (p1, p2, tau) in cfg["truths"]:
        for J in cfg["J_grid"]:
            sub = df[(df.p1_true == p1) & (df.p2_true == p2) &
                     (df.tau_true == tau) & (df.J == J)]
            if not len(sub):
                continue
            base = {"p1": p1, "p2": p2, "tau": tau, "J": J}
            for m in methods:
                for k, tv in (("p1", p1), ("p2", p2), ("tau", tau)):
                    est = sub[f"{m}_{k}"].to_numpy(float)
                    est = est[np.isfinite(est)]
                    # For a weakly identified parameter the posterior mean sits
                    # wherever the prior puts its mass, so natural-scale nRMSE
                    # explodes and conveys nothing. rmse_log -- RMSE in log10
                    # units for p1/p2 -- stays interpretable: 1.0 means "off by
                    # an order of magnitude on average".
                    if k in ("p1", "p2"):
                        rmse_log = float(np.sqrt(np.mean(
                            (np.log10(np.maximum(est, 1e-300)) - np.log10(tv)) ** 2))) if len(est) else np.nan
                    else:
                        rmse_log = float(np.sqrt(np.mean((est - tv) ** 2))) if len(est) else np.nan
                    rows.append({**base, "method": m, "param": k, "rmse_log": rmse_log,
                                 "nrmse": float(np.sqrt(np.mean((est - tv) ** 2)) / tv) if len(est) else np.nan,
                                 "ci_len": float(sub[f"{m}_{k}_cilen"].mean()),
                                 "coverage": float(sub[f"{m}_{k}_cov"].mean()),
                                 "secs": float(sub[f"{m}_secs"].mean()),
                                 "acc": float(sub[f"{m}_acc"].mean())})
            # constant-rate baselines, scored against what they actually estimate
            for m in ("MOM", "MLE"):
                est = sub[m].to_numpy(float); est = est[np.isfinite(est)]
                tv = float(sub.p_eff_true.iloc[0])
                rows.append({**base, "method": m, "param": "p_eff",
                             "rmse_log": float(np.sqrt(np.mean(
                                 (np.log10(np.maximum(est, 1e-300)) - np.log10(tv)) ** 2))) if len(est) else np.nan,
                             "nrmse": float(np.sqrt(np.mean((est - tv) ** 2)) / tv) if len(est) else np.nan,
                             "ci_len": np.nan, "coverage": np.nan, "secs": np.nan, "acc": np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--nmcmc", type=int, default=3000)
    ap.add_argument("--burnin", type=int, default=1000)
    ap.add_argument("--ns", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--gp-budget", type=int, default=GP_BUDGET)
    ap.add_argument("--J-grid", type=int, nargs="+", default=[100])
    ap.add_argument("--mut-time", default="parent", choices=["parent", "offspring"])
    ap.add_argument("--workers", type=int,
                    default=max(1, (__import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--no-sim", action="store_true",
                    help="skip the exact ABC-MCMC baseline (much faster)")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    ckpt = MODEL_DIR / "surrogate_3d.pt"
    if args.retrain or not ckpt.exists():
        print("training the DNN surrogate first...")
        train_run(str(DATA))

    cfg = dict(reps=args.reps, nmcmc=args.nmcmc, burnin=args.burnin, ns=args.ns,
               eps=args.eps, gp_budget=args.gp_budget, J_grid=list(args.J_grid),
               truths=TRUTHS, mut_time=args.mut_time, with_sim=not args.no_sim,
               workers=args.workers)
    (LOG_DIR / "experiment_config.json").write_text(json.dumps(cfg, indent=2))

    tasks = [(p1, p2, tau, J, r) for (p1, p2, tau) in TRUTHS
             for J in cfg["J_grid"] for r in range(args.reps)]
    print(f"{len(tasks)} tasks on {args.workers} workers "
          f"({'with' if cfg['with_sim'] else 'without'} the exact ABC-MCMC baseline)")

    rows, t0 = [], time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(str(ckpt), str(DATA), cfg)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one_replicate, tasks), 1):
            rows.append(r)
            if i % 5 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  [{i}/{len(tasks)}] elapsed {el/60:.1f}m  "
                      f"eta {el/i*(len(tasks)-i)/60:.1f}m", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(LOG_DIR / "raw_replicates.csv", index=False)
    tab = aggregate(raw, cfg)
    tab.to_csv(TABLE_DIR / "table1_recovery.csv", index=False)

    lines = ["# 3-D two-stage model: parameter recovery\n",
             f"Config: `{json.dumps(cfg)}`\n",
             "`rmse_log` is RMSE in log10 units for p1/p2 (so 1.0 = off by an order of "
             "magnitude on average) and in absolute time units for tau. **Prefer it to "
             "`nrmse`**: where a parameter is weakly identified the posterior mean sits "
             "wherever the prior puts its mass, and natural-scale nRMSE then explodes "
             "without conveying anything. MOM/MLE are constant-rate baselines scored "
             "against `p_eff`, the time-average rate they actually estimate -- they cannot "
             "identify p1, p2 or tau individually.\n",
             "| truth (p1, p2, tau) | J | method | param | rmse_log | nRMSE | mean 95% CI width | coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| ({r['p1']:.0e}, {r['p2']:.0e}, {r['tau']:.0f}) | {int(r['J'])} | "
                     f"{r['method']} | {r['param']} | {r['rmse_log']:.3f} | {r['nrmse']:.3f} | "
                     + (f"{r['ci_len']:.3e} | " if np.isfinite(r['ci_len']) else "- | ")
                     + (f"{r['coverage']:.2f} |" if np.isfinite(r['coverage']) else "- |"))
    (TABLE_DIR / "TABLES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"\ntotal {(time.time()-t0)/60:.1f} min; results in {RESULTS}")


if __name__ == "__main__":
    main()
