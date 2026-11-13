"""Problem 4: where does a surrogate likelihood have an edge over NPE?  (4-parameter model.)  The paper states plainly that
NPE is the better DEFAULT for calibration; this script measures the two situations in which the surrogate can still be useful.

(a) Data sets with dbar = 0 (about 30% of the prior predictive). NPE-main is trained on dbar > 0 only and cannot answer them.
    We also train NPE-Z on ALL rows with x = (zero indicator, log10 dbar, log10 s) (0 where undefined), the fair comparator, and score
    NPE-Z and DNN-ABC (whose hurdle gives p(dbar=0 | theta)) by SBC over held-out prior draws that gave dbar = 0.
(b) Changing the prior WITHOUT retraining. New prior = independent normals around a nominal truth (informative). DNN-ABC re-runs its
    chain under the new prior (cost: one chain); NPE is re-weighted by pi_new/pi_old (the old prior is uniform) -- cheap, but the effective
    sample size collapses if the new prior is tight -- or retrained on prior-matched simulations (cost: its training time, reported).
(c) Varying J: not run. The surrogate's inputs are theta only and (dbar, s) have a J=100-specific distribution, so both methods
    would be retrained for another J; stated as a limitation.
    python inference/p4_npe_vs_surrogate.py --data data/slow_data_4D.csv --surrogate results/surrogate_4d.pt --out results/p4.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "model"))
import arch_search as A
import dnn_abc
from surrogate import HurdleSurrogate
from npe import BOX
from p7_extra_features import fit, summarise
from sim import settings as S
from scipy.stats import wasserstein_distance
NAMES = ["log10 p1", "log10 p2", "tau", "log10 delta"]


def zfeat(y, zero):
    return np.column_stack([zero, np.where(zero[:, None] > .5, 0, y)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--surrogate", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=500); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--iters", type=int, default=35000); a = ap.parse_args()
    torch.set_num_threads(a.threads); X, zero, y, des, idx = A.load(a.data); box = BOX[4]; sur = HurdleSurrogate(a.surrogate); res = {}
    tr = np.concatenate([idx["train"], idx["val"]]); F = zfeat(y, zero)
    t0 = time.time(); npez = fit(X[tr], F[tr], box, 0); res["npe_z_train_s"] = time.time() - t0; print("NPE-Z trained", res["npe_z_train_s"], flush=True)
    # (a) dbar = 0 test sets
    rng = np.random.default_rng(0); te0 = idx["test"][zero[idx["test"]] > .5]; te0 = rng.choice(te0, min(a.K, len(te0)), replace=False)
    L = 1000; RN, VN = [], []
    for i in te0:
        s = npez.sample((L,), x=torch.tensor(F[i:i + 1]), show_progress_bars=False).numpy(); RN.append((s < X[i]).sum(0)); VN.append(s.var(0))
    res["a_dbar0_npe_z"] = summarise(np.array(RN), np.array(VN), L, box)
    RD, VD = [], []
    for k, i in enumerate(te0):
        ch, _ = dnn_abc.run_chain(sur, np.zeros(100), 1e-9, a.iters, np.random.default_rng(100 + k), step_scale=0.05)
        d = ch[15000::20]; RD.append((d < X[i]).sum(0)); VD.append(d.var(0))
    res["a_dbar0_dnn_abc"] = summarise(np.array(RD), np.array(VD), len(d), box); res["a_n"] = int(len(te0))
    print(json.dumps({k: res[k] for k in ("a_dbar0_npe_z", "a_dbar0_dnn_abc")}, indent=1), flush=True)
    # (b) prior change on the case-1 observed data sets 0..9
    O = dict(np.load(HERE.parent / "data" / "obs_4d_case1.npz")); mu_new = np.array([-9.0, -8.0, 14.0, np.log10(0.8)]); sd_new = np.array([0.5, 0.5, 1.5, 0.1])
    lo, hi = np.array(box).T; npe_main = __import__("pickle").load(open(HERE.parent / "results" / "npe_4d_case1.pkl", "rb"))
    log_pi = lambda t: -0.5 * (((t - mu_new) / sd_new) ** 2).sum(1)
    out = []
    for i in range(10):
        ox = O["obs"][i]; f = np.array([[np.log10(ox.mean()), np.log10(ox.std(ddof=1))]], np.float32)
        t0 = time.time(); ch, _ = dnn_abc.run_chain(sur, ox, float(O["mom"][i]), a.iters, np.random.default_rng(7 + i), sigma=sd_new, step_scale=0.05, prior_mu=mu_new); t_dnn = time.time() - t0
        dnn_new = ch[15000:]; dnn_old = np.load(HERE.parent / "results" / "dnnabc_4d_case1.npz")["samples"][i]
        t0 = time.time(); s = npe_main.sample((20000,), x=torch.tensor(f), show_progress_bars=False).numpy(); w = np.exp(log_pi(s) - log_pi(s).max()); w /= w.sum(); t_rw = time.time() - t0
        ess = 1 / (w ** 2).sum(); rs = s[np.random.default_rng(i).choice(len(s), 5000, p=w)]
        out.append(dict(idx=i, dnn_abc_rerun_s=t_dnn, npe_reweight_s=t_rw, npe_ess=float(ess), ess_frac=float(ess / len(s)),
                        w1_dnn_vs_npe_reweighted=[float(wasserstein_distance(dnn_new[:, k], rs[:, k])) for k in range(4)],
                        prior_shift_w1_dnn=[float(wasserstein_distance(dnn_old[:, k], dnn_new[:, k])) for k in range(4)]))
    res["b_prior_change"] = out; res["b_prior"] = dict(mu=mu_new.tolist(), sd=sd_new.tolist()); res["b_npe_retrain_s_reference"] = "see npe_4d_case1.npz train_s"
    res["b_npe_retrain_train_s"] = float(np.load(HERE.parent / "results" / "npe_4d_case1.npz")["train_s"])
    json.dump(res, open(a.out, "w"), indent=1); print("done")


if __name__ == "__main__":
    main()
