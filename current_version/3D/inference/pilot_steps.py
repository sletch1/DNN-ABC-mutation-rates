"""Choose the random-walk step widths by the paper's own criterion: acceptance between 20% and 40%.
The paper reports step widths s = (0.8, 3.5, 3.5, 4) without naming their order and with a GP surrogate whose predictive
variance is far wider than the network's; with the network's sharper likelihood they give a few percent acceptance, so a
single common multiplier on the paper widths is tuned on a pilot (first 6 observed datasets of both cases, 6,000 iterations).
The multiplier closest to 30% acceptance among those in [0.2, 0.4] is chosen (ties -> larger min ESS). Writes results/steps_<model>.json.
    python inference/pilot_steps.py --surrogate results/surrogate_4d.pt --obs data/obs_4d_case1.npz,data/obs_4d_case2.npz --out results/steps_4d.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_abc
from surrogate import HurdleSurrogate
from sbc import ranks  # noqa (keeps module importable)


def ess(x):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if np.allclose(x, 0): return 0.0
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n].real; acf /= acf[0]
    s, k = 0.0, 1
    while k + 1 < n:
        pr = acf[k] + acf[k + 1]
        if pr <= 0: break
        s += pr; k += 2
    return n / (1 + 2 * s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--surrogate", required=True); ap.add_argument("--obs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); sur = HurdleSurrogate(a.surrogate); rows = []
    for sc in (0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0):
        acc, es = [], []
        for f in a.obs.split(","):
            O = np.load(f)
            for i in range(6):
                ch, ac = dnn_abc.run_chain(sur, O["obs"][i], float(O["mom"][i]), 6000, np.random.default_rng(i), step_scale=sc)
                acc.append(ac); es.append(min(ess(ch[1500:, k]) for k in range(ch.shape[1])))
        rows.append(dict(scale=sc, acc=float(np.mean(acc)), min_ess=float(np.mean(es)))); print(rows[-1], flush=True)
    ok = [r for r in rows if .2 <= r["acc"] <= .4]
    best = min(ok, key=lambda r: (abs(r["acc"] - .3), -r["min_ess"])) if ok else max(rows, key=lambda r: r["min_ess"])
    json.dump(dict(chosen=best, grid=rows, paper_widths=dnn_abc.STEPS.tolist(), within_target=bool(ok)), open(a.out, "w"), indent=1); print("chosen", best)


if __name__ == "__main__":
    main()
