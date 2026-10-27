"""Architecture search for the 3-D hurdle surrogate. Run on the server:
    python model/arch_search.py --data data/slow_data_3D.csv --seeds 5 --out results/arch_search
Protocol (adapted from the previous version's benchmark_round2/benchmark_arch):
  * split BY DESIGN POINT (80/10/10, fixed split seed): ABC queries unseen theta, so the test set is
    unseen theta, not unseen replicates of seen theta. Model seed varies init/minibatching only.
  * select on VALIDATION; test is reported for every model but never used to choose.
  * noise floor estimated from within-design replicate scatter (nonzero rows), so accuracy is
    reported as excess MSE over the irreducible floor.
  * metrics: P(0) log-loss/Brier/AUC; nonzero part RMSE, MSE excess over floor, NLL, 95% coverage
    before/after split-conformal scaling, interval width; parameters, train time, latency at batch 1
    and throughput at batch 1e5.  GP baseline (GPS-ABC style, 300 points) on the nonzero part.
Cores are used dynamically: tasks go to a pool sized to the free cores, longest first.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nets import HurdleNet, hurdle_loss

FLOOR_LOG = -6.0
ARCHS = {   # name: kwargs
    "mlp_16-16_tanh":       dict(kind="mlp", hidden=(16, 16), activation="tanh"),
    "mlp_32-16_gelu":       dict(kind="mlp", hidden=(32, 16), activation="gelu"),
    "mlp_64-32_gelu-tanh":  dict(kind="mlp", hidden=(64, 32), activation=["gelu", "tanh"]),   # previous version's choice
    "mlp_128-64_gelu":      dict(kind="mlp", hidden=(128, 64), activation="gelu"),
    "mlp_128-64_tanh":      dict(kind="mlp", hidden=(128, 64), activation="tanh"),
    "mlp_128-64_silu":      dict(kind="mlp", hidden=(128, 64), activation="silu"),
    "mlp_128-64_relu":      dict(kind="mlp", hidden=(128, 64), activation="relu"),
    "mlp_256-128-64_gelu":  dict(kind="mlp", hidden=(256, 128, 64), activation="gelu"),
    "mlp_128x3_silu":       dict(kind="mlp", hidden=(128, 128, 128), activation="silu"),
    "mlp_64x4_gelu":        dict(kind="mlp", hidden=(64, 64, 64, 64), activation="gelu"),
    "res_128x3_gelu":       dict(kind="res", hidden=(128, 3), activation="gelu"),
    "res_256x4_gelu":       dict(kind="res", hidden=(256, 4), activation="gelu"),
}


def load(path, split_seed=123):
    df = pd.read_csv(path)
    cols = [np.log10(df.p1), np.log10(df.p2), df.tau] + ([np.log10(df.delta)] if df.delta.nunique() > 1 else [])
    X = np.column_stack(cols).astype(np.float32)               # 3-D: (log10 p1, log10 p2, tau); 4-D adds log10 delta
    D = df[[c for c in df.columns if c.startswith("d_") and c != "d_bar"]].to_numpy(np.float64)
    db, sdr = D.mean(1), D.std(1, ddof=1)                        # the two ABC features: mean and sd over the J cultures
    zero = (db == 0).astype(np.float32)
    y = np.column_stack([np.where(db > 0, np.log10(np.maximum(db, 1e-300)), 0.0),
                         np.where(sdr > 0, np.log10(np.maximum(sdr, 1e-300)), 0.0)]).astype(np.float32)
    des = df.design.to_numpy()
    rng = np.random.default_rng(split_seed)
    u = np.unique(des); rng.shuffle(u)
    n = len(u); sp = {"train": u[:int(.8 * n)], "val": u[int(.8 * n):int(.9 * n)], "test": u[int(.9 * n):]}
    idx = {k: np.where(np.isin(des, v))[0] for k, v in sp.items()}
    return X, zero, y, des, idx


def noise_floor(y, zero, des, idx):
    """mean within-design variance of log10 d_bar over designs with >=2 nonzero rows (per split)."""
    out = {}
    for k, ii in idx.items():
        o = []
        for f in range(y.shape[1]):
            d = pd.DataFrame(dict(des=des[ii], y=y[ii, f], nz=zero[ii] < .5))
            d = d[d.nz]; g = d.groupby("des").y
            o.append(float(g.var(ddof=1)[g.count() >= 2].mean()))
        out[k] = o
    return out


def _t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def _unpack(out):
    return (out[0], out[1], out[2], out[3] if len(out) > 3 else None)


def fit(arch, X, zero, y, idx, seed, epochs=300, patience=30, bs=512, warmup=20):
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    tr, va = idx["train"], idx["val"]
    xm, xs = X[tr].mean(0), X[tr].std(0)
    nz = zero[tr] < .5; ym, ys = y[tr][nz].mean(0), y[tr][nz].std(0)
    f = lambda ii: (_t((X[ii] - xm) / xs), _t(zero[ii]), _t((y[ii] - ym) / ys))
    xt, zt, yt = f(tr); xv, zv, yv = f(va)
    net = HurdleNet(in_dim=X.shape[1], **arch)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    best, bst, since = 1e9, None, 0; t0 = time.time(); n = len(xt)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad()
            lg, mu, lv, rho = _unpack(net(xt[b]))
            if ep < warmup:   # MSE warm-up for the mean head so the variance head cannot mask a bad fit
                nzb = zt[b] < .5
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, zt[b]) + ((mu[nzb] - yt[b][nzb]) ** 2).sum(1).mean()
            else:
                loss = hurdle_loss(lg, mu, lv, zt[b], yt[b], rho)[0]
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lgv, muv, lvv, rhov = _unpack(net(xv)); vl = hurdle_loss(lgv, muv, lvv, zv, yv, rhov)[0].item()
        sch.step(vl)
        if ep >= warmup:
            if vl < best - 1e-5: best, bst, since = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                since += 1
                if since >= patience: break
    net.load_state_dict(bst); net.eval()
    return net, dict(xm=xm, xs=xs, ym=ym, ys=ys), time.time() - t0, ep + 1, best


@torch.no_grad()
def predict(net, sc, X):
    lg, mu, lv, rho = _unpack(net(_t((X - sc["xm"]) / sc["xs"])))
    out = (torch.sigmoid(lg).numpy(), mu.numpy() * sc["ys"] + sc["ym"], np.exp(0.5 * lv.numpy()) * sc["ys"])   # (n,), (n,2), (n,2)
    return out + (rho.numpy(),) if rho is not None else out


def evaluate(net, sc, X, zero, y, des, idx, floor, tconf=None):
    from sklearn.metrics import roc_auc_score
    r = {}; nf = y.shape[1]
    for split in ("val", "test"):
        ii = idx[split]; out_ = predict(net, sc, X[ii]); p0, mu, sd = out_[:3]; rho_ = out_[3] if len(out_) > 3 else np.zeros(len(ii)); z = zero[ii]; yy = y[ii]; nz = z < .5
        if split == "val":   # split-conformal scale, per feature, for the 95% interval on nonzero rows
            tconf = [float(np.quantile(np.abs(yy[nz, f] - mu[nz, f]) / sd[nz, f], .95) / 1.959964) for f in range(nf)]
        e = np.clip(p0, 1e-7, 1 - 1e-7)
        r[f"{split}_ll0"] = float(-np.mean(z * np.log(e) + (1 - z) * np.log(1 - e)))
        r[f"{split}_brier0"] = float(np.mean((p0 - z) ** 2))
        r[f"{split}_auc0"] = float(roc_auc_score(z, p0))
        xf, cv, cr, wd, nl, rm = [], [], [], [], [], []
        for f in range(nf):
            res = yy[nz, f] - mu[nz, f]; s_ = sd[nz, f]; t = tconf[f]
            mse = float(np.mean(res ** 2)); xf.append(mse / floor[split][f]); rm.append(mse ** .5)
            nl.append(float(np.mean(0.5 * np.log(2 * np.pi * s_ ** 2) + 0.5 * (res / s_) ** 2)))
            cr.append(float(np.mean(np.abs(res) <= 1.959964 * s_))); cv.append(float(np.mean(np.abs(res) <= 1.959964 * s_ * t)))
            wd.append(float(np.mean(2 * 1.959964 * s_ * t)))
            for nm, v in (("x_floor", xf[-1]), ("rmse", rm[-1]), ("nll", nl[-1]), ("cov95_raw", cr[-1]), ("cov95_conf", cv[-1]), ("width95", wd[-1])):
                r[f"{split}_f{f+1}_{nm}"] = v
        zz = (yy[nz] - mu[nz]) / sd[nz]; rr = rho_[nz]      # joint (bivariate) negative log-likelihood of the two features, at the raw predictive sds
        r[f"{split}_joint_nll"] = float(np.mean(np.log(2 * np.pi) + np.log(sd[nz]).sum(1) + 0.5 * np.log1p(-rr ** 2) + 0.5 * (zz[:, 0] ** 2 - 2 * rr * zz[:, 0] * zz[:, 1] + zz[:, 1] ** 2) / (1 - rr ** 2)))
        r[f"{split}_x_floor"] = float(np.mean(xf)); r[f"{split}_rmse"] = float(np.mean(rm)); r[f"{split}_nll"] = float(np.sum(nl))
        r[f"{split}_cov95_raw"] = float(np.mean(cr)); r[f"{split}_cov95_conf"] = float(np.mean(cv)); r[f"{split}_width95"] = float(np.mean(wd))
    r["conformal_scale_f1"], r["conformal_scale_f2"] = tconf
    return r


def speed(net, sc, X):
    x1 = X[:1]; net.eval()
    for _ in range(50): predict(net, sc, x1)
    t = []
    for _ in range(1000):
        t0 = time.perf_counter(); predict(net, sc, x1); t.append(time.perf_counter() - t0)
    big = np.tile(X[:1000], (100, 1)); t0 = time.perf_counter(); predict(net, sc, big)
    return float(np.median(t) * 1e6), float((time.perf_counter() - t0) * 1e3)


G = {}
def _init(path):
    torch.set_num_threads(1); G["d"] = load(path); G["floor"] = noise_floor(G["d"][2], G["d"][1], G["d"][3], G["d"][4])


def task(a):
    name, seed = a
    X, zero, y, des, idx = G["d"]
    net, sc, secs, eps, vbest = fit(ARCHS[name], X, zero, y, idx, seed)
    r = evaluate(net, sc, X, zero, y, des, idx, G["floor"])
    lat, thr = speed(net, sc, X[idx["test"]])
    r.update(arch=name, seed=seed, params=sum(p.numel() for p in net.parameters()), train_s=secs, epochs=eps,
             val_loss=vbest, latency_us=lat, batch1e5_ms=thr)
    return r


def gp_baseline(X, zero, y, idx, floor, budget=300, seed=0):
    """GPS-ABC-style baseline on the nonzero part only (a GP cannot represent the zero atom):
    anisotropic RBF + white noise on `budget` space-filling training rows."""
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    y = y[:, 0]; floor = {k: v[0] for k, v in floor.items()}     # feature 1 (log10 d_bar) only
    tr = idx["train"]; tr = tr[zero[tr] < .5]
    rng = np.random.default_rng(seed); sel = rng.choice(tr, budget, replace=False)
    xm, xs = X[sel].mean(0), X[sel].std(0); ym, ys = y[sel].mean(), y[sel].std()
    t0 = time.time()
    gp = GPR(C(1.0) * RBF([1.0] * X.shape[1]) + WhiteKernel(0.1), normalize_y=False, n_restarts_optimizer=2, random_state=seed)
    gp.fit((X[sel] - xm) / xs, (y[sel] - ym) / ys); secs = time.time() - t0
    ii = idx["test"]; ii = ii[zero[ii] < .5]
    t0 = time.perf_counter(); mu, sd = gp.predict((X[ii[:1]] - xm) / xs, return_std=True); lat = time.perf_counter() - t0
    mu, sd = gp.predict((X[ii] - xm) / xs, return_std=True); mu, sd = mu * ys + ym, sd * ys
    mse = float(np.mean((y[ii] - mu) ** 2))
    return dict(arch=f"GP_{budget}pts(nonzero only)", seed=seed, params=np.nan, train_s=secs, test_mse=mse,
                test_rmse=mse ** .5, test_x_floor=mse / floor["test"], test_excess_mse=mse - floor["test"],
                test_nll=float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * ((y[ii] - mu) / sd) ** 2)),
                test_cov95_raw=float(np.mean(np.abs(y[ii] - mu) <= 1.959964 * sd)), latency_us=lat * 1e6)


def free_cores():
    n = os.cpu_count() or 2
    try: load = round(os.getloadavg()[0])
    except OSError: load = 0
    return max(2, n - load)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/arch_search"); ap.add_argument("--archs", default="all")
    ap.add_argument("--max-cores", type=int, default=None)
    a = ap.parse_args()
    names = list(ARCHS) if a.archs == "all" else a.archs.split(",")
    tasks = [(n, s) for s in range(a.seeds) for n in names]
    cost = lambda n: sum(ARCHS[n]["hidden"]) * (10 if ARCHS[n]["kind"] == "res" else 1)
    tasks.sort(key=lambda t: -cost(t[0]))      # longest first -> good dynamic balance
    C = free_cores()
    if a.max_cores: C = min(C, a.max_cores)
    print(f"{len(tasks)} fits on {C} cores", flush=True); t0 = time.time()
    rows = []
    with Pool(C, initializer=_init, initargs=(a.data,)) as pool:
        for r in pool.imap_unordered(task, tasks):
            rows.append(r); print(f"[{len(rows)}/{len(tasks)}] {r['arch']} s{r['seed']} test x_floor {r['test_x_floor']:.2f} cov {r['test_cov95_conf']:.3f}", flush=True)
    X, zero, y, des, idx = load(a.data); floor = noise_floor(y, zero, des, idx)
    rows.append(gp_baseline(X, zero, y, idx, floor, 300))
    df = pd.DataFrame(rows); df.to_csv(a.out + "_raw.csv", index=False)
    summ = {"floor": floor, "n_rows": int(len(X)), "split_sizes": {k: int(len(v)) for k, v in idx.items()},
            "frac_zero": float(zero.mean()), "wall_s": time.time() - t0, "cores": C}
    json.dump(summ, open(a.out + "_meta.json", "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
