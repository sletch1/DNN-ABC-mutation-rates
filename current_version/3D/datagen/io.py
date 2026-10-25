"""Load a training CSV (layout of slow_data_*.csv) -> theta in the sampler's coordinates
(1d: log10 p; 3d: log10 p1, log10 p2, tau; 4d: log10 p1, log10 p2, tau, log10 delta  -- paper order) and the
pooled single-culture statistics per design point."""
import numpy as np, pandas as pd


def load(path):
    df = pd.read_csv(path)
    dcols = [c for c in df.columns if c.startswith("d_") and c != "d_bar"]
    if "design" not in df:                      # 1d: group by p
        df["design"] = df.groupby("p").ngroup()
    cols = ["p"] if "p" in df else (["p1", "p2", "tau"] if df.delta.nunique() == 1 else ["p1", "p2", "tau", "delta"])
    g = df.groupby("design", sort=True)
    th = g[cols].first().to_numpy(float)
    for j, c in enumerate(cols):
        if c != "tau":
            th[:, j] = np.log10(th[:, j])
    Y = np.array([grp[dcols].to_numpy().ravel() for _, grp in g])   # (n_design, rep*J) pooled single-culture draws
    return th, Y, float(df.tp.iloc[0])
