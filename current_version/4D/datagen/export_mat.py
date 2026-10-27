"""Export observed datasets from gen_observed.py (.npz) to per-replicate .mat files, the format the
authors' MATLAB scripts load (obs_X, MOM, Nt, Xt).   python datagen/export_mat.py data/obs_4d_case1.npz data/mat/obs_4d_case1"""
import sys
from pathlib import Path
import numpy as np, scipy.io as sio
src, stem = sys.argv[1], Path(sys.argv[2]); stem.parent.mkdir(parents=True, exist_ok=True)
D = np.load(src)
for i in range(len(D["obs"])):
    sio.savemat(f"{stem}_{i+1}.mat", dict(obs_X=D["obs"][i].reshape(-1, 1), MOM=float(D["mom"][i]),
                                         Nt=D["Nt"][i].reshape(-1, 1), Xt=D["Xt"][i].reshape(-1, 1)))
print("wrote", len(D["obs"]), "files")
