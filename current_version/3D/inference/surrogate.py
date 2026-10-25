"""Load a trained hurdle surrogate (model/select_and_report.py output) and expose the ABC likelihood.

theta coordinates (paper order): 3-D (log10 p1, log10 p2, tau); 4-D adds log10 delta.
Features of an observed dataset: d_bar = mean(obs_X) and s = sd(obs_X) (ddof=1) -- the same
Feat = [mean(obs_X), std(obs_X)] the GPS-ABC sampler uses.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from nets import HurdleNet


class HurdleSurrogate:
    def __init__(self, path):
        c = torch.load(path, weights_only=False)
        self.net = HurdleNet(in_dim=c["in_dim"], **c["arch"]); self.net.load_state_dict(c["state"]); self.net.eval()
        self.sc, self.tconf, self.box, self.dim = c["scalers"], np.asarray(c["conformal_scale"]), np.asarray(c["box"], float), c["in_dim"]
        self.arch_name = c["arch_name"]

    @torch.no_grad()
    def predict(self, theta):
        x = torch.tensor((np.atleast_2d(theta) - self.sc["xm"]) / self.sc["xs"], dtype=torch.float32)
        lg, mu, lv = self.net(x)
        p0 = torch.sigmoid(lg).numpy(); mu = mu.numpy() * self.sc["ys"] + self.sc["ym"]
        sd = np.exp(0.5 * lv.numpy()) * self.sc["ys"] * self.tconf          # split-conformal scaled predictive sd
        return p0, mu, sd

    def loglik(self, theta, obs_feat, eps=0.0):
        """log p(features | theta). obs_feat = (d_bar, s) on the natural scale; d_bar == 0 -> hurdle atom."""
        p0, mu, sd = self.predict(theta)
        p0 = float(np.clip(p0[0], 1e-12, 1 - 1e-12))
        if obs_feat[0] <= 0:
            return np.log(p0)
        z = np.log10(np.maximum(np.asarray(obs_feat, float), 1e-300)); s2 = sd[0] ** 2 + eps ** 2
        return np.log1p(-p0) + float(np.sum(-0.5 * np.log(2 * np.pi * s2) - 0.5 * (z - mu[0]) ** 2 / s2))
