"""Render the surrogate architecture as results/figures/architecture.svg.

A hand-rolled SVG rather than a plotting library: the diagram is a fixed set of
labelled boxes and arrows, so emitting the markup directly keeps it crisp at any
zoom, theme-neutral, and free of a rendering dependency.

The diagram is generated FROM the live config in train.py (ARCH, USE_DERIVED),
so it cannot drift out of sync with the model that is actually trained.

Usage:
    python gen_architecture_svg.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from train import ARCH, USE_DERIVED
from model import FEATURES_ALL, FEATURES_RAW
from paths import FIG_DIR

W, H = 980, 300
BOX_W, BOX_H, GAP = 132, 58, 46
FILL_IN, FILL_HID, FILL_OUT = "#e8f0fe", "#e6f4ea", "#fce8e6"
STROKE = "#5f6368"


def box(x, y, w, h, fill, title, sub=""):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
         f'stroke="{STROKE}" stroke-width="1.5"/>'
         f'<text x="{x+w/2}" y="{y+h/2-(4 if sub else -5)}" text-anchor="middle" '
         f'font-family="Helvetica,Arial,sans-serif" font-size="13" fill="#202124">{title}</text>')
    if sub:
        s += (f'<text x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" '
              f'font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#5f6368">{sub}</text>')
    return s


def arrow(x1, y, x2):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2-7}" y2="{y}" stroke="{STROKE}" '
            f'stroke-width="1.5" marker-end="url(#a)"/>')


def main():
    feats = FEATURES_ALL if USE_DERIVED else FEATURES_RAW
    hidden = list(ARCH.get("hidden", ()))
    act = ARCH.get("activation", "gelu").upper()

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}">',
             '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" '
             f'orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="{STROKE}"/></marker></defs>',
             f'<text x="{W/2}" y="26" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" '
             f'font-size="15" fill="#202124">3-D two-stage surrogate: '
             f'(log10 p1, log10 p2, tau) -&gt; log10(d_bar) with predictive variance</text>']

    y = 110
    x = 24
    parts.append(box(x, y, BOX_W, BOX_H, FILL_IN, "inputs",
                     f"{len(feats)}: p1, p2, tau" + (" + p_eff" if USE_DERIVED else "")))
    if USE_DERIVED:
        parts.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H+22}" text-anchor="middle" '
                     f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#5f6368">'
                     f'derived: time-average rate</text>')
    x += BOX_W
    for h in hidden:
        parts.append(arrow(x, y + BOX_H / 2, x + GAP))
        x += GAP
        parts.append(box(x, y, BOX_W, BOX_H, FILL_HID, f"Dense {h}", act))
        x += BOX_W

    parts.append(arrow(x, y + BOX_H / 2, x + GAP))
    xh = x + GAP
    parts.append(box(xh, y - 42, BOX_W, BOX_H, FILL_OUT, "mean", "log10(d_bar)"))
    parts.append(box(xh, y + 42, BOX_W, BOX_H, FILL_OUT, "log variance", "soft-clamped"))
    parts.append(f'<text x="{xh+BOX_W/2}" y="{y+42+BOX_H+22}" text-anchor="middle" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#5f6368">'
                 f'+ split-conformal scale</text>')
    parts.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" '
                 f'font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#5f6368">'
                 f'trained by Gaussian NLL after an MSE warm-up; no BatchNorm</text>')
    parts.append("</svg>")

    out = FIG_DIR / "architecture.svg"
    out.write_text("\n".join(parts))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
