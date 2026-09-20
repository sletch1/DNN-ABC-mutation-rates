"""Render the surrogate architecture as results/figures/architecture.svg.

A hand-rolled SVG rather than a plotting library: the diagram is a fixed set of
labelled boxes and arrows, so emitting the markup directly keeps it crisp at any
zoom, theme-neutral, and free of a rendering dependency.

The diagram is generated FROM the live config in train.py (ARCH), so it cannot
drift out of sync with the model that is actually trained -- the layer widths
and activations below are read, never typed. Layout is computed from the number
of hidden layers, so adding or removing one re-flows the figure (and its width)
rather than overflowing it.

Deliberately the same design as the 3-D package's diagram, so the two studies
illustrate alike.

Usage:
    python gen_architecture_svg.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "network"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from train import ARCH
from paths import FIG_DIR

# The constant-rate study has exactly one input by construction: the mutation
# rate on a log scale. There is no feature list in model.py to read it from.
N_IN, IN_LABEL = 1, "log&#8321;&#8320; p"

# --- geometry ---------------------------------------------------------------
PAD = 34                      # margin around the card
BOX_W, BOX_H = 152, 84        # input and hidden blocks
HEAD_W, HEAD_H = 168, 64      # the two output heads
CAL_W, CAL_H = 150, 84        # the conformal calibration block
GAP = 56                      # horizontal space between blocks
HEAD_DY = 50                  # vertical offset of each head from the spine
ROW_Y = 168                   # top edge of the main row of blocks

FONT = "Helvetica Neue,Helvetica,Arial,sans-serif"

# --- palette: one hue per stage, tinted fill + saturated stroke --------------
PAGE_BG, CARD_BG, CARD_EDGE = "#f8fafc", "#ffffff", "#e2e8f0"
INK, MUTED = "#0f172a", "#64748b"
LINE = "#94a3b8"
IN_FILL, IN_EDGE, IN_INK = "#eef2ff", "#6366f1", "#312e81"
HID_FILL, HID_EDGE, HID_INK = "#ecfdf5", "#10b981", "#065f46"
OUT_FILL, OUT_EDGE, OUT_INK = "#fff1f2", "#f43f5e", "#881337"
CAL_FILL, CAL_EDGE, CAL_INK = "#fffbeb", "#f59e0b", "#78350f"


def text(x, y, s, size=13, fill=INK, weight="400", anchor="middle", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{ls}>{s}</text>')


def box(x, y, w, h, fill, edge, ink, title, sub="", sub2=""):
    """A rounded block with a soft shadow, a bold title and up to two sub-lines."""
    # The shadow is a second rect rather than an SVG filter: filters survive
    # browsers but not every SVG-to-PDF converter, and this does.
    parts = [f'<rect x="{x+1.5:.1f}" y="{y+3:.1f}" width="{w}" height="{h}" rx="11" '
             f'fill="#0f172a" opacity="0.05"/>',
             f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="11" '
             f'fill="{fill}" stroke="{edge}" stroke-width="1.4"/>']
    cx = x + w / 2
    lines = [l for l in (sub, sub2) if l]
    # Centre the title/sub-line stack as a block, so one- and two-line boxes
    # both sit optically centred.
    y0 = y + h / 2 - 5 * len(lines) + 5
    parts.append(text(cx, y0, title, size=14, fill=ink, weight="600"))
    for i, line in enumerate(lines):
        parts.append(text(cx, y0 + 18 + i * 15, line, size=11, fill=MUTED))
    return "".join(parts)


def arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - 8:.1f}" y2="{y2:.1f}" '
            f'stroke="{LINE}" stroke-width="1.6" marker-end="url(#arrow)"/>')


def curve(x1, y1, x2, y2):
    """A flat S-curve, for the fan-out to the heads and the fan-in from them."""
    dx = (x2 - x1) * 0.55
    return (f'<path d="M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} '
            f'{x2 - dx:.1f},{y2:.1f} {x2 - 8:.1f},{y2:.1f}" fill="none" '
            f'stroke="{LINE}" stroke-width="1.6" marker-end="url(#arrow)"/>')


def dim_label(x, y, s):
    """The tensor width riding above an arrow."""
    return text(x, y - 9, s, size=10, fill=MUTED, spacing="0.4")


def main():
    # `hidden_dims` is this package's key for the trunk widths; `activation` is
    # one name for every hidden layer here (the 3-D package allows one per
    # layer, so accept a sequence too rather than assuming a string).
    hidden = list(ARCH.get("hidden_dims", ()))
    _act = ARCH.get("activation", "gelu")
    acts = [_act] * len(hidden) if isinstance(_act, str) else list(_act)
    if len(acts) != len(hidden):
        raise ValueError(f"{len(acts)} activations for {len(hidden)} hidden layers")
    acts = [a.upper() for a in acts]
    # The figure says "no BatchNorm" in its footer, so do not let that caption
    # outlive the config it describes.
    if ARCH.get("use_bn"):
        raise ValueError("ARCH now uses BatchNorm; the diagram's footer is stale")

    # Width follows the block count, so the title never overruns the canvas.
    body_w = BOX_W * (1 + len(hidden)) + HEAD_W + CAL_W + GAP * (len(hidden) + 2)
    W = body_w + 2 * PAD
    H = 372

    spine = ROW_Y + BOX_H / 2          # vertical centre line of the main row
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H}" '
        f'viewBox="0 0 {W:.0f} {H}" role="img" '
        f'aria-label="Architecture of the 1-D constant-rate surrogate network">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        f'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="{LINE}"/></marker></defs>',
        f'<rect width="{W:.0f}" height="{H}" fill="{PAGE_BG}"/>',
        f'<rect x="{PAD / 2}" y="{PAD / 2}" width="{W - PAD:.1f}" height="{H - PAD}" '
        f'rx="16" fill="{CARD_BG}" stroke="{CARD_EDGE}"/>',
        text(W / 2, 52, "1-D constant-rate surrogate", size=18, weight="600"),
        text(W / 2, 74,
             f"{IN_LABEL} &#8594; mean and variance of log&#8321;&#8320; d&#772;, "
             "&#160; d&#772; = mean&#7522; &#8730;(X&#7522;/Z&#7522;)",
             size=12.5, fill=MUTED),
    ]

    x = PAD + 6
    parts.append(box(x, ROW_Y, BOX_W, BOX_H, IN_FILL, IN_EDGE, IN_INK, "Input",
                     IN_LABEL, "standardised"))
    x += BOX_W

    prev = N_IN
    for h, act in zip(hidden, acts):
        parts.append(arrow(x, spine, x + GAP, spine))
        parts.append(dim_label(x + GAP / 2, spine, str(prev)))
        x += GAP
        parts.append(box(x, ROW_Y, BOX_W, BOX_H, HID_FILL, HID_EDGE, HID_INK,
                         f"Dense {h}", act, "fully connected"))
        prev = h
        x += BOX_W

    # Fan out to the two heads, then back in to the conformal calibration.
    xh = x + GAP
    y_mean, y_var = spine - HEAD_DY, spine + HEAD_DY
    parts.append(curve(x, spine, xh, y_mean))
    parts.append(curve(x, spine, xh, y_var))
    # Tucked against the box and left-aligned: mid-gap it would sit on top of
    # the fan-out curves.
    parts.append(text(x + 12, spine - 18, str(prev), size=10, fill=MUTED,
                      anchor="start", spacing="0.4"))
    parts.append(box(xh, y_mean - HEAD_H / 2, HEAD_W, HEAD_H, OUT_FILL, OUT_EDGE, OUT_INK,
                     "mean", "&#956;(x) = log&#8321;&#8320; d&#772;"))
    parts.append(box(xh, y_var - HEAD_H / 2, HEAD_W, HEAD_H, OUT_FILL, OUT_EDGE, OUT_INK,
                     "log variance", "soft-clamped"))

    xc = xh + HEAD_W + GAP
    parts.append(curve(xh + HEAD_W, y_mean, xc, spine))
    parts.append(curve(xh + HEAD_W, y_var, xc, spine))
    parts.append(box(xc, ROW_Y, CAL_W, CAL_H, CAL_FILL, CAL_EDGE, CAL_INK,
                     "calibration", "split-conformal scale", "95% predictive interval"))

    parts.append(text(W / 2, H - 30,
                      "trained by Gaussian NLL after an MSE warm-up &#183; no BatchNorm "
                      "&#183; feeds the ABC acceptance step",
                      size=11, fill=MUTED))
    parts.append("</svg>")

    out = FIG_DIR / "architecture.svg"
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
