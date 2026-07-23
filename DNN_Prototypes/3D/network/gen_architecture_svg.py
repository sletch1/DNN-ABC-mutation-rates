"""Generate an aesthetic SVG of the HeteroscedasticResMLP architecture for the
3-D README. Pure layout code -> results/figures/architecture.svg. Instant.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "architecture.svg"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1180, 620
CY = 300

EDGE = "#111827"
INK = "#1f2937"
SUB = "#6b7280"
INPUT_F, INPUT_S = "#fff3e0", "#f59e0b"
PROJ_F, PROJ_S = "#e8f5e9", "#2e7d32"
BLK_F, BLK_S = "#eef2ff", "#6366f1"
SKIP = "#ef6c00"
MEAN_S = "#1e88e5"
VAR_S = "#8e44ad"

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="Helvetica Neue, Arial, sans-serif">')
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(f'<rect x="8" y="8" width="{W-16}" height="{H-16}" rx="18" '
           f'fill="#fbfcfe" stroke="#e5e9f0" stroke-width="1.5"/>')
svg.append('<defs>'
           '<linearGradient id="meanG" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#e3f2fd"/><stop offset="1" stop-color="#bbdefb"/></linearGradient>'
           '<linearGradient id="varG" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#f5e9fc"/><stop offset="1" stop-color="#e6ccf7"/></linearGradient>'
           f'<marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
           f'<path d="M0,0 L7,3 L0,6 Z" fill="{SKIP}"/></marker>'
           '</defs>')

# title
svg.append(f'<text x="{W/2}" y="46" text-anchor="middle" font-size="24" '
           f'font-weight="700" fill="{INK}">Heteroscedastic residual MLP surrogate (3-D)</text>')
svg.append(f'<text x="{W/2}" y="74" text-anchor="middle" font-size="15" '
           f'fill="{SUB}">inputs  (log&#8321;&#8320;(p), a, &#948;)   &#8594;   two outputs:  '
           f'mean and log-variance of  log&#8321;&#8320;(d&#772;),   d&#772; = mean&#7522; &#8730;(X&#7522;/Z&#7522;)</text>')

# input nodes
inx = 92
inputs = [(inx, CY - 46, "log&#8321;&#8320;(p)"), (inx, CY, "a"), (inx, CY + 46, "&#948;")]
for (x, y, lab) in inputs:
    svg.append(f'<circle cx="{x}" cy="{y}" r="15" fill="{INPUT_F}" stroke="{INPUT_S}" stroke-width="2.5"/>')
    svg.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="12.5" '
               f'font-weight="600" fill="{INK}">{lab}</text>')
svg.append(f'<text x="{inx}" y="{CY-80}" text-anchor="middle" font-size="12" fill="{SUB}">input (3)</text>')

# input projection block
px = 250
svg.append(f'<rect x="{px-42}" y="{CY-95}" width="84" height="190" rx="12" '
           f'fill="{PROJ_F}" stroke="{PROJ_S}" stroke-width="2"/>')
svg.append(f'<text x="{px}" y="{CY-4}" text-anchor="middle" font-size="14" '
           f'font-weight="700" fill="{PROJ_S}">Linear</text>')
svg.append(f'<text x="{px}" y="{CY+16}" text-anchor="middle" font-size="12.5" fill="{INK}">3 &#8594; 128</text>')
for (x, y, _l) in inputs:
    svg.append(f'<line x1="{x+15}" y1="{y}" x2="{px-42}" y2="{CY}" stroke="{EDGE}" '
               f'stroke-width="0.9" opacity="0.5"/>')

# residual blocks
block_x = [430, 640, 850]
bw, bh = 120, 210
for i, bx in enumerate(block_x):
    svg.append(f'<rect x="{bx-bw/2}" y="{CY-bh/2}" width="{bw}" height="{bh}" rx="12" '
               f'fill="{BLK_F}" stroke="{BLK_S}" stroke-width="2"/>')
    svg.append(f'<text x="{bx}" y="{CY-bh/2+24}" text-anchor="middle" font-size="13.5" '
               f'font-weight="700" fill="{BLK_S}">Res block {i+1}</text>')
    for k, txt in enumerate(["LayerNorm", "SiLU", "Linear 128", "LayerNorm", "SiLU", "Linear 128"]):
        svg.append(f'<text x="{bx}" y="{CY-bh/2+48+k*24}" text-anchor="middle" '
                   f'font-size="11.5" fill="{INK}">{txt}</text>')
    # skip arc over the block
    svg.append(f'<path d="M {bx-bw/2} {CY-bh/2-8} C {bx-40} {CY-bh/2-46}, '
               f'{bx+40} {CY-bh/2-46}, {bx+bw/2} {CY-bh/2-8}" fill="none" '
               f'stroke="{SKIP}" stroke-width="2.2" marker-end="url(#arrow)"/>')
    svg.append(f'<text x="{bx}" y="{CY-bh/2-52}" text-anchor="middle" font-size="11.5" '
               f'font-weight="700" fill="{SKIP}">+ skip</text>')

# connectors: proj -> b1 -> b2 -> b3 -> heads
xs = [px + 42] + [b + bw / 2 for b in block_x]
xt = [b - bw / 2 for b in block_x] + [1010]
for a, b in zip(xs, xt):
    svg.append(f'<line x1="{a}" y1="{CY}" x2="{b}" y2="{CY}" stroke="{EDGE}" stroke-width="1.6" opacity="0.7"/>')

# heads
hx = 1040
svg.append(f'<circle cx="{hx}" cy="{CY-52}" r="16" fill="url(#meanG)" stroke="{MEAN_S}" stroke-width="2.5"/>')
svg.append(f'<circle cx="{hx}" cy="{CY+52}" r="16" fill="url(#varG)" stroke="{VAR_S}" stroke-width="2.5"/>')
svg.append(f'<line x1="1010" y1="{CY}" x2="{hx-16}" y2="{CY-52}" stroke="{MEAN_S}" stroke-width="1.6"/>')
svg.append(f'<line x1="1010" y1="{CY}" x2="{hx-16}" y2="{CY+52}" stroke="{VAR_S}" stroke-width="1.6"/>')
svg.append(f'<text x="{hx}" y="{CY-84}" text-anchor="middle" font-size="14" '
           f'font-weight="700" fill="{MEAN_S}">&#956;</text>')
svg.append(f'<text x="{hx}" y="{CY-100}" text-anchor="middle" font-size="11.5" fill="{INK}">mean log&#8321;&#8320;(d&#772;)</text>')
svg.append(f'<text x="{hx}" y="{CY+92}" text-anchor="middle" font-size="14" '
           f'font-weight="700" fill="{VAR_S}">log &#963;&#178;</text>')
svg.append(f'<text x="{hx}" y="{CY+108}" text-anchor="middle" font-size="11.5" fill="{INK}">log pred. variance</text>')

# bottom caption band
svg.append(f'<rect x="40" y="546" width="{W-80}" height="52" rx="10" '
           f'fill="#f7f9fc" stroke="#e5e9f0" stroke-width="1"/>')
svg.append(f'<text x="60" y="568" font-size="12.5" fill="{INK}">'
           f'<tspan font-weight="700">Structure:</tspan> input projection &#8594; 3 pre-activation residual blocks '
           f'(width 128, LayerNorm + SiLU, identity skip) &#8594; two linear heads. Depth + skips fit the '
           f'(p, a, &#948;) interaction surface; LayerNorm (never BatchNorm) avoids the 1-D model&#8217;s batch-noise failure.</text>')
svg.append(f'<text x="60" y="588" font-size="12.5" fill="{INK}">'
           f'<tspan font-weight="700">Training:</tspan> Adam, MSE warm-up on the mean head, then Gaussian NLL; early stopping on val NLL; '
           f'a single conformal scale factor calibrates &#963; to 95% coverage.</text>')

svg.append('</svg>')
OUT.write_text("\n".join(svg))
print("wrote", OUT, f"({OUT.stat().st_size} bytes)")
