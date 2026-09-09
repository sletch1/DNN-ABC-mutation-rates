"""Generate an aesthetic SVG of the HeteroscedasticMLP architecture for the README.

Pure layout code -> results/architecture.svg. No training, no data; instant.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "figures" / "architecture.svg"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1120, 600
CY = 300

# palette (works on light backgrounds; GitHub/markdown default)
EDGE = "#111827"
INK = "#1f2937"
SUB = "#6b7280"
INPUT_F, INPUT_S = "#fff3e0", "#f59e0b"
HID_F, HID_S = "#eef2ff", "#6366f1"
MEAN_F, MEAN_S = "#e3f2fd", "#1e88e5"
VAR_F, VAR_S = "#f5e9fc", "#8e44ad"


def col(x, n, spacing=42, cy=CY):
    return [(x, cy + (i - (n - 1) / 2) * spacing) for i in range(n)]


input_layer = col(95, 1)
h1 = col(330, 8)
h2 = col(560, 7)
heads = [(815, 245), (815, 355)]

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="Helvetica Neue, Arial, sans-serif">')
# backdrop
svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(f'<rect x="8" y="8" width="{W-16}" height="{H-16}" rx="18" '
           f'fill="#fbfcfe" stroke="#e5e9f0" stroke-width="1.5"/>')

# defs: subtle glow gradient for heads
svg.append('<defs>'
           '<linearGradient id="meanG" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#e3f2fd"/><stop offset="1" stop-color="#bbdefb"/></linearGradient>'
           '<linearGradient id="varG" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#f5e9fc"/><stop offset="1" stop-color="#e6ccf7"/></linearGradient>'
           '</defs>')

# title
svg.append(f'<text x="{W/2}" y="46" text-anchor="middle" font-size="24" '
           f'font-weight="700" fill="{INK}">Heteroscedastic MLP surrogate</text>')
svg.append(f'<text x="{W/2}" y="74" text-anchor="middle" font-size="15" '
           f'fill="{SUB}">input  log₁₀(p)   →   two outputs:  '
           f'mean and log-variance of  log₁₀(d̄),   d̄ = meanᵢ √(Xᵢ/Zᵢ)</text>')


def edges(a, b):
    for (x1, y1) in a:
        for (x2, y2) in b:
            svg.append(f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" '
                       f'stroke="{EDGE}" stroke-width="0.6" opacity="0.45"/>')


edges(input_layer, h1)
edges(h1, h2)
edges(h2, heads)


def neurons(layer, r, fill, stroke):
    for (x, y) in layer:
        svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="{r}" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="2"/>')


neurons(input_layer, 13, INPUT_F, INPUT_S)
neurons(h1, 9, HID_F, HID_S)
neurons(h2, 9, HID_F, HID_S)
svg.append(f'<circle cx="815" cy="245" r="15" fill="url(#meanG)" stroke="{MEAN_S}" stroke-width="2.5"/>')
svg.append(f'<circle cx="815" cy="355" r="15" fill="url(#varG)" stroke="{VAR_S}" stroke-width="2.5"/>')

# input label
svg.append(f'<text x="95" y="{CY+45}" text-anchor="middle" font-size="15" '
           f'font-weight="600" fill="{INPUT_S}">log₁₀(p)</text>')
svg.append(f'<text x="95" y="{CY-38}" text-anchor="middle" font-size="12" fill="{SUB}">input</text>')

# hidden column captions
for x, width in [(330, 128), (560, 64)]:
    svg.append(f'<text x="{x}" y="{CY+180}" text-anchor="middle" font-size="15" '
               f'font-weight="700" fill="{HID_S}">Dense {width}</text>')
    svg.append(f'<text x="{x}" y="{CY+200}" text-anchor="middle" font-size="12" '
               f'fill="{SUB}">GELU activation</text>')

# head labels + boxes
svg.append(f'<text x="845" y="235" font-size="15" font-weight="700" fill="{MEAN_S}">'
           f'μ</text>')
svg.append(f'<text x="862" y="235" font-size="13" fill="{INK}">mean of log₁₀(d̄)</text>')
svg.append(f'<text x="845" y="360" font-size="15" font-weight="700" fill="{VAR_S}">'
           f'log σ²</text>')
svg.append(f'<text x="888" y="360" font-size="13" fill="{INK}">log predictive variance</text>')

# predictive-law box + downstream
bx, by, bw, bh = 845, 415, 250, 70
svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="12" '
           f'fill="#f0f7ff" stroke="{MEAN_S}" stroke-width="1.5"/>')
svg.append(f'<text x="{bx+bw/2}" y="{by+28}" text-anchor="middle" font-size="14.5" '
           f'font-weight="700" fill="{INK}">predictive law  N(μ, σ²)</text>')
svg.append(f'<text x="{bx+bw/2}" y="{by+50}" text-anchor="middle" font-size="12" '
           f'fill="{SUB}">feeds ABC acceptance (Eqs. 9–10)</text>')
# arrows from heads to box
svg.append(f'<path d="M 832 250 C 900 300, 900 360, 930 {by-2}" fill="none" '
           f'stroke="{MEAN_S}" stroke-width="1.6" opacity="0.7"/>')
svg.append(f'<path d="M 832 358 C 900 380, 905 395, 950 {by-2}" fill="none" '
           f'stroke="{VAR_S}" stroke-width="1.6" opacity="0.7"/>')

# bottom caption band
svg.append(f'<rect x="40" y="524" width="{W-80}" height="52" rx="10" '
           f'fill="#f7f9fc" stroke="#e5e9f0" stroke-width="1"/>')
svg.append(f'<text x="60" y="546" font-size="12.5" fill="{INK}">'
           f'<tspan font-weight="700">Training:</tspan> GELU activations, no BatchNorm; Adam, 60-epoch MSE warm-up on the mean '
           f'head, then Gaussian negative log-likelihood; early stopping on validation NLL.</text>')
svg.append(f'<text x="60" y="566" font-size="12.5" fill="{INK}">'
           f'<tspan font-weight="700">Calibration:</tspan> a single conformal scale factor rescales σ so the 95% '
           f'predictive interval has valid empirical coverage on held-out data.</text>')

svg.append('</svg>')
OUT.write_text("\n".join(svg))
print("wrote", OUT, f"({OUT.stat().st_size} bytes)")
