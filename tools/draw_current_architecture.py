from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
from matplotlib.lines import Line2D

OUT = Path("docs")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

COL = {
    "ink": "#172A3A",
    "muted": "#607589",
    "line": "#41637D",
    "blue": "#DDEEFF",
    "blue2": "#A9D2F5",
    "green": "#DDF3E8",
    "green2": "#9DD7BB",
    "orange": "#FFE8C6",
    "orange2": "#F5B96B",
    "purple": "#EAE2FF",
    "purple2": "#B7A1EA",
    "red": "#FBE0E1",
    "red2": "#E78A8D",
    "gray": "#F4F7FA",
    "white": "#FFFFFF",
    "train": "#A65E35",
}

fig = plt.figure(figsize=(18, 10.5), dpi=180, facecolor="white")
ax = fig.add_axes([0.025, 0.035, 0.95, 0.93])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, ec=COL["line"], lw=1.2, radius=1.2, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.22,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def txt(x, y, s, size=9, weight="normal", color=None, ha="center", va="center", z=5, style="normal", rotation=0):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or COL["ink"],
            ha=ha, va=va, zorder=z, fontstyle=style, rotation=rotation)


def arrow(x1, y1, x2, y2, color=None, lw=1.55, style="-", z=3, mutation=11):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color or COL["line"],
                                linestyle=style, shrinkA=2, shrinkB=2,
                                mutation_scale=mutation), zorder=z)


def poly_arrow(points, color=None, lw=1.45, style="-", z=3):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color or COL["line"],
                           lw=lw, linestyle=style, zorder=z))
    a, b = points[-2], points[-1]
    arrow(a[0], a[1], b[0], b[1], color=color, lw=lw, style=style, z=z)


def feature_stack(x, y, w=5.0, h=8.0, color="#A9D2F5", label=None, layers=3):
    for i in reversed(range(layers)):
        off = i * 0.65
        ax.add_patch(Polygon([[x+off, y+off], [x+w+off, y+off], [x+w+off, y+h+off], [x+off, y+h+off]],
                             closed=True, facecolor=color, edgecolor=COL["line"], lw=0.9, zorder=2+i*0.01))
    if label:
        txt(x+w/2+0.7, y-1.7, label, size=7.6, color=COL["muted"])


def tensor_pill(x, y, text, w=10, color=None):
    box(x, y, w, 3.1, color or COL["white"], ec="#9EB1C0", lw=0.8, radius=1.3, z=4)
    txt(x+w/2, y+1.55, text, size=7.3, weight="bold", color="#35536B", z=5)

# Header
ax.add_line(Line2D([1.5, 98.5], [96.2, 96.2], color="#D7E0E8", lw=1.0))
txt(2, 98.2, "RLCCD: D-FINE with Visual Prompt Encoding and Multi-head Differential Box Attention",
    size=17.5, weight="bold", ha="left")
txt(2, 95.0, "Current active architecture · CCD configuration · 10 classes · 640×640 input",
    size=9.2, color=COL["muted"], ha="left")

# Panel labels
box(1.5, 52.2, 97, 41.2, "#FBFCFE", ec="#CBD7E1", lw=1.0, radius=1.4, z=0)
box(1.5, 3.2, 97, 46.0, "#FCFBFF", ec="#D5CCE8", lw=1.0, radius=1.4, z=0)
txt(3.0, 91.3, "A", size=15, weight="bold", color="#315C80")
txt(5.0, 91.3, "End-to-end detection architecture", size=11.5, weight="bold", ha="left")
txt(3.0, 47.0, "B", size=15, weight="bold", color="#7257A5")
txt(5.0, 47.0, "Visual Prompt Encoder and differential attention block", size=11.5, weight="bold", ha="left")

# ---------------- Panel A ----------------
# Input image icon
box(3.2, 69.2, 8.0, 11.5, COL["gray"], ec="#7891A5", radius=0.8)
ax.add_patch(Polygon([[4.1,70.2],[6.1,74.2],[7.3,72.4],[9.0,76.3],[10.3,70.2]], closed=True,
                     facecolor="#8DC3E7", edgecolor="none", zorder=3))
ax.add_patch(Circle((5.2,77.7), 0.6, color="#F3B762", zorder=4))
txt(7.2, 67.1, "Input", size=8.8, weight="bold")
txt(7.2, 64.8, "B×3×640×640", size=7.5, color=COL["muted"])

arrow(11.4, 75.0, 15.0, 75.0)
box(15.0, 67.0, 13.0, 16.0, COL["blue"], ec="#5E89AB")
txt(21.5, 79.4, "HGNetv2-B2", size=10.5, weight="bold")
txt(21.5, 76.7, "Backbone", size=8, color=COL["muted"])
feature_stack(17.0, 69.4, 2.3, 6.5, COL["blue2"], layers=3)
txt(22.0, 72.8, "return_idx [1,2,3]", size=7.4, ha="left")
txt(22.0, 70.6, "384 / 768 / 1536 ch", size=7.4, ha="left")

arrow(28.1, 75.0, 31.2, 75.0)
box(31.2, 67.0, 14.5, 16.0, COL["green"], ec="#5D9B7A")
txt(38.45, 79.4, "HybridEncoder", size=10.5, weight="bold")
txt(38.45, 76.7, "3-level feature fusion", size=8, color=COL["muted"])
txt(38.45, 73.4, "C = 256", size=8.3, weight="bold")
txt(38.45, 70.9, "1 encoder layer · 8 heads", size=7.4)
txt(38.45, 68.7, "FFN 1024 · GELU", size=7.4)

# shared pyramid
arrow(45.8, 75.0, 49.0, 75.0)
feature_stack(49.2, 70.8, 3.3, 7.1, COL["green2"], label="{E₈,E₁₆,E₃₂}", layers=3)

# branch node
ax.add_patch(Circle((56.3,75.0), 0.65, facecolor=COL["white"], edgecolor=COL["line"], lw=1.2, zorder=4))
arrow(53.5, 75.0, 55.7, 75.0)
# decoder branch upper
poly_arrow([(56.9,75.0),(58.2,75.0),(58.2,82.2),(61.0,82.2)])
box(61.0, 76.3, 17.0, 12.0, COL["orange"], ec="#C58A3A")
txt(69.5, 85.1, "D-FINE Decoder", size=10.5, weight="bold")
txt(69.5, 82.5, "4 layers · 300 queries", size=8.2)
txt(69.5, 80.1, "8 heads · C=256", size=7.5)
txt(69.5, 77.9, "reg_max 32 · points [3,6,3]", size=7.2)
# VPE branch lower
poly_arrow([(56.9,75.0),(58.2,75.0),(58.2,66.5),(61.0,66.5)])
box(61.0, 59.2, 17.0, 13.5, COL["purple"], ec="#816CB0")
txt(69.5, 69.4, "Visual Prompt Encoder", size=10.5, weight="bold")
txt(69.5, 66.7, "depth 1 · C=256", size=8.2)
txt(69.5, 64.3, "MS deformable cross-attn", size=7.4)
txt(69.5, 62.1, "+ differential box attention", size=7.4)
txt(69.5, 59.9, "+ SwiGLU FFN 1024", size=7.4)
# decoder boxes to VPE
poly_arrow([(69.5,76.3),(69.5,74.0),(65.5,74.0),(65.5,72.8)], color="#8C6A42", style="--")
txt(72.0, 73.9, "boxes", size=7.0, color="#8C6A42")

# outputs
arrow(78.1, 82.2, 81.2, 82.2)
box(81.2, 77.8, 8.5, 8.8, COL["orange"], ec="#C58A3A")
txt(85.45, 84.0, "Box head", size=8.8, weight="bold")
txt(85.45, 81.6, "boxes + quality", size=7.2)
txt(85.45, 79.4, "[B,300,4/1]", size=7.0, color=COL["muted"])

arrow(78.1, 65.8, 81.2, 65.8)
box(81.2, 61.4, 8.5, 8.8, COL["purple"], ec="#816CB0")
txt(85.45, 67.6, "Cls head", size=8.8, weight="bold")
txt(85.45, 65.2, "Linear 256→10", size=7.2)
txt(85.45, 63.0, "VPE logits", size=7.0, color=COL["muted"])

# add and output
arrow(89.8, 82.2, 92.0, 78.0)
arrow(89.8, 65.8, 92.0, 74.2)
ax.add_patch(Circle((92.4,76.0), 1.25, facecolor=COL["white"], edgecolor=COL["line"], lw=1.2, zorder=4))
txt(92.4,76.0,"+",size=11,weight="bold")
arrow(93.7,76.0,96.6,76.0)
box(96.4, 70.3, 2.2, 11.5, COL["green"], ec="#5D9B7A", radius=0.7)
txt(97.5, 76.0, "Detections", size=8.2, weight="bold", rotation=90 if False else 0)
txt(92.4, 71.8, "quality + class", size=6.7, color=COL["muted"])

# Training objectives compact
box(15.0, 55.0, 30.7, 7.0, "#FFF8EE", ec="#D6A66D", radius=0.8)
txt(16.2, 60.3, "Active detection objectives", size=8.7, weight="bold", ha="left", color="#835128")
txt(16.2, 57.8, "VFL 1 · L1 5 · GIoU 2 · FGL 0.15 · DDF 1.5 · local/RL 6", size=7.2, ha="left")
poly_arrow([(69.5,59.2),(69.5,56.0),(46.0,56.0)], color=COL["train"], style="--")

box(48.7, 55.0, 30.5, 7.0, "#FFF8EE", ec="#D6A66D", radius=0.8)
txt(50.0, 60.3, "Active VPE objectives", size=8.7, weight="bold", ha="left", color="#835128")
txt(50.0, 57.8, "VPE VFL 1 · visual–text contrast 0.2 · IoU match 0.6 / 0.5", size=7.2, ha="left")

# ---------------- Panel B ----------------
# VPE input formation
box(3.5, 28.0, 17.2, 14.5, COL["gray"], ec="#7A91A4")
txt(12.1, 39.4, "Query initialization", size=10, weight="bold")
txt(4.5, 36.7, "Box encoding", size=8.0, weight="bold", ha="left")
txt(4.5, 34.5, "sine/cos + w,h,log(ar),√area", size=6.9, ha="left")
txt(4.5, 32.3, "MLP 520→256", size=7.1, ha="left")
txt(4.5, 29.7, "ROIAlign 7×7 → Conv3×3 → GN32", size=6.9, ha="left")
txt(4.5, 27.7, "ReLU → GAP → Linear 256", size=6.9, ha="left")

arrow(20.9, 35.0, 23.7, 35.0)
box(23.7, 28.0, 14.0, 14.5, COL["green"], ec="#5D9B7A")
txt(30.7, 39.4, "VPE block", size=10, weight="bold")
txt(30.7, 36.8, "Pre-Norm", size=7.4, color=COL["muted"])
txt(30.7, 34.3, "MS deformable cross-attn", size=7.4)
txt(30.7, 31.9, "8 heads · 3 levels · 6 points", size=6.9)
txt(30.7, 29.4, "residual connection", size=7.1)

arrow(37.9, 35.0, 40.8, 35.0)
# Differential module large
box(40.8, 8.0, 53.5, 34.5, "#F7F3FF", ec="#816CB0", lw=1.35)
txt(42.2, 40.1, "Multi-head Differential Box Attention", size=11, weight="bold", ha="left", color="#5B438F")
tensor_pill(42.5, 35.0, "Q ∈ ℝᴮ×³⁰⁰×²⁵⁶", w=12.5, color=COL["white"])

# Split paths
arrow(55.4, 36.6, 59.2, 36.6)
# pos and neg pool boxes
box(59.2, 31.7, 12.2, 8.0, "#E5F6ED", ec="#5D9B7A", radius=0.9)
txt(65.3, 37.5, "Positive pooling", size=8.4, weight="bold")
txt(65.3, 35.2, "Linear 256→8", size=7.0)
txt(65.3, 33.2, "softmax over 300", size=6.8)
box(59.2, 21.1, 12.2, 8.0, "#FBE5E6", ec="#C56D71", radius=0.9)
txt(65.3, 26.9, "Negative pooling", size=8.4, weight="bold")
txt(65.3, 24.6, "Linear 256→8", size=7.0)
txt(65.3, 22.6, "softmax over 300", size=6.8)
poly_arrow([(55.4,36.6),(57.0,36.6),(57.0,35.7),(59.0,35.7)])
poly_arrow([(55.4,36.6),(57.0,36.6),(57.0,25.1),(59.0,25.1)])

# token stacks
arrow(71.5, 35.7, 74.2, 35.7)
arrow(71.5, 25.1, 74.2, 25.1)
feature_stack(74.3, 32.7, 2.5, 5.0, COL["green2"], label="T₊: 8×256", layers=4)
feature_stack(74.3, 22.1, 2.5, 5.0, COL["red2"], label="T₋: 8×256", layers=4)

# head split
box(80.0, 21.0, 6.8, 18.8, COL["blue"], ec="#5E89AB", radius=0.8)
txt(83.4, 37.4, "Split", size=8.2, weight="bold")
txt(83.4, 34.9, "8 heads", size=7.1)
for i in range(8):
    yy = 22.3 + i*1.42
    ax.add_patch(FancyBboxPatch((81.1, yy), 4.6, 0.85, boxstyle="round,pad=0.05,rounding_size=.18",
                                facecolor="#FFFFFF", edgecolor="#7D9DB7", lw=0.55, zorder=4))
    txt(83.4, yy+0.42, f"h{i+1}: 32-d", size=5.6)
arrow(77.8, 35.7, 79.8, 34.0)
arrow(77.8, 25.1, 79.8, 27.0)

# differential equation
arrow(86.9, 30.4, 89.0, 30.4)
box(89.0, 20.4, 4.7, 20.0, COL["white"], ec="#816CB0", radius=0.8)
txt(91.35, 37.6, "Per head", size=7.5, weight="bold")
txt(91.35, 34.9, "A₊V₊", size=7.0, weight="bold", color="#3F8B66")
txt(91.35, 31.9, "−", size=9, weight="bold")
txt(91.35, 29.1, "λₕA₋V₋", size=7.0, weight="bold", color="#B35960")
txt(91.35, 25.8, "λₕ=softplus", size=6.2)
txt(91.35, 23.6, "init 0.5", size=6.2)

# bottom processing inside panel
poly_arrow([(91.35,20.4),(91.35,17.8),(78.5,17.8)])
box(64.0, 12.8, 14.5, 6.5, COL["purple"], ec="#816CB0", radius=0.8)
txt(71.25, 17.1, "Concat heads → RMSNorm", size=7.8, weight="bold")
txt(71.25, 14.7, "tanh(residual_scale) · Δ", size=7.1)
arrow(63.8, 16.0, 60.5, 16.0)
ax.add_patch(Circle((59.4,16.0), 1.15, facecolor=COL["white"], edgecolor=COL["line"], lw=1.0, zorder=4))
txt(59.4,16.0,"+",size=10,weight="bold")
poly_arrow([(48.7,35.0),(47.5,35.0),(47.5,16.0),(58.2,16.0)], color="#5F7890")
arrow(60.6,16.0,63.1,16.0)
txt(52.0,13.3,"outer Pre-Norm residual",size=6.5,color=COL["muted"])

# FFN and output
arrow(94.4, 30.4, 96.4, 30.4)
box(95.0, 22.8, 3.1, 15.0, COL["orange"], ec="#C58A3A", radius=0.7)
txt(96.55, 30.3, "FFN", size=8.2, weight="bold", rotation=90)
txt(96.55, 19.9, "SwiGLU 1024", size=6.3)

# Small shape annotations
tensor_pill(41.8, 9.3, "Output: B×300×256", w=13.5, color="#FFFFFF")
txt(83.8, 10.2, "8 positive tokens + 8 negative tokens · no additional geometry embedding", size=7.0, color=COL["muted"])

# legend / footer
ax.add_line(Line2D([2.5, 97.5], [1.7, 1.7], color="#D7E0E8", lw=0.9))
txt(2.5, 0.65, "Solid arrows: inference path", size=7.2, color=COL["muted"], ha="left")
txt(20.5, 0.65, "Dashed arrows: training supervision / conditioning", size=7.2, color=COL["muted"], ha="left")
txt(97.5, 0.65, "Current implementation · September 24, 2026", size=7.2, color=COL["muted"], ha="right")

for ext in ("svg", "pdf", "png"):
    path = OUT / f"current_rlccd_architecture_v2.{ext}"
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=300 if ext == "png" else None)
plt.close(fig)
