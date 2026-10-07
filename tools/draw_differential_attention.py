from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

C = {
    "ink": "#182B3A", "muted": "#617589", "line": "#496A83",
    "blue": "#DDEEFF", "blue_d": "#5597C8", "blue_q": "#78B6E5",
    "green": "#DDF4E7", "green_d": "#49A56F",
    "red": "#FBE4E5", "red_d": "#D8666B",
    "purple": "#EDE5FF", "purple_d": "#876DBB",
    "orange": "#FFEACB", "orange_d": "#D9943C",
    "gray": "#F3F6F9", "white": "#FFFFFF",
}

fig = plt.figure(figsize=(17, 9.6), dpi=180, facecolor="white")
ax = fig.add_axes([0.025, 0.04, 0.95, 0.92])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def rounded(x, y, w, h, fc, ec=None, lw=1.3, r=1.0, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.20,rounding_size={r}",
        facecolor=fc, edgecolor=ec or C["line"], linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(x, y, value, fs=9, fw="normal", color=None, ha="center", va="center", z=7):
    ax.text(x, y, value, fontsize=fs, fontweight=fw, color=color or C["ink"],
            ha=ha, va=va, zorder=z)


def arrow(x1, y1, x2, y2, color=None, lw=1.7, ls="-", z=4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color or C["line"], lw=lw,
            linestyle=ls, mutation_scale=12, shrinkA=2, shrinkB=2,
        ), zorder=z,
    )


def line(x1, y1, x2, y2, color=None, lw=1.4, ls="-", z=3):
    ax.add_line(Line2D([x1, x2], [y1, y2], color=color or C["line"],
                       lw=lw, ls=ls, zorder=z))


def query_grid(x, y, cols=12, rows=3, size=0.58, gap=0.22, color=None, edge=None):
    color = color or C["blue_q"]
    edge = edge or C["blue_d"]
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(FancyBboxPatch(
                (x + col * (size + gap), y - row * (size + gap)), size, size,
                boxstyle="round,pad=0.02,rounding_size=.10",
                fc=color, ec=edge, lw=0.48, zorder=6,
            ))


def token_row(x, y, color, edge, n=8, size=0.86, gap=0.27):
    for idx in range(n):
        ax.add_patch(FancyBboxPatch(
            (x + idx * (size + gap), y), size, size,
            boxstyle="round,pad=0.03,rounding_size=.16",
            fc=color, ec=edge, lw=0.7, zorder=6,
        ))


def numbered(x, y, number, color="#243F55"):
    ax.add_patch(Circle((x, y), 0.82, fc=color, ec="white", lw=1, zorder=9))
    text(x, y, str(number), fs=7.2, fw="bold", color="white", z=10)


# Header
text(2, 97.5, "Differential Box Attention", fs=18, fw="bold", ha="left")
text(2, 94.7,
     "Box-level semantic refinement with positive–negative multi-head differential attention",
     fs=9.5, color=C["muted"], ha="left")
line(2, 93.1, 98, 93.1, color="#D8E1E8", lw=1.0)

# Main flow panel
rounded(1.8, 51.0, 96.2, 39.8, "#FBFCFE", ec="#CBD7E0", lw=1.0, r=1.3, z=0)
text(3.2, 87.5, "A", fs=15, fw="bold", color="#35698F")
text(5.2, 87.5, "Token generation from box queries", fs=11.5, fw="bold", ha="left")

# Input query block
rounded(4.0, 66.0, 15.0, 15.2, C["blue"], ec=C["blue_d"])
text(11.5, 78.8, "Normalized box queries", fs=10.2, fw="bold")
query_grid(5.2, 75.0, cols=12, rows=3, size=0.63, gap=0.27)
text(11.5, 68.3, "Q ∈ R^(300 × 256)", fs=8.2, color=C["muted"])

# Split to two pooling branches
arrow(19.2, 73.7, 23.0, 78.2)
arrow(19.2, 73.0, 23.0, 68.3)

rounded(23.0, 76.0, 15.0, 7.8, C["green"], ec=C["green_d"])
text(30.5, 81.0, "Positive pooling", fs=10, fw="bold", color="#267548")
text(30.5, 78.4, "Linear → Softmax over 300", fs=7.5, color=C["muted"])

rounded(23.0, 64.2, 15.0, 7.8, C["red"], ec=C["red_d"])
text(30.5, 69.2, "Negative pooling", fs=10, fw="bold", color="#A84349")
text(30.5, 66.6, "Linear → Softmax over 300", fs=7.5, color=C["muted"])

arrow(38.2, 79.9, 42.0, 79.9, color=C["green_d"])
arrow(38.2, 68.1, 42.0, 68.1, color=C["red_d"])

rounded(42.0, 75.0, 17.2, 9.8, C["green"], ec=C["green_d"])
text(50.6, 82.0, "Positive tokens", fs=10, fw="bold", color="#267548")
token_row(43.4, 78.0, "#8BD4A6", C["green_d"])
text(50.6, 76.2, "T+ ∈ R^(8 × 256)", fs=7.8, color=C["muted"])

rounded(42.0, 62.8, 17.2, 9.8, C["red"], ec=C["red_d"])
text(50.6, 69.8, "Negative tokens", fs=10, fw="bold", color="#A84349")
token_row(43.4, 65.8, "#F19CA0", C["red_d"])
text(50.6, 64.0, "T− ∈ R^(8 × 256)", fs=7.8, color=C["muted"])

# Main transition to stage B
line(60.0, 60.0, 60.0, 53.4, color="#D8E1E8", lw=1.0)
text(61.3, 56.8, "same box queries are split into 8 heads", fs=7.4, color=C["muted"], ha="left")

# Detail panel
rounded(1.8, 3.0, 96.2, 47.0, "#FCFBFF", ec="#D6CDE8", lw=1.0, r=1.3, z=0)
text(3.2, 47.5, "B", fs=15, fw="bold", color="#7355A5")
text(5.2, 47.5, "Multi-head differential read and residual update", fs=11.5, fw="bold", ha="left")

# Query head input
rounded(3.5, 28.0, 13.0, 13.0, C["blue"], ec=C["blue_d"])
text(10.0, 38.0, "Query heads", fs=9.8, fw="bold")
query_grid(5.0, 35.0, cols=8, rows=3, size=0.66, gap=0.28)
text(10.0, 30.2, "8 heads × 32 dim", fs=7.5, color=C["muted"])

# Positive and negative read branches
arrow(16.6, 35.0, 20.2, 39.2, color=C["green_d"])
arrow(16.6, 34.0, 20.2, 30.1, color=C["red_d"])

rounded(20.2, 35.5, 16.0, 8.0, C["green"], ec=C["green_d"])
text(28.2, 40.8, "Positive read", fs=9.8, fw="bold", color="#267548")
text(28.2, 38.2, "Softmax(QK+ / √32) V+", fs=7.7, color=C["muted"])

rounded(20.2, 25.5, 16.0, 8.0, C["red"], ec=C["red_d"])
text(28.2, 30.8, "Negative read", fs=9.8, fw="bold", color="#A84349")
text(28.2, 28.2, "Softmax(QK− / √32) V−", fs=7.7, color=C["muted"])

arrow(36.5, 39.5, 40.0, 39.5, color=C["green_d"])
arrow(36.5, 29.5, 40.0, 29.5, color=C["red_d"])

rounded(40.0, 35.5, 15.0, 8.0, C["green"], ec=C["green_d"])
text(47.5, 40.8, "Positive output", fs=9.8, fw="bold", color="#267548")
text(47.5, 38.2, "O+ ∈ R^(8×300×32)", fs=7.4, color=C["muted"])

rounded(40.0, 25.5, 15.0, 8.0, C["red"], ec=C["red_d"])
text(47.5, 30.8, "Negative output", fs=9.8, fw="bold", color="#A84349")
text(47.5, 28.2, "O− ∈ R^(8×300×32)", fs=7.4, color=C["muted"])

# Differential operation
arrow(55.5, 39.5, 59.0, 39.5, color=C["green_d"])
arrow(55.5, 29.5, 57.0, 29.5, color=C["red_d"])
line(57.0, 29.5, 57.0, 36.4, color=C["red_d"], lw=1.4)
arrow(57.0, 36.4, 59.0, 36.4, color=C["red_d"])
rounded(59.0, 31.8, 15.0, 12.0, C["purple"], ec=C["purple_d"])
text(66.5, 41.0, "Differential", fs=10, fw="bold")
text(66.5, 38.0, "O+ − λh · O−", fs=9.2, fw="bold", color=C["ink"])
text(66.5, 35.1, "λh = softplus(λh)", fs=7.4, color=C["muted"])
text(66.5, 33.0, "8 learnable head-wise λ", fs=7.2, color=C["muted"])

# Merge and normalization
arrow(74.2, 37.8, 77.4, 37.8)
rounded(77.4, 32.4, 9.0, 10.8, C["gray"], ec="#8296A7")
text(81.9, 39.2, "Merge", fs=9.4, fw="bold")
text(81.9, 36.6, "8 × 32 → 256", fs=7.5, color=C["muted"])
text(81.9, 34.4, "+ RMSNorm", fs=7.5, color=C["muted"])

# Residual scaling and outer residual
arrow(86.7, 37.8, 89.1, 37.8)
rounded(89.1, 32.4, 7.5, 10.8, C["orange"], ec=C["orange_d"])
text(92.85, 39.2, "Update", fs=9.4, fw="bold")
text(92.85, 36.8, "tanh(s)", fs=8.2, fw="bold")
text(92.85, 34.5, "s starts at 0", fs=7.2, color=C["muted"])

# Outer residual visual: query skips around differential branch
line(10.0, 27.8, 10.0, 17.5, color=C["blue_d"], lw=1.4, ls="--")
line(10.0, 17.5, 94.8, 17.5, color=C["blue_d"], lw=1.4, ls="--")
arrow(94.8, 17.5, 94.8, 32.0, color=C["blue_d"], lw=1.4, ls="--")
text(11.2, 18.8, "residual query", fs=7.2, color=C["blue_d"], ha="left")

arrow(96.6, 37.8, 98.0, 37.8, color=C["orange_d"])
ax.add_patch(Circle((98.0, 37.8), 1.0, fc=C["white"], ec=C["line"], lw=1.1, zorder=6))
text(98.0, 37.8, "+", fs=10, fw="bold")
text(92.5, 14.6, "Refined box semantics", fs=9.5, fw="bold", color=C["ink"])
arrow(98.0, 36.6, 98.0, 16.0, color=C["line"], lw=1.2)

# Numbered stages
for x, y, number in [
    (4.0, 84.6, 1), (21.8, 84.6, 2), (40.8, 84.6, 3),
    (3.0, 43.2, 4), (19.7, 43.2, 5), (39.5, 43.2, 6),
    (58.5, 45.2, 7), (76.8, 45.2, 8), (88.5, 45.2, 9),
]:
    numbered(x, y, number)

# Footer legend and note
line(2, 1.7, 98, 1.7, color="#D8E1E8", lw=0.9)
text(2.2, 0.65, "Blue: box queries    Green: positive path    Red: negative path    Purple: differential operation",
     fs=7.4, color=C["muted"], ha="left")
text(97.8, 0.65, "Current implementation", fs=7.4, color=C["muted"], ha="right")

for ext in ("png", "svg", "pdf"):
    fig.savefig(
        OUT / f"differential_box_attention.{ext}",
        bbox_inches="tight", facecolor="white", dpi=300 if ext == "png" else None,
    )
plt.close(fig)
