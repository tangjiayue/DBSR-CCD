from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
IMG = OUT / "assets" / "tct_val00001.jpg"

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
    "orange": "#FFEACB", "orange_d": "#D9943C",
    "purple": "#EDE5FF", "purple_d": "#876DBB",
    "gray": "#F3F6F9", "white": "#FFFFFF", "panel": "#FBFCFE",
}

fig = plt.figure(figsize=(18, 9.6), dpi=180, facecolor="white")
ax = fig.add_axes([0.025, 0.04, 0.95, 0.92])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def rounded(x, y, w, h, fc, ec=None, lw=1.3, r=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.20,rounding_size={r}",
                       facecolor=fc, edgecolor=ec or C["line"], linewidth=lw, zorder=z)
    ax.add_patch(p); return p


def text(x, y, s, fs=9, fw="normal", color=None, ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=color or C["ink"],
            ha=ha, va=va, zorder=z)


def arrow(x1, y1, x2, y2, color=None, lw=1.7, ls="-", z=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color or C["line"], lw=lw,
                                linestyle=ls, mutation_scale=12, shrinkA=2, shrinkB=2), zorder=z)


def path(points, color=None, lw=1.55, ls="-"):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color or C["line"], lw=lw, ls=ls, zorder=3))
    a, b = points[-2], points[-1]
    arrow(a[0], a[1], b[0], b[1], color=color, lw=lw, ls=ls)


def feature_maps(x, y, color, label=None, scale=1.0):
    w, h = 4.2*scale, 6.4*scale
    for i in range(3, -1, -1):
        o = i*0.48*scale
        ax.add_patch(Polygon([[x+o,y+o],[x+w+o,y+o],[x+w+o,y+h+o],[x+o,y+h+o]],
                             closed=True, fc=color, ec=C["line"], lw=0.8, zorder=2+i*0.02))
    if label: text(x+w/2+0.7, y-1.2, label, fs=7.2, color=C["muted"])


def query_grid(x, y, cols=10, rows=3, size=0.75, gap=0.32, color=None, edge=None):
    color = color or C["blue_q"]; edge = edge or C["blue_d"]
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(FancyBboxPatch((x+c*(size+gap), y-r*(size+gap)), size, size,
                                       boxstyle="round,pad=0.02,rounding_size=.12",
                                       fc=color, ec=edge, lw=0.55, zorder=5))


def token_row(x, y, color, edge, n=8, size=1.05, gap=0.35):
    for i in range(n):
        ax.add_patch(FancyBboxPatch((x+i*(size+gap), y), size, size,
                                   boxstyle="round,pad=0.03,rounding_size=.20",
                                   fc=color, ec=edge, lw=0.8, zorder=5))

# Title
text(2, 97.7, "RLCCD Model Architecture", fs=18, fw="bold", ha="left")
text(2, 94.9, "D-FINE detector with a Visual Prompt Encoder and multi-head differential box attention",
     fs=9.5, color=C["muted"], ha="left")
ax.add_line(Line2D([2, 98], [93.2, 93.2], color="#D8E1E8", lw=1.0))

# Panels
rounded(1.8, 58.5, 96.2, 32.7, C["panel"], ec="#CBD7E0", lw=1.0, r=1.3, z=0)
rounded(1.8, 3.0, 96.2, 51.5, "#FCFBFF", ec="#D6CDE8", lw=1.0, r=1.3, z=0)
text(3.3, 88.9, "A", fs=15, fw="bold", color="#35698F")
text(5.3, 88.9, "Overall architecture", fs=11.5, fw="bold", ha="left")
text(3.3, 52.1, "B", fs=15, fw="bold", color="#7355A5")
text(5.3, 52.1, "Visual Prompt Encoder", fs=11.5, fw="bold", ha="left")

# --- Top overall architecture ---
# Dataset image
img = mpimg.imread(IMG)
img_ax = fig.add_axes([0.055, 0.625, 0.115, 0.235], zorder=2)
img_ax.imshow(img); img_ax.axis("off")
for spine in img_ax.spines.values(): spine.set_visible(False)
rounded(3.5, 65.0, 11.8, 20.0, "none", ec="#6E879A", lw=1.2, r=0.7, z=1)
text(9.4, 62.8, "Cervical cytology image", fs=8.3, fw="bold")

arrow(15.8, 75.0, 20.0, 75.0)
rounded(20.0, 68.0, 13.0, 14.0, C["blue"], ec=C["blue_d"])
text(26.5, 76.5, "Backbone", fs=12, fw="bold")
text(26.5, 72.8, "HGNetv2-B2", fs=8.5, color=C["muted"])

arrow(33.2, 75.0, 37.0, 75.0)
rounded(37.0, 68.0, 13.0, 14.0, C["green"], ec=C["green_d"])
text(43.5, 76.5, "Encoder", fs=12, fw="bold")
text(43.5, 72.8, "HybridEncoder", fs=8.5, color=C["muted"])

arrow(50.2, 75.0, 55.0, 75.0)
rounded(55.0, 68.0, 13.0, 14.0, C["orange"], ec=C["orange_d"])
text(61.5, 76.5, "Decoder", fs=12, fw="bold")
text(61.5, 72.8, "300 object queries", fs=8.5, color=C["muted"])

# VPE below decoder/encoder
rounded(55.0, 59.8, 13.0, 5.5, C["purple"], ec=C["purple_d"])
text(61.5, 62.55, "VPE", fs=11, fw="bold")
path([(43.5,68.0),(43.5,62.55),(54.8,62.55)], color=C["green_d"])
path([(61.5,68.0),(61.5,65.5)], color=C["orange_d"], ls="--")
text(50.4, 64.1, "features", fs=7.0, color=C["green_d"])
text(64.2, 66.3, "boxes", fs=7.0, color=C["orange_d"])

# heads and merge
arrow(68.2, 75.0, 72.2, 75.0)
rounded(72.2, 70.8, 10.2, 8.4, C["orange"], ec=C["orange_d"])
text(77.3, 76.3, "Box prediction", fs=9.2, fw="bold")
text(77.3, 73.5, "+ quality", fs=7.8, color=C["muted"])
arrow(68.2, 62.55, 72.2, 62.55)
rounded(72.2, 58.35, 10.2, 8.4, C["purple"], ec=C["purple_d"])
text(77.3, 63.9, "Classification", fs=9.2, fw="bold")
text(77.3, 61.1, "10 classes", fs=7.8, color=C["muted"])

path([(82.6,75.0),(85.0,75.0),(85.0,70.0),(87.1,70.0)])
path([(82.6,62.55),(85.0,62.55),(85.0,68.0),(87.1,68.0)])
ax.add_patch(Circle((88.1,69.0), 1.2, fc=C["white"], ec=C["line"], lw=1.1, zorder=5))
text(88.1, 69.0, "+", fs=11, fw="bold")
arrow(89.4, 69.0, 92.0, 69.0)
rounded(92.0, 64.7, 4.5, 8.6, C["green"], ec=C["green_d"], r=0.7)
text(94.25, 69.0, "Output", fs=8.8, fw="bold")

# --- Bottom VPE expanded ---
# Sources
feature_maps(4.5, 35.0, "#A7D8BE", label="Encoder features", scale=0.9)
rounded(3.5, 14.5, 13.0, 12.0, C["gray"], ec="#7A91A4")
text(10.0, 23.5, "Query initialization", fs=9.5, fw="bold")
text(10.0, 20.5, "box encoding", fs=7.7)
text(10.0, 18.1, "+ ROI texture", fs=7.7)
text(10.0, 15.8, "300 box queries", fs=7.7, color=C["muted"])
path([(8.0,34.5),(8.0,26.7)], color=C["green_d"])

arrow(16.8, 20.5, 20.0, 20.5)
rounded(20.0, 15.0, 12.0, 11.0, C["green"], ec=C["green_d"])
text(26.0, 22.9, "Cross-Attention", fs=9.8, fw="bold")
text(26.0, 19.9, "read multi-scale", fs=7.5)
text(26.0, 17.5, "encoder features", fs=7.5)
path([(10.0,38.5),(18.0,38.5),(18.0,23.0),(19.8,23.0)], color=C["green_d"])

arrow(32.2, 20.5, 35.0, 20.5)
# query set block
rounded(35.0, 13.2, 14.2, 14.7, C["blue"], ec=C["blue_d"])
text(42.1, 25.2, "Query features", fs=9.8, fw="bold")
query_grid(36.3, 22.0, cols=10, rows=3, size=0.62, gap=0.28)
text(42.1, 15.2, "300 × 256", fs=7.5, color=C["muted"])

# Pool split
arrow(49.4, 20.5, 52.0, 20.5)
rounded(52.0, 13.2, 11.8, 14.7, C["gray"], ec="#8296A7")
text(57.9, 25.2, "Attention Pool", fs=9.5, fw="bold")
text(57.9, 21.8, "summarize queries", fs=7.3)
text(57.9, 18.8, "positive / negative", fs=7.3)
text(57.9, 15.7, "two learned routes", fs=7.3, color=C["muted"])

# Pos/neg tokens visually
path([(63.9,22.3),(66.0,22.3),(66.0,31.2),(68.0,31.2)], color=C["green_d"])
path([(63.9,18.7),(66.0,18.7),(66.0,10.4),(68.0,10.4)], color=C["red_d"])
rounded(68.0, 27.4, 14.5, 8.0, C["green"], ec=C["green_d"])
text(75.25, 33.3, "Positive tokens", fs=8.8, fw="bold", color="#267548")
token_row(69.5, 29.2, "#8BD4A6", C["green_d"], n=8, size=0.92, gap=0.34)
text(75.25, 27.9, "8 tokens", fs=6.8, color=C["muted"])

rounded(68.0, 6.6, 14.5, 8.0, C["red"], ec=C["red_d"])
text(75.25, 12.5, "Negative tokens", fs=8.8, fw="bold", color="#A84349")
token_row(69.5, 8.4, "#F19CA0", C["red_d"], n=8, size=0.92, gap=0.34)
text(75.25, 7.1, "8 tokens", fs=6.8, color=C["muted"])

# Differential read
path([(82.7,31.2),(85.0,31.2),(85.0,23.0),(86.7,23.0)], color=C["green_d"])
path([(82.7,10.4),(85.0,10.4),(85.0,18.0),(86.7,18.0)], color=C["red_d"])
path([(49.2,20.5),(50.5,20.5),(50.5,5.0),(84.5,5.0),(84.5,20.5),(86.7,20.5)], color=C["blue_d"])
rounded(86.7, 14.5, 9.2, 12.0, C["purple"], ec=C["purple_d"])
text(91.3, 23.4, "Differential", fs=9.3, fw="bold")
text(91.3, 20.5, "8-head read", fs=7.5)
text(91.3, 18.0, "positive − λ·negative", fs=6.8)
text(91.3, 15.7, "+ residual", fs=7.3)

# FFN and classifier continuation
arrow(91.3, 14.3, 91.3, 11.5)
rounded(87.5, 6.2, 7.6, 5.2, C["orange"], ec=C["orange_d"])
text(91.3, 8.8, "FFN", fs=8.7, fw="bold")
arrow(95.2, 8.8, 97.0, 8.8)
text(97.4, 8.8, "VPE\nfeatures", fs=7.2, fw="bold", ha="left")

# Step numbers
steps = [(3.0,20.5,"1"),(19.1,20.5,"2"),(34.1,20.5,"3"),(51.1,20.5,"4"),(67.1,20.5,"5"),(85.8,20.5,"6")]
for x,y,n in steps:
    ax.add_patch(Circle((x,y),0.85,fc="#243F55",ec="white",lw=1,zorder=8)); text(x,y,n,fs=7,fw="bold",color="white",z=9)

# Footer
ax.add_line(Line2D([2,98],[1.6,1.6],color="#D8E1E8",lw=0.9))
text(2,0.55,"Blue: box queries    Green: positive tokens    Red: negative tokens",fs=7.4,color=C["muted"],ha="left")
text(98,0.55,"Current active model",fs=7.4,color=C["muted"],ha="right")

for ext in ("png","svg","pdf"):
    fig.savefig(OUT / f"current_rlccd_architecture_v3.{ext}", bbox_inches="tight",
                facecolor="white", dpi=300 if ext=="png" else None)
plt.close(fig)
