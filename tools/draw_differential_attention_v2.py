from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
fontManager.addfont(FONT)
FONT_PROP = FontProperties(fname=FONT)

plt.rcParams.update({
    "font.family": ["DejaVu Sans", FONT_PROP.get_name()],
    "font.size": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

C = {
    "ink": "#173042",
    "muted": "#607486",
    "line": "#46677E",
    "panel": "#FAFCFE",
    "blue": "#E2F1FC",
    "blue_d": "#4F94C4",
    "blue_q": "#79B8E3",
    "green": "#E1F5E9",
    "green_d": "#42A36A",
    "green_t": "#86D29F",
    "red": "#FCE7E8",
    "red_d": "#D66168",
    "red_t": "#F09A9F",
    "purple": "#EEE7FC",
    "purple_d": "#8065B3",
    "orange": "#FFF0D8",
    "orange_d": "#D58A2F",
    "gray": "#F2F5F8",
    "gray_d": "#7C91A2",
    "white": "#FFFFFF",
}

fig = plt.figure(figsize=(19, 10.8), dpi=180, facecolor="white")
ax = fig.add_axes([0.025, 0.035, 0.95, 0.93])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def rounded(x, y, w, h, fc, ec=None, lw=1.25, radius=0.9, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or C["line"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(x, y, value, size=9, weight="normal", color=None,
         align="center", valign="center", z=7):
    ax.text(
        x, y, value,
        fontsize=size,
        fontweight=weight,
        color=color or C["ink"],
        ha=align,
        va=valign,
        zorder=z,
    )


def arrow(x1, y1, x2, y2, color=None, width=1.65, style="-", z=5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color or C["line"],
            "lw": width,
            "linestyle": style,
            "mutation_scale": 12,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=z,
    )


def line(x1, y1, x2, y2, color=None, width=1.35, style="-", z=4):
    ax.add_line(Line2D(
        [x1, x2], [y1, y2],
        color=color or C["line"],
        lw=width,
        ls=style,
        zorder=z,
    ))


def path(points, color=None, width=1.5, style="-"):
    for start, end in zip(points[:-2], points[1:-1]):
        line(start[0], start[1], end[0], end[1], color, width, style)
    start, end = points[-2], points[-1]
    arrow(start[0], start[1], end[0], end[1], color, width, style)


def query_grid(x, y, columns=12, rows=3, size=0.62, gap=0.25):
    for row in range(rows):
        for column in range(columns):
            ax.add_patch(FancyBboxPatch(
                (x + column * (size + gap), y - row * (size + gap)),
                size, size,
                boxstyle="round,pad=0.02,rounding_size=.10",
                fc=C["blue_q"], ec=C["blue_d"], lw=0.5, zorder=7,
            ))


def token_row(x, y, fill, edge, count=8, size=0.92, gap=0.30):
    for index in range(count):
        ax.add_patch(FancyBboxPatch(
            (x + index * (size + gap), y), size, size,
            boxstyle="round,pad=0.03,rounding_size=.17",
            fc=fill, ec=edge, lw=0.75, zorder=7,
        ))


def step(x, y, number):
    ax.add_patch(Circle((x, y), 0.82, fc="#243F55", ec="white", lw=1, zorder=10))
    text(x, y, str(number), size=7.2, weight="bold", color="white", z=11)


# Title
text(2, 97.3, "差分框注意力结构", size=19, weight="bold", align="left")
text(
    2, 94.5,
    "从归一化框特征中生成正、负语义 Token，并通过逐头差分精炼框语义",
    size=9.7, color=C["muted"], align="left",
)
line(2, 92.8, 98, 92.8, color="#D7E1E8", width=1.0)

# Stage I
rounded(1.8, 55.0, 96.2, 35.5, C["panel"], ec="#CAD7E0", lw=1.0, radius=1.25, z=0)
text(3.2, 87.4, "A", size=15, weight="bold", color="#35698F")
text(5.2, 87.4, "正、负语义 Token 生成", size=12, weight="bold", align="left")

rounded(4.0, 65.0, 16.0, 16.0, C["blue"], ec=C["blue_d"])
text(12.0, 78.3, "归一化框特征", size=10.5, weight="bold")
query_grid(5.2, 74.9, columns=12, rows=3, size=0.64, gap=0.27)
text(12.0, 67.2, "Q ∈ R^(B×300×256)", size=8.0, color=C["muted"])
text(12.0, 63.2, "Cross-Attention 后经 norm_self", size=7.6, color=C["muted"])

path([(20.2, 73.7), (23.0, 78.5), (26.0, 78.5)], color=C["green_d"])
path([(20.2, 72.4), (23.0, 64.5), (26.0, 64.5)], color=C["red_d"])

rounded(26.0, 74.0, 20.0, 9.0, C["green"], ec=C["green_d"])
text(36.0, 80.8, "正向注意力池化", size=10.2, weight="bold", color="#267548")
text(36.0, 77.8, "Linear(256→8) → Softmax(300)", size=8.0)
text(36.0, 75.5, "对全部框特征加权汇聚", size=7.4, color=C["muted"])

rounded(26.0, 60.0, 20.0, 9.0, C["red"], ec=C["red_d"])
text(36.0, 66.8, "负向注意力池化", size=10.2, weight="bold", color="#A84349")
text(36.0, 63.8, "Linear(256→8) → Softmax(300)", size=8.0)
text(36.0, 61.5, "对全部框特征加权汇聚", size=7.4, color=C["muted"])

arrow(46.2, 78.5, 50.0, 78.5, color=C["green_d"])
arrow(46.2, 64.5, 50.0, 64.5, color=C["red_d"])

rounded(50.0, 74.0, 18.0, 9.0, C["green"], ec=C["green_d"])
text(59.0, 81.0, "8 个正 Token", size=10.2, weight="bold", color="#267548")
token_row(52.2, 77.2, C["green_t"], C["green_d"], size=0.90, gap=0.27)
text(59.0, 75.3, "T+ ∈ R^(B×8×256)", size=7.5, color=C["muted"])

rounded(50.0, 60.0, 18.0, 9.0, C["red"], ec=C["red_d"])
text(59.0, 67.0, "8 个负 Token", size=10.2, weight="bold", color="#A84349")
token_row(52.2, 63.2, C["red_t"], C["red_d"], size=0.90, gap=0.27)
text(59.0, 61.3, "T− ∈ R^(B×8×256)", size=7.5, color=C["muted"])

rounded(73.0, 65.0, 20.5, 16.0, C["gray"], ec=C["gray_d"])
text(83.25, 78.4, "划分为 8 个注意力头", size=10.4, weight="bold")
text(83.25, 74.9, "256 = 8 × 32", size=9.1, weight="bold", color=C["purple_d"])
text(83.25, 71.7, "Qh:  [B, 8, 300, 32]", size=8.0)
text(83.25, 68.9, "T+h: [B, 8, 8, 32]", size=8.0, color="#267548")
text(83.25, 66.3, "T−h: [B, 8, 8, 32]", size=8.0, color="#A84349")

path([(68.2, 78.5), (70.5, 78.5), (70.5, 75.5), (72.8, 75.5)], color=C["green_d"])
path([(68.2, 64.5), (70.5, 64.5), (70.5, 69.0), (72.8, 69.0)], color=C["red_d"])
path([(20.2, 70.8), (22.0, 70.8), (22.0, 57.3), (83.2, 57.3), (83.2, 64.8)], color=C["blue_d"], style="--")

for x, y, number in [(3.0, 83.5, 1), (24.8, 85.0, 2), (48.8, 85.0, 3), (71.8, 83.5, 4)]:
    step(x, y, number)

# Stage II
rounded(1.8, 3.2, 96.2, 48.8, "#FCFBFF", ec="#D7CEE8", lw=1.0, radius=1.25, z=0)
text(3.2, 49.0, "B", size=15, weight="bold", color="#7355A5")
text(5.2, 49.0, "多头正负读取、逐头差分与残差更新", size=12, weight="bold", align="left")

rounded(4.0, 25.8, 13.0, 13.5, C["blue"], ec=C["blue_d"])
text(10.5, 36.5, "框 Query Heads", size=9.7, weight="bold")
query_grid(5.1, 33.5, columns=8, rows=3, size=0.67, gap=0.28)
text(10.5, 28.1, "8 heads × 32 dim", size=7.6, color=C["muted"])

path([(17.2, 33.5), (20.0, 39.0), (22.5, 39.0)], color=C["green_d"])
path([(17.2, 31.8), (20.0, 23.4), (22.5, 23.4)], color=C["red_d"])

rounded(22.5, 34.5, 20.0, 9.0, C["green"], ec=C["green_d"])
text(32.5, 41.1, "读取正 Token", size=9.9, weight="bold", color="#267548")
text(32.5, 38.5, "Softmax(Qh T+hᵀ / √32)", size=8.0)
text(32.5, 36.2, "加权读取 T+h → O+", size=7.6, color=C["muted"])

rounded(22.5, 18.9, 20.0, 9.0, C["red"], ec=C["red_d"])
text(32.5, 25.5, "读取负 Token", size=9.9, weight="bold", color="#A84349")
text(32.5, 22.9, "Softmax(Qh T−hᵀ / √32)", size=8.0)
text(32.5, 20.6, "加权读取 T−h → O−", size=7.6, color=C["muted"])

arrow(42.7, 39.0, 46.5, 39.0, color=C["green_d"])
arrow(42.7, 23.4, 46.5, 23.4, color=C["red_d"])

rounded(46.5, 34.5, 13.5, 9.0, C["green"], ec=C["green_d"])
text(53.25, 40.4, "正向语义 O+", size=9.5, weight="bold", color="#267548")
text(53.25, 37.3, "[B,8,300,32]", size=7.6, color=C["muted"])

rounded(46.5, 18.9, 13.5, 9.0, C["red"], ec=C["red_d"])
text(53.25, 24.8, "负向语义 O−", size=9.5, weight="bold", color="#A84349")
text(53.25, 21.7, "[B,8,300,32]", size=7.6, color=C["muted"])

path([(60.2, 39.0), (62.0, 39.0), (62.0, 34.4), (64.0, 34.4)], color=C["green_d"])
path([(60.2, 23.4), (62.0, 23.4), (62.0, 29.8), (64.0, 29.8)], color=C["red_d"])

rounded(64.0, 26.8, 14.5, 11.7, C["purple"], ec=C["purple_d"])
text(71.25, 35.8, "逐头正负差分", size=10.1, weight="bold")
text(71.25, 32.5, "Oh = O+ − λh · O−", size=9.0, weight="bold")
text(71.25, 29.4, "8 个独立可学习 λh", size=7.5, color=C["muted"])

arrow(78.7, 32.7, 81.0, 32.7)
rounded(81.0, 26.8, 8.0, 11.7, C["gray"], ec=C["gray_d"])
text(85.0, 35.8, "合并 8 头", size=8.9, weight="bold")
text(85.0, 32.6, "→ 256维", size=8.0)
text(85.0, 29.4, "RMSNorm", size=7.7, color=C["muted"])

arrow(89.2, 32.7, 91.2, 32.7)
rounded(91.2, 26.8, 6.0, 11.7, C["orange"], ec=C["orange_d"])
text(94.2, 35.7, "缩放", size=8.8, weight="bold")
text(94.2, 32.5, "tanh(s)", size=8.1, weight="bold")
text(94.2, 29.4, "可学习", size=7.3, color=C["muted"])

# External residual
line(10.5, 25.5, 10.5, 10.5, color=C["blue_d"], width=1.5, style="--")
line(10.5, 10.5, 94.2, 10.5, color=C["blue_d"], width=1.5, style="--")
arrow(94.2, 10.5, 94.2, 25.8, color=C["blue_d"], width=1.5, style="--")
text(12.0, 12.0, "原始 Query 外部残差", size=7.8, color=C["blue_d"], align="left")

arrow(97.3, 32.7, 98.4, 32.7, color=C["orange_d"])
ax.add_patch(Circle((98.4, 32.7), 0.95, fc=C["white"], ec=C["line"], lw=1.1, zorder=8))
text(98.4, 32.7, "+", size=10, weight="bold", z=9)
arrow(98.4, 31.5, 98.4, 17.0, color=C["line"], width=1.3)
text(96.8, 15.2, "精炼后的框语义", size=9.2, weight="bold", align="right")

for x, y, number in [(3.0, 42.0, 5), (21.3, 45.0, 6), (45.3, 45.0, 7), (62.8, 41.0, 8), (80.0, 41.0, 9)]:
    step(x, y, number)

line(2, 1.8, 98, 1.8, color="#D7E1E8", width=0.9)
text(
    2.2, 0.7,
    "蓝色：框 Query    绿色：正向语义路径    红色：负向语义路径    紫色：逐头差分",
    size=7.6, color=C["muted"], align="left",
)
text(97.8, 0.7, "当前 DifferentialBoxAttention 实现", size=7.6, color=C["muted"], align="right")

for extension in ("png", "svg", "pdf"):
    fig.savefig(
        OUT / f"differential_box_attention_v2.{extension}",
        bbox_inches="tight",
        facecolor="white",
        dpi=300 if extension == "png" else None,
    )

plt.close(fig)
