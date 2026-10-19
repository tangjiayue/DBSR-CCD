#!/usr/bin/env python3
"""绘制分类置信度与定位质量的联合热力图。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DFINE = Path("/root/userfolder/Projects/DBSR-CCD/output/dbsr_hgnetv2_m_ccd/d-fine/ccd-ComparisonDetectorDataset/detection_results/detection_results_epoch-1_20261016_103050.json")
DBSR = Path("/root/userfolder/Projects/DBSR-CCD/output/dbsr_hgnetv2_m_cdetectord/self_attn/加vpe加fusion/token8/0/detection_results/detection_results_epoch-1_20261016_101753.json")
OUT = Path("/root/userfolder/Projects/DBSR-CCD/output/metric_analysis")
BINS = np.linspace(0.0, 1.0, 21)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def load_points(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores, qualities = [], []
    for record in payload["detections"]:
        # 后处理使用分类-Query 展平 Top-K；同一框的多类别条目合并为一个 Query。
        merged = {}
        for prediction in record.get("predictions", []):
            key = tuple(float(value) for value in prediction["bbox"])
            current = merged.get(key)
            if current is None or float(prediction["score"]) > current["score"]:
                merged[key] = {
                    "bbox": prediction["bbox"],
                    "score": float(prediction["score"]),
                }
        ground_truths = record.get("ground_truths", [])
        for prediction in merged.values():
            quality = max((iou(prediction["bbox"], gt["bbox"]) for gt in ground_truths), default=0.0)
            scores.append(prediction["score"])
            qualities.append(quality)
    return np.asarray(scores), np.asarray(qualities)


def build_hist(scores, qualities):
    counts, _, _ = np.histogram2d(qualities, scores, bins=(BINS, BINS))
    return counts / max(float(counts.sum()), 1.0) * 100.0


def stats(scores, qualities):
    high_conf_low_quality = (scores >= 0.5) & (qualities < 0.5)
    low_conf_high_quality = (scores < 0.5) & (qualities >= 0.5)
    corr = float(np.corrcoef(scores, qualities)[0, 1]) if len(scores) > 1 else 0.0
    return {
        "queries": int(len(scores)),
        "high_conf_low_iou_percent": float(high_conf_low_quality.mean() * 100.0),
        "low_conf_high_iou_percent": float(low_conf_high_quality.mean() * 100.0),
        "score_iou_pearson": corr,
        "median_score": float(np.median(scores)),
        "median_iou": float(np.median(qualities)),
    }


def annotate(ax, method, values):
    text = (
        f"N={values['queries']:,}\n"
        f"High confidence / low IoU: {values['high_conf_low_iou_percent']:.1f}%\n"
        f"Low confidence / high IoU: {values['low_conf_high_iou_percent']:.1f}%\n"
        f"Score–IoU Pearson: {values['score_iou_pearson']:.3f}"
    )
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#B8C0C8", "alpha": 0.9},
    )
    ax.set_title(method, fontsize=12, fontweight="bold")
    ax.set_xlabel("Max foreground classification confidence")
    ax.set_ylabel("Max box IoU (localization quality)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(color="white", alpha=0.18, linewidth=0.6)
    ax.axvline(0.5, color="#263238", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(0.5, color="#263238", linestyle="--", linewidth=0.8, alpha=0.8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dfine_scores, dfine_quality = load_points(DFINE)
    dbsr_scores, dbsr_quality = load_points(DBSR)
    dfine_hist = build_hist(dfine_scores, dfine_quality)
    dbsr_hist = build_hist(dbsr_scores, dbsr_quality)
    diff = dbsr_hist - dfine_hist
    dfine_stats = stats(dfine_scores, dfine_quality)
    dbsr_stats = stats(dbsr_scores, dbsr_quality)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    panels = [
        (dfine_hist, "D-FINE", dfine_stats, "magma", None),
        (dbsr_hist, "DBSR-CCD", dbsr_stats, "magma", None),
        (diff, "DBSR-CCD − D-FINE", None, "RdBu_r", max(float(np.abs(diff).max()), 0.5)),
    ]
    images = []
    for ax, (matrix, title, values, cmap, limit) in zip(axes, panels):
        if limit is None:
            image = ax.imshow(np.log10(matrix + 1e-3), origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap=cmap)
        else:
            image = ax.imshow(matrix, origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap=cmap, vmin=-limit, vmax=limit)
        images.append(image)
        if values is not None:
            annotate(ax, title, values)
        else:
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Max foreground classification confidence")
            ax.set_ylabel("Max box IoU (localization quality)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.axvline(0.5, color="#263238", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.axhline(0.5, color="#263238", linestyle="--", linewidth=0.8, alpha=0.8)
            ax.grid(color="white", alpha=0.18, linewidth=0.6)
    axes[0].text(0.5, 1.08, "Color: log10(Query fraction + 0.001)", transform=axes[0].transAxes, ha="center", fontsize=9)
    axes[2].text(0.5, 1.08, "Red: increased Query fraction; blue: decreased", transform=axes[2].transAxes, ha="center", fontsize=9)
    fig.suptitle("Classification confidence vs. localization quality", fontsize=15, fontweight="bold")
    cb1 = fig.colorbar(images[0], ax=axes[:2], shrink=0.88, pad=0.02)
    cb1.set_label("log10(Query fraction + 0.001)")
    cb2 = fig.colorbar(images[2], ax=axes[2], shrink=0.88, pad=0.02)
    cb2.set_label("Difference (percentage points)")
    png = OUT / "confidence_iou_coupling_heatmap.png"
    pdf = OUT / "confidence_iou_coupling_heatmap.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    summary = {
        "dfine": dfine_stats,
        "dbsr_ccd": dbsr_stats,
        "difference_percentage_points": {
            "high_conf_low_iou": dbsr_stats["high_conf_low_iou_percent"] - dfine_stats["high_conf_low_iou_percent"],
            "low_conf_high_iou": dbsr_stats["low_conf_high_iou_percent"] - dfine_stats["low_conf_high_iou_percent"],
            "score_iou_pearson": dbsr_stats["score_iou_pearson"] - dfine_stats["score_iou_pearson"],
        },
    }
    (OUT / "confidence_iou_coupling_heatmap.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
