import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d



def box_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_confidence_groups(path, iou_threshold=0.5):
    with open(path, "r") as file:
        payload = json.load(file)

    matched_scores = []
    unmatched_scores = []
    raw_predictions = 0
    merged_predictions = 0
    total_gt = 0

    for image_record in payload["detections"]:
        ground_truths = image_record.get("ground_truths", [])
        total_gt += len(ground_truths)
        predictions = image_record.get("predictions", [])
        raw_predictions += len(predictions)

        # Merge repeated top-K class entries with exactly the same predicted box.
        # Keep the maximum class confidence as the query-level score.
        merged = {}
        for prediction in predictions:
            key = tuple(float(value) for value in prediction["bbox"])
            current = merged.get(key)
            if current is None or float(prediction["score"]) > current["score"]:
                merged[key] = {
                    "bbox": prediction["bbox"],
                    "score": float(prediction["score"]),
                }
        merged_predictions += len(merged)

        # Class-agnostic, descending-score, one-to-one spatial matching.
        ordered_predictions = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        assigned = [False] * len(ground_truths)
        for prediction in ordered_predictions:
            best_index = -1
            best_iou = iou_threshold
            for gt_index, ground_truth in enumerate(ground_truths):
                if assigned[gt_index]:
                    continue
                overlap = box_iou(prediction["bbox"], ground_truth["bbox"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_index >= 0:
                assigned[best_index] = True
                matched_scores.append(prediction["score"])
            else:
                unmatched_scores.append(prediction["score"])

    return {
        "matched": np.asarray(matched_scores, dtype=np.float64),
        "unmatched": np.asarray(unmatched_scores, dtype=np.float64),
        "metadata": payload.get("metadata", {}),
        "raw_predictions": raw_predictions,
        "merged_predictions": merged_predictions,
        "total_gt": total_gt,
    }


def density_curve(values, bins=500):
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=2.0, mode="nearest")
    bin_width = edges[1] - edges[0]
    density = smoothed / max(smoothed.sum() * bin_width, 1e-12)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers, density


def summary(values):
    return {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else 0.0,
        "fraction_ge_0.5": float(np.mean(values >= 0.5)) if values.size else 0.0,
        "fraction_ge_0.1": float(np.mean(values >= 0.1)) if values.size else 0.0,
    }


def draw_density_plot(dfine, dbsr, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    blue = "#1787A5"
    red = "#E84C4C"
    fill_blue = "#8FC9D8"
    fill_red = "#F3B1B1"
    edge = "#C7CDD3"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.25), gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.16, top=0.82, wspace=0.18)

    # Matched queries.
    ax = axes[0]
    x_d, y_d = density_curve(dfine["matched"])
    x_r, y_r = density_curve(dbsr["matched"])
    ax.fill_between(x_d, y_d, color=fill_blue, alpha=0.28)
    ax.fill_between(x_r, y_r, color=fill_red, alpha=0.23)
    ax.plot(x_d, y_d, color=blue, lw=2.4, label="D-FINE")
    ax.plot(x_r, y_r, color=red, lw=2.4, label="DBSR-CCD")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Final classification score")
    ax.set_ylabel("Probability density")
    ax.set_title("Spatially matched queries (IoU ≥ 0.50)", fontsize=11, fontweight="bold", pad=7)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ds, rs = summary(dfine["matched"]), summary(dbsr["matched"])
    ax.text(
        0.015, 0.98,
        f"D-FINE: n={ds['n']:,} | median={ds['median']:.3f} | P≥0.5: {100*ds['fraction_ge_0.5']:.1f}%\n"
        f"DBSR-CCD: n={rs['n']:,} | median={rs['median']:.3f} | P≥0.5: {100*rs['fraction_ge_0.5']:.1f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=edge, alpha=0.9),
    )

    # Unmatched queries and log-tail inset.
    ax = axes[1]
    x_d, y_d = density_curve(dfine["unmatched"])
    x_r, y_r = density_curve(dbsr["unmatched"])
    ax.fill_between(x_d, y_d, color=fill_blue, alpha=0.28)
    ax.fill_between(x_r, y_r, color=fill_red, alpha=0.23)
    ax.plot(x_d, y_d, color=blue, lw=2.4, label="D-FINE")
    ax.plot(x_r, y_r, color=red, lw=2.4, label="DBSR-CCD")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Final classification score")
    ax.set_ylabel("Probability density")
    ax.set_title("Unmatched queries (not assigned at IoU ≥ 0.50)", fontsize=11, fontweight="bold", pad=7)

    us, ur = summary(dfine["unmatched"]), summary(dbsr["unmatched"])
    ax.text(
        0.985, 0.10,
        f"D-FINE: n={us['n']:,} | median={us['median']:.3f} | P≥0.5: {100*us['fraction_ge_0.5']:.1f}%\n"
        f"DBSR-CCD: n={ur['n']:,} | median={ur['median']:.3f} | P≥0.5: {100*ur['fraction_ge_0.5']:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=edge, alpha=0.9),
    )

    inset = ax.inset_axes([0.50, 0.56, 0.46, 0.38])
    inset.plot(x_d[x_d >= 0.1], y_d[x_d >= 0.1], color=blue, lw=1.35)
    inset.plot(x_r[x_r >= 0.1], y_r[x_r >= 0.1], color=red, lw=1.35)
    inset.set_yscale("log")
    inset.set_xlim(0.1, 1.0)
    inset.set_ylim(1e-8, max(float(y_d.max()), float(y_r.max()), 1e-7))
    inset.set_title("High-score tail (score ≥ 0.1)", fontsize=7.3, pad=2)
    inset.tick_params(axis="both", labelsize=6)
    inset.grid(True, which="both", alpha=0.2, linewidth=0.45)
    for spine in inset.spines.values():
        spine.set_color("#9CA8B2")
        spine.set_linewidth(0.7)

    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfine", required=True, type=Path)
    parser.add_argument("--dbsr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dfine = load_confidence_groups(args.dfine)
    dbsr = load_confidence_groups(args.dbsr)
    draw_density_plot(dfine, dbsr, args.output)

    result = {
        "iou_threshold": 0.5,
        "matching": "class-agnostic, descending-score, one-to-one GT assignment",
        "merge": "exact duplicate predicted boxes, retaining maximum class score",
        "dfine": {
            "path": str(args.dfine),
            "raw_predictions": dfine["raw_predictions"],
            "merged_predictions": dfine["merged_predictions"],
            "total_gt": dfine["total_gt"],
            "matched": summary(dfine["matched"]),
            "unmatched": summary(dfine["unmatched"]),
        },
        "dbsr_ccd": {
            "path": str(args.dbsr),
            "raw_predictions": dbsr["raw_predictions"],
            "merged_predictions": dbsr["merged_predictions"],
            "total_gt": dbsr["total_gt"],
            "matched": summary(dbsr["matched"]),
            "unmatched": summary(dbsr["unmatched"]),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
