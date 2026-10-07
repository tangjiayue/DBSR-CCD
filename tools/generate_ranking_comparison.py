import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = [
    "normal", "ascus", "asch", "lsil", "hsil_scc_omn",
    "agc_adenocarcinoma_em", "vaginalis", "monilia",
    "dysbacteriosis_herpes_act", "ec",
]

SHORT_CLASS_NAMES = [
    "Normal", "ASC-US", "ASC-H", "LSIL", "HSIL",
    "AGC", "Vaginalis", "Monilia", "Dysb./Herpes", "EC",
]


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_records(path):
    with open(path, "r") as file:
        payload = json.load(file)
    return payload["detections"], payload.get("metadata", {})


def prepare(records):
    gt_by_image_class = {}
    predictions = []
    total_gt = 0
    for image_record in records:
        image_id = image_record["image_id"]
        for gt in image_record.get("ground_truths", []):
            label = int(gt["label"])
            gt_by_image_class.setdefault((image_id, label), []).append(gt["bbox"])
            total_gt += 1
        for pred in image_record.get("predictions", []):
            predictions.append({
                "image_id": image_id,
                "label": int(pred["label"]),
                "score": float(pred["score"]),
                "bbox": pred["bbox"],
            })
    return gt_by_image_class, predictions, total_gt


def match_predictions(gt_by_image_class, predictions, iou_threshold=0.5, label=None):
    if label is not None:
        predictions = [p for p in predictions if p["label"] == label]
    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)
    matched = {key: np.zeros(len(boxes), dtype=bool)
               for key, boxes in gt_by_image_class.items()
               if label is None or key[1] == label}
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)
    for index, pred in enumerate(predictions):
        key = (pred["image_id"], pred["label"])
        gt_boxes = gt_by_image_class.get(key, [])
        if not gt_boxes:
            fp[index] = 1.0
            continue
        best_iou = iou_threshold
        best_index = -1
        for gt_index, gt_box in enumerate(gt_boxes):
            if matched[key][gt_index]:
                continue
            overlap = iou_xyxy(pred["bbox"], gt_box)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index >= 0:
            matched[key][best_index] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0
    return predictions, tp, fp


def precision_recall(tp, fp, total_gt):
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall = cumulative_tp / max(total_gt, 1)
    return precision, recall


def ap_101(precision, recall):
    if len(precision) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = np.linspace(0.0, 1.0, 101)
    values = []
    for target in samples:
        valid = np.where(recall >= target)[0]
        values.append(float(envelope[valid[0]]) if len(valid) else 0.0)
    return float(np.mean(values))


def precision_at_recall(precision, recall, target):
    if len(precision) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    valid = np.where(recall >= target)[0]
    return float(envelope[valid[0]]) if len(valid) else 0.0


def compute_method(path):
    records, metadata = load_records(path)
    gt_by_image_class, predictions, total_gt = prepare(records)
    ordered, tp, fp = match_predictions(gt_by_image_class, predictions)
    precision, recall = precision_recall(tp, fp, total_gt)
    overall_ap = ap_101(precision, recall)

    class_ap = []
    class_curves = {}
    for label in range(len(CLASS_NAMES)):
        gt_count = sum(len(boxes) for (image_id, class_id), boxes in gt_by_image_class.items()
                       if class_id == label)
        class_ordered, class_tp, class_fp = match_predictions(
            gt_by_image_class, predictions, label=label
        )
        class_precision, class_recall = precision_recall(class_tp, class_fp, gt_count)
        class_ap.append(ap_101(class_precision, class_recall))
        class_curves[label] = (class_precision, class_recall, gt_count, len(class_ordered))

    return {
        "path": str(path),
        "metadata": metadata,
        "total_gt": total_gt,
        "pred_count": len(predictions),
        "precision": precision,
        "recall": recall,
        "ap50_micro": overall_ap,
        "class_ap50": np.asarray(class_ap),
        "class_curves": class_curves,
    }


def plot_results(dfine, dbsr, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    blue = "#2563EB"
    red = "#D9485F"
    dark = "#243447"
    class_colors = [red if value >= 0 else "#718096" for value in dbsr["class_ap50"] - dfine["class_ap50"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.19, top=0.88, wspace=0.30)

    # Left: micro ranking PR curve.
    ax = axes[0]
    ax.plot(dfine["recall"], dfine["precision"], color="#2AA6C5", lw=1.8, label="D-FINE")
    ax.plot(dbsr["recall"], dbsr["precision"], color="#D94B4F", lw=1.8, label="DBSR-CCD")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", labelpad=2)
    ax.set_ylabel("Macro precision", labelpad=2)
    ax.set_title("IoU=0.50 ranking curve", fontsize=10, pad=5)
    ax.legend(frameon=True, framealpha=0.90, edgecolor="#D8DDE3", fontsize=7.2,
              loc="upper right", borderpad=0.35, handlelength=1.5)

    # Middle: precision at fixed recalls.
    ax = axes[1]
    targets = np.asarray([0.50, 0.75, 0.90])
    dfine_values = np.asarray([precision_at_recall(dfine["precision"], dfine["recall"], target) for target in targets])
    dbsr_values = np.asarray([precision_at_recall(dbsr["precision"], dbsr["recall"], target) for target in targets])
    x = np.arange(len(targets))
    width = 0.34
    ax.bar(x - width / 2, dfine_values, width, color=blue, label="D-FINE")
    ax.bar(x + width / 2, dbsr_values, width, color=red, label="DBSR-CCD")
    for index, value in enumerate(dfine_values):
        ax.text(index - width / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color=blue)
    for index, value in enumerate(dbsr_values):
        ax.text(index + width / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color=red)
    ax.set_xticks(x, [f"R={value:.2f}" for value in targets])
    ax.set_ylim(0, max(1.0, float(max(dbsr_values.max(), dfine_values.max()) + 0.16)))
    ax.set_ylabel("Precision")
    ax.set_title("Precision at the same recall", fontsize=10, pad=5)
    ax.set_ylabel("Macro precision", labelpad=2)
    ax.legend(frameon=True, framealpha=0.90, edgecolor="#D8DDE3", fontsize=7.2,
              loc="upper right", borderpad=0.35, handlelength=1.5)

    # Right: per-class AP50 gain, matching the compact vertical-bar reference style.
    ax = axes[2]
    gains = (dbsr["class_ap50"] - dfine["class_ap50"]) * 100.0
    x = np.arange(len(SHORT_CLASS_NAMES))
    ax.bar(x, gains, color="#3FA06B", edgecolor="white", linewidth=0.55, width=0.70)
    for index, gain in enumerate(gains):
        ax.text(index, gain + 0.07, f"{gain:.1f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(0, color=dark, lw=0.8)
    ax.set_xticks(x, SHORT_CLASS_NAMES, rotation=55, ha="right", fontsize=6.4)
    ax.set_ylabel("AP50 gain (percentage points)", labelpad=2)
    ax.set_title("Class-wise ranking gain", fontsize=10, pad=5)
    ax.set_ylim(0, max(3.9, float(gains.max()) + 0.45))
    ax.tick_params(axis="y", labelsize=7)

    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    return targets, dfine_values, dbsr_values, gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfine", required=True, type=Path)
    parser.add_argument("--dbsr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dfine = compute_method(args.dfine)
    dbsr = compute_method(args.dbsr)
    targets, dfine_values, dbsr_values, gains = plot_results(dfine, dbsr, args.output)

    summary = {
        "iou_threshold": 0.5,
        "dfine": {
            "path": dfine["path"],
            "total_gt": dfine["total_gt"],
            "pred_count": dfine["pred_count"],
            "micro_ap50_101_point": dfine["ap50_micro"],
            "class_ap50_101_point": dfine["class_ap50"].tolist(),
            "precision_at_recall": dict(zip([str(x) for x in targets], dfine_values.tolist())),
        },
        "dbsr_ccd": {
            "path": dbsr["path"],
            "total_gt": dbsr["total_gt"],
            "pred_count": dbsr["pred_count"],
            "micro_ap50_101_point": dbsr["ap50_micro"],
            "class_ap50_101_point": dbsr["class_ap50"].tolist(),
            "precision_at_recall": dict(zip([str(x) for x in targets], dbsr_values.tolist())),
        },
        "class_names": CLASS_NAMES,
        "dbsr_minus_dfine_class_ap50_percentage_points": gains.tolist(),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
