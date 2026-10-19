#!/usr/bin/env python3
"""从原图中抽取与分类-定位耦合相关的典型错误样本。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DFINE = Path("/root/userfolder/Projects/DBSR-CCD/output/dbsr_hgnetv2_m_ccd/d-fine/ccd-ComparisonDetectorDataset/detection_results/detection_results_epoch-1_20261016_103050.json")
DBSR = Path("/root/userfolder/Projects/DBSR-CCD/output/dbsr_hgnetv2_m_cdetectord/self_attn/加vpe加fusion/token8/0/detection_results/detection_results_epoch-1_20261016_101753.json")
DATASET = Path("/root/userfolder/Dataset/ObjectDetection/ComparisonDetectorDataset")
OUT = Path("/root/userfolder/Projects/DBSR-CCD/output/metric_analysis")
T = 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))["detections"]


def merge(record):
    groups = {}
    for pred in record.get("predictions", []):
        key = tuple(float(value) for value in pred["bbox"])
        current = groups.get(key)
        if current is None or float(pred["score"]) > current["score"]:
            groups[key] = {
                "bbox": pred["bbox"],
                "label": int(pred["label"]),
                "score": float(pred["score"]),
            }
    return list(groups.values())


def quality(pred, ground_truths):
    return max((iou(pred["bbox"], gt["bbox"]) for gt in ground_truths), default=0.0)


def matches(record, predictions):
    candidates = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(record["ground_truths"]):
            overlap = iou(pred["bbox"], gt["bbox"])
            if overlap >= T:
                candidates.append((overlap, pi, gi))
    candidates.sort(reverse=True)
    used_predictions, used_gt = set(), set()
    output = []
    for overlap, pi, gi in candidates:
        if pi in used_predictions or gi in used_gt:
            continue
        used_predictions.add(pi)
        used_gt.add(gi)
        output.append((gi, pi, overlap))
    return output


def nearest(pred, predictions):
    return max(predictions, key=lambda item: iou(pred["bbox"], item["bbox"]), default=None)


def select_cases(dfine, dbsr):
    background = None
    for image_index, record in enumerate(dfine):
        predictions = merge(record)
        for pred in predictions:
            q = quality(pred, record["ground_truths"])
            if q < T and (background is None or pred["score"] > background["pred"]["score"]):
                background = {"kind": "Background contamination", "image_index": image_index, "pred": pred, "quality": q}

    correction = None
    for image_index, (rb, rm) in enumerate(zip(dfine, dbsr)):
        pb, pm = merge(rb), merge(rm)
        model_matches = {gi: (pi, overlap) for gi, pi, overlap in matches(rm, pm)}
        for gi, pi, overlap in matches(rb, pb):
            gt_label = int(rb["ground_truths"][gi]["label"])
            if int(pb[pi]["label"]) == gt_label:
                continue
            if gi not in model_matches:
                continue
            model_pi, model_overlap = model_matches[gi]
            if int(pm[model_pi]["label"]) == int(rm["ground_truths"][gi]["label"]):
                candidate = {
                    "kind": "Classification correction",
                    "image_index": image_index,
                    "gt_index": gi,
                    "baseline": pb[pi],
                    "dbsr": pm[model_pi],
                    "baseline_iou": overlap,
                    "dbsr_iou": model_overlap,
                }
                if correction is None or candidate["baseline"]["score"] > correction["baseline"]["score"]:
                    correction = candidate

    low_confidence = None
    for image_index, record in enumerate(dfine):
        for pred in merge(record):
            q = quality(pred, record["ground_truths"])
            if q >= T and pred["score"] < 0.5:
                candidate = {"kind": "Good localization, low confidence", "image_index": image_index, "pred": pred, "quality": q}
                if low_confidence is None or (candidate["quality"] - candidate["pred"]["score"]) > (low_confidence["quality"] - low_confidence["pred"]["score"]):
                    low_confidence = candidate
    return [background, correction, low_confidence]


def font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_panel(path, record, method, prediction, target_gt_index=None, subtitle=""):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    gt_font = font(max(14, round(image.width / 110)))
    # 绿色框表示真实目标；彩色框表示待分析的预测框。
    for index, gt in enumerate(record["ground_truths"]):
        box = [round(value) for value in gt["bbox"]]
        color = "#18A558" if target_gt_index is None or index == target_gt_index else "#77B98A"
        width = 5 if target_gt_index is not None and index == target_gt_index else 2
        draw.rectangle(box, outline=color, width=width)
    if prediction is not None:
        box = [round(value) for value in prediction["bbox"]]
        q = quality(prediction, record["ground_truths"])
        color = "#E45756" if q < T or (target_gt_index is not None and prediction["label"] != record["ground_truths"][target_gt_index]["label"]) else "#247BA0"
        draw.rectangle(box, outline=color, width=6)
        label = f"{method} | cls={prediction['label']} | score={prediction['score']:.3f} | IoU={q:.3f}"
        draw.rectangle((10, 10, min(image.width - 10, 10 + max(420, len(label) * 10)), 45), fill=(20, 20, 20))
        draw.text((18, 17), label, fill="white", font=gt_font)
    if subtitle:
        draw.rectangle((10, image.height - 42, min(image.width - 10, 10 + max(420, len(subtitle) * 9)), image.height - 10), fill=(20, 20, 20))
        draw.text((18, image.height - 36), subtitle, fill="white", font=gt_font)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dfine, dbsr = load(DFINE), load(DBSR)
    image_records = json.loads((DATASET / "test.json").read_text(encoding="utf-8"))["images"]
    cases = select_cases(dfine, dbsr)
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    report = []
    for row, case in enumerate(cases):
        if case is None:
            for col in range(2):
                axes[row, col].axis("off")
            continue
        image_index = case["image_index"]
        image_path = DATASET / "test" / image_records[image_index]["file_name"]
        rb, rm = dfine[image_index], dbsr[image_index]
        if case["kind"] == "Classification correction":
            gi = case["gt_index"]
            baseline_pred = case["baseline"]
            dbsr_pred = case["dbsr"]
            title_left = f"{case['kind']} — D-FINE"
            title_right = f"{case['kind']} — DBSR-CCD"
            left = draw_panel(image_path, rb, "D-FINE", baseline_pred, gi, f"baseline wrong class | GT={rb['ground_truths'][gi]['label']}")
            right = draw_panel(image_path, rm, "DBSR-CCD", dbsr_pred, gi, f"refined correct class | GT={rm['ground_truths'][gi]['label']}")
            report.append({"kind": case["kind"], "image_index": image_index, "file_name": image_records[image_index]["file_name"], "gt_index": gi, "baseline": baseline_pred, "dbsr": dbsr_pred, "baseline_iou": case["baseline_iou"], "dbsr_iou": case["dbsr_iou"]})
        else:
            baseline_pred = case["pred"]
            dbsr_pred = nearest(baseline_pred, merge(rm))
            left = draw_panel(image_path, rb, "D-FINE", baseline_pred, None, f"{case['kind']} | baseline")
            right = draw_panel(image_path, rm, "DBSR-CCD", dbsr_pred, None, f"nearest spatial candidate | DBSR")
            report.append({"kind": case["kind"], "image_index": image_index, "file_name": image_records[image_index]["file_name"], "baseline": baseline_pred, "baseline_iou": case["quality"], "dbsr": dbsr_pred, "dbsr_iou": quality(dbsr_pred, rm["ground_truths"]) if dbsr_pred else None})
        axes[row, 0].imshow(left)
        axes[row, 1].imshow(right)
        axes[row, 0].set_title(title_left if case["kind"] == "Classification correction" else f"{case['kind']} — D-FINE", fontsize=12, fontweight="bold")
        axes[row, 1].set_title(title_right if case["kind"] == "Classification correction" else f"{case['kind']} — DBSR-CCD", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.suptitle("Representative errors related to confidence–localization coupling", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "error_analysis_gallery.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "error_analysis_gallery.pdf", facecolor="white")
    plt.close(fig)
    (OUT / "error_analysis_gallery.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
