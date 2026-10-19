#!/usr/bin/env python3
"""按 Decoupled DETR Figure 1 风格绘制共享/解耦响应图。"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from plot_branch_gradcam import load_model, run_one, read_samples, anns_for_image, make_hook, fmap_from_grads

ORANGE = "#D55E00"
BLUE = "#0072B2"
PURPLE = "#7A3E9D"
GREEN = "#00E676"


def draw_boxes(ax, anns, info):
    sx, sy = 640 / info["width"], 640 / info["height"]
    for ann in anns:
        x, y, w, h = ann["bbox"]
        ax.add_patch(patches.Rectangle((x*sx, y*sy), w*sx, h*sy, fill=False, edgecolor=GREEN, linewidth=1.8))


def run_shared_decoder(model, image_tensor):
    """D-FINE 共享 decoder 响应：对共享 query 表征反向传播。"""
    storage = {}
    handle = model.encoder.register_forward_hook(make_hook(storage))
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        feats = model.backbone(image_tensor)
        enc = model.encoder(feats)
        out = model.decoder(enc)
        probs = out["pred_logits"].sigmoid()
        vals, _ = probs[0].max(dim=1)
        q = int(vals.argmax().item())
        # 共享 decoder query 的能量，避免人为选择分类或回归目标。
        target = out["output"][0, q].pow(2).mean()
        target.backward()
        cam = fmap_from_grads(storage["acts"]).cpu().numpy()
        score = float(vals[q].detach().cpu())
    handle.remove()
    return cam, q, score


def show_cam(ax, base, cam, color, title, anns, info):
    gray = np.asarray(base.convert("L").resize((640, 640)))
    cmap = mcolors.LinearSegmentedColormap.from_list("cam", ["#222222", color])
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(cam, cmap=cmap, vmin=0, vmax=1, alpha=0.86)
    mask = cam >= np.quantile(cam, 0.85)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)
    draw_boxes(ax, anns, info)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def show_overlap(ax, base, cls_cam, reg_cam, anns, info, overlap):
    cls = cls_cam >= np.quantile(cls_cam, 0.85)
    reg = reg_cam >= np.quantile(reg_cam, 0.85)
    state = cls.astype(np.uint8) + 2 * reg.astype(np.uint8)
    cmap = mcolors.ListedColormap(["#E5E5E5", ORANGE, BLUE, PURPLE])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(base.resize((640, 640)).convert("L"), cmap="gray", vmin=0, vmax=255)
    ax.imshow(state, cmap=cmap, norm=norm, alpha=0.82)
    draw_boxes(ax, anns, info)
    ax.set_title(f"DBSR overlap\nTop-15% IoU={overlap:.3f}", fontsize=10.5, fontweight="bold")
    ax.axis("off")


def main(args):
    device = torch.device(args.device)
    samples = read_samples(args.ann_file, args.image_root, args.result_json, args.num_samples, args.image_ids)
    if not samples:
        raise RuntimeError("没有找到指定样本")
    results = {}
    for name, cfg, ckpt, baseline in [("dfine", args.baseline_config, args.baseline_checkpoint, True), ("dbsr", args.dbsr_config, args.dbsr_checkpoint, False)]:
        model = load_model(cfg, ckpt, device, baseline=baseline)
        for i, (_iid, path, _info) in enumerate(samples):
            base = Image.open(path).convert("RGB")
            inp = TF.resize(TF.to_tensor(base), [640, 640]).unsqueeze(0).to(device)
            if baseline:
                results[(name, i)] = run_shared_decoder(model, inp)
            else:
                results[(name, i)] = run_one(model, inp, baseline=False)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(samples), 5, figsize=(13.5, 3.9 * len(samples)), squeeze=False)
    headers = ["Image / GT", "D-FINE\nshared decoder", "DBSR-CCD\nclassification", "DBSR-CCD\nlocalization", "DBSR-CCD\nspatial overlap"]
    for j, h in enumerate(headers): axes[0, j].set_title(h, fontsize=12, fontweight="bold", pad=10)
    summary = []
    for r, (iid, path, info) in enumerate(samples):
        base = Image.open(path).convert("RGB")
        anns = anns_for_image(args.ann_file, iid)
        disp = base.resize((640, 640)).copy()
        # input 列也画 GT 框
        from PIL import ImageDraw
        dr = ImageDraw.Draw(disp)
        sx, sy = 640 / info["width"], 640 / info["height"]
        for a in anns:
            x, y, w, h = a["bbox"]; dr.rectangle([x*sx, y*sy, (x+w)*sx, (y+h)*sy], outline=GREEN, width=2)
        axes[r, 0].imshow(disp); axes[r, 0].axis("off"); axes[r, 0].set_ylabel(f"image_id={iid}", fontsize=10)
        shared, q, score = results[("dfine", r)]
        cls, reg, cq, cs, ov = results[("dbsr", r)]
        show_cam(axes[r, 1], base, shared, "#555555", f"q={q}, score={score:.2f}", anns, info)
        show_cam(axes[r, 2], base, cls, ORANGE, f"q={cq}, score={cs:.2f}", anns, info)
        show_cam(axes[r, 3], base, reg, BLUE, "localization", anns, info)
        show_overlap(axes[r, 4], base, cls, reg, anns, info, ov)
        summary.append({"image_id": iid, "file_name": info["file_name"], "dfine_shared_query": q, "dbsr_query": cq, "dbsr_overlap_top15": ov})
    fig.suptitle("Shared decoder versus spatially decoupled classification/localization responses", fontsize=15, y=0.995)
    fig.text(0.5, 0.006, "Green box = GT   Orange = classification   Blue = localization   Purple = overlap", ha="center", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    out.with_suffix(".json").write_text(json.dumps({"method": "shared decoder vs DBSR branch Grad-CAM-style FPN response", "samples": summary}, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(out), "samples": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", required=True); p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--dbsr-config", required=True); p.add_argument("--dbsr-checkpoint", required=True)
    p.add_argument("--ann-file", required=True); p.add_argument("--image-root", required=True); p.add_argument("--result-json", required=True)
    p.add_argument("--output", required=True); p.add_argument("--num-samples", type=int, default=1); p.add_argument("--image-ids", nargs="*", type=int, default=None); p.add_argument("--device", default="cpu")
    main(p.parse_args())
