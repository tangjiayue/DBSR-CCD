#!/usr/bin/env python3
"""以 DDOD 风格绘制更清晰的分类/回归梯度响应图。"""
import argparse
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
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from plot_branch_gradcam import load_model, run_one, read_samples, anns_for_image
from torchvision.transforms import functional as TF

CLS = "#D55E00"       # 橙红：分类关注区域
REG = "#0072B2"       # 蓝色：回归关注区域
OVERLAP = "#7A3E9D"   # 紫色：二者重叠


def top_mask(cam, fraction=0.15):
    return cam >= np.quantile(cam, 1.0 - fraction)


def draw_boxes(ax, anns, info):
    for ann in anns:
        x, y, w, h = ann["bbox"]
        sx, sy = 640 / info["width"], 640 / info["height"]
        rect = patches.Rectangle((x*sx, y*sy), w*sx, h*sy, fill=False, edgecolor="#00E676", linewidth=1.8)
        ax.add_patch(rect)


def show_response(ax, image, cam, color, title, anns, info):
    gray = np.asarray(image.convert("L").resize((640, 640)))
    cmap = mcolors.LinearSegmentedColormap.from_list("response", ["#202020", color])
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.imshow(cam, cmap=cmap, vmin=0, vmax=1, alpha=0.86)
    mask = top_mask(cam)
    # 仅画高响应边界，避免红蓝透明叠加造成“彩色糊图”。
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.15)
    draw_boxes(ax, anns, info)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def show_coupling(ax, image, cls_cam, reg_cam, title, overlap, anns, info):
    a, b = top_mask(cls_cam), top_mask(reg_cam)
    state = a.astype(np.uint8) + 2 * b.astype(np.uint8)
    # 0=背景，1=分类独占，2=回归独占，3=重叠。
    cmap = mcolors.ListedColormap(["#E5E5E5", CLS, REG, OVERLAP])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(image.resize((640, 640)).convert("L"), cmap="gray", vmin=0, vmax=255)
    ax.imshow(state, cmap=cmap, norm=norm, alpha=0.82)
    draw_boxes(ax, anns, info)
    ax.set_title(f"{title}\nTop-15% overlap={overlap:.3f}", fontsize=10.5, fontweight="bold")
    ax.axis("off")


def main(args):
    device = torch.device(args.device)
    samples = read_samples(args.ann_file, args.image_root, args.result_json, args.num_samples, args.image_ids)
    if not samples:
        raise RuntimeError("没有找到指定图像")

    results = {}
    for name, cfg, ckpt, is_baseline in [
        ("D-FINE", args.baseline_config, args.baseline_checkpoint, True),
        ("DBSR-CCD", args.dbsr_config, args.dbsr_checkpoint, False),
    ]:
        model = load_model(cfg, ckpt, device, baseline=is_baseline)
        for i, (_iid, path, _info) in enumerate(samples):
            image = Image.open(path).convert("RGB")
            inp = TF.resize(TF.to_tensor(image), [640, 640]).unsqueeze(0).to(device)
            with torch.no_grad():
                results[(name, i)] = run_one(model, inp, baseline=is_baseline)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(samples), 7, figsize=(17.5, 4.1 * len(samples)), squeeze=False)
    headers = ["Input / GT", "D-FINE\nclassification", "D-FINE\nregression", "D-FINE\ncoupling", "DBSR-CCD\nclassification", "DBSR-CCD\nregression", "DBSR-CCD\ncoupling"]
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    summaries = []
    for row, (iid, path, info) in enumerate(samples):
        image = Image.open(path).convert("RGB")
        base = image.resize((640, 640))
        disp = base.copy()
        draw = ImageDraw.Draw(disp)
        for ann in anns_for_image(args.ann_file, iid):
            x, y, w, h = ann["bbox"]
            sx, sy = 640 / info["width"], 640 / info["height"]
            draw.rectangle([x*sx, y*sy, (x+w)*sx, (y+h)*sy], outline="#00A65A", width=2)
        axes[row, 0].imshow(disp); axes[row, 0].axis("off"); axes[row, 0].set_ylabel(f"image_id={iid}", fontsize=10)
        dc, dr, dq, ds, dov = results[("D-FINE", row)]
        mc, mr, mq, ms, mov = results[("DBSR-CCD", row)]
        anns = anns_for_image(args.ann_file, iid)
        show_response(axes[row, 1], base, dc, CLS, f"q={dq}, score={ds:.2f}", anns, info)
        show_response(axes[row, 2], base, dr, REG, "regression", anns, info)
        show_coupling(axes[row, 3], base, dc, dr, "D-FINE", dov, anns, info)
        show_response(axes[row, 4], base, mc, CLS, f"q={mq}, score={ms:.2f}", anns, info)
        show_response(axes[row, 5], base, mr, REG, "regression", anns, info)
        show_coupling(axes[row, 6], base, mc, mr, "DBSR-CCD", mov, anns, info)
        summaries.append({"image_id": iid, "file_name": info["file_name"], "dfine_overlap_top15": dov, "dbsr_overlap_top15": mov})

    fig.suptitle("Spatial feature coupling visualized by backward gradients on FPN features", fontsize=16, y=0.995)
    fig.text(0.5, 0.006, "Orange = classification response   Blue = regression response   Purple = overlap", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    import json
    out.with_suffix(".json").write_text(json.dumps({"method": "DDOD-style Grad-CAM response; top-15% masks", "samples": summaries}, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(out), "samples": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", required=True); p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--dbsr-config", required=True); p.add_argument("--dbsr-checkpoint", required=True)
    p.add_argument("--ann-file", required=True); p.add_argument("--image-root", required=True); p.add_argument("--result-json", required=True)
    p.add_argument("--output", required=True); p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--image-ids", nargs="*", type=int, default=None); p.add_argument("--device", default="cpu")
    main(p.parse_args())
