#!/usr/bin/env python3
"""生成分类/回归分支梯度响应与耦合可视化。"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.core import YAMLConfig  # noqa: E402


def load_model(config_path, checkpoint_path, device, baseline=False):
    cfg = YAMLConfig(config_path)
    # ccd 配置的历史文件为 10 类，而当前 ComparisonDetectorDataset 检查点为 11 类。
    if baseline:
        cfg.yaml_cfg["num_classes"] = 11
    model = cfg.model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[{Path(checkpoint_path).name}] missing={len(missing)} (baseline classifier is bypassed)")
    if unexpected:
        print(f"[{Path(checkpoint_path).name}] unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def make_hook(storage):
    def hook(_module, _inputs, output):
        storage["acts"] = output
        for x in output:
            if torch.is_grad_enabled() and x.requires_grad:
                x.retain_grad()
    return hook


def fmap_from_grads(acts):
    maps = []
    for act in acts:
        if act.grad is None:
            continue
        # Grad-CAM 风格的 |d y / d A| * A，保留各 FPN 层的空间结构。
        w = act.grad.abs().mean(dim=(2, 3), keepdim=True)
        cam = (w * act.abs()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(640, 640), mode="bilinear", align_corners=False)
        maps.append(cam[0, 0])
    if not maps:
        return torch.zeros((640, 640), device=acts[0].device)
    out = torch.stack(maps).sum(0)
    out = out - out.min()
    out = out / (out.max() + 1e-8)
    return out.detach()


def run_one(model, image_tensor, baseline=False):
    """对同一个 query 分别反传分类分数和回归量。"""
    storage = {}
    handle = model.encoder.register_forward_hook(make_hook(storage))
    x = image_tensor.clone().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        if baseline:
            feats = model.backbone(x)
            enc = model.encoder(feats)
            out = model.decoder(enc)
        else:
            out, _ = model(x)
        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        probs = logits.sigmoid()
        flat = probs[0].max(dim=1)
        q = int(flat.values.argmax().item())
        cls_score = logits[0, q].sigmoid().max()
        # 对同一 query 的四个回归参数求和，得到回归分支响应。
        reg_score = boxes[0, q].sum()
        cls_score.backward(retain_graph=True)
        cls_map = fmap_from_grads(storage["acts"])
        model.zero_grad(set_to_none=True)
        for a in storage["acts"]:
            if a.grad is not None:
                a.grad.zero_()
        reg_score.backward()
        reg_map = fmap_from_grads(storage["acts"])
        score = float(cls_score.detach().cpu())
    handle.remove()
    # top-20% 显著区域 IoU，用于图中给出耦合程度。
    a = cls_map >= torch.quantile(cls_map, 0.80)
    b = reg_map >= torch.quantile(reg_map, 0.80)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    overlap = inter / max(union, 1)
    return cls_map.cpu().numpy(), reg_map.cpu().numpy(), q, score, overlap


def overlay(base, cmap_a, cmap_b):
    base = np.asarray(base).astype(np.float32) / 255.0
    # 红色=分类，蓝色=回归；重叠区域呈紫色。
    a = np.clip(cmap_a, 0, 1)[..., None]
    b = np.clip(cmap_b, 0, 1)[..., None]
    out = base * 0.45 + np.concatenate([a, np.zeros_like(a), b], axis=-1) * 0.55
    return np.clip(out, 0, 1)


def read_samples(ann_file, image_root, result_json, count, image_ids=None):
    anns = json.load(open(ann_file, "r"))
    # detection_results 的 image_id 是测试集顺序索引；标注文件中的 id 通常从 1 开始，不能直接混用。
    imgs = list(anns["images"])
    result = json.load(open(result_json, "r"))
    dets = result.get("detections", [])
    # 优先选有多个 GT 的样本，能同时展示前景和背景干扰；其余按 id 补齐。
    gt_count = {int(d["image_id"]): len(d.get("ground_truths", [])) for d in dets}
    if image_ids:
        ids = [int(i) for i in image_ids if 0 <= int(i) < len(imgs)]
    else:
        ids = sorted(range(len(imgs)), key=lambda i: (-gt_count.get(i, 0), i))[:count]
    out = []
    for iid in ids:
        info = imgs[iid]
        path = os.path.join(image_root, info["file_name"])
        if os.path.exists(path):
            out.append((iid, path, info))
    return out


def main(args):
    device = torch.device(args.device)
    samples = read_samples(args.ann_file, args.image_root, args.result_json, args.num_samples, args.image_ids)
    if not samples:
        raise RuntimeError("没有找到可用图像")

    rows = len(samples)
    fig, axes = plt.subplots(rows, 7, figsize=(18, 4.0 * rows), squeeze=False)
    titles = ["Input / GT", "D-FINE\nclassification", "D-FINE\nregression", "D-FINE\noverlap", "DBSR-CCD\nclassification", "DBSR-CCD\nregression", "DBSR-CCD\noverlap"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")
    summaries = [{"image_id": iid, "file_name": info["file_name"]} for iid, _path, info in samples]
    # 两个模型依次加载，避免在 2080 Ti 的有限空闲显存中同时驻留。
    model_results = {}
    for name, cfg_path, ckpt_path, is_baseline in [
        ("dfine", args.baseline_config, args.baseline_checkpoint, True),
        ("dbsr", args.dbsr_config, args.dbsr_checkpoint, False),
    ]:
        model = load_model(cfg_path, ckpt_path, device, baseline=is_baseline)
        for idx, (_iid, path, _info) in enumerate(samples):
            im = Image.open(path).convert("RGB")
            inp = TF.resize(TF.to_tensor(im), [640, 640]).unsqueeze(0).to(device)
            with torch.no_grad():
                model_results[(name, idx)] = run_one(model, inp, baseline=is_baseline)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for r, (iid, path, info) in enumerate(samples):
        im = Image.open(path).convert("RGB")
        dc, dr, dq, ds, do = model_results[("dfine", r)]
        mc, mr, mq, ms, mo = model_results[("dbsr", r)]
        # 第一列保留原图与 GT 框，便于读者核对响应区域。
        disp = im.copy()
        draw = ImageDraw.Draw(disp)
        for ann in anns_for_image(args.ann_file, info["id"]):
            x, y, w, h = ann["bbox"]
            sx, sy = disp.width / info["width"], disp.height / info["height"]
            draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline="#00ff55", width=max(2, disp.width // 500))
        axes[r, 0].imshow(disp); axes[r, 0].axis("off"); axes[r, 0].set_ylabel(f"image_id={iid}", fontsize=10)
        axes[r, 1].imshow(overlay(np.array(im.resize((640, 640))), dc, np.zeros_like(dc)), cmap="magma"); axes[r, 1].axis("off"); axes[r, 1].text(0.02, 0.97, f"q={dq}, s={ds:.2f}", color="white", transform=axes[r,1].transAxes, va="top", fontsize=9)
        axes[r, 2].imshow(overlay(np.array(im.resize((640, 640))), np.zeros_like(dr), dr)); axes[r, 2].axis("off")
        axes[r, 3].imshow(overlay(np.array(im.resize((640, 640))), dc, dr)); axes[r, 3].axis("off"); axes[r,3].text(0.02,0.97,f"top20% IoU={do:.3f}",color="white",transform=axes[r,3].transAxes,va="top",fontsize=9)
        axes[r, 4].imshow(overlay(np.array(im.resize((640, 640))), mc, np.zeros_like(mc))); axes[r, 4].axis("off"); axes[r, 4].text(0.02,0.97,f"q={mq}, s={ms:.2f}",color="white",transform=axes[r,4].transAxes,va="top",fontsize=9)
        axes[r, 5].imshow(overlay(np.array(im.resize((640, 640))), np.zeros_like(mr), mr)); axes[r, 5].axis("off")
        axes[r, 6].imshow(overlay(np.array(im.resize((640, 640))), mc, mr)); axes[r, 6].axis("off"); axes[r,6].text(0.02,0.97,f"top20% IoU={mo:.3f}",color="white",transform=axes[r,6].transAxes,va="top",fontsize=9)
        summaries[r].update({"dfine_query": dq, "dbsr_query": mq, "dfine_overlap_top20": do, "dbsr_overlap_top20": mo})
    fig.suptitle("Classification–regression branch response (Grad-CAM-style gradient on FPN features)", fontsize=16, y=0.995)
    fig.text(0.5, 0.006, "Red: classification response   Blue: regression response   Purple: spatial overlap", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"method": "Grad-CAM-style |d y/d A|*|A| on three encoder FPN levels", "top_fraction": 0.20, "samples": summaries}, f, indent=2)
    print(json.dumps({"output": str(out), "samples": summaries}, ensure_ascii=False, indent=2))


def anns_for_image(ann_file, image_id):
    d = json.load(open(ann_file, "r"))
    return [a for a in d["annotations"] if int(a["image_id"]) == int(image_id)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", required=True)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--dbsr-config", required=True)
    p.add_argument("--dbsr-checkpoint", required=True)
    p.add_argument("--ann-file", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--image-ids", nargs="*", type=int, default=None)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())

