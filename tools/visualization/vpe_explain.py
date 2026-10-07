#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_model(config_path, checkpoint_path, device):
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path, device=device)
    model = cfg.model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    return cfg, model


def build_val_loader(cfg, batch_size, num_workers):
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = batch_size
    cfg.yaml_cfg["val_dataloader"].pop("batch_size", None)
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = num_workers
    cfg.yaml_cfg["val_dataloader"]["shuffle"] = False
    return cfg.val_dataloader


def set_vpe_attention_debug(model, enabled=True):
    for module in model.modules():
        if hasattr(module, "last_attn") and hasattr(module, "last_sampling_locations"):
            module.debug = enabled
            if not enabled:
                module.last_attn = None
                module.last_sampling_locations = None


def get_last_attention_module(model):
    found = None
    for module in model.modules():
        if getattr(module, "last_attn", None) is not None:
            found = module
    return found


def tensor_to_image(tensor):
    arr = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def normalize_heatmap(heat):
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = np.maximum(heat, 0)
    max_val = float(heat.max())
    if max_val <= 1e-8:
        return np.zeros_like(heat, dtype=np.float32)
    return (heat / max_val).astype(np.float32)


def overlay_heatmap(image_np, heat, alpha=0.55):
    heat = normalize_heatmap(heat)
    color = np.zeros_like(image_np, dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = 220.0 * heat
    base = image_np.astype(np.float32)
    mix = base * (1.0 - alpha * heat[..., None]) + color * (alpha * heat[..., None])
    return np.clip(mix, 0, 255).astype(np.uint8)


def draw_box_and_label(image_np, box, text, color=(0, 255, 255)):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x1, max(0, y1 - 18)), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((x1, max(0, y1 - 18)), text, fill=color, font=font)
    return np.asarray(img)


def add_gaussian(heat, x, y, weight, radius):
    h, w = heat.shape
    x = int(round(float(x)))
    y = int(round(float(y)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sigma = max(radius / 2.0, 1.0)
    kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    heat[y0:y1, x0:x1] += float(weight) * kernel


def build_attention_heatmaps(attn, locations, batch_idx, query_idx, image_h, image_w, radius):
    weights = attn[batch_idx, query_idx].detach().float().cpu().numpy()
    points = locations[batch_idx, query_idx].detach().float().cpu().numpy()
    num_levels = weights.shape[1]
    all_heat = np.zeros((image_h, image_w), dtype=np.float32)
    level_heats = []
    for level in range(num_levels):
        heat = np.zeros((image_h, image_w), dtype=np.float32)
        level_weights = weights[:, level, :]
        level_points = points[:, level, :, :]
        for head in range(level_weights.shape[0]):
            for point_idx in range(level_weights.shape[1]):
                x_norm, y_norm = level_points[head, point_idx]
                x = (x_norm + 1.0) * 0.5 * (image_w - 1)
                y = (y_norm + 1.0) * 0.5 * (image_h - 1)
                add_gaussian(heat, x, y, level_weights[head, point_idx], radius)
        all_heat += heat
        level_heats.append(heat)
    return all_heat, level_heats


def grad_cam(feature, grad, image_h, image_w):
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * feature).sum(dim=0, keepdim=True)).unsqueeze(0)
    cam = F.interpolate(cam, size=(image_h, image_w), mode="bilinear", align_corners=False)
    return cam.squeeze().detach().float().cpu().numpy()


def expand_box(box, image_w, image_h, scale=1.25):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max((x2 - x1) * scale, 1.0)
    h = max((y2 - y1) * scale, 1.0)
    return [
        max(0, int(round(cx - 0.5 * w))),
        max(0, int(round(cy - 0.5 * h))),
        min(image_w, int(round(cx + 0.5 * w))),
        min(image_h, int(round(cy + 0.5 * h))),
    ]


def mask_heatmap_to_box(heat, box, image_w, image_h, scale=1.25):
    x1, y1, x2, y2 = expand_box(box, image_w, image_h, scale=scale)
    masked = np.zeros_like(heat, dtype=np.float32)
    if x2 > x1 and y2 > y1:
        masked[y1:y2, x1:x2] = heat[y1:y2, x1:x2]
    return masked


def forward_with_grad(model, samples):
    # Build Grad-CAM on encoder features, independent of whether model
    # parameters were frozen by the training/eval pipeline.
    with torch.no_grad():
        feats = model.backbone(samples)
        feats = model.encoder(feats)
    feats = [feat.detach().requires_grad_(True) for feat in feats]
    for feat in feats:
        feat.retain_grad()
    outputs = model.decoder(feats)
    outputs = model.VisualClassifier(feats, outputs, targets=None)
    return outputs, feats


def class_name(dataset, label):
    if hasattr(dataset, "category2name"):
        return dataset.category2name.get(int(label), str(int(label)))
    return str(int(label))


def box_cxcywh_to_xyxy_pixels(box, image_w, image_h):
    cx, cy, w, h = [float(v) for v in box]
    return [
        (cx - 0.5 * w) * image_w,
        (cy - 0.5 * h) * image_h,
        (cx + 0.5 * w) * image_w,
        (cy + 0.5 * h) * image_h,
    ]


def select_predictions(outputs, post_result, batch_idx, args, image_w, image_h):
    if args.query_index is None:
        labels = post_result["labels"].detach().cpu()
        scores = post_result["scores"].detach().cpu()
        boxes = post_result["boxes"].detach().cpu()
        query_indices = post_result["query_index"].detach().cpu()
        keep = torch.where(scores >= args.score_threshold)[0]
        keep = keep[: args.max_boxes]
        return [
            {
                "rank": rank,
                "query_idx": int(query_indices[item_idx].item()),
                "label": int(labels[item_idx].item()),
                "score": float(scores[item_idx].item()),
                "box": boxes[item_idx].numpy().tolist(),
            }
            for rank, item_idx in enumerate(keep.tolist())
        ]

    query_idx = int(args.query_index)
    final_scores = torch.sigmoid(outputs["vpe_logits"][batch_idx, query_idx]).detach().cpu()
    label = int(args.class_id) if args.class_id is not None else int(final_scores.argmax().item())
    box = box_cxcywh_to_xyxy_pixels(
        outputs["pred_boxes"][batch_idx, query_idx].detach().cpu().tolist(),
        image_w,
        image_h,
    )
    return [{
        "rank": 0,
        "query_idx": query_idx,
        "label": label,
        "score": float(final_scores[label].item()),
        "box": box,
    }]


def explain_batch(model, postprocessor, dataset, samples, targets, args, out_dir, start_image_idx):
    device = samples.device
    image_h, image_w = samples.shape[-2:]
    image_sizes = torch.tensor([[image_w, image_h]] * samples.shape[0], device=device)

    set_vpe_attention_debug(model, True)
    outputs, feats = forward_with_grad(model, samples)
    with torch.no_grad():
        results = postprocessor(outputs, image_sizes)

    attn_module = get_last_attention_module(model)
    if attn_module is None:
        raise RuntimeError("No VPE attention cache found. Check SimpleMSDeformableAttention.debug.")
    attn = attn_module.last_attn
    locations = attn_module.last_sampling_locations

    records = []
    for b_idx, result in enumerate(results):
        if b_idx + start_image_idx >= args.max_images:
            break
        image_np = tensor_to_image(samples[b_idx])
        image_id = int(targets[b_idx].get("image_id", torch.tensor([start_image_idx + b_idx])).flatten()[0].item())
        sample_dir = out_dir / f"image_{image_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_np).save(sample_dir / "image.jpg", quality=95)

        selected = select_predictions(outputs, result, b_idx, args, image_w, image_h)

        for selected_item in selected:
            rank = selected_item["rank"]
            query_idx = selected_item["query_idx"]
            label = selected_item["label"]
            score = selected_item["score"]
            box = selected_item["box"]
            name = class_name(dataset, label)
            text = f"q{query_idx} {name} {score:.3f}"

            model.zero_grad(set_to_none=True)
            if feats[0].grad is not None:
                feats[0].grad.zero_()
            # Use raw VPE class logits for classification evidence. The final
            # vpe_logits also include quality_score, which is useful for ranking
            # boxes but can make Grad-CAM look image-wide.
            class_logits = outputs.get("vpe_logits_raw", outputs["vpe_logits"])
            target_logit = class_logits[b_idx, query_idx, label]
            target_logit.backward(retain_graph=True)
            cam = grad_cam(feats[0][b_idx], feats[0].grad[b_idx], image_h, image_w)
            cam_box = mask_heatmap_to_box(cam, box, image_w, image_h, scale=args.cam_box_scale)
            cam_img = overlay_heatmap(image_np, cam_box, alpha=args.alpha)
            cam_img = draw_box_and_label(cam_img, box, text)

            attn_heat, level_heats = build_attention_heatmaps(
                attn, locations, b_idx, query_idx, image_h, image_w, args.attn_radius
            )
            attn_img = overlay_heatmap(image_np, attn_heat, alpha=args.alpha)
            attn_img = draw_box_and_label(attn_img, box, text)

            prefix = f"rank{rank:02d}_q{query_idx}_cls{label}"
            Image.fromarray(cam_img).save(sample_dir / f"{prefix}_gradcam.jpg", quality=95)
            Image.fromarray(attn_img).save(sample_dir / f"{prefix}_attn_all.jpg", quality=95)
            for level_idx, heat in enumerate(level_heats):
                lvl_img = overlay_heatmap(image_np, heat, alpha=args.alpha)
                lvl_img = draw_box_and_label(lvl_img, box, f"{text} L{level_idx}")
                Image.fromarray(lvl_img).save(sample_dir / f"{prefix}_attn_level{level_idx}.jpg", quality=95)

            final_scores = torch.sigmoid(outputs["vpe_logits"][b_idx, query_idx]).detach().cpu()
            raw_scores = torch.sigmoid(outputs.get("vpe_logits_raw", outputs["vpe_logits"])[b_idx, query_idx]).detach().cpu()
            top_scores, top_labels = torch.topk(final_scores, k=min(args.topk, final_scores.numel()))
            quality = outputs.get("vpe_quality_score")
            quality_val = None
            if quality is not None:
                quality_val = float(quality[b_idx, query_idx].detach().cpu().flatten()[0].item())

            records.append({
                "image_id": image_id,
                "query_index": query_idx,
                "rank": rank,
                "pred_label": label,
                "pred_name": name,
                "score": score,
                "box_xyxy": [float(v) for v in box],
                "quality_logit": quality_val,
                "raw_vpe_score_for_pred": float(raw_scores[label].item()),
                "final_score_for_pred": float(final_scores[label].item()),
                "top_classes": [
                    {
                        "label": int(lbl.item()),
                        "name": class_name(dataset, int(lbl.item())),
                        "final_score": float(scr.item()),
                        "raw_vpe_score": float(raw_scores[int(lbl.item())].item()),
                    }
                    for scr, lbl in zip(top_scores, top_labels)
                ],
                "files": {
                    "gradcam": str(sample_dir / f"{prefix}_gradcam.jpg"),
                    "attn_all": str(sample_dir / f"{prefix}_attn_all.jpg"),
                },
            })
    set_vpe_attention_debug(model, False)
    return records


def main():
    parser = argparse.ArgumentParser(description="Visualize VPE classification evidence.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, help="checkpoint path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/vpe_explain")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=5)
    parser.add_argument("--image-index", type=int, default=None, help="visualize a specific val dataset index")
    parser.add_argument("--query-index", type=int, default=None, help="visualize a specific decoder query")
    parser.add_argument("--class-id", type=int, default=None, help="class logit for Grad-CAM; defaults to query argmax")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--attn-radius", type=int, default=10)
    parser.add_argument("--cam-box-scale", type=float, default=1.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, model = load_model(args.config, args.resume, device)
    postprocessor = cfg.postprocessor.to(device).eval()
    loader = build_val_loader(cfg, args.batch_size, args.num_workers)

    if args.image_index is not None:
        sample, target = loader.dataset[args.image_index]
        samples = sample.unsqueeze(0).to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}]
        all_records = explain_batch(model, postprocessor, loader.dataset, samples, targets, args, out_dir, 0)
    else:
        all_records = []
        seen = 0
        for samples, targets in loader:
            if seen >= args.max_images:
                break
            samples = samples.to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            records = explain_batch(model, postprocessor, loader.dataset, samples, targets, args, out_dir, seen)
            all_records.extend(records)
            seen += samples.shape[0]

    json_path = out_dir / "explanations.json"
    json_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"saved {len(all_records)} explanations to {out_dir}")


if __name__ == "__main__":
    main()
