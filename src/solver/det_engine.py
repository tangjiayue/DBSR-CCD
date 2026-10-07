import math
import sys
from typing import Dict, Iterable, List
import gc
import os
import json
from pathlib import Path
import torch.distributed as dist

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_convert, box_iou


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    postprocessor = kwargs.get("postprocessor", None)
    cfg = kwargs.get("cfg", None)

    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    accum_steps = cfg.yaml_cfg["train_dataloader"].get("accum_steps", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=False):
                outputs, outputs1= model(samples, targets=targets)

            vis_outputs = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in outputs.items()
            }
            loss_dict = criterion(outputs, targets, **metas)
            loss_dict.update(model.module.get_losses(outputs1))
                
            loss_raw = sum(loss_dict.values())
            loss = loss_raw / accum_steps 
            scaler.scale(loss).backward()   

            if (i + 1) % accum_steps == 0:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            del outputs, loss_raw, loss


        else:
            outputs, outputs1 = model(samples, targets=targets)
            vis_outputs = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in outputs.items()
            }
            loss_dict = criterion(outputs, targets, **metas)
            loss_dict.update(model.module.get_losses(outputs1))

            loss_raw: torch.Tensor = sum(loss_dict.values())
            loss = loss_raw / accum_steps            
            loss.backward()
            
            if (i + 1) % accum_steps == 0:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad()

            del outputs, loss_raw, loss


        if (i + 1) % accum_steps == 0:
            # ema
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())
        del loss_dict

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

        # gc.collect()
        # torch.cuda.empty_cache()

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # ========================== [绘制特征分布图] ==========================
    if dist_utils.is_main_process():
        try:
            from ..zoo.dbsr.plot_distribution import epoch_visualizer
            # 使用引擎传入的 output_dir 作为保存路径
            save_path = output_dir if output_dir is not None else "./output"
            epoch_visualizer.plot_and_clear(save_dir=save_path, epoch=epoch)
        except Exception as e:
            print(f"绘制特征分布图失败: {e}")
    # =========================================================================

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


_CCD_CLASS_NAMES = ["Normal0", "ASC-US1", "ASC-H2", "LSIL3", "HSIL/SCC4", "AGC5", "VAG6", "MON7", "DYS8", "EC9"]

def _label_name(label):
    label = int(label)
    return _CCD_CLASS_NAMES[label] if 0 <= label < len(_CCD_CLASS_NAMES) else str(label)


def _collect_matched_query_predictions(outputs, targets, samples, epoch, batch_idx, iou_thresh=0.5, topk=3):
    records = []
    if "pred_logits" not in outputs or "pred_boxes" not in outputs:
        return records

    pred_logits = outputs["pred_logits"].detach()
    pred_boxes_cxcywh = outputs["pred_boxes"].detach()
    pred_scores_all = pred_logits.sigmoid()
    B = pred_logits.shape[0]

    for img_idx in range(B):
        target = targets[img_idx]
        gt_labels_all = target["labels"].detach().long()
        gt_keep = gt_labels_all != -1
        if not gt_keep.any():
            continue

        gt_boxes = scale_boxes(
            target["boxes"].detach()[gt_keep].clone(),
            (target["orig_size"][1], target["orig_size"][0]),
            (samples[img_idx].shape[-2], samples[img_idx].shape[-1]),
        )
        gt_labels = gt_labels_all[gt_keep]
        gt_indices = torch.arange(gt_labels_all.shape[0], device=gt_labels_all.device)[gt_keep]

        orig_size = target["orig_size"].to(device=pred_boxes_cxcywh.device, dtype=pred_boxes_cxcywh.dtype)
        pred_boxes = box_convert(pred_boxes_cxcywh[img_idx], in_fmt="cxcywh", out_fmt="xyxy")
        pred_boxes = pred_boxes * orig_size.repeat(2)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        scores = pred_scores_all[img_idx]
        pred_scores, pred_labels = scores.max(dim=-1)
        k = min(int(topk), scores.shape[-1])
        top_scores, top_labels = torch.topk(scores, k=k, dim=-1)

        ious = box_iou(pred_boxes, gt_boxes)
        query_indices, gt_match_indices = torch.nonzero(ious >= iou_thresh, as_tuple=True)
        if query_indices.numel() == 0:
            continue

        iou_values = ious[query_indices, gt_match_indices]
        order = torch.argsort(iou_values, descending=True, stable=True)
        query_indices = query_indices[order]
        gt_match_indices = gt_match_indices[order]
        iou_values = iou_values[order]

        image_id_obj = target.get("image_id", torch.tensor(-1, device=gt_labels.device))
        image_id = int(image_id_obj.item()) if torch.is_tensor(image_id_obj) else int(image_id_obj)

        gt_seen_counts = {}
        for query_idx_t, gt_idx_t, iou_t in zip(query_indices, gt_match_indices, iou_values):
            query_idx = int(query_idx_t.item())
            gt_local_idx = int(gt_idx_t.item())
            gt_index = int(gt_indices[gt_local_idx].item())
            gt_seen_counts[gt_index] = gt_seen_counts.get(gt_index, 0) + 1

            pred_label = int(pred_labels[query_idx].item())
            gt_label = int(gt_labels[gt_local_idx].item())
            is_correct = pred_label == gt_label

            top3 = []
            for score, label in zip(top_scores[query_idx], top_labels[query_idx]):
                label_int = int(label.item())
                top3.append({"label": label_int, "name": _label_name(label_int), "score": float(score.item())})

            records.append({
                "epoch": int(epoch),
                "batch_idx": int(batch_idx),
                "batch_image_idx": int(img_idx),
                "image_id": image_id,
                "gt_index": gt_index,
                "gt_uid": f"{image_id}:{gt_index}",
                "gt_match_rank": gt_seen_counts[gt_index],
                "query_index": query_idx,
                "iou": float(iou_t.item()),
                "is_correct": bool(is_correct),
                "gt_label": gt_label,
                "gt_name": _label_name(gt_label),
                "pred_label": pred_label,
                "pred_name": _label_name(pred_label),
                "pred_score": float(pred_scores[query_idx].item()),
                "gt_class_score": float(scores[query_idx, gt_label].item()),
                "top3": top3,
                "gt_box_xyxy": [float(x) for x in gt_boxes[gt_local_idx].detach().cpu().tolist()],
                "pred_box_xyxy": [float(x) for x in pred_boxes[query_idx].detach().cpu().tolist()],
            })
    return records


def _deduplicate_matched_query_records(records):
    deduped = {}
    for item in records:
        key = (item["image_id"], item["gt_index"], item["query_index"])
        old = deduped.get(key)
        if old is None:
            deduped[key] = item
            continue
        if (item.get("iou", 0.0), item.get("pred_score", 0.0)) > (old.get("iou", 0.0), old.get("pred_score", 0.0)):
            deduped[key] = item
    return list(deduped.values())

def _group_records_by_gt(records):
    groups = []
    group_map = {}
    for item in records:
        key = item["gt_uid"]
        if key not in group_map:
            group = {
                "image_id": item["image_id"],
                "gt_index": item["gt_index"],
                "gt_uid": item["gt_uid"],
                "gt_label": item["gt_label"],
                "gt_name": item["gt_name"],
                "gt_box_xyxy": item["gt_box_xyxy"],
                "num_predictions": 0,
                "num_correct": 0,
                "num_wrong": 0,
                "predictions": [],
            }
            group_map[key] = group
            groups.append(group)
        group = group_map[key]
        pred_record = {
            "image_id": item["image_id"],
            "gt_index": item["gt_index"],
            "gt_label": item["gt_label"],
            "gt_name": item["gt_name"],
            "query_index": item["query_index"],
            "gt_match_rank": item["gt_match_rank"],
            "iou": item["iou"],
            "is_correct": item["is_correct"],
            "pred_label": item["pred_label"],
            "pred_name": item["pred_name"],
            "pred_score": item["pred_score"],
            "gt_class_score": item.get("gt_class_score"),
            "top3": item["top3"],
            "pred_box_xyxy": item["pred_box_xyxy"],
        }
        group["predictions"].append(pred_record)
        group["num_predictions"] += 1
        if item["is_correct"]:
            group["num_correct"] += 1
        else:
            group["num_wrong"] += 1
    return groups

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types
   
    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    matched_prediction_records = []

    output_dir = kwargs.get("output_dir", None)
    m_output_dir = kwargs.get("m_output_dir", None)

    # VPE explanation output during validation. Comment out this block, or set
    # explain_max_images = 0, if you do not want explanation images for a run.
    explain_max_images = 2
    explain_max_boxes = 5
    explain_score_threshold = 0.05
    explain_topk = 5
    explain_alpha = 0.55
    explain_attn_radius = 10
    explain_cam_box_scale = 1.25
    explain_records = []
    explain_seen_images = 0
    explain_dir = None
    if explain_max_images > 0 and dist_utils.is_main_process():
        base_dir = Path(m_output_dir or output_dir or "./output")
        explain_dir = base_dir / "vpe_explain" / f"epoch_{epoch}"
        explain_dir.mkdir(parents=True, exist_ok=True)

    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    # # ==================== 初始化可视化器 ====================
    # # 只在主进程且指定 epoch 时初始化（比如每 5 个 epoch 可视化一次）
    # visualize_this_epoch = (epoch % 5 == 0) or (epoch == -1)  # 每 5 个 epoch 可视化一次，可以调整
    # visualizer = None
    # # print(dist_utils.is_main_process(), visualize_this_epoch, m_output_dir is not None)
    # if dist_utils.is_main_process() and visualize_this_epoch and m_output_dir is not None:
    #     from ..misc.visualizer import InferenceVisualizer  
    #     visualizer = InferenceVisualizer(score_threshold=0., top_k=20)
    #     print(f"\n[可视化] 初始化可视化器，epoch={epoch}")

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")


        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        raw_model = model.module if hasattr(model, "module") else model
        outputs, feats = raw_model.sample(samples, targets=targets)
        results = postprocessor(outputs, orig_target_sizes)
        matched_prediction_records.extend(
            _collect_matched_query_predictions(outputs, targets, samples, epoch=epoch, batch_idx=i)
        )

        # #输出所有预测框
        # if dist_utils.is_main_process() and m_output_dir is not None and i < 2:
        #     save_all_dbsr_query_boxes(
        #         samples=samples,
        #         outputs=outputs,
        #         targets=targets,
        #         output_dir=m_output_dir,
        #         epoch=epoch,
        #         batch_idx=i,
        #         logits_key="pred_logits",   # DBSR 原始预测类别和置信度
        #     )

        # if (
        #     explain_dir is not None
        #     and explain_seen_images < explain_max_images
        #     and dist_utils.is_main_process()
        # ):
        #     try:
        #         from types import SimpleNamespace
        #         from tools.visualization.vpe_explain import explain_batch

        #         explain_args = SimpleNamespace(
        #             max_images=explain_max_images,
        #             max_boxes=explain_max_boxes,
        #             query_index=None,
        #             class_id=None,
        #             score_threshold=explain_score_threshold,
        #             topk=explain_topk,
        #             alpha=explain_alpha,
        #             attn_radius=explain_attn_radius,
        #             cam_box_scale=explain_cam_box_scale,
        #         )
        #         raw_model.zero_grad(set_to_none=True)
        #         with torch.enable_grad():
        #             records = explain_batch(
        #                 raw_model,
        #                 postprocessor,
        #                 data_loader.dataset,
        #                 samples,
        #                 targets,
        #                 explain_args,
        #                 explain_dir,
        #                 explain_seen_images,
        #             )
        #         explain_records.extend(records)
        #         explain_seen_images += samples.shape[0]
        #     except Exception as e:
        #         print(f"[VPE Explain] failed at batch {i}: {e}")
        #         explain_dir = None

        # results = model.module.predict_refine(feats, results, 1)

        # loss_dict = model.module.get_losses(outputs)
        # loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        # loss_value = sum(loss_dict_reduced.values())

        # if dist_utils.is_main_process() and global_step % 10 == 0:
        #     print(f"loss_value: {loss_value.item():.4f}")
        #  # ==================== 可视化推理结果 ====================
        # # 只在主进程、前几个 batch、且可视化器已初始化时执行
        # if dist_utils.is_main_process() and visualizer is not None and i < 2:  # 只可视化前 2 个 batch
        #     try:
        #         save_dir = os.path.join(m_output_dir, "inference_vis", f"epoch_{epoch}")                
        #         # 并排对比（需要 targets）
        #         visualizer.visualize_batch(
        #             images=samples,
        #             outputs=outputs,
        #             save_dir=save_dir,
        #             batch_idx=i,
        #             max_images=4,
        #             gt_targets=targets,
        #             draw_mode="compare",
        #             nms_threshold=0.5
        #         )
                
        #         print(f"[可视化] 已保存 batch {i} 的可视化结果到 {save_dir}")
        #     except Exception as e:
        #         print(f"[可视化] 警告: 可视化失败: {e}")
        # # ========================================================

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
            # # =========================================
            # # save coco predictions
            # # =========================================
            # if dist_utils.is_main_process():
            #     import json

            #     coco_results = []
            #     for image_id, output in res.items():
            #         labels = output["labels"].cpu().numpy()
            #         boxes = output["boxes"].cpu().numpy()
            #         scores = output["scores"].cpu().numpy()
            #         for lab, box, score in zip(labels, boxes, scores):
            #             x1, y1, x2, y2 = box
            #             coco_results.append({
            #                 "image_id": int(image_id),
            #                 "category_id": int(lab),
            #                 "bbox": [
            #                     float(x1),
            #                     float(y1),
            #                     float(x2 - x1),
            #                     float(y2 - y1)
            #                 ],
            #                 "score": float(score)
            #             })

            #     os.makedirs(str(m_output_dir) + "/detection_results", exist_ok=True)
            #     path = str(m_output_dir) + "/detection_results/" + f"epoch{epoch}-predictions.json"

            #     with open(path, "w") as f:
            #         json.dump(coco_results, f)
            #     print(f"saved prediction json: {path}")

            #     from tidecv import TIDE, datasets
            #     tide = TIDE()
            #     tide.evaluate_range(
            #         datasets.COCO("/root/userfolder/Dataset/ObjectDetection/TCT_JPEGImages/val5000-cocolike-cat10.json"),
            #         datasets.COCOResult(path),
            #         mode=TIDE.BOX
            #     )
            #     tide.summarize()

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            labels = target["labels"]
            boxes = target["boxes"]

            # 过滤掉 label == 0
            keep = labels != -1

            gt.append(
                {
                    "boxes": scale_boxes(
                        boxes[keep],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": labels[keep],
                    "gt_indices": torch.arange(labels.shape[0], device=labels.device)[keep],
                    "image_id": int(target["image_id"].item()),
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            top3_records = []
            class_score_records = []
            query_indices = result.get("query_index", None)
            if query_indices is not None and "pred_logits" in outputs:
                query_scores = outputs["pred_logits"].detach().sigmoid()[idx, query_indices.long()]
                class_score_records = [[float(x) for x in row] for row in query_scores.detach().cpu().tolist()]
                top_scores, top_labels = torch.topk(query_scores, k=min(3, query_scores.shape[-1]), dim=-1)
                for scores_row, labels_row in zip(top_scores, top_labels):
                    row = []
                    for score, label in zip(scores_row, labels_row):
                        label_int = int(label.item())
                        row.append({"label": label_int, "name": _label_name(label_int), "score": float(score.item())})
                    top3_records.append(row)
            preds.append(
                {
                    "boxes": result["boxes"],
                    "labels": labels,
                    "scores": result["scores"],
                    "top3": top3_records,
                    "class_scores": class_score_records,
                    "query_index": query_indices if query_indices is not None else torch.empty(0, device=result["boxes"].device, dtype=torch.long),
                    "image_id": int(target["image_id"].item()),
                }
            )

    gathered_matched_predictions = dist_utils.all_gather(matched_prediction_records)
    if dist_utils.is_main_process():
        merged_matched_predictions = []
        for records_part in gathered_matched_predictions:
            merged_matched_predictions.extend(records_part)
        raw_merged_count = len(merged_matched_predictions)
        merged_matched_predictions = _deduplicate_matched_query_records(merged_matched_predictions)
        duplicate_count = raw_merged_count - len(merged_matched_predictions)
        merged_matched_predictions.sort(key=lambda item: (item["image_id"], item.get("gt_index", -1), -item["iou"], item["query_index"]))
        correct_count = sum(1 for item in merged_matched_predictions if item.get("is_correct", False))
        wrong_count = len(merged_matched_predictions) - correct_count
        grouped_records = _group_records_by_gt(merged_matched_predictions)
        unique_gt_count = len(grouped_records)
        low_score_count = sum(1 for item in merged_matched_predictions if item.get("pred_score", 1.0) < 0.05)
        base_dir = Path(m_output_dir or output_dir or "./output")
        save_dir = base_dir / "matched_predictions"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"epoch_{epoch}_top3.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(grouped_records, f, ensure_ascii=False, indent=2)
        print(f"[Matched Predictions] saved {len(merged_matched_predictions)} records to {save_path}; correct: {correct_count}, wrong: {wrong_count}, unique_gt: {unique_gt_count}, score<0.05: {low_score_count}, dedup_removed: {duplicate_count}")

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds, conf_thresh=0).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if explain_dir is not None and dist_utils.is_main_process():
        json_path = explain_dir / "explanations.json"
        json_path.write_text(json.dumps(explain_records, indent=2, ensure_ascii=False))
        print(f"[VPE Explain] saved {len(explain_records)} explanations to {explain_dir}")

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()      
        coco_evaluator.summarize()

        stats_obj = coco_evaluator.coco_eval["bbox"]    # 获取 bbox 评估结果对象
        cat_ids = stats_obj.params.catIds
        precision = stats_obj.eval['precision']

        # 用于计算排除 label 0 后的平均值
        valid_aps_50_95 = []
        valid_aps_50 = []
        for i, cat_id in enumerate(cat_ids):
            ap_50_95 = precision[:, :, i, 0, -1].mean()
            ap_50 = precision[0, :, i, 0, -1].mean()
            
            if cat_id != 0:  # 排除类别0
                valid_aps_50_95.append(ap_50_95)
                valid_aps_50.append(ap_50)

        # 计算排除类别0后的 mAP（所有进程都计算）
        if len(valid_aps_50_95) > 0:
            mean_ap_50_95 = sum(valid_aps_50_95) / len(valid_aps_50_95)
            mean_ap_50 = sum(valid_aps_50) / len(valid_aps_50)
        else:
            mean_ap_50_95 = 0.0
            mean_ap_50 = 0.0

        if dist_utils.is_main_process():
            print("\n" + "="*50)
            print(f"{'CatID':<10} | {'AP@50:95':<12} | {'AP@50':<10}")
            print("-" * 50)
            
            for i, cat_id in enumerate(cat_ids):
                ap_50_95 = precision[:, :, i, 0, -1].mean()
                ap_50 = precision[0, :, i, 0, -1].mean()
                print(f"{cat_id:<10} | {ap_50_95:<12.4f} | {ap_50:<10.4f}")
            
            print("="*50)
            print(f"{'mAP(excl.0)':<10} | {mean_ap_50_95:<12.4f} | {mean_ap_50:<10.4f}")
            print("="*50 + "\n")


    # # 保存检测结果
    # if dist_utils.is_dist_available_and_initialized():
    #     # 收集所有rank的gt和preds
    #     all_gt = gather_all_ranks_data(gt)
    #     all_preds = gather_all_ranks_data(preds)
        
    #     if dist_utils.is_main_process():
    #         # 只有主进程保存汇总结果
    #         json_path = dist_utils.save_detection_results(
    #             gt=all_gt,
    #             preds=all_preds,
    #             save_dir=str(m_output_dir) + "/detection_results",
    #             epoch=epoch,
    #         )

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            # stats["bbox_mean_ap_50"] = [mean_ap_50]

            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    del gt,preds,res,outputs,results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================== [绘制验证集推理特征分布图] ==========================
    if dist_utils.is_main_process():
        try:
            from ..zoo.dbsr.plot_distribution import epoch_visualizer
            save_path = output_dir if output_dir is not None else "./output"
            epoch_visualizer.plot_and_clear(save_dir=save_path, epoch=f"{epoch}_eval")
        except Exception as e:
            print(f"绘制验证集分布图失败: {e}")
    # ======================================================================================

    return stats, coco_evaluator


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def save_all_dbsr_query_boxes(samples, outputs, targets, output_dir, epoch, batch_idx, logits_key="pred_logits"):
    """
    保存 DBSR 原始所有 query 框，不做 NMS，不做 topk，不做 score 过滤。
    左图：所有 query 预测框；右图：GT 框。
    """
    if output_dir is None:
        return

    save_dir = Path(output_dir) / "all_dbsr_query_boxes" / f"epoch_{epoch}"
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_boxes = outputs["pred_boxes"].detach().cpu()       # [B, N, 4], cxcywh, normalized
    pred_logits = outputs[logits_key].detach().cpu()        # [B, N, C]

    scores_all = pred_logits.sigmoid()
    scores, labels = scores_all.max(dim=-1)                 # [B, N]

    # 高对比颜色，避免亮绿色和浅色文字看不清
    BOX_COLORS = [
        (230, 30, 30),     # red
        (30, 90, 255),     # strong blue
        (230, 0, 230),     # magenta
        (245, 150, 0),     # orange
        (140, 0, 255),     # purple
        (0, 145, 210),     # cyan-blue
        (210, 60, 90),     # rose
        (80, 80, 255),     # blue-violet
        (210, 110, 0),     # dark orange
        (0, 120, 120),     # teal
    ]
    GT_COLOR = (20, 90, 230)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        font = ImageFont.load_default()

    for img_idx in range(samples.shape[0]):
        # 参考 src/misc/visualizer.py：直接转 PIL，不手动反归一化
        base_img = to_pil_image(samples[img_idx].detach().cpu())
        pred_img = base_img.copy()
        gt_img = base_img.copy()
        pred_draw = ImageDraw.Draw(pred_img)
        gt_draw = ImageDraw.Draw(gt_img)
        w, h = pred_img.size

        boxes_i = box_convert(pred_boxes[img_idx], in_fmt="cxcywh", out_fmt="xyxy")
        boxes_i[:, [0, 2]] *= w
        boxes_i[:, [1, 3]] *= h
        labels_i = labels[img_idx]
        scores_i = scores[img_idx]

        # 左图：画所有 query，一个都不过滤
        for qid in range(boxes_i.shape[0]):
            x1, y1, x2, y2 = boxes_i[qid].tolist()
            x1 = int(max(0, min(x1, w - 1)))
            y1 = int(max(0, min(y1, h - 1)))
            x2 = int(max(0, min(x2, w - 1)))
            y2 = int(max(0, min(y2, h - 1)))

            if x2 <= x1:
                x2 = min(w - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(h - 1, y1 + 1)

            cls_id = int(labels_i[qid].item())
            score = float(scores_i[qid].item())
            color = BOX_COLORS[cls_id % len(BOX_COLORS)]
            text = f"q{qid} c{cls_id} {score:.2f}"

            pred_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 字体本身带颜色；不画背景，保持透明/原图背景
            tb = pred_draw.textbbox((x1, y1), text, font=font)
            th = tb[3] - tb[1]
            text_y = max(0, y1 - th - 1)
            pred_draw.text((x1, text_y), text, fill=color, font=font)

        # 右图：画 GT 框
        gt_boxes = targets[img_idx]["boxes"].detach().cpu().clone()
        gt_labels = targets[img_idx]["labels"].detach().cpu().clone()
        if gt_boxes.numel() > 0:
            # 正常验证流里 GT 是 xyxy 像素坐标；这里兜底兼容 normalized cxcywh
            if float(gt_boxes.max()) <= 1.5:
                gt_boxes = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                gt_boxes[:, [0, 2]] *= w
                gt_boxes[:, [1, 3]] *= h

            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.tolist()
                x1 = int(max(0, min(x1, w - 1)))
                y1 = int(max(0, min(y1, h - 1)))
                x2 = int(max(0, min(x2, w - 1)))
                y2 = int(max(0, min(y2, h - 1)))
                if x2 <= x1:
                    x2 = min(w - 1, x1 + 1)
                if y2 <= y1:
                    y2 = min(h - 1, y1 + 1)

                cls_id = int(label.item())
                text = f"GT c{cls_id}"
                gt_draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)
                tb = gt_draw.textbbox((x1, y1), text, font=font)
                th = tb[3] - tb[1]
                text_y = max(0, y1 - th - 1)
                gt_draw.text((x1, text_y), text, fill=GT_COLOR, font=font)

        # 左右拼接：左预测，右 GT
        comparison = Image.new("RGB", (w * 2, h), (0, 0, 0))
        comparison.paste(pred_img, (0, 0))
        comparison.paste(gt_img, (w, 0))
        comp_draw = ImageDraw.Draw(comparison)
        comp_draw.text((5, 5), f"ALL DBSR QUERIES ({logits_key})", fill=(255, 255, 255), font=font)
        comp_draw.text((w + 5, 5), "GT", fill=(255, 255, 255), font=font)

        image_id = int(targets[img_idx]["image_id"].item()) if "image_id" in targets[img_idx] else img_idx
        save_path = save_dir / f"batch{batch_idx}_img{img_idx}_id{image_id}_{logits_key}_all_queries_vs_gt.jpg"
        comparison.save(save_path, quality=95)


def gather_all_ranks_data(data_list):
    """收集所有rank的数据到主进程（支持任意Python对象）"""
    if not dist.is_initialized():
        return data_list
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 创建存储所有rank数据的列表
    gathered_data = [None] * world_size
    
    # 使用 all_gather_object 收集所有数据
    dist.all_gather_object(gathered_data, data_list)
    
    # 展平所有数据
    if rank == 0:
        all_data = []
        for r_data in gathered_data:
            all_data.extend(r_data)
        return all_data
    else:
        return None
