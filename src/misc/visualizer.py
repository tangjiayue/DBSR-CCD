import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    box_fmt: 'xyxy', 'xywh', 'cxcywh', D-FINE uses 'cxcywh' for training, 'xyxy' for validation
    '''
    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    # Predefined colors (standard color names recognized by PIL)
    BOX_COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "lavender", "brown", "beige", "maroon",
        "navy", "olive", "coral", "turquoise", "gold"
    ]

    LABEL_TEXT_COLOR = "white"

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = sample.clone().cpu()
        target_boxes = target["boxes"].clone().cpu()
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        sample_visualization = to_pil_image(sample_visualization)
        sample_visualization_w, sample_visualization_h = sample_visualization.size

        # normalized to pixel space
        if normalized:
            target_boxes[:, 0] = target_boxes[:, 0] * sample_visualization_w
            target_boxes[:, 2] = target_boxes[:, 2] * sample_visualization_w
            target_boxes[:, 1] = target_boxes[:, 1] * sample_visualization_h
            target_boxes[:, 3] = target_boxes[:, 3] * sample_visualization_h

        # any box format -> xyxy
        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")

        # clip to image size
        target_boxes[:, 0] = torch.clamp(target_boxes[:, 0], 0, sample_visualization_w)
        target_boxes[:, 1] = torch.clamp(target_boxes[:, 1], 0, sample_visualization_h)
        target_boxes[:, 2] = torch.clamp(target_boxes[:, 2], 0, sample_visualization_w)
        target_boxes[:, 3] = torch.clamp(target_boxes[:, 3], 0, sample_visualization_h)

        target_boxes = target_boxes.numpy().astype(np.int32)
        target_labels = target_labels.numpy().astype(np.int32)

        draw = ImageDraw.Draw(sample_visualization)

        # draw target boxes
        for box, label in zip(target_boxes, target_labels):
            x1, y1, x2, y2 = box

            # Select color based on class ID
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]

            # Draw box (thick)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            label_text = f"{label}"

            # Measure text size
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]

            # Draw text background
            padding = 2
            draw.rectangle(
                [x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1],
                fill=box_color
            )

            # Draw text (LABEL_TEXT_COLOR)
            draw.text((x1 + padding, y1 - text_height - padding), label_text,
                     fill=LABEL_TEXT_COLOR, font=font)

        save_path = Path(output_dir) / f"{split}_samples" / f"{target_image_id}_{target_image_path_stem}.webp"
        sample_visualization.save(save_path)

def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()



# import torch
# import numpy as np
# import json
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# from torchvision.ops import box_convert
# from PIL import Image, ImageDraw, ImageFont
# import warnings

# class InferenceVisualizer:
#     """
#     推理时可视化工具（支持 PIL）
#     输出英文类别名称，只显示全局置信度前 K 个预测框
#     """
    
#     # 类别名称（根据你的 COCO categories）
#     CLASS_NAMES = [
#         "normal",                    # 0
#         "ascus",                     # 1
#         "asch",                      # 2
#         "lsil",                      # 3
#         "hsil_scc_omn",              # 4
#         "agc_adenocarcinoma_em",     # 5
#         "vaginalis",                 # 6
#         "monilia",                   # 7
#         "dysbacteriosis_herpes_act", # 8
#         "ec"                         # 9
#     ]
    
#     # 每类固定颜色（RGB格式）
#     CLASS_COLORS = [
#         (255, 0, 0),    # 0: normal - 红色
#         (0, 100, 255),    # 1: ascus - 绿色
#         (0, 0, 255),    # 2: asch - 蓝色
#         (255, 255, 0),  # 3: lsil - 青色
#         (255, 0, 255),  # 4: hsil_scc_omn - 品红
#         (0, 255, 255),  # 5: agc_adenocarcinoma_em - 黄色
#         (128, 0, 128),  # 6: vaginalis - 紫色
#         (255, 128, 0),  # 7: monilia - 橙色
#         (0, 128, 255),  # 8: dysbacteriosis_herpes_act - 天蓝
#         (128, 128, 0),  # 9: ec - 橄榄
#     ]
    
#     def __init__(
#         self,
#         score_threshold: float = 0.05,
#         top_k: int = 10,
#         line_width: int = 2,
#         font_size: int = 12,
#     ):
#         """
#         Args:
#             score_threshold: 显示框的最小置信度阈值
#             top_k: 全局置信度最高的 K 个框
#             line_width: 边框线宽
#             font_size: 文字大小
#         """
#         self.score_threshold = score_threshold
#         self.top_k = top_k
#         self.line_width = line_width
#         self.font_size = font_size
        
#         # 尝试加载字体
#         try:
#             self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
#         except:
#             self.font = ImageFont.load_default()
    
#     @staticmethod
#     def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         """将归一化的 tensor 转回 PIL Image"""
#         if isinstance(image_tensor, torch.Tensor):
#             image = image_tensor.cpu().clone()
#             if image.dim() == 4:
#                 image = image[0]
#             for i in range(3):
#                 image[i] = image[i] * std[i] + mean[i]
#             image = image.permute(1, 2, 0).numpy()
#             image = np.clip(image * 255, 0, 255).astype(np.uint8)
#             return Image.fromarray(image)
#         return image_tensor
    
#     def draw_pred_boxes_pil(
#         self,
#         image: Image.Image,
#         boxes: np.ndarray,
#         scores: np.ndarray,
#         labels: np.ndarray,
#     ) -> Image.Image:
#         """使用 PIL 绘制预测框"""
#         img_copy = image.copy()
#         draw = ImageDraw.Draw(img_copy)
#         w, h = img_copy.size
        
#         # 按置信度排序，取前 top_k 个
#         if len(scores) > self.top_k:
#             idx = np.argsort(scores)[::-1][:self.top_k]
#             boxes = boxes[idx]
#             scores = scores[idx]
#             labels = labels[idx]
        
#         for box, score, label in zip(boxes, scores, labels):
#             if score < self.score_threshold:
#                 continue
            
#             x1, y1, x2, y2 = map(int, box)
#             x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
#             y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
#             color = self.CLASS_COLORS[label % len(self.CLASS_COLORS)]
#             class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
#             text = f"{class_name}: {score:.2f}"
            
#             # 绘制边框
#             draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
            
#             # 测量文字大小
#             bbox = draw.textbbox((x1, y1), text, font=self.font)
#             text_w = bbox[2] - bbox[0]
#             text_h = bbox[3] - bbox[1]
            
#             # 绘制文字背景
#             draw.rectangle(
#                 [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
#                 fill=color
#             )
            
#             # 绘制文字
#             draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
#         return img_copy
    
#     def draw_gt_boxes_pil(
#         self,
#         image: Image.Image,
#         gt_boxes: np.ndarray,
#         gt_labels: np.ndarray,
#     ) -> Image.Image:
#         """使用 PIL 绘制 GT 框"""
#         img_copy = image.copy()
#         draw = ImageDraw.Draw(img_copy)
#         w, h = img_copy.size
#         gt_color = (0, 0, 255)
        
#         for box, label in zip(gt_boxes, gt_labels):
#             x1, y1, x2, y2 = map(int, box)
#             x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
#             y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
#             draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width)
            
#             class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
#             text = f"GT: {class_name}"
            
#             bbox = draw.textbbox((x1, y1), text, font=self.font)
#             text_w = bbox[2] - bbox[0]
#             text_h = bbox[3] - bbox[1]
            
#             draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
#             draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
#         return img_copy
    
#     def draw_gt_and_pred_combined_pil(
#         self,
#         image: Image.Image,
#         pred_boxes: np.ndarray,
#         pred_scores: np.ndarray,
#         pred_labels: np.ndarray,
#         gt_boxes: np.ndarray,
#         gt_labels: np.ndarray,
#     ) -> Image.Image:
#         """在一张图上同时绘制 GT 框和预测框"""
#         img_copy = self.draw_pred_boxes_pil(image, pred_boxes, pred_scores, pred_labels)
#         draw = ImageDraw.Draw(img_copy)
#         w, h = img_copy.size
#         gt_color = (0, 100, 255)
        
#         for box, label in zip(gt_boxes, gt_labels):
#             x1, y1, x2, y2 = map(int, box)
#             x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
#             y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
#             draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width + 1)
            
#             class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
#             text = f"GT: {class_name}"
            
#             bbox = draw.textbbox((x1, y1), text, font=self.font)
#             text_w = bbox[2] - bbox[0]
#             text_h = bbox[3] - bbox[1]
            
#             draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
#             draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
#         return img_copy
    
#     def create_comparison_image_pil(
#         self,
#         image: Image.Image,
#         pred_boxes: np.ndarray,
#         pred_scores: np.ndarray,
#         pred_labels: np.ndarray,
#         gt_boxes: Optional[np.ndarray] = None,
#         gt_labels: Optional[np.ndarray] = None,
#     ) -> Image.Image:
#         """创建对比图：左侧预测框，右侧GT框"""
#         w, h = image.size
        
#         left_img = self.draw_pred_boxes_pil(image.copy(), pred_boxes, pred_scores, pred_labels)
        
#         if gt_boxes is not None:
#             right_img = self.draw_gt_boxes_pil(image.copy(), gt_boxes, gt_labels)
#             comparison = Image.new('RGB', (w * 2, h))
#             comparison.paste(left_img, (0, 0))
#             comparison.paste(right_img, (w, 0))
#             return comparison
#         else:
#             return left_img
    
#     # def visualize_batch(
#     #     self,
#     #     images: torch.Tensor,
#     #     outputs: Dict[str, torch.Tensor],
#     #     save_dir: str,
#     #     batch_idx: int = 0,
#     #     max_images: int = 8,
#     #     gt_targets: Optional[List[Dict]] = None,
#     #     draw_mode: str = "pred_only",
#     #     random_sample: bool = True,  # 是否随机采样
#     # ):
#     #     """批量可视化推理结果"""
#     #     save_path = Path(save_dir)
#     #     save_path.mkdir(parents=True, exist_ok=True)
        
#     #     pred_logits = outputs['pred_logits']
#     #     pred_boxes = outputs['pred_boxes']
#     #     pred_scores = torch.sigmoid(pred_logits)
        
#     #     batch_total = images.shape[0]
#     #     # 确定要可视化的图片数量（不超过 batch 总数）
#     #     num_to_vis = min(batch_total, max_images)  
#     #     if random_sample:
#     #         # 从 batch 中随机选择 num_to_vis 张（不重复）
#     #         indices = np.random.choice(batch_total, num_to_vis, replace=False)
#     #     else:
#     #         indices = range(min(batch_total, max_images))

#     #     for idx_in_batch in indices:
#     #         i = idx_in_batch  # 图片在 batch 中的索引
            
#     #         img_pil = self.denormalize_image(images[i])
#     #         w, h = img_pil.size
            
#     #         scores_i, labels_i = pred_scores[i].max(dim=-1)
#     #         scores_i = scores_i.flatten()
#     #         labels_i = labels_i.flatten()
#     #         boxes_i = pred_boxes[i]
            
#     #         boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
#     #         boxes_xyxy[:, [0, 2]] *= w
#     #         boxes_xyxy[:, [1, 3]] *= h
#     #         boxes_xyxy = boxes_xyxy.cpu().numpy().astype(np.int32)
#     #         scores_i = scores_i.cpu().numpy()
#     #         labels_i = labels_i.cpu().numpy()
            
#     #         gt_boxes = None
#     #         gt_labels = None
#     #         if gt_targets is not None and i < len(gt_targets):
#     #             gt = gt_targets[i]
#     #             if 'boxes' in gt:
#     #                 gt_boxes = gt['boxes'].cpu().numpy()
#     #                 if gt_boxes.max() <= 1.0:
#     #                     gt_boxes[:, [0, 2]] *= w
#     #                     gt_boxes[:, [1, 3]] *= h
#     #                 gt_boxes = gt_boxes.astype(np.int32)
#     #                 gt_labels = gt['labels'].cpu().numpy() if 'labels' in gt else None
            
#     #         # 使用 image_id 作为文件名的一部分，避免重复
#     #         img_id = gt_targets[i].get("image_id", i) if gt_targets else i
            
#     #         if draw_mode == "pred_only":
#     #             vis = self.draw_pred_boxes_pil(img_pil.copy(), boxes_xyxy, scores_i, labels_i)
#     #             filename = f"img{img_id}_pred.jpg"
#     #         elif draw_mode == "gt_only" and gt_boxes is not None:
#     #             vis = self.draw_gt_boxes_pil(img_pil.copy(), gt_boxes, gt_labels)
#     #             filename = f"img{img_id}_gt.jpg"
#     #         elif draw_mode == "combined" and gt_boxes is not None:
#     #             vis = self.draw_gt_and_pred_combined_pil(
#     #                 img_pil.copy(), boxes_xyxy, scores_i, labels_i, gt_boxes, gt_labels
#     #             )
#     #             filename = f"img{img_id}_combined.jpg"
#     #         elif draw_mode == "compare" and gt_boxes is not None:
#     #             vis = self.create_comparison_image_pil(
#     #                 img_pil.copy(), boxes_xyxy, scores_i, labels_i, gt_boxes, gt_labels
#     #             )
#     #             filename = f"img{img_id}_compare.jpg"
#     #         else:
#     #             continue
            
#     #         vis.save(str(save_path / filename))
            
#     #         # print(f"\n[图像 {img_id}] 检测到 {len(boxes_xyxy)} 个目标，显示前 {self.top_k} 个")
#     #         top_idx = np.argsort(scores_i)[::-1][:self.top_k]
#     #         for rank, idx in enumerate(top_idx):
#     #             if scores_i[idx] < self.score_threshold:
#     #                 continue
#     #             class_name = self.CLASS_NAMES[labels_i[idx]] if labels_i[idx] < len(self.CLASS_NAMES) else str(labels_i[idx])
#     #             # print(f"  {rank+1}. {class_name}: {scores_i[idx]:.4f}")

#     def visualize_batch(
#         self,
#         images: torch.Tensor,
#         outputs: Dict[str, torch.Tensor],
#         save_dir: str,
#         batch_idx: int = 0,
#         max_images: int = 8,
#         gt_targets: Optional[List[Dict]] = None,
#         draw_mode: str = "pred_only",
#         random_sample: bool = True,
#         nms_threshold: float = 0.5,      # 新增：NMS 阈值
#         conf_threshold: float = 0.05,     # 新增：置信度阈值
#     ):
#         """批量可视化推理结果（支持 NMS 过滤）"""
#         save_path = Path(save_dir)
#         save_path.mkdir(parents=True, exist_ok=True)
        
#         pred_logits = outputs['pred_logits']
#         pred_boxes = outputs['pred_boxes']
#         pred_scores = torch.sigmoid(pred_logits)
        
#         batch_total = images.shape[0]
#         num_to_vis = min(batch_total, max_images)  
#         if random_sample:
#             indices = np.random.choice(batch_total, num_to_vis, replace=False)
#         else:
#             indices = range(min(batch_total, max_images))

#         for idx_in_batch in indices:
#             i = idx_in_batch
            
#             img_pil = self.denormalize_image(images[i])
#             w, h = img_pil.size
            
#             scores_i, labels_i = pred_scores[i].max(dim=-1)
#             scores_i = scores_i.flatten()
#             labels_i = labels_i.flatten()
#             boxes_i = pred_boxes[i]
            
#             boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
#             boxes_xyxy[:, [0, 2]] *= w
#             boxes_xyxy[:, [1, 3]] *= h
#             boxes_xyxy = boxes_xyxy.cpu().numpy().astype(np.int32)
#             scores_i = scores_i.cpu().numpy()
#             labels_i = labels_i.cpu().numpy()
            
#             # ========== 新增：NMS 过滤 ==========
#             # 1. 置信度过滤
#             keep_conf = scores_i >= conf_threshold
#             boxes_filtered = boxes_xyxy[keep_conf]
#             scores_filtered = scores_i[keep_conf]
#             labels_filtered = labels_i[keep_conf]
            
#             # 2. NMS 过滤
#             if len(boxes_filtered) > 0:
#                 # 转换为 tensor 进行 NMS
#                 boxes_tensor = torch.tensor(boxes_filtered).float()
#                 scores_tensor = torch.tensor(scores_filtered).float()
#                 labels_tensor = torch.tensor(labels_filtered).long()
                
#                 keep_nms = torchvision.ops.batched_nms(
#                     boxes_tensor, scores_tensor, labels_tensor, nms_threshold
#                 )
                
#                 final_boxes = boxes_filtered[keep_nms.cpu().numpy()]
#                 final_scores = scores_filtered[keep_nms.cpu().numpy()]
#                 final_labels = labels_filtered[keep_nms.cpu().numpy()]

#                 print(f"\n[DEBUG] 图像 {i}:")
#                 print(f"  原始框数: {len(boxes_xyxy)}")
#                 print(f"  置信度过滤后: {len(boxes_filtered)} (阈值={conf_threshold})")
#                 print(f"  NMS 后: {len(final_boxes)} (阈值={nms_threshold})")

#                 # 打印前几个框的坐标，看是否真的重叠
#                 if len(final_boxes) > 0:
#                     print(f"  保留的框 (前5个):")
#                     for j in range(min(5, len(final_boxes))):
#                         print(f"    {final_boxes[j]}, score={final_scores[j]:.3f}")
#             else:
#                 final_boxes = boxes_xyxy
#                 final_scores = scores_i
#                 final_labels = labels_i
#             # ==================================
            
#             gt_boxes = None
#             gt_labels = None
#             if gt_targets is not None and i < len(gt_targets):
#                 gt = gt_targets[i]
#                 if 'boxes' in gt:
#                     gt_boxes = gt['boxes'].cpu().numpy()
#                     if gt_boxes.max() <= 1.0:
#                         gt_boxes[:, [0, 2]] *= w
#                         gt_boxes[:, [1, 3]] *= h
#                     gt_boxes = gt_boxes.astype(np.int32)
#                     gt_labels = gt['labels'].cpu().numpy() if 'labels' in gt else None
            
#             img_id = gt_targets[i].get("image_id", i) if gt_targets else i
            
#             # 使用 NMS 后的结果进行绘制
#             if draw_mode == "pred_only":
#                 vis = self.draw_pred_boxes_pil(img_pil.copy(), final_boxes, final_scores, final_labels)
#                 filename = f"img{img_id}_pred_nms{nms_threshold}.jpg"
#             elif draw_mode == "gt_only" and gt_boxes is not None:
#                 vis = self.draw_gt_boxes_pil(img_pil.copy(), gt_boxes, gt_labels)
#                 filename = f"img{img_id}_gt.jpg"
#             elif draw_mode == "combined" and gt_boxes is not None:
#                 vis = self.draw_gt_and_pred_combined_pil(
#                     img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
#                 )
#                 filename = f"img{img_id}_combined_nms{nms_threshold}.jpg"
#             elif draw_mode == "compare" and gt_boxes is not None:
#                 vis = self.create_comparison_image_pil(
#                     img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
#                 )
#                 filename = f"img{img_id}_compare_nms{nms_threshold}.jpg"
#             else:
#                 continue
            
#             vis.save(str(save_path / filename))
            
#             # 可选：打印统计信息
#             if len(final_boxes) != len(boxes_xyxy):
#                 print(f"[图像 {img_id}] NMS: {len(boxes_xyxy)} -> {len(final_boxes)} (阈值={nms_threshold})")

#     def save_detection_json(self, images, outputs, save_path, image_ids=None, score_threshold=0.05):
#         """保存检测结果为 JSON"""
#         pred_logits = outputs['pred_logits']
#         pred_boxes = outputs['pred_boxes']
#         pred_scores = torch.sigmoid(pred_logits)
        
#         results = []
#         batch_size = images.shape[0]
        
#         for i in range(batch_size):
#             img = self.denormalize_image(images[i])
#             w, h = img.size
            
#             scores_i, labels_i = pred_scores[i].max(dim=-1)
#             scores_i = scores_i.flatten()
#             labels_i = labels_i.flatten()
#             boxes_i = pred_boxes[i]
            
#             boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
#             boxes_xyxy[:, [0, 2]] *= w
#             boxes_xyxy[:, [1, 3]] *= h
            
#             keep = scores_i > score_threshold
#             boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
#             scores_i = scores_i[keep].cpu().numpy()
#             labels_i = labels_i[keep].cpu().numpy()
            
#             img_id = image_ids[i] if image_ids else i
            
#             for box, score, label in zip(boxes_xyxy, scores_i, labels_i):
#                 x1, y1, x2, y2 = box
#                 results.append({
#                     "image_id": img_id,
#                     "category_id": int(label),
#                     "category_name": self.CLASS_NAMES[int(label)] if int(label) < len(self.CLASS_NAMES) else str(int(label)),
#                     "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
#                     "score": float(score),
#                 })
        
#         with open(save_path, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"\n保存检测结果到: {save_path}")
#         return results

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import torchvision  # 添加这个导入

class InferenceVisualizer:
    """
    推理时可视化工具（支持 PIL）
    输出英文类别名称，只显示全局置信度前 K 个预测框
    """
    
    # 类别名称
    CLASS_NAMES = [
        "normal",                    # 0
        "ascus",                     # 1
        "asch",                      # 2
        "lsil",                      # 3
        "hsil_scc_omn",              # 4
        "agc_adenocarcinoma_em",     # 5
        "vaginalis",                 # 6
        "monilia",                   # 7
        "dysbacteriosis_herpes_act", # 8
        "ec"                         # 9
    ]
    
    # 每类固定颜色（RGB格式）
    CLASS_COLORS = [
        (255, 0, 0),      # 0: normal - 红色
        (0, 100, 255),    # 1: ascus - 绿色
        (0, 0, 255),      # 2: asch - 蓝色
        (255, 255, 0),    # 3: lsil - 青色
        (255, 0, 255),    # 4: hsil_scc_omn - 品红
        (0, 255, 255),    # 5: agc_adenocarcinoma_em - 黄色
        (128, 0, 128),    # 6: vaginalis - 紫色
        (255, 128, 0),    # 7: monilia - 橙色
        (0, 128, 255),    # 8: dysbacteriosis_herpes_act - 天蓝
        (128, 128, 0),    # 9: ec - 橄榄
    ]
    
    def __init__(
        self,
        score_threshold: float = 0.05,
        top_k: int = 10,
        line_width: int = 2,
        font_size: int = 12,
    ):
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.line_width = line_width
        self.font_size = font_size
        
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            self.font = ImageFont.load_default()
    
    @staticmethod
    def tensor_to_pil(image_tensor):
        """将 tensor 转换为 PIL Image（与 save_samples 保持一致）"""
        if isinstance(image_tensor, torch.Tensor):
            # to_pil_image 会自动处理 [0,1] 或 [0,255] 范围的 tensor
            return to_pil_image(image_tensor.cpu())
        return image_tensor
    
    def draw_pred_boxes_pil(
        self,
        image: Image.Image,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Image.Image:
        """使用 PIL 绘制预测框"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        
        # 按置信度排序，取前 top_k 个
        if len(scores) > self.top_k:
            idx = np.argsort(scores)[::-1][:self.top_k]
            boxes = boxes[idx]
            scores = scores[idx]
            labels = labels[idx]
        
        for box, score, label in zip(boxes, scores, labels):
            if score < self.score_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            color = self.CLASS_COLORS[label % len(self.CLASS_COLORS)]
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"{class_name}: {score:.2f}"
            
            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
            
            # 测量文字大小
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # 绘制文字背景
            draw.rectangle(
                [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
                fill=color
            )
            
            # 绘制文字
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    def draw_gt_boxes_pil(
        self,
        image: Image.Image,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Image.Image:
        """使用 PIL 绘制 GT 框"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        gt_color = (0, 255, 0)  # 改为绿色，更明显
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width)
            
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"GT: {class_name}"
            
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    def draw_gt_and_pred_combined_pil(
        self,
        image: Image.Image,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Image.Image:
        """在一张图上同时绘制 GT 框和预测框"""
        # 先画预测框
        img_copy = self.draw_pred_boxes_pil(image, pred_boxes, pred_scores, pred_labels)
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        gt_color = (0, 255, 0)  # GT 用绿色
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
            
            # GT 框用虚线或更粗的线（PIL 不支持虚线，用粗线代替）
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=self.line_width + 1)
            
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else str(label)
            text = f"GT: {class_name}"
            
            bbox = draw.textbbox((x1, y1), text, font=self.font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=gt_color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=self.font)
        
        return img_copy
    
    def create_comparison_image_pil(
        self,
        image: Image.Image,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """创建对比图：左侧预测框，右侧GT框"""
        w, h = image.size
        
        left_img = self.draw_pred_boxes_pil(image.copy(), pred_boxes, pred_scores, pred_labels)
        
        if gt_boxes is not None:
            right_img = self.draw_gt_boxes_pil(image.copy(), gt_boxes, gt_labels)
            comparison = Image.new('RGB', (w * 2, h))
            comparison.paste(left_img, (0, 0))
            comparison.paste(right_img, (w, 0))
            return comparison
        else:
            return left_img
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        save_dir: str,
        batch_idx: int = 0,
        max_images: int = 8,
        gt_targets: Optional[List[Dict]] = None,
        draw_mode: str = "pred_only",
        random_sample: bool = True,
        nms_threshold: float = 0.5,
        conf_threshold: float = 0.05,
    ):
        """批量可视化推理结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_scores = torch.sigmoid(pred_logits)
        
        batch_total = images.shape[0]
        num_to_vis = min(batch_total, max_images)  
        if random_sample:
            indices = np.random.choice(batch_total, num_to_vis, replace=False)
        else:
            indices = range(min(batch_total, max_images))

        for idx_in_batch in indices:
            i = idx_in_batch
            
            # 关键：直接转换，不做任何归一化/反归一化
            img_pil = self.tensor_to_pil(images[i])
            w, h = img_pil.size
            
            # 获取预测结果
            scores_i, labels_i = pred_scores[i].max(dim=-1)
            scores_i = scores_i.flatten()
            labels_i = labels_i.flatten()
            boxes_i = pred_boxes[i]
            
            # 预测框是归一化的 [0,1]，需要转换到像素坐标
            boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h
            boxes_xyxy = boxes_xyxy.cpu().numpy().astype(np.int32)
            scores_i = scores_i.cpu().numpy()
            labels_i = labels_i.cpu().numpy()
            
            # NMS 过滤
            keep_conf = scores_i >= conf_threshold
            boxes_filtered = boxes_xyxy[keep_conf]
            scores_filtered = scores_i[keep_conf]
            labels_filtered = labels_i[keep_conf]
            
            if len(boxes_filtered) > 0:
                boxes_tensor = torch.tensor(boxes_filtered).float()
                scores_tensor = torch.tensor(scores_filtered).float()
                labels_tensor = torch.tensor(labels_filtered).long()
                
                keep_nms = torchvision.ops.batched_nms(
                    boxes_tensor, scores_tensor, labels_tensor, nms_threshold
                )
                
                final_boxes = boxes_filtered[keep_nms.cpu().numpy()]
                final_scores = scores_filtered[keep_nms.cpu().numpy()]
                final_labels = labels_filtered[keep_nms.cpu().numpy()]
            else:
                final_boxes = boxes_filtered
                final_scores = scores_filtered
                final_labels = labels_filtered
            
            # GT boxes 已经是像素坐标，不需要转换
            gt_boxes = None
            gt_labels = None
            if gt_targets is not None and i < len(gt_targets):
                gt = gt_targets[i]
                if 'boxes' in gt:
                    gt_boxes = gt['boxes'].cpu().numpy()
                    # GT boxes 已经是像素坐标，直接使用
                    gt_boxes = gt_boxes.astype(np.int32)
                    gt_labels = gt['labels'].cpu().numpy() if 'labels' in gt else None
            
            img_id = gt_targets[i].get("image_id", i) if gt_targets else i
            
            # 绘制并保存
            if draw_mode == "pred_only":
                vis = self.draw_pred_boxes_pil(img_pil.copy(), final_boxes, final_scores, final_labels)
                filename = f"img{img_id}_pred_nms{nms_threshold}.jpg"
            elif draw_mode == "gt_only" and gt_boxes is not None:
                vis = self.draw_gt_boxes_pil(img_pil.copy(), gt_boxes, gt_labels)
                filename = f"img{img_id}_gt.jpg"
            elif draw_mode == "combined" and gt_boxes is not None:
                vis = self.draw_gt_and_pred_combined_pil(
                    img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
                )
                filename = f"img{img_id}_combined_nms{nms_threshold}.jpg"
            elif draw_mode == "compare" and gt_boxes is not None:
                vis = self.create_comparison_image_pil(
                    img_pil.copy(), final_boxes, final_scores, final_labels, gt_boxes, gt_labels
                )
                filename = f"img{img_id}_compare_nms{nms_threshold}.jpg"
            else:
                continue
            
            vis.save(str(save_path / filename), quality=95)
    
    def save_detection_json(self, images, outputs, save_path, image_ids=None, score_threshold=0.05):
        """保存检测结果为 JSON"""
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        pred_scores = torch.sigmoid(pred_logits)
        
        results = []
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img = self.tensor_to_pil(images[i])
            w, h = img.size
            
            scores_i, labels_i = pred_scores[i].max(dim=-1)
            scores_i = scores_i.flatten()
            labels_i = labels_i.flatten()
            boxes_i = pred_boxes[i]
            
            boxes_xyxy = box_convert(boxes_i, in_fmt='cxcywh', out_fmt='xyxy')
            boxes_xyxy[:, [0, 2]] *= w
            boxes_xyxy[:, [1, 3]] *= h
            
            keep = scores_i > score_threshold
            boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
            scores_i = scores_i[keep].cpu().numpy()
            labels_i = labels_i[keep].cpu().numpy()
            
            img_id = image_ids[i] if image_ids else i
            
            for box, score, label in zip(boxes_xyxy, scores_i, labels_i):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "category_name": self.CLASS_NAMES[int(label)] if int(label) < len(self.CLASS_NAMES) else str(int(label)),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n保存检测结果到: {save_path}")
        return results



# ==================== 使用示例 ====================
def visualize_inference_example(model, samples, targets, output_dir, epoch=0):
    """
    在你的推理代码中集成
    """
    # 1. 初始化可视化器（全局置信度前10）
    visualizer = InferenceVisualizer(
        score_threshold=0.05,  # 低于0.05不显示
        top_k=10,              # 只显示前10个
        line_width=2,
    )
    
    # 2. 模型推理
    model.eval()
    with torch.no_grad():
        outputs, _ = model.module.sample(samples)  # 根据你的模型调整
    
    # 3. 可视化（多种模式可选）
    save_dir = f"{output_dir}/inference_vis_epoch{epoch}"
    
    # 模式1：只显示预测框
    visualizer.visualize_batch(
        images=samples,
        outputs=outputs,
        save_dir=save_dir,
        batch_idx=0,
        max_images=8,
        draw_mode="pred_only"
    )
    
    # 模式2：如果有GT，可以同时显示（并排对比）
    if targets is not None:
        visualizer.visualize_batch(
            images=samples,
            outputs=outputs,
            save_dir=save_dir,
            batch_idx=0,
            max_images=8,
            gt_targets=targets,
            draw_mode="compare"  # 或 "combined"
        )
    
    # 4. 保存 JSON 结果
    visualizer.save_detection_json(
        images=samples,
        outputs=outputs,
        save_path=f"{save_dir}/detections.json",
        score_threshold=0.1,
    )
    
    return visualizer

   