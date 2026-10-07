import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, ModuleList
import torch.fft
import torchvision.ops as ops
from torchvision.ops import roi_align
import math

from ...core import register
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .utils import get_activation

__all__ = [
    "VisualClassifier",
    ]

@register()
class VisualClassifier(nn.Module):
    __inject__ = [
        "matcher",
    ]

    def __init__(self, matcher, weight_dict, num_classes=10, hidden_dim=256, alpha=0.2, gamma=2.0,
            text_feats_path="/root/userfolder/Dataset/ObjectDetection/ComparisonDetectorDataset/pubmed_text11_new_feats.pt",
            num_decoder_layers=6,
            ):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.matcher = matcher
        self.num_decoder_layers = max(1, int(num_decoder_layers))

        self.vpe = VisualPromptEncoder(hidden_dim=hidden_dim, num_levels=3, depth=1, return_intermediate=True) 
        self.cls_head = ClassificationHead(hidden_dim, num_classes)
        self.cls_head_aux = ClassificationHead(hidden_dim, num_classes)
        self.aux_cls_head = nn.Linear(hidden_dim, num_classes)
        
        text_feats = torch.load(text_feats_path, map_location='cpu')["text_feats"]
        text_dim = text_feats.shape[1]

        self.text_adapter = TextAdapter(text_dim=text_dim, img_dim=hidden_dim, num_layers=1)
        self.register_buffer("class_text_feats", text_feats)
        # 可学习的 Logit Scale (初始化为 1/0.1 ≈ 10，即 ln(10) ≈ 2.3)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3026) 

        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.fusion_cls_head = ClassificationHead(hidden_dim, num_classes)


    def forward(self, feats, outputs, targets=None):
        #========开始
        pred_boxes = outputs['pred_boxes'] # [B, 300, 4] (cxcywh)
        quality_scores = outputs['quality_score'] # [B, 300, 1]
        query_feats = outputs["output"] # [B, N, C] D-FINE 解码器输出的 query 特征
        decoder_query_feats = outputs.get("decoder_query_feats")
        if decoder_query_feats is None:
            raise ValueError("decoder_query_feats must contain all decoder layers with shape [L, B, N, C].")
        # decoder_query_feats = decoder_query_feats.detach()
        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        
        # 提取特征并分类
        vpe_output = self.vpe(reference_boxes=pred_boxes, multi_scale_feats=feats)  # [B,N,C] 
        if not isinstance(vpe_output, list):
            vpe_output = [vpe_output]
        
        all_layer_vpe_feats = vpe_output  # List of [B, N, hidden_dim]
        num_layers = len(all_layer_vpe_feats)
        
        # 计算每一层的 logits（使用不同的分类头）
        all_layer_logits = []
        contrast_vpe_feats = None
        for layer_idx, layer_feats in enumerate(all_layer_vpe_feats):
            is_last_layer = (layer_idx == num_layers - 1)
            
            if is_last_layer:
                contrast_vpe_feats = layer_feats
                layer_logits = self.cls_head(layer_feats)
                aux_cls_logits = self.aux_cls_head(layer_feats)
                pred_vpe_feats = layer_feats
                pred_vpe_logits = layer_logits

                fusion_feats = self._dynamic_gated_residual_fusion(query_feats, layer_feats)
                fusion_logits = self.fusion_cls_head(fusion_feats)
            else:
                # 中间层使用辅助分类头
                layer_logits = self.cls_head_aux(layer_feats)
            
            all_layer_logits.append(layer_logits)
       
        # # 融合质量分数作为最终匹配/预测的依据
        refined_logits = fusion_logits + quality_scores
        # refined_logits = pred_vpe_logits + quality_scores
        refined_logits1 = pred_vpe_logits + quality_scores

        if self.training:            
            outputs_for_matcher = {
                'pred_logits': fusion_logits + quality_scores,
                'pred_boxes': pred_boxes
            }
            fusion_indices = self.matcher(outputs_for_matcher, targets)["indices"]
            outputs['fusion_logits'] = fusion_logits + quality_scores
            outputs['fusion_feats'] = fusion_feats          # [B, N, C]


            #=====================iou匹配===========================
            # iou一对多匹配版本
            indices, _ = rcnn_iou_match(
                pred_boxes=pred_boxes,
                targets=targets,
                pos_threshold=0.6,
                neg_threshold=0.5,
            )

            #=======================构造正样本的 Labels 和 IoUs===================
            flat_boxes_cxcywh = pred_boxes.reshape(-1, 4)
            flat_pred_boxes_xyxy = box_cxcywh_to_xyxy(flat_boxes_cxcywh)
            
            flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
            flatten_ious = torch.zeros((B, N), device=device)
            for i in range(B):
                src_idx, tgt_idx = indices[i]
                if len(src_idx) > 0:
                    flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
                    p_boxes = flat_pred_boxes_xyxy.view(B, N, 4)[i, src_idx]
                    t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx])
                    ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
                    flatten_ious[i, src_idx] = ious
            flatten_labels = flatten_labels.reshape(-1)
            flatten_ious = flatten_ious.reshape(-1)
            flat_pred_quality = quality_scores.reshape(-1, 1)
            flatten_vpe_logits = pred_vpe_logits.reshape(-1, pred_vpe_logits.shape[-1])

            # 存储中间层的 matched 数据
            layer_matched_aux = []
            for layer_idx, layer_logits in enumerate(all_layer_logits):
                is_last_layer = (layer_idx == num_layers - 1)
                if is_last_layer:
                    continue  # 最后一层的 loss 在主分类头里计算，不参与辅助分类头的损失计算

                indices_aux, _ = rcnn_iou_match(
                    pred_boxes=pred_boxes,
                    targets=targets,
                    pos_threshold=0.6,
                    neg_threshold=0.5,
                )
                flatten_labels_aux = torch.full((B, N), -1, dtype=torch.long, device=device)
                flatten_ious_aux = torch.zeros((B, N), device=device)
                for i in range(B):
                    src_idx_aux, tgt_idx_aux = indices_aux[i]
                    if len(src_idx_aux) > 0:
                        flatten_labels_aux[i, src_idx_aux] = targets[i]["labels"][tgt_idx_aux]
                        p_boxes = box_cxcywh_to_xyxy(pred_boxes[i, src_idx_aux])
                        t_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"][tgt_idx_aux])
                        ious = torch.diag(box_iou(p_boxes, t_boxes)[0])
                        flatten_ious_aux[i, src_idx_aux] = ious      
                flatten_labels_aux = flatten_labels_aux.reshape(-1)
                flatten_ious_aux = flatten_ious_aux.reshape(-1)
                flat_pred_quality_aux = quality_scores.reshape(-1, 1)
                flatten_vpe_logits_aux = layer_logits.reshape(-1, layer_logits.shape[-1])

                layer_result = {
                    "matched_vpe_logits_aux": flatten_vpe_logits_aux,
                    "matched_labels_aux": flatten_labels_aux,
                    "matched_ious_aux": flatten_ious_aux,
                    "matched_quality_aux": flat_pred_quality_aux,
                    "layer_idx": layer_idx,
                }
                layer_matched_aux.append(layer_result)

            # --- 获取匹配上的正样本特征 ---
            flatten_feats = contrast_vpe_feats.reshape(B*N, -1)  # [BN,C] 
            flatten_fusion_feats = fusion_feats.reshape(B*N, -1)
            pos_mask = (flatten_labels != -1).reshape(-1)

            pred_pos_feats = flatten_feats[pos_mask]
            pred_pos_labels = flatten_labels[pos_mask]       # [Num_Pred_Pos]
            pred_pos_ious = flatten_ious[pos_mask]

            # --------------------------------------------------
            # 在构造 pred_pos_feats 时顺便保存对应 GT index
            pred_pos_gt_indices = []
            offset = 0

            for i in range(B):
                src_idx, tgt_idx = indices[i]
                if len(src_idx) > 0:
                    pred_pos_gt_indices.append(tgt_idx + offset)
                offset += len(targets[i]["boxes"])

            if len(pred_pos_gt_indices) > 0:
                pred_pos_gt_indices = torch.cat(pred_pos_gt_indices)
            else:
                pred_pos_gt_indices = torch.empty(0, dtype=torch.long, device=device)

            # ================= [添加到全局收集器] =================
            # 根据特征获取当前的预测类别 (用于画混淆矩阵)
            pred_pos_preds = (flatten_vpe_logits+flat_pred_quality)[pos_mask].detach().argmax(dim=-1)
            if pred_pos_feats.shape[0] > 0:
                try:
                    from .plot_distribution import epoch_visualizer
                    epoch_visualizer.update(
                        feats=pred_pos_feats.detach(),
                        true_labels=pred_pos_labels.detach(),
                        pred_labels=pred_pos_preds,
                        all_logits=(flatten_vpe_logits+flat_pred_quality)[pos_mask].detach(),
                    )
                except Exception:
                    pass
            # ==============================================================
            
            
            # --- 获取 GT 框的特征 ---
            gt_boxes_list = [t["boxes"] for t in targets] # list of [Ni, 4]
            valid_gt_feats = flatten_fusion_feats.new_zeros((0, flatten_fusion_feats.shape[-1]))
            valid_gt_labels = torch.empty(0, dtype=torch.long, device=device)
            
            # 构造 Batch 化 GT 
            max_gts = max([len(b) for b in gt_boxes_list])
            if max_gts > 0:
                gt_boxes_padded = torch.zeros((B, max_gts, 4), device=device)
                gt_labels_padded = torch.full((B, max_gts), -1, dtype=torch.long, device=device)
                gt_mask = torch.zeros((B, max_gts), dtype=torch.bool, device=device)

                for i, boxes in enumerate(gt_boxes_list):
                    n_gt = len(boxes)
                    if n_gt > 0:
                        gt_boxes_padded[i, :n_gt] = boxes
                        gt_labels_padded[i, :n_gt] = targets[i]["labels"]
                        gt_mask[i, :n_gt] = True
                
                # 提取 GT 特征
                gt_feats= self.vpe(reference_boxes=gt_boxes_padded, multi_scale_feats=feats) # [B, Max_GT, C]
                
                if isinstance(gt_feats, list):
                    gt_feats = gt_feats[-1]  # 只取最后一层

                #只取有效的 GT 特征
                valid_gt_feats = gt_feats[gt_mask]  # [Total_GT, C]
                valid_gt_labels = gt_labels_padded[gt_mask] # [Total_GT]

                all_contrast_feats = valid_gt_feats # [Total_GT, C]
                all_contrast_labels = valid_gt_labels # [Total_GT]
            else:
                # 如果没有 GT (极端情况)，只用 pred
                all_contrast_feats = pred_pos_feats
                all_contrast_labels = pred_pos_labels

        
            # --- 计算对比损失所需的 Logits ---
            matched_sim_logits = None
            if all_contrast_feats.shape[0] > 0:
                v_proj = F.normalize(all_contrast_feats, dim=-1)
                adapted_text = self.text_adapter(self.class_text_feats)
                t_norm = F.normalize(adapted_text, dim=-1)

                # 使用可学习温度，并进行钳位防止溢出
                logit_scale = self.logit_scale.exp().clamp(max=100.0)
                # 计算相似度矩阵 [Total_Samples, Num_Classes]
                matched_sim_logits = logit_scale * torch.matmul(v_proj, t_norm.t())
            
            #把对比学习的数据单独存一个字段
            adapted_text = self.text_adapter(self.class_text_feats)  # [num_cls, C]
            t_norm = F.normalize(adapted_text, dim=-1)

            
            contrast_data = {
                "contrast_feats": all_contrast_feats,     # [Total_Samples, C]
                "contrast_labels": all_contrast_labels,   # [Total_Samples]
                

                "pred_pos_feats": pred_pos_feats,
                "pred_pos_ious": pred_pos_ious,
                "gt_feats": valid_gt_feats,
                "gt_labels": valid_gt_labels,
                "pred_pos_gt_indices": pred_pos_gt_indices,
                "text_feats": t_norm,
                "contrast_logits": matched_sim_logits,  

            }
            
            # outputs['pred_logits'] = refined_logits
            return {
                "matched_vpe_logits": flatten_vpe_logits,
                "aux_cls_logits": aux_cls_logits.reshape(-1, aux_cls_logits.shape[-1]),
                "matched_labels": flatten_labels,
                "matched_ious": flatten_ious,  #[BN]
                "matched_quality": flat_pred_quality,

                "layer_matched_aux": layer_matched_aux,
                "fusion_indices": fusion_indices,
                "targets": targets,

                **contrast_data,
                "num_boxes": self._get_num_boxes(targets, device),
                "outputs":outputs,  #联调用
                "feats": feats  #多尺度特征
            }
        else:
            # 推理模式：直接使用前面算好的 refined_logits
            outputs['pred_logits'] = refined_logits
            outputs['vpe_logits'] = refined_logits1      

            
            if targets is not None:
                # ================= [推理阶段特征与混淆矩阵收集] =================
                try:
                    fake_targets = []
                    for t in targets:
                        boxes_xyxy_norm = t["boxes"] / 640.                     
                        boxes_cxcywh_norm = box_xyxy_to_cxcywh(boxes_xyxy_norm)
                        fake_targets.append({"labels": t["labels"], "boxes": boxes_cxcywh_norm})
       
                    outputs_for_matcher = {
                        'pred_logits': refined_logits, 
                        'pred_boxes': pred_boxes
                    }
                    indices = self.matcher(outputs_for_matcher, targets)["indices"]
                    # indices = rcnn_iou_match(
                    #     pred_boxes=pred_boxes,
                    #     targets=fake_targets,
                    #     pos_threshold=0.6,   
                    # )[0]
                    
                    flatten_labels = torch.full((B, N), -1, dtype=torch.long, device=device)
                    for i in range(B):
                        src_idx, tgt_idx = indices[i]
                        if len(src_idx) > 0:
                            flatten_labels[i, src_idx] = targets[i]["labels"][tgt_idx]
                    
                    flatten_labels = flatten_labels.reshape(-1)
                    flatten_vpe_logits = refined_logits.reshape(-1, self.num_classes)
                    flatten_feats = fusion_feats.reshape(B*N, -1)
                    
                    pos_mask = (flatten_labels != -1).reshape(-1)
                    if pos_mask.any():
                        pred_pos_feats = flatten_feats[pos_mask]
                        pred_pos_labels = flatten_labels[pos_mask]
                        pred_pos_preds = flatten_vpe_logits[pos_mask].detach().argmax(dim=-1)
                        
                        from .plot_distribution import epoch_visualizer
                        epoch_visualizer.update(
                            feats=pred_pos_feats, 
                            true_labels=pred_pos_labels, 
                            pred_labels=pred_pos_preds,
                            all_logits=flatten_vpe_logits[pos_mask]
                        )
                except Exception as e:
                    pass
                
            return outputs

    def _dynamic_gated_residual_fusion(self, query_feats, vpe_feats):
        gate = torch.sigmoid(self.gate_proj(query_feats))
        return query_feats + 0.5 * gate * vpe_feats

    def _get_num_boxes(self, targets, device):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        return torch.clamp(num_boxes / get_world_size(), min=1).item()


    def get_losses(self, outputs,  **kwargs):
        num_boxes = outputs["num_boxes"]
        losses = {}
        if self.training:
            # 主分支
            losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
            losses.update(losses_main)

            losses.update(self.loss_labels_vfl(outputs["outputs"], outputs["targets"], outputs["fusion_indices"], num_boxes))

            # Contrastive Loss
            losses.update(self.loss_contrast_matched(outputs, num_boxes))

            losses = {
                k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
            }
            losses = {k + "_vpe_cls": v for k, v in losses.items()}

        else:
            # 主分支
            losses_main = self.loss_labels_matched_branch(outputs, num_boxes, branch="main")
            losses.update(losses_main)
            losses = {
                k: losses[k] * self.weight_dict[k] for k in losses if k in self.weight_dict
            }
            losses = {k + "_test_vpe_cls": v for k, v in losses.items()}
        
        return losses

    def loss_labels_matched_branch(self, outputs, num_boxes, branch="main"):
        if branch == "main":
            src_logits = outputs["matched_vpe_logits"]
            target_labels = outputs["matched_labels"]
            ious = outputs["matched_ious"].detach()
            quality = outputs["matched_quality"]
            loss_name = "loss_cls"


        if quality.dim() == 1:
            quality = quality.unsqueeze(-1)
        fused_logits = src_logits + quality

        target_classes = target_labels.clone()
        target_classes[target_classes == -1] = self.num_classes 
        # target 形状为 [4800, 10]
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].float()

        # 构造 Target Score (IoU 软标签)
        target_score = target * ious.view(-1, 1).pow(0.5)
        # # target_score = target * ious.view(-1, 1)

        with torch.no_grad():
            pred_score = fused_logits.sigmoid().detach()
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            fused_logits, target_score, weight=weight, reduction="none"
        )   # [B*N,10]

        return {loss_name: loss.sum() / num_boxes}


    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["fusion_logits"]  #[B,N,C]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype).pow(0.5)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()

        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_fusion_cls": loss}



    def loss_contrast_matched(self, outputs, num_boxes):
        pred_feats = outputs["pred_pos_feats"]
        gt_feats = outputs["gt_feats"]
        gt_labels = outputs["gt_labels"]
        match_idx = outputs["pred_pos_gt_indices"]
        t_norm = outputs["text_feats"]
        pred_pos_ious = outputs["pred_pos_ious"]  #[num_pos], aligned with pred_pos_feats

        gt_text_sim_logits = outputs.get("contrast_logits") # [Total_Samples, 10]

        device = gt_feats.device
        dtype = gt_feats.dtype 

        if gt_feats.shape[0] == 0:
            return {"loss_contrast": torch.tensor(0.0, device=device, dtype=dtype)}

        # 统一 normalize
        v_gt = F.normalize(gt_feats, dim=-1)
        v_pred = F.normalize(pred_feats, dim=-1) if pred_feats is not None else None

        loss = 0.0

        # 权重
        w_inst = 1.0
        w_cls = 0.3
        w_struct = 3

        # ================= ① Pred ↔ GT =================
        if v_pred is not None and match_idx.shape[0] > 0:
            if match_idx.shape[0] != v_pred.shape[0] or pred_pos_ious.shape[0] != v_pred.shape[0]:
                raise RuntimeError(
                    "Contrastive positive tensors are misaligned: "
                    + "pred_feats={}, gt_indices={}, ious={}".format(
                        v_pred.shape[0], match_idx.shape[0], pred_pos_ious.shape[0]
                    )
                )

            matched_gt = v_gt[match_idx]           # [num_pos, C]
            pos_ious = pred_pos_ious             # [num_pos], already aligned with pred_feats

            sim = F.cosine_similarity(v_pred, matched_gt, dim=-1)

            target = (pos_ious * 2 - 1).to(dtype)               # 映射到 [-1,1]
            weight = pos_ious.detach().to(dtype)             # IoU加权
            sim = sim.to(dtype)
            loss_inst = (weight * (sim - target.detach()) ** 2).mean()

            loss += w_inst * loss_inst

        # # ================= ② GT ↔ Text =================
        loss += w_cls * F.cross_entropy(gt_text_sim_logits, gt_labels).to(dtype)

        # ================= ③ 结构对齐 =================
        if v_gt.shape[0] > 1:
            sim_gt = torch.matmul(v_gt, v_gt.t()).float()

            text_selected = t_norm[gt_labels]
            sim_text = torch.matmul(text_selected, text_selected.t()).float()

            loss += w_struct * F.mse_loss(sim_gt, sim_text.detach())

        return {"loss_contrast": loss}


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def rcnn_iou_match(pred_boxes, targets, pos_threshold=0.6, neg_threshold=0.3):
    """
    Mask R-CNN / Faster R-CNN 风格匹配
    
    Args:
        pred_boxes: [B, N, 4] (cxcywh)
        targets: list of dicts, 每个包含 'boxes' (cxcywh)
        pos_threshold: 正样本 IoU 阈值 (默认 0.7)
        neg_threshold: 负样本 IoU 阈值 (默认 0.3)
    
    Returns:
        pos_indices: list of (src_idx, tgt_idx) - 正样本匹配
        neg_indices: list of src_idx - 负样本索引（无 GT 匹配）
    """
    B, N, _ = pred_boxes.shape
    device = pred_boxes.device

    # 转 xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

    pos_indices = []
    neg_indices = []

    for i in range(B):
        gt_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"])

        # 没有 GT 的情况：所有 query 都是负样本
        if len(gt_boxes) == 0:
            pos_indices.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            ))
            neg_indices.append(torch.arange(N, device=device))
            continue

        # IoU: [N, M]
        ious = box_iou(boxes_xyxy[i], gt_boxes)[0]

        # 每个 query 找最大 IoU 的 GT
        max_ious, matched_gt_idx = ious.max(dim=1)

        # 正样本：IoU >= pos_threshold
        pos_mask = max_ious >= pos_threshold
        pos_src_idx = torch.where(pos_mask)[0]
        pos_tgt_idx = matched_gt_idx[pos_mask]
        pos_indices.append((pos_src_idx, pos_tgt_idx))

        # 负样本：IoU < neg_threshold
        neg_mask = max_ious < neg_threshold
        neg_src_idx = torch.where(neg_mask)[0]
        neg_indices.append(neg_src_idx)

    return pos_indices, neg_indices

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SimpleMSDeformableAttention(nn.Module):
    """
    简化版 Multi-Scale Deformable Attention
    适合 ROI-like feature extraction
    """

    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.head_dim = embed_dim // num_heads

        # offset prediction
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )

        # attention weights
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points, multi_scale_feats):
        B, N, C = query.shape
        num_levels = len(multi_scale_feats)
        
        # 计算 offsets 和 attention weights
        offsets = self.sampling_offsets(query)  # [B, N, num_heads * num_levels * num_points * 2]
        offsets = offsets.view(B, N, self.num_heads, num_levels, self.num_points, 2)

        attn = self.attention_weights(query)  # [B, N, num_heads * num_levels * num_points]
        attn = attn.view(B, N, self.num_heads, num_levels * self.num_points)
        attn = attn.softmax(-1).view(B, N, self.num_heads, num_levels, self.num_points)

        # 统一对所有尺度的特征进行投影
        value_proj_list = []
        for feat in multi_scale_feats:
            Bf, Cf, H, W = feat.shape
            # 投影并保持空间结构
            feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            feat_proj = self.value_proj(feat_flat)  # [B, H*W, C]
            feat_proj = feat_proj.view(B, -1, self.num_heads, self.head_dim)   # [B, H*W, num_heads, head_dim]
            value_proj_list.append(feat_proj)
        
        # 初始化输出 [B, N, num_heads, head_dim]
        output = torch.zeros(B, N, self.num_heads, self.head_dim, device=query.device)

        # 分离中心点和宽高
        ref_cxcy = reference_points[..., :2]  # [B, N, num_levels, 2]
        ref_wh = reference_points[..., 2:4]   # [B, N, num_levels, 2]
        
        # 对每个尺度进行采样
        for lvl in range(num_levels):
            value = value_proj_list[lvl]   # 当前尺度的value: [B, H*W, num_heads, head_dim]
            H, W = multi_scale_feats[lvl].shape[-2:]
            num_tokens = H * W
            
            ref_lvl = ref_cxcy[:, :, lvl]  # [B, N, 2]
            wh_lvl = ref_wh[:, :, lvl]     # [B, N, 2]
            
            # 将参考点从[0,1]映射到[-1,1]用于grid_sample
            ref_lvl = ref_lvl * 2 - 1  # [B, N, 2]

            # 将box_wh也映射到[-1,1]空间（实际上范围是[0,2]）
            # 因为wh在[0,1]，乘以2后范围[0,2]
            box_wh = wh_lvl * 2 # [B, N, 2]
            
            offset_lvl = offsets[:, :, :, lvl]  # 当前尺度的offsets: [B, N, num_heads, num_points, 2]
            
            # 对每个头独立处理
            for h in range(self.num_heads):
                # 当前头的value: [B, H*W, head_dim]
                value_h = value[..., h, :]  # [B, num_tokens, head_dim]
                
                # 当前头的offsets: [B, N, num_points, 2]
                offset_h = offset_lvl[:, :, h]  # [B, N, num_points, 2]
                                           
                # 计算采样点位置
                sample_points = ref_lvl[:, :, None, :] + offset_h * box_wh[:, :, None, :]
                sample_points = sample_points.clamp(-1, 1)  # [B, N, num_points, 2]
                
                # 重塑为grid_sample格式: [B, N*num_points, 1, 2]
                grid = sample_points.view(B, N * self.num_points, 1, 2)
                
                # 准备value用于采样: [B, head_dim, H, W]
                value_h_spatial = value_h.transpose(1, 2).reshape(B, self.head_dim, H, W)
                
                # 采样: [B, head_dim, N*num_points, 1]
                sampled = F.grid_sample(
                    value_h_spatial,
                    grid,
                    align_corners=False,
                    mode='bilinear',
                    padding_mode='zeros'
                )
                
                sampled = sampled.view(B, self.head_dim, N, self.num_points)  #[B, head_dim, N, num_points]               
                sampled = sampled.permute(0, 2, 3, 1)  # [B, N, num_points, head_dim]
                
                attn_h = attn[:, :, h, lvl]  # 当前头的attention weights: [B, N, num_points]
                # 加权求和: [B, N, head_dim]
                weighted_sampled = (sampled * attn_h.unsqueeze(-1)).sum(dim=2)
                
                # 累加到对应头的输出
                output[:, :, h, :] += weighted_sampled
        
        # 合并多头并输出
        output = output.view(B, N, C)  # [B, N, num_heads, head_dim] -> [B, N, C]
        
        return self.output_proj(output)

class DifferentialBoxAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_tokens=8, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pos_pool_proj = nn.Linear(hidden_dim, num_tokens)
        self.neg_pool_proj = nn.Linear(hidden_dim, num_tokens)
        self.out_norm = RMSNorm(hidden_dim)
        self.lambda_param = nn.Parameter(torch.full((num_heads,), math.log(math.exp(0.5) - 1))) 
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, query):
        batch_size, num_queries, hidden_dim = query.shape
        pos_pool_logits = self.pos_pool_proj(query).transpose(1, 2)
        neg_pool_logits = self.neg_pool_proj(query).transpose(1, 2)
        pos_pool_weights = pos_pool_logits.softmax(dim=-1)
        neg_pool_weights = neg_pool_logits.softmax(dim=-1)
        pos_tokens = torch.matmul(pos_pool_weights, query)
        neg_tokens = torch.matmul(neg_pool_weights, query)

        query_heads = query.view(
            batch_size, num_queries, self.num_heads, self.head_dim
        ).transpose(1, 2)
        pos_token_heads = pos_tokens.view(
            batch_size, self.num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        neg_token_heads = neg_tokens.view(
            batch_size, self.num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scale = self.head_dim ** -0.5
        pos_scores = torch.matmul(
            query_heads, pos_token_heads.transpose(-1, -2)
        ) * scale
        neg_scores = torch.matmul(
            query_heads, neg_token_heads.transpose(-1, -2)
        ) * scale
        pos_weights = pos_scores.softmax(dim=-1)
        neg_weights = neg_scores.softmax(dim=-1)
        pos_out = torch.matmul(pos_weights, pos_token_heads)
        neg_out = torch.matmul(neg_weights, neg_token_heads)

        lambda_value = F.softplus(self.lambda_param).view(1, self.num_heads, 1, 1)
        differential_heads = pos_out - lambda_value * neg_out
        differential_out = differential_heads.transpose(1, 2).reshape(
            batch_size, num_queries, hidden_dim
        )
        differential_out = self.out_norm(differential_out)

        return self.residual_scale.tanh() * differential_out


class VisualPromptEncoderBlock(nn.Module):
    """
    单个 Visual Prompt Encoder Block
    包含：Cross-Attention + Self-Attention + FFN
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=6, ffn_dim=1024):
        super().__init__()
        
        # Cross-Attention
        self.cross_attn = SimpleMSDeformableAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        
        # 原始 Self-Attention
        # self.self_attn = MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=num_heads,
        #     batch_first=True
        # )
        self.differential_attn = DifferentialBoxAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_tokens=8,
            dropout=0.1,
        )
        
        # FFN
        self.ffn = SwiGLUFFN(
            in_features=hidden_dim, 
            hidden_features=ffn_dim, 
            out_features=hidden_dim
        )
        
        # Layer Norms (Pre-Norm 风格)
        self.norm_cross = RMSNorm(hidden_dim)
        self.norm_self = RMSNorm(hidden_dim)
        self.norm_ffn = RMSNorm(hidden_dim)
        
        # Dropout
        self.dropout_cross = nn.Dropout(0.1)
        self.dropout_self = nn.Dropout(0.1)
        self.dropout_ffn = nn.Dropout(0.1)
    
    def forward(self, query, multi_scale_feats, reference_points):
        """
        Args:
            query: [B, N, C]
            multi_scale_feats: List of multi-scale features
            reference_points: [B, N, num_levels, 2] or [B, N, 2]
        """
        # Pre-Norm Cross-Attention
        residual = query
        query = self.norm_cross(query)
        attn_out = self.cross_attn(
            query=query,
            reference_points=reference_points,
            multi_scale_feats=multi_scale_feats
        )
        query = residual + self.dropout_cross(attn_out)
        
        # Pre-Norm Differential Attention
        residual = query
        normalized_query = self.norm_self(query)
        differential_out = self.differential_attn(normalized_query)
        query = residual + self.dropout_self(differential_out)
        # 原始自注意力调用保留，不删除：
        # residual = query
        # query = self.norm_self(query)
        # self_attn_out, _ = self.self_attn(query=query, key=query, value=query)
        # query = residual + self.dropout_self(self_attn_out)
        
        # Pre-Norm FFN
        residual = query
        query = self.norm_ffn(query)
        ffn_out = self.ffn(query)
        query = residual + self.dropout_ffn(ffn_out)
        
        return query

class VisualPromptEncoder(nn.Module):
    """
    基于 multi-scale 注意力提取预测框特征，支持多层深度
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=6, 
                 ffn_dim=1024, depth=1, return_intermediate=False):  # 新增 depth 参数
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.depth = depth
        
        # 框坐标编码
        self.visual_bbox_encoding = MLP(hidden_dim * 2 + 8, hidden_dim, hidden_dim, num_layers=2)
        
        # 纹理特征辅助提取分支
        self.texture_extract = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(inplace=True),
            # CBAM(hidden_dim),
            nn.AdaptiveAvgPool2d(1)
        )
        self.texture_fusion = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.texture_fusion.weight)
        nn.init.zeros_(self.texture_fusion.bias)

        
        # 堆叠多个 Block
        self.blocks = nn.ModuleList([
            VisualPromptEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                ffn_dim=ffn_dim
            )
            for _ in range(depth)
        ])

        self.return_intermediate = return_intermediate  # 是否返回中间层特征
        
        # 最终的 LayerNorm（可选）
        self.final_norm = RMSNorm(hidden_dim) if depth > 0 else nn.Identity()
    
    def forward(self, reference_boxes: torch.Tensor, multi_scale_feats: list):
        """
        reference_boxes: [B, N, 4], cxcywh, 归一化到[0,1]
        multi_scale_feats: List[B, C, H, W], 多尺度特征
        """
        B, N, _ = reference_boxes.shape
        device = reference_boxes.device

        # --- 坐标编码 ---
        query_sine_embed = self.coordinate_to_encoding(reference_boxes)
        query = self.visual_bbox_encoding(query_sine_embed)

        # --- 纹理特征辅助 ---
        finest_feat = multi_scale_feats[0]
        _, _, H_feat, W_feat = finest_feat.shape
        
        # 构建 RoIs
        rois_list = []
        for i in range(B):
            boxes_pixels = box_cxcywh_to_xyxy(reference_boxes[i])
            boxes_pixels[:, 0::2] *= W_feat
            boxes_pixels[:, 1::2] *= H_feat
            idx_col = torch.full((N, 1), i, device=device)
            rois_list.append(torch.cat([idx_col, boxes_pixels], dim=1))
        
        all_rois = torch.cat(rois_list, dim=0)
        
        texture_feats = roi_align(
            finest_feat, all_rois, output_size=(7, 7), 
            spatial_scale=1.0, sampling_ratio=2, aligned=True
        )
        texture_vector = self.texture_extract(texture_feats)
        texture_vector = texture_vector.flatten(1).view(B, N, -1)
        content_embed = self.texture_fusion(texture_vector)
        query = query + content_embed

        # 准备参考点（多尺度）
        reference_points = reference_boxes[:, :, None, :].repeat(1, 1, self.num_levels, 1)
        
        intermediate_outputs = []

        # 通过多个 Block 处理
        for block in self.blocks:
            query = block(query, multi_scale_feats, reference_points)
            # 收集每一层的输出（包括最后一层）
            if self.return_intermediate:
                intermediate_outputs.append(query)
        
        # # 最终归一化
        # if self.final_norm is not None:
        #     query = self.final_norm(query)
        if self.return_intermediate:
            intermediate_outputs[-1] = query  # 替换最后一层为 final_norm 后的版本
        
        if self.return_intermediate:
            return intermediate_outputs  # List of [B, N, hidden_dim], 长度 = depth
        else:
            return query  # [B, N, hidden_dim]
    
    @staticmethod
    def coordinate_to_encoding(boxes: torch.Tensor):
        """坐标编码"""
        boxes = boxes.clamp(min=1e-6, max=1.0)
        B, N, _ = boxes.shape
        device = boxes.device
        
        scale = 1000.0
        boxes_scaled = boxes * scale
        
        dim_t_orig = torch.arange(0, 64, device=device)
        dim_t_orig = 10000 ** (2 * (dim_t_orig // 2) / 128)
        boxes_scaled_orig = boxes_scaled[:, :, :, None] / dim_t_orig
        pos_sine = torch.stack([
            boxes_scaled_orig[:, :, 0].sin(), boxes_scaled_orig[:, :, 0].cos(),
            boxes_scaled_orig[:, :, 1].sin(), boxes_scaled_orig[:, :, 1].cos(),
            boxes_scaled_orig[:, :, 2].sin(), boxes_scaled_orig[:, :, 2].cos(),
            boxes_scaled_orig[:, :, 3].sin(), boxes_scaled_orig[:, :, 3].cos()
        ], dim=-1).flatten(2)
        
        w, h = boxes[..., 2], boxes[..., 3]
        aspect_ratio = w / (h + 1e-6)
        area = w * h
        aspect_ratio = aspect_ratio.clamp(min=1e-5, max=1e5)
        area = area.clamp(min=1e-6)
        geo_feats = torch.stack([
            w, h, 
            aspect_ratio.log(), 
            area.sqrt()
        ], dim=-1)
        geo_embed = torch.cat([geo_feats.sin(), geo_feats.cos()], dim=-1)
        
        pos = torch.cat([pos_sine, geo_embed], dim=-1)
        return pos

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(hidden_dim, num_classes)
        
        prior_prob = 0.01
        # 根据 sigmoid 反函数计算 bias: bias = -log((1 - p) / p)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        # 初始化权重为很小的随机值
        nn.init.normal_(self.head.weight, std=0.01)
        # 初始化偏置，使模型初始输出的概率接近 0.01
        nn.init.constant_(self.head.bias, bias_value)

    def forward(self, x):
        return self.head(x)

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.constant_(self.w12.bias, 0)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.avg_pool(x).view(B, C)
        max_ = self.max_pool(x).view(B, C)

        attn = self.mlp(avg) + self.mlp(max_)
        attn = torch.sigmoid(attn).view(B, C, 1, 1)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(concat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class GatedFFNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * dim * 2) 
        self.w3 = nn.Linear(dim * 2, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x1 = self.norm(x)
        x12 = self.w12(x1)
        x1, x2 = x12.chunk(2, dim=-1)
        gated = F.silu(x1) * x2
        out = x + self.w3(gated)
        return out

class TextAdapter(nn.Module):
    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedFFNBlock(text_dim) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(text_dim, img_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.w12.weight)
            nn.init.constant_(layer.w12.bias, 0)
            nn.init.xavier_uniform_(layer.w3.weight)
            nn.init.constant_(layer.w3.bias, 0)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, text_feats):
        x = text_feats
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)
