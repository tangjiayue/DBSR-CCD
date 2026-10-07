import argparse
import sys
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchvision.ops import box_convert, box_iou, roi_align

from src.core import YAMLConfig


CLASS_NAMES = ['Normal0', 'ASC-US1', 'ASC-H2', 'LSIL3', 'HSIL/SCC4', 'AGC5', 'VAG6', 'MON7', 'DYS8', 'EC9']


def load_tuning_state(model, path):
    state = torch.load(path, map_location='cpu')
    params = state.get('ema', {}).get('module', None)
    if params is None:
        params = state.get('model', state)
    cur = model.state_dict()
    matched = {}
    missed = []
    unmatched = []
    for k, v in cur.items():
        if k not in params:
            missed.append(k)
        elif params[k].shape == v.shape:
            matched[k] = params[k]
        else:
            unmatched.append(k)
    model.load_state_dict(matched, strict=False)
    print(f"Loaded checkpoint: {path}")
    print(f"matched={len(matched)}, missed={len(missed)}, unmatched_shape={len(unmatched)}")
    if missed[:8]:
        print('missed sample:', missed[:8])
    if unmatched[:8]:
        print('unmatched sample:', unmatched[:8])


def build_cfg(args):
    cfg = YAMLConfig(args.config)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    if 'val_dataloader' in cfg.yaml_cfg:
        vd = cfg.yaml_cfg['val_dataloader']
        vd.pop('total_batch_size', None)
        vd['batch_size'] = args.batch_size
        vd['num_workers'] = args.num_workers
        vd['drop_last'] = False
        vd['shuffle'] = False
        if args.num_workers == 0:
            vd['persistent_workers'] = False
    cfg.device = args.device
    return cfg


def bucket_name(score):
    if score < 0.3:
        return 'easy'
    if score < 0.7:
        return 'medium'
    return 'high'


def label_name(label):
    label = int(label)
    if 0 <= label < len(CLASS_NAMES):
        return CLASS_NAMES[label]
    return str(label)


def normalized_boxes_to_level_rois(boxes_cxcywh, batch_idx, feat_hw):
    h, w = feat_hw
    boxes_xyxy = box_convert(boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
    boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)
    boxes_xyxy[:, 0::2] *= w
    boxes_xyxy[:, 1::2] *= h
    batch_col = batch_idx.to(boxes_xyxy.device, dtype=boxes_xyxy.dtype).view(-1, 1)
    return torch.cat([batch_col, boxes_xyxy], dim=1)


def roi_pool_multiscale(feats, boxes_cxcywh, batch_idx):
    pooled = []
    for feat in feats:
        _, _, h, w = feat.shape
        rois = normalized_boxes_to_level_rois(boxes_cxcywh, batch_idx, (h, w))
        out = roi_align(feat.float(), rois.float(), output_size=(1, 1), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        pooled.append(out.flatten(1))
    return torch.cat(pooled, dim=1)


def choose_indices(statuses, max_per_group, current_counts, rng):
    selected = []
    by_group = defaultdict(list)
    for idx, status in enumerate(statuses):
        by_group[status].append(idx)
    for status, idxs in by_group.items():
        remain = max_per_group - current_counts[status]
        if remain <= 0:
            continue
        if len(idxs) > remain:
            idxs = rng.sample(idxs, remain)
        selected.extend(idxs)
        current_counts[status] += len(idxs)
    selected.sort()
    return selected


def flatten_jsonable(records):
    out = []
    for r in records:
        item = {}
        for k, v in r.items():
            if isinstance(v, np.integer):
                item[k] = int(v)
            elif isinstance(v, np.floating):
                item[k] = float(v)
            else:
                item[k] = v
        out.append(item)
    return out


def run_linear_probe(features, labels, seed):
    labels = np.asarray(labels)
    keep = labels != 'ignore'
    features = np.asarray(features)[keep]
    labels = labels[keep]
    counts = Counter(labels.tolist())
    valid_classes = [k for k, v in counts.items() if v >= 8]
    keep = np.asarray([x in valid_classes for x in labels])
    features = features[keep]
    labels = labels[keep]
    if len(valid_classes) < 2 or features.shape[0] < 30:
        return {'skipped': True, 'reason': 'not enough classes/samples', 'counts': dict(counts)}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {'skipped': True, 'reason': f'sklearn import failed: {e}', 'counts': dict(counts)}

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=seed, stratify=labels
        )
    except Exception as e:
        return {'skipped': True, 'reason': f'train_test_split failed: {e}', 'counts': dict(counts)}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        'skipped': False,
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'classes': sorted(valid_classes),
        'counts': dict(Counter(labels.tolist())),
        'accuracy': float(accuracy_score(y_test, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
        'macro_f1': float(f1_score(y_test, pred, average='macro')),
    }


def plot_tsne(features, labels, save_path, seed, max_points=3000):
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except Exception as e:
        print(f'Skip t-SNE {save_path}: {e}')
        return False

    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.shape[0] < 10:
        return False
    rng = np.random.default_rng(seed)
    if features.shape[0] > max_points:
        idx = rng.choice(features.shape[0], max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
    perplexity = min(30, max(2, features.shape[0] // 20))
    emb = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perplexity, random_state=seed).fit_transform(features)
    uniq = sorted(set(labels.tolist()))
    cmap = plt.get_cmap('tab10')
    plt.figure(figsize=(10, 8))
    for i, lab in enumerate(uniq):
        m = labels == lab
        plt.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.65, label=lab, color=cmap(i % 10))
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    return True


def summarize(records):
    total = Counter(r['status'] for r in records)
    by_bucket = Counter(f"{r['status']}:{r['bucket']}" for r in records)
    by_pred = Counter(f"{r['status']}:{r['pred_label']}" for r in records)
    score_stats = {}
    for status in sorted(total):
        vals = np.asarray([r['score'] for r in records if r['status'] == status], dtype=np.float32)
        ious = np.asarray([r['max_iou'] for r in records if r['status'] == status], dtype=np.float32)
        score_stats[status] = {
            'count': int(vals.shape[0]),
            'score_mean': float(vals.mean()) if vals.size else 0.0,
            'score_p50': float(np.median(vals)) if vals.size else 0.0,
            'score_p90': float(np.quantile(vals, 0.9)) if vals.size else 0.0,
            'iou_mean': float(ious.mean()) if ious.size else 0.0,
        }
    return {
        'status_counts': dict(total),
        'status_bucket_counts': dict(by_bucket),
        'status_pred_label_counts': dict(by_pred),
        'score_stats': score_stats,
    }


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = cfg.model
    load_tuning_state(model, args.checkpoint)
    model.to(device).eval()

    loader = cfg.val_dataloader
    print(f'Analyzing val loader: batches={len(loader)}, batch_size={args.batch_size}, device={device}')

    all_records = []
    selected_records = []
    selected_counts = defaultdict(int)
    feature_store = {'vpe': [], 'backbone_roi': [], 'pre_fpn_roi': [], 'encoder_roi': []}
    status_labels = []
    binary_tp_vs_fpbg = []
    binary_tp_vs_allfp = []

    total_seen = 0
    for batch_i, (samples, targets) in enumerate(loader):
        if args.max_batches is not None and batch_i >= args.max_batches:
            break
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        backbone_feats = model.backbone(samples)
        encoder_feats = model.encoder(backbone_feats)
        outputs = model.decoder(encoder_feats)
        outputs = model.VisualClassifier(encoder_feats, outputs, targets)
        logits = outputs['pred_logits'].detach()
        probs = logits.sigmoid()
        scores, pred_labels = probs.max(dim=-1)
        pred_boxes = outputs['pred_boxes'].detach()

        vpe_feats = model.VisualClassifier.vpe(reference_boxes=pred_boxes, multi_scale_feats=encoder_feats)
        if isinstance(vpe_feats, list):
            vpe_feats = vpe_feats[-1]
        vpe_feats = vpe_feats.detach()

        bsz, nq = pred_boxes.shape[:2]
        sample_h, sample_w = samples.shape[-2:]
        per_query = []

        for b in range(bsz):
            gt_labels_all = targets[b]['labels']
            gt_boxes_all = targets[b]['boxes']
            valid_gt = gt_labels_all >= 0
            gt_labels = gt_labels_all[valid_gt]
            gt_boxes = gt_boxes_all[valid_gt]

            pred_xyxy = box_convert(pred_boxes[b], in_fmt='cxcywh', out_fmt='xyxy').clamp(0.0, 1.0)
            pred_xyxy_abs = pred_xyxy.clone()
            pred_xyxy_abs[:, 0::2] *= sample_w
            pred_xyxy_abs[:, 1::2] *= sample_h

            if gt_boxes.numel() > 0:
                ious = box_iou(pred_xyxy_abs, gt_boxes)
                max_iou, gt_idx = ious.max(dim=1)
                best_gt_labels = gt_labels[gt_idx]
            else:
                max_iou = torch.zeros(nq, device=device)
                gt_idx = torch.full((nq,), -1, device=device, dtype=torch.long)
                best_gt_labels = torch.full((nq,), -1, device=device, dtype=torch.long)

            image_id_t = targets[b].get('image_id', torch.tensor(-1, device=device))
            image_id = int(image_id_t.item()) if torch.is_tensor(image_id_t) else int(image_id_t)

            for q in range(nq):
                score = float(scores[b, q].item())
                pred_label = int(pred_labels[b, q].item())
                miou = float(max_iou[q].item())
                gt_label = int(best_gt_labels[q].item()) if best_gt_labels.numel() else -1
                bucket = bucket_name(score)
                if miou >= args.iou_thr and gt_label == pred_label:
                    status = 'tp'
                elif miou >= args.iou_thr:
                    status = f'fp_cls_{bucket}'
                else:
                    status = f'fp_bg_{bucket}'
                rec = {
                    'image_id': image_id,
                    'batch_idx': int(batch_i),
                    'batch_image_idx': int(b),
                    'query_index': int(q),
                    'status': status,
                    'bucket': bucket,
                    'score': score,
                    'max_iou': miou,
                    'pred_label': pred_label,
                    'pred_name': label_name(pred_label),
                    'gt_label': gt_label,
                    'gt_name': label_name(gt_label) if gt_label >= 0 else 'background',
                    'gt_index_in_valid': int(gt_idx[q].item()) if gt_idx.numel() else -1,
                }
                all_records.append(rec)
                per_query.append((b, q, rec))

        statuses = [r['status'] for _, _, r in per_query]
        chosen_local = choose_indices(statuses, args.max_per_group, selected_counts, rng)
        if chosen_local:
            batch_idx_t = torch.tensor([per_query[j][0] for j in chosen_local], device=device, dtype=torch.long)
            query_idx_t = torch.tensor([per_query[j][1] for j in chosen_local], device=device, dtype=torch.long)
            sel_boxes = pred_boxes[batch_idx_t, query_idx_t]

            feature_store['vpe'].append(vpe_feats[batch_idx_t, query_idx_t].float().cpu())
            feature_store['backbone_roi'].append(roi_pool_multiscale(backbone_feats, sel_boxes, batch_idx_t).cpu())
            pre_fpn_feats = getattr(model.encoder, "pre_fpn_feats", None)
            if pre_fpn_feats is not None:
                feature_store['pre_fpn_roi'].append(roi_pool_multiscale(pre_fpn_feats, sel_boxes, batch_idx_t).cpu())
            feature_store['encoder_roi'].append(roi_pool_multiscale(encoder_feats, sel_boxes, batch_idx_t).cpu())

            for j in chosen_local:
                rec = per_query[j][2]
                selected_records.append(rec)
                status_labels.append(rec['status'])
                if rec['status'] == 'tp':
                    binary_tp_vs_fpbg.append('tp')
                    binary_tp_vs_allfp.append('tp')
                elif rec['status'].startswith('fp_bg'):
                    binary_tp_vs_fpbg.append('fp_bg')
                    binary_tp_vs_allfp.append('fp')
                else:
                    binary_tp_vs_fpbg.append('ignore')
                    binary_tp_vs_allfp.append('fp')

        total_seen += bsz * nq
        if batch_i % 10 == 0:
            print(f'batch {batch_i}/{len(loader)}: seen_queries={total_seen}, selected={len(selected_records)}')

    features_np = {}
    for name, chunks in feature_store.items():
        if chunks:
            features_np[name] = torch.cat(chunks, dim=0).numpy().astype(np.float32)
        else:
            features_np[name] = np.zeros((0, 1), dtype=np.float32)

    status_arr = np.asarray(status_labels)
    np.savez_compressed(
        out_dir / 'fp_feature_samples.npz',
        status=status_arr,
        binary_tp_vs_fpbg=np.asarray(binary_tp_vs_fpbg),
        binary_tp_vs_allfp=np.asarray(binary_tp_vs_allfp),
        **{f'{k}_features': v for k, v in features_np.items()},
    )

    full_summary = summarize(all_records)
    selected_summary = summarize(selected_records)
    probes = {}
    for feat_name, feat in features_np.items():
        probes[feat_name] = {
            'status_multiclass': run_linear_probe(feat, status_arr, args.seed),
            'tp_vs_fp_bg': run_linear_probe(feat, binary_tp_vs_fpbg, args.seed),
            'tp_vs_all_fp': run_linear_probe(feat, binary_tp_vs_allfp, args.seed),
        }
        plot_tsne(feat, status_arr, out_dir / f'{feat_name}_tsne_status.png', args.seed, args.max_tsne)

    (out_dir / 'summary.json').write_text(json.dumps({
        'config': args.config,
        'checkpoint': args.checkpoint,
        'iou_thr': args.iou_thr,
        'score_buckets': {'easy': '<0.3', 'medium': '[0.3,0.7)', 'high': '>=0.7'},
        'total_queries_seen': len(all_records),
        'selected_feature_records': len(selected_records),
        'all_query_summary': full_summary,
        'selected_summary': selected_summary,
        'linear_probes': probes,
    }, indent=2, ensure_ascii=False))
    (out_dir / 'selected_records.json').write_text(json.dumps(flatten_jsonable(selected_records), indent=2, ensure_ascii=False))

    print(f'Saved summary: {out_dir / "summary.json"}')
    print(f'Saved features: {out_dir / "fp_feature_samples.npz"}')
    print(json.dumps(full_summary['status_counts'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-t', '--checkpoint', required=True)
    parser.add_argument('--output-dir', default='output/fp_feature_analysis')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--max-per-group', type=int, default=1500)
    parser.add_argument('--max-tsne', type=int, default=3000)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
