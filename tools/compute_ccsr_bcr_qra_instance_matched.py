#!/usr/bin/env python3
"""实例/空间匹配版 CCSR、BCR、QRA，不跨模型使用 query_index 对齐。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., float(a[2])-float(a[0])) * max(0., float(a[3])-float(a[1]))
    bb = max(0., float(b[2])-float(b[0])) * max(0., float(b[3])-float(b[1]))
    return inter / max(aa + bb - inter, 1e-12)


def aggregate_queries(record):
    # query_index 只用于合并同一模型中同一 query 的多类别 Top-K 条目；不用于跨模型配对。
    groups = {}
    for p in record.get('predictions', []):
        q = p.get('query_index')
        if q is None:
            # 仅为旧结果保留退化模式；本次输入结果均包含 query_index。
            q = 'bbox:' + ','.join(f'{float(v):.5f}' for v in p['bbox'])
        q = str(q)
        item = {'key': q, 'bbox': p['bbox'], 'label': int(p['label']), 'score': float(p['score'])}
        if q not in groups or item['score'] > groups[q]['score']:
            groups[q] = item
    return list(groups.values())


def quality(p, gts):
    vals = [iou(p['bbox'], g['bbox']) for g in gts]
    return (max(vals), int(np.argmax(vals))) if vals else (0.0, None)


def one_to_one_prediction_gt(preds, gts, threshold):
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            pairs.append((iou(p['bbox'], g['bbox']), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for val, pi, gi in pairs:
        if val < threshold or pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); out.append((pi, gi, val))
    return out


def one_to_one_box_match(src, dst, threshold):
    pairs = []
    for si, s in enumerate(src):
        for di, d in enumerate(dst):
            pairs.append((iou(s['bbox'], d['bbox']), si, di))
    pairs.sort(reverse=True)
    used_s, used_d, out = set(), set(), []
    for val, si, di in pairs:
        if val < threshold or si in used_s or di in used_d:
            continue
        used_s.add(si); used_d.add(di); out.append((si, di, val))
    return out


def qra(preds, gts, quality_margin, score_eps):
    vals = []
    for p in preds:
        qi, _ = quality(p, gts)
        vals.append((p['score'], qi))
    correct = total = 0
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            si, qi = vals[i]; sj, qj = vals[j]
            if abs(qi-qj) < quality_margin:
                continue
            total += 1
            higher_score = si if qi > qj else sj
            lower_score = sj if qi > qj else si
            if higher_score > lower_score + score_eps:
                correct += 1
    return correct, total


def main(args):
    b = json.load(open(args.baseline, encoding='utf-8'))
    d = json.load(open(args.dbsr, encoding='utf-8'))
    bm = {int(r['image_id']): r for r in b['detections']}
    dm = {int(r['image_id']): r for r in d['detections']}
    image_ids = sorted(set(bm) & set(dm))

    # CCSR：GT 实例级对齐。
    ccsr_den = ccsr_num = 0
    ccsr_rows = []
    # BCR：预测框空间一对一对齐。
    bcr_num = bcr_den = bcr_all_num = bcr_all_den = 0
    bcr_rows = []
    # QRA：模型内部独立排序。
    qra_sum = {'D-FINE': [0, 0], 'DBSR-CCD': [0, 0]}
    qra_rows = []

    for image_id in image_ids:
        br, dr = bm[image_id], dm[image_id]
        gts = br.get('ground_truths', [])
        bp, dp = aggregate_queries(br), aggregate_queries(dr)

        b_assign = one_to_one_prediction_gt(bp, gts, args.loc_thr)
        d_assign = one_to_one_prediction_gt(dp, gts, args.loc_thr)
        d_by_gt = {gi: dp[pi] for pi, gi, _ in d_assign}
        wrong = [(bp[pi], gi, val) for pi, gi, val in b_assign if bp[pi]['label'] != int(gts[gi]['label'])]
        c_num = 0
        for _p, gi, _val in wrong:
            ccsr_den += 1
            if gi in d_by_gt and d_by_gt[gi]['label'] == int(gts[gi]['label']):
                ccsr_num += 1; c_num += 1
        ccsr_rows.append({'image_id': image_id, 'baseline_wrong_localized': len(wrong), 'corrected_by_dbsr': c_num})

        # 背景定义：baseline 候选与所有 GT 的最大 IoU < bg_thr。
        bg = [p for p in bp if quality(p, gts)[0] < args.bg_thr]
        matches = one_to_one_box_match(bg, dp, args.box_match_thr)
        matched = 0; rise = 0
        for si, di, _ in matches:
            matched += 1
            if dp[di]['score'] > bg[si]['score'] + args.score_eps:
                rise += 1
        bcr_num += rise; bcr_den += matched
        bcr_all_num += rise; bcr_all_den += len(bg)
        bcr_rows.append({'image_id': image_id, 'baseline_background': len(bg), 'spatially_matched': matched, 'score_rise': rise})

        qr = {'image_id': image_id}
        for name, preds in [('D-FINE', bp), ('DBSR-CCD', dp)]:
            c, t = qra(preds, gts, args.quality_margin, args.score_eps)
            qra_sum[name][0] += c; qra_sum[name][1] += t
            qr[name+'_correct_pairs'] = c; qr[name+'_valid_pairs'] = t
        qra_rows.append(qr)

    pct = lambda n, d: (100.0*n/d) if d else None
    result = {
        'alignment': {
            'CCSR': '以同一 GT 实例对齐 baseline 和 DBSR，允许两个模型使用不同 query_index',
            'BCR': '以预测框 IoU 做跨模型一对一空间匹配，不使用 query_index',
            'QRA': '每个模型内部独立按自身预测框 IoU 计算置信度排序准确率',
        },
        'thresholds': {'localization_iou': args.loc_thr, 'background_iou': args.bg_thr, 'box_match_iou': args.box_match_thr, 'quality_margin': args.quality_margin, 'score_eps': args.score_eps},
        'images': len(image_ids),
        'CCSR': {'numerator': ccsr_num, 'denominator': ccsr_den, 'percent': pct(ccsr_num, ccsr_den)},
        'BCR': {'numerator': bcr_num, 'matched_background_denominator': bcr_den, 'percent_matched': pct(bcr_num, bcr_den), 'all_background_numerator': bcr_all_num, 'all_background_denominator': bcr_all_den, 'percent_all_background_conservative': pct(bcr_all_num, bcr_all_den), 'spatial_match_coverage': pct(bcr_den, bcr_all_den)},
        'QRA': {name: {'correct_pairs': val[0], 'valid_pairs': val[1], 'percent': pct(val[0], val[1])} for name, val in qra_sum.items()},
        'per_image_ccsr': ccsr_rows, 'per_image_bcr': bcr_rows, 'per_image_qra': qra_rows,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(out.with_suffix('.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['metric','numerator','denominator','percent'])
        w.writerow(['CCSR',ccsr_num,ccsr_den,pct(ccsr_num,ccsr_den)])
        w.writerow(['BCR_matched',bcr_num,bcr_den,pct(bcr_num,bcr_den)])
        w.writerow(['BCR_all_background_conservative',bcr_all_num,bcr_all_den,pct(bcr_all_num,bcr_all_den)])
        w.writerow(['QRA_D-FINE',qra_sum['D-FINE'][0],qra_sum['D-FINE'][1],pct(*qra_sum['D-FINE'])])
        w.writerow(['QRA_DBSR-CCD',qra_sum['DBSR-CCD'][0],qra_sum['DBSR-CCD'][1],pct(*qra_sum['DBSR-CCD'])])
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('per_image')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True); p.add_argument('--dbsr', required=True); p.add_argument('--output', required=True)
    p.add_argument('--loc-thr', type=float, default=0.5); p.add_argument('--bg-thr', type=float, default=0.1); p.add_argument('--box-match-thr', type=float, default=0.5); p.add_argument('--quality-margin', type=float, default=0.05); p.add_argument('--score-eps', type=float, default=1e-6)
    main(p.parse_args())

