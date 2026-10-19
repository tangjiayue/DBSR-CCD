import json
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = map(float, a)
    xb1, yb1, xb2, yb2 = map(float, b)
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return inter / max(area_a + area_b - inter, 1e-12)


def best_iou(box, gts):
    vals = [iou_xyxy(box, g['bbox']) for g in gts]
    return max(vals) if vals else 0.0


def choose_candidates(preds, gts):
    scored = [(p, best_iou(p['bbox'], gts)) for p in preds]
    # 最高置信度低质量候选，展示排序错误。
    bad = next((x for x in sorted(scored, key=lambda z: z[0]['score'], reverse=True)
                if x[1] < 0.50), None)
    # 质量最好的候选中，置信度最高者作为同模型对照。
    good_pool = [x for x in scored if x[1] >= 0.50]
    good = max(good_pool, key=lambda z: z[0]['score']) if good_pool else max(scored, key=lambda z: z[1])
    # DBSR 右侧还显示最高分低质量候选；如果没有，则使用分数最低的低质量候选。
    low_pool = [x for x in scored if x[1] < 0.50]
    low = max(low_pool, key=lambda z: z[0]['score']) if low_pool else None
    return bad, good, low


def draw_box(ax, box, color, lw=2.6, ls='-', label=None, text_color=None, alpha=1.0):
    x1, y1, x2, y2 = map(float, box)
    ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                           edgecolor=color, linewidth=lw, linestyle=ls,
                           alpha=alpha, zorder=5))
    if label:
        tc = text_color or color
        ax.text(x1 + 3, max(2, y1 - 6), label, color=tc, fontsize=9,
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='none',
                alpha=0.82, pad=1.5), zorder=8)


def setup_ax(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    if hasattr(img, 'width'):
        w, h = img.width, img.height
    else:
        h, w = img.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.2)
        s.set_color('#4b5563')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/userfolder/Dataset/ObjectDetection/ComparisonDetectorDataset')
    ap.add_argument('--df', required=True)
    ap.add_argument('--db', required=True)
    ap.add_argument('--image-ids', default='605')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    image_id = int(args.image_ids.split(',')[0])

    ann_path = os.path.join(args.root, 'test_intid_label0.json')
    ann = json.load(open(ann_path))
    # detection_results 中 image_id 是测试集顺序索引，而不是 COCO annotation id。
    im_info = ann['images'][image_id]
    filename = im_info['file_name']
    img_path = os.path.join(args.root, 'test', filename)
    img = Image.open(img_path).convert('RGB')

    def read(path):
        d = json.load(open(path))
        return next(x for x in d['detections'] if int(x['image_id']) == image_id)

    df = read(args.df)
    db = read(args.db)
    gts = df['ground_truths']
    df_bad, df_good, _ = choose_candidates(df['predictions'], gts)
    db_bad, db_good, db_low = choose_candidates(db['predictions'], gts)
    if df_bad is None or df_good is None or db_good is None:
        raise RuntimeError('该图没有足够的高/低质量候选，无法生成对比图')

    # 论文风格：低饱和灰度背景，GT 绿色，错误高分红色，正确高分蓝色。
    fig = plt.figure(figsize=(16.5, 7.3), dpi=220, facecolor='white')
    gs = fig.add_gridspec(2, 3, height_ratios=[4.8, 1.35], hspace=0.28, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bars = fig.add_subplot(gs[1, :])
    gray = np.asarray(img.convert('L'))
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    for ax in axes:
        ax.set_facecolor('#f5f5f5')

    setup_ax(axes[0], img, 'Input and ground truth')
    for g in gts:
        draw_box(axes[0], g['bbox'], '#16a34a', lw=2.8, label='GT', text_color='#15803d')
    axes[0].text(0.02, 0.96, f'image {image_id}  |  {filename}', transform=axes[0].transAxes,
                 fontsize=10, color='#111827', va='top', bbox=dict(facecolor='white', alpha=.85, pad=3))

    setup_ax(axes[1], gray_rgb, 'D-FINE: confidence–quality mismatch')
    for g in gts:
        draw_box(axes[1], g['bbox'], '#16a34a', lw=2.4, label='GT', text_color='#15803d')
    draw_box(axes[1], df_bad[0]['bbox'], '#dc2626', lw=3.2, ls='--',
             label=f"high score {df_bad[0]['score']:.3f} | IoU {df_bad[1]:.3f}", text_color='#b91c1c')
    draw_box(axes[1], df_good[0]['bbox'], '#2563eb', lw=2.5,
             label=f"better box {df_good[0]['score']:.3f} | IoU {df_good[1]:.3f}", text_color='#1d4ed8')
    axes[1].text(0.03, 0.96, 'ranking inversion', transform=axes[1].transAxes,
                 fontsize=11, color='#b91c1c', va='top', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=.9, pad=3))

    setup_ax(axes[2], gray_rgb, 'DBSR-CCD: ranking corrected')
    for g in gts:
        draw_box(axes[2], g['bbox'], '#16a34a', lw=2.4, label='GT', text_color='#15803d')
    draw_box(axes[2], db_good[0]['bbox'], '#2563eb', lw=3.2,
             label=f"high score {db_good[0]['score']:.3f} | IoU {db_good[1]:.3f}", text_color='#1d4ed8')
    if db_low is not None:
        draw_box(axes[2], db_low[0]['bbox'], '#f97316', lw=2.4, ls='--',
                 label=f"background {db_low[0]['score']:.3f} | IoU {db_low[1]:.3f}", text_color='#c2410c')
    axes[2].text(0.03, 0.96, 'quality-aware ranking', transform=axes[2].transAxes,
                 fontsize=11, color='#1d4ed8', va='top', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=.9, pad=3))

    # 下方用同一坐标系给出最直观的排序变化。
    names = ['D-FINE\nlow-quality', 'D-FINE\nbetter box', 'DBSR-CCD\nhigh-quality', 'DBSR-CCD\nbackground']
    scores = [df_bad[0]['score'], df_good[0]['score'], db_good[0]['score'], db_low[0]['score'] if db_low else 0.0]
    ious = [df_bad[1], df_good[1], db_good[1], db_low[1] if db_low else 0.0]
    x = np.arange(4)
    width = 0.34
    bars.bar(x-width/2, scores, width, color=['#dc2626','#2563eb','#2563eb','#f97316'], alpha=.9, label='classification confidence')
    bars.bar(x+width/2, ious, width, color=['#fca5a5','#93c5fd','#60a5fa','#fdba74'], alpha=.95, label='box IoU')
    bars.set_ylim(0, 1.08)
    bars.set_ylabel('value', fontsize=10)
    bars.set_xticks(x, names, fontsize=9)
    bars.grid(axis='y', alpha=.22)
    bars.set_axisbelow(True)
    bars.legend(loc='upper right', frameon=False, ncol=2, fontsize=9)
    bars.text(-0.02, 1.04, 'Higher confidence should correspond to higher localization quality',
              transform=bars.transAxes, fontsize=10, color='#374151', va='bottom')

    fig.suptitle('Qualitative evidence of classification–localization coupling', fontsize=16,
                 fontweight='bold', y=0.985)
    fig.text(0.5, 0.012,
             'Red dashed: high-confidence low-IoU candidate; blue: high-quality candidate; green: ground truth.',
             ha='center', fontsize=10, color='#4b5563')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    stem, _ = os.path.splitext(args.out)
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.svg', bbox_inches='tight')
    with open(stem + '.json', 'w') as f:
        json.dump({'image_id': image_id, 'filename': filename,
                   'df_bad': {'score': df_bad[0]['score'], 'iou': df_bad[1], 'bbox': df_bad[0]['bbox']},
                   'df_good': {'score': df_good[0]['score'], 'iou': df_good[1], 'bbox': df_good[0]['bbox']},
                   'db_good': {'score': db_good[0]['score'], 'iou': db_good[1], 'bbox': db_good[0]['bbox']},
                   'db_background': ({'score': db_low[0]['score'], 'iou': db_low[1], 'bbox': db_low[0]['bbox']} if db_low else None)}, f, indent=2)
    print(json.dumps({'image_id': image_id, 'filename': filename,
                      'df_bad_score': df_bad[0]['score'], 'df_bad_iou': df_bad[1],
                      'df_good_score': df_good[0]['score'], 'df_good_iou': df_good[1],
                      'db_good_score': db_good[0]['score'], 'db_good_iou': db_good[1],
                      'db_bg_score': db_low[0]['score'] if db_low else None,
                      'db_bg_iou': db_low[1] if db_low else None}, ensure_ascii=False))


if __name__ == '__main__':
    main()

