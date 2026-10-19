import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from tools.plot_branch_gradcam import load_model, run_one

def region_stats(cmap, box, size=640):
    x1,y1,x2,y2=box
    x1=int(np.clip(x1,0,size-1)); y1=int(np.clip(y1,0,size-1))
    x2=int(np.clip(x2, x1+1, size)); y2=int(np.clip(y2, y1+1, size))
    w=max(2,x2-x1); h=max(2,y2-y1)
    # 中心语义区域与边界环带分开统计。
    mx=max(1,int(0.20*w)); my=max(1,int(0.20*h))
    inner=cmap[y1+my:y2-my, x1+mx:x2-mx]
    ring=np.zeros((size,size),dtype=bool); ring[y1:y2,x1:x2]=True
    ring[y1+my:y2-my,x1+mx:x2-mx]=False
    rv=cmap[ring]
    return float(inner.mean()) if inner.size else 0., float(rv.mean()) if rv.size else 0.

def main(a):
    ann=json.load(open(a.ann_file)); imgs=ann['images']
    ids=[int(x) for x in a.image_ids]
    result=json.load(open(a.result_json)); dets={int(x['image_id']):x for x in result['detections']}
    device=torch.device(a.device)
    model=load_model(a.config,a.checkpoint,device,baseline=True)
    rows=[]
    for iid in ids:
        info=imgs[iid]
        path=os.path.join(a.image_root,info['file_name'])
        im=Image.open(path).convert('RGB')
        inp=torch.from_numpy(np.asarray(im.resize((640,640)),dtype=np.float32).transpose(2,0,1)/255.).unsqueeze(0).to(device)
        with torch.no_grad():
            cls,reg,q,score,overlap=run_one(model,inp,baseline=True)
        # 使用标注文件中的 GT，选择能同时体现内部/边界差异的目标。
        gts=[x for x in ann['annotations'] if int(x['image_id'])==int(info['id'])]
        best=None
        for g in gts:
            x,y,w,h=map(float,g['bbox'])
            box=[x/info['width']*640,y/info['height']*640,(x+w)/info['width']*640,(y+h)/info['height']*640]
            ci,ce=region_stats(cls,box); ri,re=region_stats(reg,box)
            # 分类内部偏好 + 回归边界偏好，同时惩罚响应完全重叠。
            score_e=(ci-ce)+(re-ri)+0.25*(1-overlap)
            item={'image_id':iid,'file_name':info['file_name'],'gt_id':g['id'],'bbox':box,
                  'cls_in':ci,'cls_edge':ce,'reg_in':ri,'reg_edge':re,
                  'evidence_score':score_e,'query':q,'query_score':score,'overlap':overlap}
            if best is None or item['evidence_score']>best['evidence_score']: best=item
        if best: rows.append(best)
        print(json.dumps(best,ensure_ascii=False))
    rows.sort(key=lambda x:x['evidence_score'],reverse=True)
    print('BEST',json.dumps(rows[:5],ensure_ascii=False,indent=2))
    json.dump(rows,open(a.output,'w'),ensure_ascii=False,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--ann-file',required=True); p.add_argument('--image-root',required=True); p.add_argument('--result-json',required=True); p.add_argument('--output',required=True); p.add_argument('--image-ids',nargs='+',type=int,required=True); p.add_argument('--device',default='cuda'); main(p.parse_args())

