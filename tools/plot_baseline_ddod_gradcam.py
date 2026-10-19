#!/usr/bin/env python3
"""在 baseline D-FINE 上复现 DDOD Figure 2 风格的分类/回归 Grad-CAM。"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,ROOT)
from src.core import YAMLConfig

def load_model(cfg_path, ckpt_path, device):
    cfg=YAMLConfig(cfg_path); cfg.yaml_cfg['num_classes']=11
    model=cfg.model
    ck=torch.load(ckpt_path,map_location='cpu')
    st=ck.get('ema',{}).get('module',ck.get('model',ck))
    model.load_state_dict(st,strict=False); model.to(device).eval(); return model

def hook_store(store):
    def h(_m,_i,out):
        store['acts']=out
        for a in out:
            if torch.is_grad_enabled() and a.requires_grad: a.retain_grad()
    return h

def make_cam(store):
    cams=[]
    for a in store['acts']:
        if a.grad is None: continue
        w=a.grad.abs().mean(dim=(2,3),keepdim=True)
        c=F.relu((w*a.abs()).sum(dim=1,keepdim=True))
        c=F.interpolate(c,size=(640,640),mode='bilinear',align_corners=False)[0,0]
        cams.append(c)
    c=torch.stack(cams).sum(0) if cams else torch.zeros(640,640,device=next(iter(store['acts'])).device)
    c=c-c.min(); return (c/(c.max()+1e-8)).detach().cpu().numpy()

def xyxy_from_cxcywh(b):
    cx,cy,w,h=b
    return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-8)

def target_cam(model, im, gt_boxes):
    x=torch.from_numpy(np.asarray(im.resize((640,640)),dtype=np.float32).transpose(2,0,1)/255.).unsqueeze(0).to(next(model.parameters()).device)
    store={}; handle=model.encoder.register_forward_hook(hook_store(store))
    with torch.enable_grad():
        feats=model.backbone(x); enc=model.encoder(feats); out=model.decoder(enc)
        logits=out['pred_logits']; boxes=out['pred_boxes']
        probs=logits.sigmoid(); q=int(probs[0].max(dim=1).values.argmax())
        pred=boxes[0,q]
        pxy=xyxy_from_cxcywh(pred.detach().cpu().tolist())
        gi=max(range(len(gt_boxes)),key=lambda j:iou(pxy,gt_boxes[j])) if gt_boxes else 0
        target=torch.tensor(gt_boxes[gi],device=pred.device,dtype=pred.dtype)
        tx=(target[0]+target[2])/2; ty=(target[1]+target[3])/2; tw=target[2]-target[0]; th=target[3]-target[1]
        tgt=torch.stack([tx,ty,tw,th])
        cls=logits[0,q].sigmoid().max()
        reg=F.smooth_l1_loss(pred,tgt,reduction='sum')
        cls.backward(retain_graph=True); cls_cam=make_cam(store)
        model.zero_grad(set_to_none=True)
        for a in store['acts']:
            if a.grad is not None: a.grad.zero_()
        reg.backward(); reg_cam=make_cam(store)
    handle.remove()
    aa=cls_cam>=np.quantile(cls_cam,.8); bb=reg_cam>=np.quantile(reg_cam,.8)
    ov=float((aa&bb).sum()/max((aa|bb).sum(),1))
    return cls_cam,reg_cam,q,float(cls.detach()),ov,gi,pxy

def add_box(ax,b,color,lw=2.4,label=None):
    x1,y1,x2,y2=b; ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=lw))
    if label: ax.text(x1+2,max(4,y1-5),label,color=color,fontsize=8,fontweight='bold',bbox=dict(facecolor='white',alpha=.8,pad=1,edgecolor='none'))

def main(a):
    ann=json.load(open(a.ann_file)); imgs=ann['images']; device=torch.device(a.device)
    model=load_model(a.config,a.checkpoint,device)
    samples=[]
    for iid in a.image_ids:
        info=imgs[int(iid)]; path=os.path.join(a.image_root,info['file_name']); im=Image.open(path).convert('RGB')
        gts=[g for g in ann['annotations'] if int(g['image_id'])==int(info['id'])]
        gt_boxes=[]
        for g in gts:
            x,y,w,h=map(float,g['bbox']); gt_boxes.append([x/info['width'],y/info['height'],(x+w)/info['width'],(y+h)/info['height']])
        cls,reg,q,score,ov,gi,pred=target_cam(model,im,gt_boxes)
        samples.append((int(iid),info,im,gt_boxes,cls,reg,q,score,ov,gi,pred))
        print(json.dumps({'image_id':int(iid),'file_name':info['file_name'],'query':q,'score':score,'overlap':ov,'gt_index':gi},ensure_ascii=False))
    samples.sort(key=lambda z:z[8])
    rows=len(samples); fig,ax=plt.subplots(rows,3,figsize=(11,3.45*rows),squeeze=False)
    for r,(iid,info,im,gts,cls,reg,q,score,ov,gi,pred) in enumerate(samples):
        arr=np.asarray(im.resize((640,640)))
        pred_pix=[v*640 for v in pred]
        px1,py1,px2,py2=map(int,pred_pix)
        px1=max(0,min(639,px1)); py1=max(0,min(639,py1)); px2=max(px1+1,min(640,px2)); py2=max(py1+1,min(640,py2))
        magma=plt.get_cmap('magma')
        if getattr(a,'full',False):
            # 全图版本：显示该 query 对整幅输入的原始响应。
            x0=y0=0; crop=arr
            cls_roi=magma(np.clip(cls,0,1))[...,:3].astype(np.float32)
            reg_roi=magma(np.clip(reg,0,1))[...,:3].astype(np.float32)
        else:
            # 局部版本：以预测框为中心放大，框外响应固定为深蓝紫色。
            side=int(max(px2-px1,py2-py1)*2.2)
            side=max(32,min(640,side)); cx=(px1+px2)//2; cy=(py1+py2)//2
            x0=max(0,min(640-side,cx-side//2)); y0=max(0,min(640-side,cy-side//2)); x1=x0+side; y1=y0+side
            crop=arr[y0:y1,x0:x1]
            roi=np.zeros((side,side),dtype=bool); roi[max(0,py1-y0):min(side,py2-y0),max(0,px1-x0):min(side,px2-x0)]=True
            dark=np.array([0.035,0.012,0.090],dtype=np.float32)
            cls_roi=magma(np.clip(cls[y0:y1,x0:x1],0,1))[...,:3].astype(np.float32)
            reg_roi=magma(np.clip(reg[y0:y1,x0:x1],0,1))[...,:3].astype(np.float32)
            cls_roi[~roi]=dark; reg_roi[~roi]=dark
        # 只显示 baseline，保持 DDOD Figure 2 的问题分析定位。
        ax[r,0].imshow(crop)
        if getattr(a,'full',False):
            gray=np.asarray(crop).mean(2)
            ax[r,1].imshow(gray,cmap='gray',vmin=0,vmax=255)
            ax[r,1].imshow(cls,cmap='magma',alpha=.78,vmin=0,vmax=1)
            ax[r,2].imshow(gray,cmap='gray',vmin=0,vmax=255)
            ax[r,2].imshow(reg,cmap='magma',alpha=.78,vmin=0,vmax=1)
        else:
            ax[r,1].imshow(cls_roi)
            ax[r,2].imshow(reg_roi)
        for c in range(3): ax[r,c].axis('off')
        if r == 0:
            ax[r,0].set_title('Image',fontweight='bold',fontsize=14,pad=10)
            ax[r,1].set_title('Classification',fontweight='bold',fontsize=14,pad=10)
            ax[r,2].set_title('Localization',fontweight='bold',fontsize=14,pad=10)
        if gts:
            b=gts[gi]; add_box(ax[r,0],[b[0]*640-x0,b[1]*640-y0,b[2]*640-x0,b[3]*640-y0],'#26d34e',2.2)
        add_box(ax[r,0],[pred_pix[0]-x0,pred_pix[1]-y0,pred_pix[2]-x0,pred_pix[3]-y0],'#ff3b30',2.2)
        if getattr(a,'full',False):
            add_box(ax[r,1],pred_pix,'#ff3b30',2.2)
            add_box(ax[r,2],pred_pix,'#ff3b30',2.2)
    fig.text(.5,.025,'Input images with ground-truth boxes; warm colors indicate larger backward sensitivity on the shared encoder/FPN features.',ha='center',fontsize=10,color='#4b5563')
    fig.text(.5,.010,'Different and mixed spatial responses for classification and localization motivate task-specific feature disentanglement.',ha='center',fontsize=10,color='#4b5563')
    fig.tight_layout(rect=[0,.06,1,.98]); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=240,bbox_inches='tight'); fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight'); fig.savefig(out.with_suffix('.svg'),bbox_inches='tight')
    json.dump([{'image_id':x[0],'file_name':x[1]['file_name'],'query':x[6],'score':x[7],'overlap':x[8],'gt_index':x[9]} for x in samples],open(out.with_suffix('.json'),'w'),indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--ann-file',required=True); p.add_argument('--image-root',required=True); p.add_argument('--output',required=True); p.add_argument('--image-ids',nargs='+',type=int,required=True); p.add_argument('--device',default='cuda'); p.add_argument('--full',action='store_true'); main(p.parse_args())

