"""Typing-trained pixel residual with whole-session holdouts and temporal controls."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from phase0.analysis import seqctc as S, ctcv4 as V
from phase0.analysis.visual_fusion import greedy
from phase0.analysis.text_contract import edits

ROOT = Path('.cache/nextgen/typing_pixels')
KBDS = ['20260910-021315-kbd', '20260910-131629-kbd']


def crops(sid):
    folder = S.SESS/sid
    signature = dict(recipe='masked-hand-rgb24-v1', **{n: hashlib.sha256((folder/n).read_bytes()).hexdigest()
                    for n in ['video.mp4', 'landmarks.parquet', 'frames.jsonl']})
    target = ROOT/f'{sid}.npz'
    if target.exists():
        with np.load(target) as z:
            if json.loads(str(z['signature'])) != signature:
                raise ValueError('Stale crops')
            return z['crops']
    frames = [json.loads(l)['i'] for l in (folder/'frames.jsonl').read_text().splitlines()]
    tb = pq.read_table(folder/'landmarks.parquet').to_pydict()
    points = {}
    for i,h,j,x,y in zip(*(tb[n] for n in ['i','hand','joint','x','y'])):
        points.setdefault(i, {}).setdefault(h, np.full((21,2), np.nan))[j] = [x,y]
    out = np.zeros((len(frames[::2]),6,24,24), np.uint8)
    wanted = {int(f):k for k,f in enumerate(frames[::2])}
    cap = cv2.VideoCapture(str(folder/'video.mp4'))
    i = seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            hs = sorted([p for p in points.get(i,{}).values() if np.isfinite(p).all()], key=lambda p:p[0,1])
            # Single-hand frames are omitted to avoid future-dependent identity assignment.
            if len(hs)==2:
                for h,p in enumerate(hs):
                    lo,hi=p.min(0),p.max(0)
                    scale=24/max(16.,float((hi-lo).max())*1.25)
                    center=(lo+hi)/2
                    mat=np.array([[scale,0,12-scale*center[0]],[0,scale,12-scale*center[1]]],np.float32)
                    crop=cv2.warpAffine(frame,mat,(24,24))
                    mask=np.zeros((24,24),np.uint8)
                    cv2.fillConvexPoly(mask,cv2.convexHull((p@mat[:,:2].T+mat[:,2]).astype(np.int32)),255)
                    crop[mask==0]=0
                    out[wanted[i],h*3:(h+1)*3]=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).transpose(2,0,1)
            seen+=1
        i+=1
    cap.release()
    if seen!=len(out):
        raise ValueError('Video timestamps do not match frames')
    np.savez_compressed(target,crops=out,signature=json.dumps(signature))
    print(f'crops {sid}: {len(out)}',flush=True)
    return out


class PixelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(12,16,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.image_norm=nn.LayerNorm(96)
        self.temporal=nn.Sequential(nn.Conv1d(602,96,1),nn.GELU(),nn.Conv1d(96,96,5,padding=2),
                                   nn.GELU(),nn.Conv1d(96,29,1))
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)

    def forward(self, rgb, geo, base, pixels=True):
        delta=torch.cat([torch.zeros_like(rgb[:1]),rgb[1:]-rgb[:-1]])
        z=self.image_norm(self.image(torch.cat([rgb,delta],1)))
        if not pixels:
            z=z*0
        x=torch.cat([geo,z],1).T[None]
        return (base+self.temporal(x)[0].T).log_softmax(-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--pretrain',type=int,default=150)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--anchor',type=float,default=1.)
    a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    # CPU CTC avoids backend-specific CTC fallback; the small encoder fits in RAM.
    desk=V.load_data(Path('.cache/ctc_v4/desk_data.json'))
    config=dict(**vars(a),keyboard=KBDS,desk=list(desk),crop=24,
        evaluation='whole-session holdout; development; greedy phrase windows',not_deployed=True)
    (ROOT/f'config_s{a.seed}.json').write_text(json.dumps(config,indent=2))
    models=[V.V3/'runs'/'hwtmix'/f'seed{s}'/'all.pt' for s in range(3)]
    examples={}
    for sid in list(desk)+KBDS:
        rgb=crops(sid)
        st=S.load_session(sid)
        geo=S.featurize(torch.from_numpy(st.A)[None],torch.from_numpy(st.M)[None],False)[0,::2].numpy()
        base=V.ensemble_cont(ROOT,models,sid)
        if len(rgb)!=len(base) or len(rgb)!=len(geo):
            raise ValueError('Clock mismatch')
        windows=desk[sid]['windows'] if sid in desk else [(lo,hi,S.sym_text(sy)) for lo,hi,sy in S.eval_windows(st)]
        examples[sid]=[]
        for lo,hi,text in windows:
            i,j=np.searchsorted(st.t[::2],[lo,hi])
            if len(text)<2 or j-i<2*len(text):
                continue
            examples[sid].append((torch.tensor(rgb[i:j],dtype=torch.float32)/255,
                torch.tensor(geo[i:j]),torch.tensor(np.array(base[i:j]),dtype=torch.float32),text))
        print(f'loaded {sid}: {len(examples[sid])} phrases',flush=True)
    keyboard=[e for sid in KBDS for e in examples[sid]]
    results={}
    started=time.time()
    for pixels in [False,True]:
        torch.manual_seed(a.seed)
        model=PixelResidual()
        rng=np.random.default_rng(a.seed)
        def train(es,steps):
            model.train()
            opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
            for step in range(steps):
                rgb,geo,base,text=es[rng.integers(len(es))]
                lp=model(rgb,geo,base,pixels)
                target=torch.tensor(S.text_syms(text),dtype=torch.long)
                loss=nn.functional.ctc_loss(lp[:,None],target,[len(lp)],[len(target)])
                loss=loss+a.anchor*nn.functional.kl_div(lp,base.softmax(-1),reduction='batchmean')
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
        train(keyboard,a.pretrain)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        for held in desk:
            model.load_state_dict(initial)
            rng=np.random.default_rng(a.seed)
            train([e for sid in desk if sid!=held for e in examples[sid]]+keyboard,a.steps)
            model.eval()
            rows=[]
            with torch.inference_mode():
                for rgb,geo,base,text in examples[held]:
                    pred=greedy(model(rgb,geo,base,pixels))
                    shuffled=greedy(model(rgb[torch.randperm(len(rgb))],geo,base,pixels))
                    static=greedy(model(rgb.mean(0,keepdim=True).expand_as(rgb),geo,base,pixels))
                    rows.append(dict(chars=len(text),ce=edits(text,pred),baseline_ce=edits(text,greedy(base)),
                                     shuffled_ce=edits(text,shuffled),static_ce=edits(text,static)))
            mode='pixels' if pixels else 'geometry'
            results.setdefault(held,{})[mode]=rows
            torch.save(dict(state=model.state_dict(),config=config,heldout=held),ROOT/f'{held}_{mode}_s{a.seed}.pt')
            print(held,mode,{k:sum(r[k] for r in rows)/sum(r['chars'] for r in rows)
                              for k in ['ce','baseline_ce','shuffled_ce','static_ce']},flush=True)
    target=Path(f'results/nextgen/typing_pixels_s{a.seed}.json')
    target.write_text(json.dumps(dict(config=config,seconds=time.time()-started,results=results),indent=2))


if __name__=='__main__':
    main()
