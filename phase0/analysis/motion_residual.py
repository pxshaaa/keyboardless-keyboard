"""Optical-flow residual CTC with whole-recording holdouts and timing controls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from phase0.analysis import seqctc as S,ctcv4 as V
from phase0.analysis.typing_pixels import KBDS
from phase0.analysis.visual_fusion import greedy
from phase0.analysis.text_contract import edits
ROOT=Path('.cache/nextgen/motion_residual')
RESOLUTION=.5


def track(previous,current,points,resolution=.5):
    p=np.asarray(points,np.float32).reshape(-1,1,2)
    win=15 if resolution==.5 else 31
    kw=dict(winSize=(win,win),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03))
    nxt,ok,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,**kw)
    if nxt is None:return np.zeros((len(p),2),np.float32),np.zeros(len(p),bool)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None,**kw)
    if back is None:return np.zeros((len(p),2),np.float32),np.zeros(len(p),bool)
    good=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<3*resolution)
    flow=nxt[:,0]-p[:,0];flow[~good]=0
    return flow,good


def extract(sid):
    folder=S.SESS/sid;target=ROOT/f'{sid}_flow.npz'
    signature=dict(recipe='lk-half-res-fb1.5-joints-v1' if RESOLUTION==.5 else 'lk-full-res-fb3-level2-v2',**{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['video.mp4','frames.jsonl','landmarks.parquet']})
    if target.exists():
        z=np.load(target)
        if json.loads(str(z['signature']))!=signature:raise ValueError('Stale motion cache')
        return z['features']
    frames=[json.loads(l)['i'] for l in (folder/'frames.jsonl').read_text().splitlines()]
    wanted={int(f):k for k,f in enumerate(frames[::2])}
    tb=pq.read_table(folder/'landmarks.parquet',columns=['i','hand','joint','x','y']).to_pydict();points={}
    for i,h,j,x,y in zip(*(tb[k] for k in ['i','hand','joint','x','y'])):
        points.setdefault(i,{}).setdefault(h,np.full((21,2),np.nan,np.float32))[j]=[x*RESOLUTION,y*RESOLUTION]
    out=np.zeros((len(wanted),2,21,5),np.float32)
    cap=cv2.VideoCapture(str(folder/'video.mp4'));i=seen=0;prev=None;prevpts=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        if i in wanted:
            gray=cv2.cvtColor(cv2.resize(frame,None,fx=RESOLUTION,fy=RESOLUTION),cv2.COLOR_BGR2GRAY)
            hs=sorted([p for p in points.get(i,{}).values() if np.isfinite(p).all()],key=lambda p:p[0,1])
            pts=np.stack(hs) if len(hs)==2 else None
            if pts is not None and prev is not None and prevpts is not None:
                flow,good=track(prev,gray,prevpts,RESOLUTION);flow=flow.reshape(2,21,2);good=good.reshape(2,21)
                for h in range(2):
                    scale=max(16*RESOLUTION,np.linalg.norm(prevpts[h,5]-prevpts[h,17]))
                    v=flow[h]/scale*10;landmark=(pts[h]-prevpts[h])/scale*10
                    out[wanted[i],h,:,:2]=np.clip(v,-3,3)
                    out[wanted[i],h,:,2:4]=np.clip(v-landmark,-3,3)*good[h,:,None]
                    out[wanted[i],h,:,4]=good[h]
            prev,prevpts=gray,pts;seen+=1
        i+=1
    cap.release()
    if seen!=len(out):raise ValueError('Video/landmark clock mismatch')
    features=out.reshape(len(out),-1)
    np.savez_compressed(target,features=features,signature=json.dumps(signature))
    print(sid,'flow frames',len(out),'valid',float(out[:,:,:,4].mean()),flush=True)
    return features


class MotionResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo=nn.LayerNorm(506)
        self.input=nn.Conv1d(506+210+29,64,1)
        self.blocks=nn.ModuleList([nn.Conv1d(64,64,5,padding=2*d,dilation=d) for d in [1,2,4,8]])
        self.out=nn.Conv1d(64,29,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,geo,flow,base,use_flow=True):
        x=torch.cat([self.geo(geo),flow if use_flow else flow*0,base.softmax(-1)],1).T[None]
        x=torch.nn.functional.gelu(self.input(x))
        for block in self.blocks:x=x+.2*torch.nn.functional.gelu(block(x))
        return (base+self.out(x)[0].T).log_softmax(-1)


def load_examples():
    desk=V.load_data(Path('.cache/ctc_v4/desk_data.json'));examples={}
    models=[V.V3/'runs'/'hwtmix'/f'seed{s}'/'all.pt' for s in range(3)]
    for sid in list(desk)+KBDS:
        flow=extract(sid);st=S.load_session(sid)
        geo=S.featurize(torch.from_numpy(st.A)[None],torch.from_numpy(st.M)[None],False)[0,::2]
        base=V.ensemble_cont(Path('.cache/nextgen/typing_pixels'),models,sid)
        if not len(flow)==len(geo)==len(base):raise ValueError('Feature clock mismatch')
        windows=desk[sid]['windows'] if sid in desk else [(lo,hi,S.sym_text(sy)) for lo,hi,sy in S.eval_windows(st)]
        examples[sid]=[]
        for lo,hi,text in windows:
            i,j=np.searchsorted(st.t[::2],[lo,hi])
            if len(text)<2 or j-i<2*len(text):continue
            examples[sid].append((geo[i:j].clone(),torch.tensor(flow[i:j]),torch.tensor(np.array(base[i:j]),dtype=torch.float32),text))
        print(sid,'windows',len(examples[sid]),flush=True)
    return desk,examples


def main():
    global ROOT,RESOLUTION
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=600);p.add_argument('--pretrain',type=int,default=300);p.add_argument('--resolution',type=float,choices=[.5,1.],default=.5);a=p.parse_args()
    RESOLUTION=a.resolution
    if RESOLUTION==1.:ROOT=Path('.cache/nextgen/motion_fullres_v2')
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    config=dict(**vars(a),evaluation='whole-session heldout, phrase windows, development only',anchor=1.,desk_probability=.5,receptive_field_frames=61,not_deployed=True)
    (ROOT/f'config_s{a.seed}.json').write_text(json.dumps(config,indent=2))
    desk,examples=load_examples();keyboard=[e for sid in KBDS for e in examples[sid]];results={};started=time.time()
    for use_flow in [False,True]:
        torch.manual_seed(a.seed);model=MotionResidual();rng=np.random.default_rng(a.seed)
        def train(es,steps,adapt=False):
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);model.train()
            for step in range(steps):
                source=keyboard if adapt and rng.random()<.5 else es
                geo,flow,base,text=source[rng.integers(len(source))];lp=model(geo,flow,base,use_flow)
                target=torch.tensor(S.text_syms(text),dtype=torch.long)
                loss=nn.functional.ctc_loss(lp[:,None],target,[len(lp)],[len(target)])+nn.functional.kl_div(lp,base.softmax(-1),reduction='batchmean')
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        train(keyboard,a.pretrain);initial={k:v.clone() for k,v in model.state_dict().items()}
        for held in desk:
            model.load_state_dict(initial);rng=np.random.default_rng(a.seed);train([e for sid in desk if sid!=held for e in examples[sid]],a.steps,True);model.eval();rows=[];post={}
            with torch.inference_mode():
                for n,(geo,flow,base,text) in enumerate(examples[held]):
                    lp=model(geo,flow,base,use_flow);pred=greedy(lp)
                    shuf=greedy(model(geo,flow[torch.randperm(len(flow))],base,use_flow));zero=greedy(model(geo,flow*0,base,use_flow))
                    rows.append(dict(truth=text,hypothesis=pred,chars=len(text),ce=edits(text,pred),baseline_ce=edits(text,greedy(base)),shuffled_ce=edits(text,shuf),zero_ce=edits(text,zero)))
                    post[f'phrase{n}']=lp.numpy()
            mode='flow' if use_flow else 'geometry';results.setdefault(held,{})[mode]=rows
            torch.save(dict(state=model.state_dict(),config=config,heldout=held),ROOT/f'{held}_{mode}_s{a.seed}.pt');np.savez_compressed(ROOT/f'{held}_{mode}_s{a.seed}_lp.npz',**post)
            print(held,mode,{k:sum(r[k] for r in rows)/sum(r['chars'] for r in rows) for k in ['ce','baseline_ce','shuffled_ce','zero_ce']},flush=True)
            prefix='motion_residual' if RESOLUTION==.5 else 'motion_fullres_v2'
            Path(f'results/nextgen/{prefix}_s{a.seed}.json').write_text(json.dumps(dict(config=config,seconds=time.time()-started,results=results),indent=2))

if __name__=='__main__':main()
