"""Causal GRU distillation pilot with fixed first-observation calibration and session holdouts."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from phase0.analysis import seqctc as S,ctcv4 as V
from phase0.analysis.streaming_replay import raw_points
from phase0.analysis.typing_pixels import KBDS
from phase0.analysis.visual_fusion import greedy
from phase0.analysis.text_contract import edits

ROOT=Path('.cache/nextgen/causal_student')


def causal_features(points):
    A=np.zeros_like(points);M=np.zeros(points.shape[:2],bool)
    anchor=scale=mid=None
    for i,row in enumerate(points):
        hs=sorted([p for p in row if np.isfinite(p[:,:2]).all()],key=lambda p:p[0,1])
        if anchor is None:
            if len(hs)!=2:
                continue
            width=np.median([np.linalg.norm(p[5,:2]-p[17,:2]) for p in hs])
            if width<1 or np.linalg.norm(hs[0][0,:2]-hs[1][0,:2])<.6*width:
                continue
            scale=float(width)
            anchor=np.array([p[S.PALM,:2].mean(0) for p in hs])
            mid=np.mean([p[0,1] for p in hs])
        for j,p in enumerate(hs):
            h=j if len(hs)==2 else int(p[0,1]>mid)
            A[i,h,:,:2]=(p[:,:2]-anchor[h])/scale
            M[i,h]=True
    return S.featurize(torch.tensor(A)[None],torch.tensor(M)[None],False)[0,::2]


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(506),nn.Linear(506,96),nn.GELU())
        self.rnn=nn.GRU(96,128,batch_first=True)
        self.head=nn.Linear(128,29)

    def forward(self,x,state=None):
        y,state=self.rnn(self.input(x),state)
        return self.head(y).log_softmax(-1),state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args()
    suffix='_balanced' if args.balanced else ''
    torch.set_num_threads(2)
    ROOT.mkdir(parents=True,exist_ok=True)
    config=dict(steps=400,seed=0,teacher='three-seed hwtmix zero-shot',not_deployed=True,
        normalization='fixed first valid two-hand observation; no future frames',loss='frame KL distillation',balanced=args.balanced)
    (ROOT/f'config{suffix}.json').write_text(json.dumps(config,indent=2))
    desk=V.load_data(Path('.cache/ctc_v4/desk_data.json'))
    examples={}
    models=[V.V3/'runs/hwtmix'/f'seed{s}'/'all.pt' for s in range(3)]
    for sid in list(desk)+KBDS:
        t,raw=raw_points(S.SESS/sid)
        features=causal_features(raw)
        lp=V.ensemble_cont(Path('.cache/nextgen/typing_pixels'),models,sid)
        examples[sid]=(features,torch.tensor(lp),t)
    results={}
    for held in desk:
        torch.manual_seed(0)
        rng=np.random.default_rng(0)
        model=Student()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        train=[v for k,v in examples.items() if k!=held]
        for step in range(config['steps']):
            x,teacher,_=train[rng.integers(len(train))]
            a=int(rng.integers(max(1,len(x)-240)))
            lp,_=model(x[a:a+240][None])
            target=teacher[a:a+240].softmax(-1)
            kl=nn.functional.kl_div(lp[0],target,reduction='none').sum(-1)
            weight=.05+(1-target[:,0]-target[:,28]).clamp_min(0) if args.balanced else torch.ones_like(kl)
            loss=(kl*weight).sum()/weight.sum()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        x,teacher,t=examples[held]
        tick=time.perf_counter()
        state=None;chunks=[]
        with torch.inference_mode():
            for a in range(0,len(x),15):
                lp,state=model(x[a:a+15][None],state)
                chunks.append(lp[0])
        prediction=torch.cat(chunks)
        elapsed=time.perf_counter()-tick
        rows=[]
        for lo,hi,text in desk[held]['windows']:
            a,b=np.searchsorted(t[::2],[lo,hi])
            rows.append(dict(chars=len(text),ce=edits(text,greedy(prediction[a:b])),
                             teacher_ce=edits(text,greedy(teacher[a:b]))))
        results[held]=dict(rows=rows,inference_seconds=elapsed,video_seconds=float(t[-1]-t[0]))
        torch.save(dict(state=model.state_dict(),config=config,heldout=held),ROOT/f'{held}{suffix}.pt')
        print(held,'CER',sum(r['ce'] for r in rows)/sum(r['chars'] for r in rows),flush=True)
    Path(f'results/nextgen/causal_student{suffix}.json').write_text(json.dumps(dict(config=config,results=results),indent=2))


if __name__=='__main__':
    main()
