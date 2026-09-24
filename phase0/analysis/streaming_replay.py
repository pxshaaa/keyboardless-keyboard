"""Bounded-context camera-model replay using only landmarks already received."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from phase0.analysis import seqctc as S, ctcv3 as C, ctcv4 as V
from phase0.analysis.visual_fusion import greedy
from phase0.analysis.text_contract import edits


def ranges(n,hop=30,history=360,lookahead=30):
    if min(n,hop,history)<1 or lookahead<0 or any(x%2 for x in (hop,history,lookahead)):
        raise ValueError('Use positive even frame counts; nonnegative even lookahead')
    for start in range(0,n,hop):
        end=min(start+hop,n)
        yield max(0,start-history), min(n,end+lookahead), start, end


def raw_points(folder):
    frames=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]
    ids=np.array([r['i'] for r in frames])
    t=np.array([r['t'] for r in frames])
    if np.any(np.diff(ids)<=0) or np.any(np.diff(t)<=0):
        raise ValueError('Frame clocks must increase')
    tb=pq.read_table(folder/'landmarks.parquet').to_pydict()
    points=np.full((len(t),2,21,3),np.nan,np.float32)
    for i,h,j,x,y,z in zip(*(tb[k] for k in ['i','hand','joint','x','y','z'])):
        row=np.searchsorted(ids,i)
        if row<len(ids) and ids[row]==i:
            points[row,h,j]=[x,y,z]
    return t,points


def replay(model,points,t,lookahead=30):
    outputs=[]
    events=[]
    cumulative=0.
    for lo,hi,start,end in ranges(len(t),lookahead=lookahead):
        tick=time.perf_counter()
        available=points[lo:hi]
        if not np.isfinite(available).any():
            lp=np.full(((end-start+1)//2,29),-30.,np.float32)
            lp[:,0]=0
        else:
            A,M=S.normalise(available.copy())
            f=S.featurize(torch.tensor(A)[None],torch.tensor(M)[None],False)
            with torch.inference_mode():
                pred=model(f,[len(A)])[0].numpy()
            lp=pred[(start-lo)//2:(end-lo+1)//2]
        elapsed=time.perf_counter()-tick
        cumulative+=elapsed
        outputs.append(lp)
        events.append(dict(time=float(t[hi-1]-t[0]),through=float(t[end-1]-t[0]),
            compute_seconds=elapsed,text=greedy(torch.tensor(np.concatenate(outputs))),final=end==len(t)))
    return np.concatenate(outputs),events,cumulative


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session')
    p.add_argument('--lookahead',type=int,default=30)
    a=p.parse_args()
    torch.set_num_threads(2)
    sid=Path(a.session).name
    model=C.load(C.recipe('hwtmix'),V.V3/'runs/hwtmix/seed0/all.pt')
    t,points=raw_points(S.SESS/sid)
    lp,events,compute=replay(model,points,t,a.lookahead)
    data=V.load_data(Path('.cache/ctc_v4/desk_data.json'))
    rows=[]
    if sid in data:
        for lo,hi,text in data[sid]['windows']:
            i,j=np.searchsorted(t[::2],[lo,hi])
            hyp=greedy(torch.tensor(lp[i:j]))
            rows.append(dict(chars=len(text),ce=edits(text,hyp)))
    out=Path(f'results/nextgen/stream_{sid}_l{a.lookahead}.json')
    out.write_text(json.dumps(dict(session=sid,events=events,rows=rows,compute_seconds=compute,
        video_seconds=float(t[-1]-t[0]),lookahead_frames=a.lookahead,hop_frames=30,
        note='Recorded-landmark replay; excludes capture/extraction; legacy offline-trained model; greedy; not deployed'),indent=2))
    print(json.dumps(dict(output=str(out),compute_seconds=compute,
        cer=sum(r['ce'] for r in rows)/sum(r['chars'] for r in rows) if rows else None)))


if __name__=='__main__':
    main()
