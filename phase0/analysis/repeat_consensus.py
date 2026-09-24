"""Unsupervised detection and posterior alignment of repeated phrases within a recording."""
import json
from pathlib import Path
import numpy as np
from phase0.analysis import swipe_common as C,wordbeam_v4 as W
from phase0.analysis.text_contract import edits


def fuse(a,b):
    cost=1-np.sqrt(np.exp(a))@np.sqrt(np.exp(b)).T
    n,m=cost.shape
    dp=np.full((n+1,m+1),np.inf);dp[0,0]=0
    back=np.zeros((n,m),np.uint8)
    for i in range(n):
        for j in range(m):
            options=[dp[i,j],dp[i,j+1]+.05,dp[i+1,j]+.05]
            choice=int(np.argmin(options));back[i,j]=choice
            dp[i+1,j+1]=options[choice]+cost[i,j]
    mapping=[[] for _ in range(n)];i,j=n-1,m-1
    while i>=0 and j>=0:
        mapping[i].append(j)
        step=back[i,j]
        if step==0:i,j=i-1,j-1
        elif step==1:i-=1
        else:j-=1
    out=np.empty_like(a)
    for i,js in enumerate(mapping):
        if not js:raise ValueError('Incomplete temporal alignment')
        out[i]=np.logaddexp(a[i],np.logaddexp.reduce(b[js],axis=0)-np.log(len(js)))-np.log(2)
    return out


def main():
    predictions=json.loads(Path('results/nextgen/consensus/predictions.json').read_text())
    cfg=dict(W.CONFIGS['small'],**W.SEARCH)
    lm=W.LM();results={}
    for name,(setname,*_) in C.SETS.items():
        meta=json.loads((C.LLM/'sets'/f'{setname}.json').read_text())
        z=np.load(C.LLM/'sets'/f'{setname}_lp.npz')
        rows=predictions[name];results[name]=[]
        for i,a in enumerate(rows):
            for j in range(i+1,len(rows)):
                b=rows[j];ta=a['hyps']['map'];tb=b['hyps']['map']
                if min(len(ta.split()),len(tb.split()))<8 or edits(ta,tb)/max(len(ta),len(tb))>.3:
                    continue
                aa=C.ens28([z[f'{a["id"]}__{g}'] for g in meta['items'][i]['c']])
                bb=C.ens28([z[f'{b["id"]}__{g}'] for g in meta['items'][j]['c']])
                baseline=W.beam(aa,lm,cfg)[:10]
                combined=W.beam(fuse(aa,bb),lm,cfg)[:10]
                results[name].append(dict(first=a['id'],second=b['id'],baseline=baseline,combined=combined))
                print(name,a['id'],b['id'],baseline[:1],combined[:1],flush=True)
    Path('results/nextgen/repeat_consensus.json').write_text(json.dumps(dict(results=results,
        note='Pairs chosen without truth by >=8 words and <=0.3 normalized character distance; may merge distinct similar phrases; not deployed'),indent=2))


if __name__=='__main__':
    main()
