"""Decode uniformly expanded automatic segments, scoring without per-phrase tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from phase0.analysis import seqctc as S,ctcv4 as V,decipher as D


def main():
    torch.set_num_threads(2);lm=S.LMScorer();a,b=D.char_params();results={}
    for sid,info in V.load_data(Path('.cache/ctc_v4/desk_data.json')).items():
        name='s_zz-v4loso-'+sid;data=json.loads((Path('.cache/llmdec/sets')/(name+'.json')).read_text());z=np.load(f'.cache/decipher/out/zz-v4loso-{sid}_lp.npz')
        lp=(z['zs']+z['desk'])/2;lp-=np.logaddexp.reduce(lp,axis=1,keepdims=True);results[sid]={}
        for pad in [0,15,30]:
            h=[S.beam_lm(lp[max(0,lo-pad):min(len(lp),hi+pad)],lm,a,b,16) for lo,hi in data['meta']['segments']]
            sc=D.score_variant([w[2] for w in info['windows']],h)
            results[sid][str(pad)]=dict(hyps=h,errors=int(sc['wed'].sum()),words=int(sc['wn'].sum()))
        print(sid,'boundary sweep complete',flush=True)
    selected={}
    for held in results:
        pad=min(['0','15','30'],key=lambda k:sum(results[s][k]['errors'] for s in results if s!=held))
        selected[held]=dict(pad_frames=pad,**results[held][pad])
    Path('results/nextgen/boundary_probe.json').write_text(json.dumps(dict(results=results,other_session_selection=selected,note='Development only, character beam; uniform extensions, no reference-dependent clipping; overlap counted as insertion errors. No 8B claim.'),indent=2));print([(s,r['pad_frames'],r['errors']) for s,r in selected.items()])

if __name__=='__main__':main()
