"""Reference-assisted boundary and candidate audit; never used to generate predictions."""
import json
from pathlib import Path
import os
import numpy as np
import torch
from phase0.analysis import seqctc as S,ctcv4 as V,swipe_common as C,llmdec as LD,v4_suggest as VS


def main():
    torch.set_num_threads(2);os.environ['LLMDEC_FUZZY_TAG']='';LD._FZ.clear()
    run=json.loads(Path('results/llmdec/frozen_config_v4.json').read_text())['run'];out={}
    for sid,info in V.load_data(Path('.cache/ctc_v4/desk_data.json')).items():
        name='s_zz-v4loso-'+sid;d,pools,_=LD.build_pools(name,('8b','p05b'));full=np.load(f'.cache/decipher/out/zz-v4loso-{sid}_lp.npz');segs=[]
        for (lo,hi),p in zip(d['meta']['segments'],pools):
            P=p['P']['ens'];sc=VS.pool_scores(P,run);hyp=P['texts'][int(sc.argmax())]
            segs.append(dict(hyp=hyp,lps=[full[g][lo:hi] for g in ['zs','desk']]))
        C.assign_truth(segs,[w[2] for w in info['windows']]);rows=[]
        for (lo,hi),p,s in zip(d['meta']['segments'],pools,segs):
            truth=s['truth'];hyp=s['hyp'];P=p['P']['ens'];normal=C.ctc_ens(s['lps'],[truth,hyp]);expanded=C.ctc_ens([full[g][max(0,lo-30):min(len(full[g]),hi+30)] for g in ['zs','desk']],[truth,hyp])
            words=set(w for text in P['texts'] for w in text.split());missing=[w for w in truth.split() if w not in words]
            _,states,labels=C.viterbi(C.ens28(s['lps']),truth)
            emitted=np.flatnonzero(states%2==1)
            rows.append(dict(id=p['id'],truth=truth,hypothesis=hyp,frames=[lo,hi],reference_words_absent_from_pool=missing,
                reference_minus_hyp_ctc=float(normal[0]-normal[1]),expanded_reference_minus_hyp_ctc=float(expanded[0]-expanded[1]),
                expansion_margin_gain=float((expanded[0]-expanded[1])-(normal[0]-normal[1])),
                forced_first_frame=int(emitted[0]) if len(emitted) else None,forced_last_frame=int(emitted[-1]) if len(emitted) else None,
                segment_frames=hi-lo))
        out[sid]=rows;print(sid,'audited',flush=True)
    Path('results/nextgen/boundary_diagnosis.json').write_text(json.dumps(dict(results=out,note='Diagnostic uses reference text and forced alignment, not predictive accuracy. +1s each side; margin changes may include neighboring activity. Baseline whole-session holdouts, cached pre-wordbeam candidate pools.'),indent=2))

if __name__=='__main__':main()
