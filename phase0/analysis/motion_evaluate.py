"""Compare optical-flow ensembles to zero-shot and existing session-held-out fine-tunes."""
import json
from pathlib import Path
import numpy as np
import torch
from phase0.analysis import seqctc as S,ctcv4 as V
from phase0.analysis.motion_residual import ROOT
from phase0.analysis.decipher import char_params
from phase0.analysis.text_contract import edits


def main():
    torch.set_num_threads(2);desk=V.load_data(Path('.cache/ctc_v4/desk_data.json'));lm=S.LMScorer();alpha,beta=char_params();results={}
    for sid in desk:
        st=S.load_session(sid)
        zs=V.ensemble_cont(Path('.cache/nextgen/typing_pixels'),[V.V3/'runs'/'hwtmix'/f'seed{s}'/'all.pt' for s in range(3)],sid)
        models=[Path('.cache/ctc_v4/runs/hwtmix')/f'seed{s}'/f'loso_{sid}'/'model.pt' for s in range(3)]
        for model in models:
            meta=json.loads((model.parent/'meta.json').read_text())
            if sid not in meta['excluded']:raise ValueError('Fine-tune trained on held-out session')
        ft=V.ensemble_cont(Path('.cache/ctc_v4'),models,sid)
        saved={mode:[np.load(ROOT/f'{sid}_{mode}_s{s}_lp.npz') for s in range(3)] for mode in ['geometry','flow']}
        results[sid]=[]
        for n,(lo,hi,truth) in enumerate(desk[sid]['windows']):
            i,j=np.searchsorted(st.t[::2],[lo,hi]);pools=dict(zero_shot=zs[i:j],existing_finetune=ft[i:j])
            for mode in saved:
                lp=np.mean([z[f'phrase{n}'] for z in saved[mode]],0);lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True);pools[mode]=lp
            # Fixed interpolation, evaluated as an experiment rather than selected on this phrase.
            for mode in ['geometry','flow']:
                lp=(pools[mode]+pools['existing_finetune'])/2;lp-=np.logaddexp.reduce(lp,axis=-1,keepdims=True);pools[mode+'_plus_finetune']=lp
            row=dict(truth=truth,chars=len(truth),words=len(truth.split()),methods={})
            for mode,lp in pools.items():
                raw=S.greedy(lp);hyp=S.beam_lm(lp,lm,alpha=alpha,beta=beta,beam=16)
                row['methods'][mode]=dict(greedy_ce=edits(truth,raw),char_ce=edits(truth,hyp),word_errors=edits(truth.split(),hyp.split()),hypothesis=hyp)
            results[sid].append(row)
        print(sid,'decoded',flush=True)
    summary={}
    rows=[r for rs in results.values() for r in rs]
    for mode in rows[0]['methods']:
        summary[mode]={k:sum(r['methods'][mode][k] for r in rows)/sum(r['words' if k=='word_errors' else 'chars'] for r in rows) for k in ['greedy_ce','char_ce','word_errors']}
    rng=np.random.default_rng(42);paired={}
    for mode in ['geometry_plus_finetune','flow_plus_finetune']:
        samples=[]
        for _ in range(10000):
            delta=words=0
            for session_rows in results.values():
                for i in rng.integers(len(session_rows),size=len(session_rows)):
                    r=session_rows[i];delta+=r['methods'][mode]['word_errors']-r['methods']['existing_finetune']['word_errors'];words+=r['words']
            samples.append(delta/words)
        paired[mode]=dict(delta_wer=summary[mode]['word_errors']-summary['existing_finetune']['word_errors'],ci95=np.quantile(samples,[.025,.975]).tolist())
    Path('results/nextgen/motion_evaluation.json').write_text(json.dumps(dict(results=results,summary=summary,paired_phrase_bootstrap=paired,bootstrap_note='10000 paired phrase resamples within recording; conditional on these three recordings, not new-session uncertainty.',note='Development only; whole-session visual-model holdouts; frozen char decoder; historical language model exposure remains. Three-seed ensembles; fixed 0.5 interpolation.'),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
