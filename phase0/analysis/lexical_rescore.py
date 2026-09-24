"""Local 8B rescoring of a truth-independent shortlist of lexical repairs."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path('.cache/nextgen/lexical_rescore')

def generate():
    sys.path.insert(0,str(Path('.cache/personal_llm').resolve()))
    from score import PersonalLM
    import mlx.core as mx
    ROOT.mkdir(parents=True,exist_ok=True)
    lm=PersonalLM('mlx-community/Qwen3-8B-4bit',mem_gb=7,cache_gb=.5)
    for path in sorted(Path('.cache/nextgen/lexical_cloze').glob('*.json')):
        out=ROOT/path.name
        if out.exists():continue
        r=json.loads(path.read_text());ctc=np.array(r['ctc']);language=np.array(r['lm'])
        ids=[0]+sorted((set(np.argsort(-ctc)[:24])|set(np.argsort(-language)[:24]))-{0})
        texts=[r['texts'][i] for i in ids]
        scores=lm.score(texts,bs=2)
        out.write_text(json.dumps(dict(texts=texts,ctc=ctc[ids].tolist(),lm=scores,shortlist='baseline plus top24 camera and top24 bigram; no reference input'),indent=2))
        mx.clear_cache();print(path.name,len(texts),flush=True)

def evaluate():
    from phase0.analysis import swipe_common as C
    from phase0.analysis.cloze_probe import compose
    from phase0.analysis.decipher import score_variant
    from phase0.analysis.text_contract import edits
    configs=[(l,w,g) for l in [.25,.5,1.,2.,4.,8.] for w in [0,2,4] for g in [-1,0,2,4]]
    output={};audit={}
    for name in C.SETS:
        segs,lines=C.load(name);hyps={'baseline':[]};hyps.update({str(c):[] for c in configs});audit[name]=[]
        for s in segs:
            r=json.loads((ROOT/f'{name}_{s["id"]}.json').read_text());texts=r['texts'];ctc=np.array(r['ctc']);lm=np.array(r['lm']);length=np.array([len(t.split()) for t in texts])
            hyps['baseline'].append(texts[0])
            for l,w,g in configs:
                sc=ctc+l*lm+w*length
                hyps[str((l,w,g))].append(texts[int(sc.argmax())] if g<0 else compose(texts[0],texts,sc,g))
            audit[name].append(dict(id=s['id'],oracle_errors=min(edits(s['truth'].split(),t.split()) for t in texts)))
        output[name]={}
        for key,h in hyps.items():
            sc=score_variant(lines,h);output[name][key]=dict(errors=int(sc['wed'].sum()),words=int(sc['wn'].sum()),hypotheses=h)
    cross={}
    for held in output:
        chosen=min(output[held],key=lambda k:sum(output[s][k]['errors'] for s in output if s!=held))
        cross[held]=dict(config=chosen,**output[held][chosen])
    Path('results/nextgen/lexical_rescore.json').write_text(json.dumps(dict(results=output,cross_session=cross,audit=audit,note='Development only; other-session weight selection; historical truth exposure; approximate composition'),indent=2))
    print(json.dumps(cross,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['generate','eval']);a=p.parse_args()
    {'generate':generate,'eval':evaluate}[a.step]()
