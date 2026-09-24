"""Recover alternatives from raw character hypotheses using a local 8B model, then verify with CTC."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT=Path('.cache/nextgen/span_repair')
PROMPTS={
 'coverage':'Recover the intended typed sentence from noisy camera character recognition. The polished guess may have discarded words. Inspect the raw character strings for missing opening words, short words between other words, and damaged suffixes. Preserve every supported part. Do not simply polish the guess. Give four plausible transcriptions, best first. Return ONLY a JSON array of four lowercase strings. No explanation.',
 'redundancy':'Recover one typed sentence from noisy camera character recognition. Other predictions from the SAME recording may contain repeated wording that helps resolve it. They may also be wrong. Use the raw character strings as evidence; do not invent a topic. Give four plausible transcriptions, best first. Return ONLY a JSON array of four lowercase strings. No explanation.'}


def parse(raw):
    try:
        match=re.search(r'\[.*\]',raw,re.S)
        parsed=json.loads(match.group()) if match else []
        values=[x if isinstance(x,str) else next((x[k] for k in ['transcription','text','sentence','hypothesis'] if k in x),'')
                if isinstance(x,dict) else '' for x in parsed]
        return [x for x in values if isinstance(x,str) and 0<len(x)<300][:4]
    except (ValueError,TypeError):
        return []


def prep():
    from phase0.analysis import swipe_common as C
    pred=json.loads(Path('results/nextgen/consensus/predictions.json').read_text())
    items=[]
    for name,(setname,*_) in C.SETS.items():
        data=json.loads((C.LLM/'sets'/f'{setname}.json').read_text())
        context=[r['hyps']['map'] for r in pred[name]]
        for row,previous in zip(data['items'],pred[name]):
            raw=[v['char'][0][0] for v in row['c'].values()]
            items.append(dict(set=name,id=row['id'],raw=raw,baseline=previous['hyps']['map'],context=context))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'input.json').write_text(json.dumps(dict(prompts=PROMPTS,items=items),indent=2))


def generate_all():
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    sys.path.insert(0,str(Path('.cache/personal_llm').resolve()))
    from score import PersonalLM
    data=json.loads((ROOT/'input.json').read_text())
    lm=PersonalLM('mlx-community/Qwen3-8B-4bit',mem_gb=7,cache_gb=.5)
    for item in data['items']:
        for kind,prompt in data['prompts'].items():
            out=ROOT/f'{item["set"]}_{item["id"]}_{kind}.json'
            if out.exists() and json.loads(out.read_text()).get('parser_version')==2:
                continue
            supplied={k:item[k] for k in ['raw','baseline']}
            if kind=='redundancy':
                supplied['other_predictions']=item['context']
            message=lm.tok.apply_chat_template([dict(role='system',content=prompt),
                dict(role='user',content=json.dumps(supplied))],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            raw=json.loads(out.read_text())['raw'] if out.exists() else generate(lm.model,lm.tok,prompt=message,max_tokens=256,sampler=make_sampler(temp=0),verbose=False)
            alternatives=parse(raw)
            alternatives=list(dict.fromkeys([' '.join(re.sub('[^a-z ]','',x.lower()).split()) for x in alternatives]))
            alternatives=list(dict.fromkeys([item['baseline']]+[x for x in alternatives if x]))
            scores=lm.score(alternatives,bs=2)
            out.write_text(json.dumps(dict(alternatives=alternatives,lm=scores,raw=raw,parser_version=2),indent=2))
            mx.clear_cache()
            print(out.name,len(alternatives),flush=True)


def evaluate():
    import numpy as np
    import torch
    from phase0.analysis import swipe_common as C
    from phase0.analysis.decipher import score_variant
    from phase0.analysis.text_contract import edits
    torch.set_num_threads(2)
    inputs=json.loads((ROOT/'input.json').read_text())['items']
    configs=[(k,l,w) for k in PROMPTS for l in [0.,.25,.5,1.,2.,4.] for w in [0.,1.,2.]]
    output={};oracle={}
    for name in C.SETS:
        segs,lines=C.load(name)
        items=[x for x in inputs if x['set']==name]
        hyps={str(c):[] for c in configs};hyps['baseline']=[];poolrows=[]
        for seg,item in zip(segs,items):
            hyps['baseline'].append(item['baseline'])
            alltexts=[]
            for kind in PROMPTS:
                r=json.loads((ROOT/f'{name}_{item["id"]}_{kind}.json').read_text())
                if r.get('parser_version')!=2:
                    raise ValueError('Reprocess cached generations with the current parser before scoring')
                texts=r['alternatives'];alltexts+=texts
                ctc=C.ctc_ens(seg['lps'],texts)
                for k,l,w in configs:
                    if k==kind:
                        score=ctc+l*np.array(r['lm'])+w*np.array([len(t.split()) for t in texts])
                        hyps[str((k,l,w))].append(texts[int(score.argmax())])
            poolrows.append(dict(id=item['id'],oracle_errors=min(edits(seg['truth'].split(),t.split()) for t in alltexts)))
        output[name]={}
        for key,h in hyps.items():
            sc=score_variant(lines,h)
            output[name][key]=dict(errors=int(sc['wed'].sum()),words=int(sc['wn'].sum()),hypotheses=h)
        oracle[name]=poolrows
    cross={}
    for held in output:
        keys=['baseline']+[str(c) for c in configs]
        chosen=min(keys,key=lambda k:sum(output[s][k]['errors'] for s in output if s!=held))
        cross[held]=dict(config=chosen,**output[held][chosen])
    Path('results/nextgen/span_repair.json').write_text(json.dumps(dict(results=output,cross_session=cross,oracle=oracle,
        note='Development only; generator designed after error audit; weights chosen on other sessions'),indent=2))
    print(json.dumps(cross,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['prep','generate','eval']);a=p.parse_args()
    {'prep':prep,'generate':generate_all,'eval':evaluate}[a.step]()
