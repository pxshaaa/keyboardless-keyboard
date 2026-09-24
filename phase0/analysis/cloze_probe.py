"""Generate single-word repairs at every word and gap; validate them with held-session CTC/LM selection."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT=Path('.cache/nextgen/cloze')
PROMPT='For each numbered sentence, suggest three likely single English words for <blank>. Empty string is allowed if no word belongs there. Each sentence is independent. Do not repeat the whole sentence. Return ONLY a JSON object mapping each number to an array of three words. No explanation.'


def slots(text):
    words=text.split();out=[]
    for i in range(len(words)):
        out.append(dict(kind='replace',i=i,prompt=' '.join(words[:i]+['<blank>']+words[i+1:])))
    for i in range(len(words)+1):
        out.append(dict(kind='insert',i=i,prompt=' '.join(words[:i]+['<blank>']+words[i:])))
    return out


def parse(raw):
    try:
        match=re.search(r'\{.*\}',raw,re.S)
        result=json.loads(match.group()) if match else {}
        return result if isinstance(result,dict) else {}
    except ValueError:
        return {}


def compose(baseline,texts,scores,margin):
    original=baseline.split();choices={};base=float(scores[0])
    for text,score in zip(texts[1:],scores[1:]):
        gain=float(score)-base
        if gain<=margin:continue
        words=text.split();start=0
        while start<min(len(original),len(words)) and original[start]==words[start]:start+=1
        end_a,end_b=len(original),len(words)
        while end_a>start and end_b>start and original[end_a-1]==words[end_b-1]:end_a,end_b=end_a-1,end_b-1
        if end_a-start>1 or end_b-start>1:continue
        key=('insert' if end_a==start else 'replace',start)
        replacement=words[start:end_b]
        if key not in choices or gain>choices[key][0]:choices[key]=(gain,replacement)
    output=[]
    for i in range(len(original)+1):
        output+=choices.get(('insert',i),(0,[]))[1]
        if i<len(original):output+=choices.get(('replace',i),(0,[original[i]]))[1]
    return ' '.join(output)


def generate_all():
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    sys.path.insert(0,str(Path('.cache/personal_llm').resolve()))
    from score import PersonalLM
    ROOT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads(Path('.cache/nextgen/span_repair/input.json').read_text())['items']
    (ROOT/'config.json').write_text(json.dumps(dict(prompt=PROMPT,max_tokens=512,model='mlx-community/Qwen3-8B-4bit',
        candidate_policy='all word positions and all gaps, single edits, no truth input'),indent=2))
    lm=PersonalLM('mlx-community/Qwen3-8B-4bit',mem_gb=7,cache_gb=.5)
    for item in inputs:
        out=ROOT/f'{item["set"]}_{item["id"]}.json'
        ss=slots(item['baseline'])
        if out.exists():
            saved=json.loads(out.read_text())
            if saved.get('parsed_slots')==len(ss):continue
        request={str(i):s['prompt'] for i,s in enumerate(ss)}
        prompt=lm.tok.apply_chat_template([dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(request))],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        raw=generate(lm.model,lm.tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False)
        proposed=parse(raw)
        # Retry incomplete structured output in small batches, without changing
        # the prompt or selecting positions based on reference errors.
        missing=[k for k in request if k not in proposed]
        retries=[]
        for start in range(0,len(missing),6):
            batch={k:request[k] for k in missing[start:start+6]}
            retry_prompt=lm.tok.apply_chat_template([dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(batch))],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            retry=generate(lm.model,lm.tok,prompt=retry_prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False)
            proposed.update({k:v for k,v in parse(retry).items() if k in batch})
            retries.append(retry)
        words=item['baseline'].split();texts={item['baseline']}
        for key,alts in proposed.items():
            if not str(key).isdigit() or int(key)>=len(ss) or not isinstance(alts,list):continue
            slot=ss[int(key)];i=slot['i'];end=i+(slot['kind']=='replace')
            for w in alts[:3]:
                if not isinstance(w,str) or not re.fullmatch('[a-zA-Z]{0,25}',w):continue
                text=' '.join(words[:i]+([w.lower()] if w else [])+words[end:])
                if text:texts.add(text)
        texts=[item['baseline']]+sorted(texts-{item['baseline']})
        # Score all proposed single edits, not only those already favored by the camera.
        scores=lm.score(texts,bs=2)
        out.write_text(json.dumps(dict(baseline=item['baseline'],texts=texts,lm=scores,raw=raw,
            parsed_slots=len(proposed),expected_slots=len(ss),retries=retries),indent=2))
        mx.clear_cache();print(out.name,len(texts),len(proposed),flush=True)


def evaluate():
    import numpy as np
    import torch
    from phase0.analysis import swipe_common as C
    from phase0.analysis.decipher import score_variant
    from phase0.analysis.text_contract import edits
    torch.set_num_threads(2)
    configs=[(l,w,margin) for l in [.25,.5,1.,2.,4.,8.] for w in [0.,1.,2.] for margin in [-1,0,2,4]]
    output={};audits={}
    for name in C.SETS:
        segs,lines=C.load(name);hyps={'baseline':[]};hyps.update({str(c):[] for c in configs});audit=[]
        for s in segs:
            path=ROOT/f'{name}_{s["id"]}.json';r=json.loads(path.read_text());texts=r['texts']
            ctc=C.ctc_ens(s['lps'],texts);r['ctc']=ctc.tolist();path.write_text(json.dumps(r,indent=2))
            hyps['baseline'].append(r['baseline'])
            for l,w,margin in configs:
                sc=ctc+l*np.array(r['lm'])+w*np.array([len(t.split()) for t in texts])
                hyp=texts[int(sc.argmax())] if margin<0 else compose(r['baseline'],texts,sc,margin)
                hyps[str((l,w,margin))].append(hyp)
            audit.append(dict(id=s['id'],oracle_errors=min(edits(s['truth'].split(),t.split()) for t in texts),
                parsed_slots=r['parsed_slots'],expected_slots=r['expected_slots']))
        output[name]={}
        for key,h in hyps.items():
            sc=score_variant(lines,h);output[name][key]=dict(errors=int(sc['wed'].sum()),words=int(sc['wn'].sum()),hypotheses=h)
        audits[name]=audit
    cross={}
    for held in output:
        chosen=min(output[held],key=lambda k:sum(output[s][k]['errors'] for s in output if s!=held))
        cross[held]=dict(config=chosen,**output[held][chosen])
    Path('results/nextgen/cloze_probe.json').write_text(json.dumps(dict(results=output,cross_session=cross,audit=audits,
        note='Development only; generator after error audit; selection weights fit on the other two sessions; multi-edit composition uses additive single-edit gains, not exact joint LM scores'),indent=2))
    print(json.dumps(cross,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['generate','eval']);a=p.parse_args()
    {'generate':generate_all,'eval':evaluate}[a.step]()
