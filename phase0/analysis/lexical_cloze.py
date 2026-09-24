"""Exact full-vocabulary two-sided bigram proposals with CTC verification and cross-session selection."""
import json
from pathlib import Path
import re
import numpy as np
import torch
from phase0.analysis import seqctc2 as S2,swipe_common as C
from phase0.analysis.cloze_probe import slots,compose
from phase0.analysis.decipher import score_variant
from phase0.analysis.text_contract import edits


class Cloze:
    def __init__(self):
        self.lm=S2.BigramScorer(Path('.cache/personal/bigram_personal.npz'))
        self.cache={}

    def suggest(self,previous,following):
        key=(previous,following)
        if key in self.cache:return self.cache[key]
        m=self.lm;n=len(m.vocab);p=m.puni[:n].copy();ids=np.arange(n)
        ci=m.ix.get(previous)
        if ci is not None and m.ctx[ci]>0:
            p=m.D*m.types[ci]/m.ctx[ci]*p
            lo,hi=np.searchsorted(m.codes,[ci*m.V,(ci+1)*m.V])
            for code,count in zip(m.codes[lo:hi],m.cnt[lo:hi]):
                wi=int(code)%m.V
                if wi<n:p[wi]+=max(count-m.D,0)/m.ctx[ci]
        wi=m.ix.get(following)
        if wi is not None:
            codes=ids*m.V+wi;k=np.searchsorted(m.codes,codes)
            valid=k<len(m.codes);counts=np.zeros(n)
            valid[valid]&=m.codes[k[valid]]==codes[valid]
            counts[valid]=m.cnt[k[valid]]
            den=m.ctx[:n];q=np.full(n,m.puni[wi]);nz=den>0
            q[nz]=(np.maximum(counts[nz]-m.D,0)+m.D*m.types[:n][nz]*m.puni[wi])/den[nz]
            p*=q
        out=[m.vocab[i] for i in np.argsort(-p) if re.fullmatch('[a-z]+',m.vocab[i])][:20]
        self.cache[key]=out;return out

    def score(self,text):
        words=text.split();m=self.lm
        return sum(m.score([(tuple(words[:i]),w)])[0] if w in m.ix else -20 for i,w in enumerate(words))


def main():
    torch.set_num_threads(2)
    lm=Cloze();inputs=json.loads(Path('.cache/nextgen/span_repair/input.json').read_text())['items']
    configs=[(l,w,g) for l in [.25,.5,1.,2.,4.] for w in [0,2,4] for g in [-1,0,2,4]]
    root=Path('.cache/nextgen/lexical_cloze');root.mkdir(parents=True,exist_ok=True)
    output={};audit={}
    for name in C.SETS:
        segs,lines=C.load(name);items=[x for x in inputs if x['set']==name]
        hyps={'baseline':[]};hyps.update({str(c):[] for c in configs});audit[name]=[]
        for s,item in zip(segs,items):
            words=item['baseline'].split();pool={item['baseline']}
            for slot in slots(item['baseline']):
                i=slot['i'];end=i+(slot['kind']=='replace')
                previous=words[i-1] if i else '';following=words[end] if end<len(words) else ''
                for word in lm.suggest(previous,following)+([''] if slot['kind']=='replace' else []):
                    text=' '.join(words[:i]+([word] if word else [])+words[end:])
                    if text:pool.add(text)
            texts=[item['baseline']]+sorted(pool-{item['baseline']})
            ctc=C.ctc_ens(s['lps'],texts);language=np.array([lm.score(t) for t in texts]);length=np.array([len(t.split()) for t in texts])
            (root/f'{name}_{item["id"]}.json').write_text(json.dumps(dict(texts=texts,ctc=ctc.tolist(),lm=language.tolist())))
            hyps['baseline'].append(item['baseline'])
            for l,w,g in configs:
                sc=ctc+l*language+w*length
                hyps[str((l,w,g))].append(texts[int(sc.argmax())] if g<0 else compose(item['baseline'],texts,sc,g))
            audit[name].append(dict(id=item['id'],candidates=len(texts),oracle_errors=min(edits(s['truth'].split(),t.split()) for t in texts)))
        output[name]={}
        for key,h in hyps.items():
            sc=score_variant(lines,h);output[name][key]=dict(errors=int(sc['wed'].sum()),words=int(sc['wn'].sum()),hypotheses=h)
        print(name,'complete',flush=True)
    cross={}
    for held in output:
        selected=min(output[held],key=lambda k:sum(output[s][k]['errors'] for s in output if s!=held))
        cross[held]=dict(config=selected,**output[held][selected])
    Path('results/nextgen/lexical_cloze.json').write_text(json.dumps(dict(results=output,cross_session=cross,audit=audit,
        note='Development data; full vocabulary proposals based on neighboring words; weights chosen on other recordings; composition uses additive gains'),indent=2))
    print(json.dumps(cross,indent=2))


if __name__=='__main__':main()
