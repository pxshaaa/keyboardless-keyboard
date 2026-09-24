"""Cross-session selection of minimum-risk and word-consensus decoders on existing pools."""
import json
from pathlib import Path
import numpy as np
from phase0.analysis import v4_dev as D,v4_suggest as V,swipe_common as C
from phase0.analysis.decipher import score_variant
from phase0.analysis.text_contract import edits

OUT=Path('results/nextgen/consensus')
TEMPS=[.5,1.,2.,4.,8.,16.]


def decode(texts,scores,temp,kind):
    idx=np.flatnonzero(np.isfinite(scores))
    idx=idx[np.argsort(-scores[idx],kind='stable')][:100]
    if not len(idx):
        raise ValueError('No finite candidates')
    chosen=[texts[i] for i in idx]
    p=np.exp((scores[idx]-scores[idx[0]])/temp);p/=p.sum()
    if kind=='map':
        return chosen[0]
    if kind=='mbr':
        words=[x.split() for x in chosen]
        risk=np.array([[edits(a,b)/max(1,len(b)) for b in words] for a in words])@p
        return chosen[int(risk.argmin())]
    slots=[{} for _ in chosen[0].split()]
    for text,weight in zip(chosen,p):
        spans=V.spans(chosen[0].split(),text.split())[0]
        for slot,word in zip(slots,spans):
            slot[word]=slot.get(word,0.)+float(weight)
    return ' '.join(w for slot in slots for w in max(slot,key=slot.get).split())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run=json.loads(D.FROZEN3.read_text())['run']
    config=dict(temperatures=TEMPS,methods=['map','mbr','consensus'],top_candidates=100,
        selection='other two sessions only; historical decoder development exposure remains',frozen_run=run)
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    candidates={};truth={}
    for name in C.SETS:
        pools=D.pools_for(name,'v4s')
        candidates[name]=[]
        for pool in pools:
            p=pool['P'][run['groups']]
            scores=V.pool_scores(p,run)
            texts=list(p['texts'])
            hyps={'map':decode(texts,scores,1.,'map')}
            for kind in ['mbr','consensus']:
                for t in TEMPS:
                    hyps[f'{kind}_{t}']=decode(texts,scores,t,kind)
            candidates[name].append(dict(id=pool['id'],hyps=hyps,
                texts=[text for text,s in zip(texts,scores) if np.isfinite(s)]))
        print(name,'predictions complete',flush=True)
    (OUT/'predictions.json').write_text(json.dumps(candidates,indent=2))
    results={};audit={}
    for name,rows in candidates.items():
        segments,lines=C.load(name)
        truth[name]=lines
        results[name]={}
        for key in rows[0]['hyps']:
            score=score_variant(lines,[r['hyps'][key] for r in rows])
            results[name][key]=dict(errors=int(score['wed'].sum()),words=int(score['wn'].sum()),
                by_line=score['hyp_by_line'])
        audit[name]=[]
        for row,segment in zip(rows,segments):
            ref=segment['truth']
            errors=[edits(ref.split(),h.split()) for h in row['texts']]
            audit[name].append(dict(id=row['id'],truth=ref,prediction=row['hyps']['map'],
                errors=edits(ref.split(),row['hyps']['map'].split()),
                pool_oracle_errors=min(errors),truth_in_pool=ref in row['texts']))
    cross={}
    for held in results:
        keys=list(results[held])
        best=min(keys,key=lambda k:sum(results[s][k]['errors'] for s in results if s!=held))
        cross[held]=dict(selected=best,**results[held][best],baseline_errors=results[held]['map']['errors'])
    report=dict(config=config,results=results,cross_session=cross,audit=audit,
        note='No fresh blind claim; oracle is diagnostic only; word consensus can synthesize new sentences')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(cross,indent=2))


if __name__=='__main__':
    main()
