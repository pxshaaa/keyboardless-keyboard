"""Aggregate matched visual pilots and candidate-pool diagnostics without further model selection."""
import json
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C


def oracle_errors(reference, pools):
    costs = np.arange(len(reference)+1)
    for pool in pools:
        rows = []
        for candidate in pool:
            row = costs.copy()
            for word in candidate.split():
                nxt = np.empty_like(row)
                nxt[0] = row[0]+1
                for j,ref in enumerate(reference,1):
                    nxt[j] = min(row[j]+1,nxt[j-1]+1,row[j-1]+int(ref!=word))
                row = nxt
            rows.append(row)
        costs = np.min(rows,axis=0)
    return int(costs[-1])


def main():
    root = Path('results/nextgen')
    runs = [json.loads((root/f'visual_fusion_s{i}.json').read_text()) for i in range(3)]
    per_session = {}
    diffs,denoms = [],[]
    totals = dict(geometry=0.,rgb=0.,shuffled=0.,baseline=0.,chars=0.)
    for sid in runs[0]['results']:
        per_session[sid] = {mode:float(np.mean([r['results'][sid][mode]['cer'] for r in runs]))
                            for mode in ('geometry','rgb')}
        per_session[sid]['rgb_shuffled'] = float(np.mean([r['results'][sid]['rgb']['shuffled_cer'] for r in runs]))
        gg = np.array([[p['ce'] for p in r['results'][sid]['geometry']['rows']] for r in runs]).mean(0)
        rr = np.array([[p['ce'] for p in r['results'][sid]['rgb']['rows']] for r in runs]).mean(0)
        ss = np.array([[p['shuffled_ce'] for p in r['results'][sid]['rgb']['rows']] for r in runs]).mean(0)
        nn = np.array([p['chars'] for p in runs[0]['results'][sid]['rgb']['rows']])
        bb = np.array([p['baseline_ce'] for p in runs[0]['results'][sid]['rgb']['rows']])
        diffs.extend(rr-gg)
        denoms.extend(nn)
        for k,v in [('geometry',gg),('rgb',rr),('shuffled',ss),('baseline',bb),('chars',nn)]:
            totals[k] += float(v.sum())
    rng=np.random.default_rng(0)
    ix=rng.integers(0,len(diffs),(5000,len(diffs)))
    delta=np.asarray(diffs)[ix].sum(1)/np.asarray(denoms)[ix].sum(1)
    summary=dict(sessions=per_session,pooled_cer={k:v/totals['chars'] for k,v in totals.items() if k!='chars'},
        delta_rgb_geometry=sum(diffs)/sum(denoms),descriptive_phrase_bootstrap_95=np.quantile(delta,[.025,.975]).tolist(),
        caveat='31 development phrases, three sessions and seeds; phrase bootstrap does not establish new-session generalization. Greedy phrase-window pilot, not deployed-decoder accuracy.',
        conclusion='Shuffled RGB is similar: no demonstrated gain from temporal image evidence. Do not deploy this pilot.')
    context=json.loads((root/'context_spotter.json').read_text())
    for name,(_,sdir,sid) in C.SETS.items():
        path=Path('data/sessions')/sdir/'truth.txt'
        if not path.exists():
            path=Path('data/sessions')/sid/'truth.txt'
        ref=C.norm(path.read_text()).split()
        rows=json.loads((Path('.cache/nextgen/context_spotter')/f'{name}.json').read_text())
        context['sets'][name]['pool_oracle_errors']=oracle_errors(ref,[[x['text'] for x in r['pool']] for r in rows])
    report=dict(visual=summary,context=context,production_model_changed=False)
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
