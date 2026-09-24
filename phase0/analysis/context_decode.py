"""Generate task-context alternatives from a supplied posterior file without changing frozen decoders."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase0.analysis.context_spotter import candidates
from phase0.analysis.ctcv4 import merged,ctc_logp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('posteriors',type=Path)
    p.add_argument('--baseline',required=True)
    p.add_argument('--context',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    lp=np.load(a.posteriors,allow_pickle=False)
    if lp.ndim!=2 or lp.shape[1]!=29 or not np.isfinite(lp).all():
        raise ValueError('Expected finite [frames,29] log probabilities')
    context=json.loads(a.context.read_text())
    terms=context['terms']
    if not isinstance(terms,list) or len(terms)>512 or any(not isinstance(t,str) or len(t)>64 for t in terms):
        raise ValueError('Invalid context vocabulary')
    rows=candidates(merged(lp),a.baseline,terms)
    for row in rows:
        row['ctc_logp']=float(ctc_logp(lp,row['text']))
    rows.sort(key=lambda r:r['ctc_logp'],reverse=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:
        json.dump(dict(baseline=a.baseline,alternatives=rows,context_sources=context['sources'],
            note='Camera-only ranking; suggestions, not automatically accepted corrections'),f,indent=2)


if __name__=='__main__':
    main()
