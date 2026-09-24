"""Select short calibration blocks by diminishing-return character-transition coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from phase0.analysis.text_contract import legacy_training_text


def transitions(text):
    return Counter(zip(text,text[1:]))


def select(prompts, existing, count):
    covered = Counter()
    for text in existing:
        covered.update(transitions(text))
    remaining = list(dict.fromkeys(prompts))
    chosen = []
    for _ in range(min(count,len(remaining))):
        def gain(text):
            return sum(1/(1+covered[pair]) for pair in transitions(text))/max(1,len(text))**.5
        best = max(remaining,key=lambda t:(gain(t),t))
        chosen.append(best)
        covered.update(transitions(best))
        remaining.remove(best)
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--phrases-per-block',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('.cache/nextgen/calibration'))
    a=p.parse_args()
    if a.blocks<1 or a.phrases_per_block<1:
        p.error('Block sizes must be positive')
    paths=[Path('phase0/phrases_calib.txt'),Path('phase0/phrases_calib_natural.txt')]
    good,excluded=[],[]
    for path in paths:
        for text in path.read_text().splitlines():
            if not text.strip() or text.lstrip().startswith('#'):
                continue
            try:
                good.append(legacy_training_text(text))
            except ValueError:
                excluded.append(text)
    d=json.loads(Path('.cache/ctc_v4/desk_data.json').read_text())
    existing=[w[2] for s in d['sessions'].values() for w in s['windows']]
    chosen=select(good,existing,a.blocks*a.phrases_per_block)
    a.out.mkdir(parents=True,exist_ok=True)
    for i in range(a.blocks):
        (a.out/f'part{i+1}.txt').write_text('\n'.join(chosen[i::a.blocks])+'\n')
    summary=dict(blocks=a.blocks,selected=len(chosen),excluded_unsupported=len(excluded),
                 selection='Diminishing-return bigram coverage against existing labels, no new test truth.',
                 estimated_minutes_per_block=a.phrases_per_block*14/60,
                 collection='Separate sessions; remount between blocks, not during typing. Human review required.',
                 sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (a.out/'manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
