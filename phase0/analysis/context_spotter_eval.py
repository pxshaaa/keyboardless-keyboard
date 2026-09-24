"""Development-only word-spotting ablation on saved v4 inputs; no LLM or remote calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase0.analysis.context_spotter import candidates
from phase0.analysis.text_contract import edits
from phase0.analysis import swipe_common as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terms', type=int, default=64)
    p.add_argument('--out', type=Path, default=Path('results/nextgen/context_spotter.json'))
    a = p.parse_args()
    torch.set_num_threads(2)
    from phase0.analysis.llmdec import ctc_ll
    lexpath = Path('.cache/personal/lexicon_personal.json')
    lex = json.loads(lexpath.read_text())['personal']
    counts = {w: sum(v['sources'].get(s,0) for s in ('notion_user','cc_local','cc_mini','git'))
              for w,v in lex.items() if w.isascii() and w.isalpha() and 3 <= len(w) <= 20
              and not v.get('in_generic') and not v.get('blocked')}
    terms = sorted((w for w,n in counts.items() if n >= 5), key=lambda w: (-counts[w],w))[:a.terms]
    if not terms:
        raise ValueError('No eligible context terms')
    private = Path('.cache/nextgen/context_spotter')
    private.mkdir(parents=True, exist_ok=True)
    cfg = dict(terms=terms, cost=3., lexicon_sha256=hashlib.sha256(lexpath.read_bytes()).hexdigest(),
               note='Development data; vocabulary excludes keys/other-author Notion counts. No truth-derived term list.')
    (private/'config.json').write_text(json.dumps(cfg,indent=2))
    dev = json.loads(Path('results/v4/dev_eval.json').read_text())
    t0 = time.time()
    allrows = {}
    for name,(setname,sdir,sid) in C.SETS.items():
        source = C.LLM/'sets'/f'{setname}.json'
        items = json.loads(source.read_text())['items']
        z = np.load(source.with_name(source.stem+'_lp.npz'))
        baselines = dev['sets'][name]['v4s']['hyps']
        if len(items) != len(baselines):
            raise ValueError('Baseline and posterior item counts differ')
        rows = []
        for item,base in zip(items,baselines):
            lps = [z[f"{item['id']}__{g}"] for g in item['c']]
            pool = candidates(C.ens28(lps),base,terms)
            scores = np.mean([ctc_ll(lp,[r['text'] for r in pool]) for lp in lps],axis=0)
            for r,sc in zip(pool,scores):
                r['ctc'] = float(sc)
            rows.append(dict(id=item['id'],baseline=base,pool=pool))
        allrows[name] = rows
        (private/f'{name}.json').write_text(json.dumps(rows))
        print(f'{name}: {len(rows)} segments, {sum(len(r["pool"])-1 for r in rows)} new candidates',flush=True)
    report = dict(disclosure='All three sessions are development data, never a blind accuracy claim.',
                  config_sha256=hashlib.sha256((private/'config.json').read_bytes()).hexdigest(),
                  term_count=len(terms),seconds=time.time()-t0,sets={},grid=[])
    refs = {}
    for name,(_,sdir,sid) in C.SETS.items():
        tf = Path('data/sessions')/sdir/'truth.txt'
        if not tf.exists():
            tf = Path('data/sessions')/sid/'truth.txt'
        refs[name] = C.norm(tf.read_text()).split()
        baseline = ' '.join(r['baseline'] for r in allrows[name]).split()
        report['sets'][name] = dict(words=len(refs[name]),baseline_errors=edits(refs[name],baseline),
                                    added_candidates=sum(len(r['pool'])-1 for r in allrows[name]))
    for margin in (0.,2.,5.,10.):
        row = dict(margin=margin,sets={})
        for name,rows in allrows.items():
            hyps,changes = [],0
            for r in rows:
                best = max(r['pool'],key=lambda x:x['ctc'])
                take = best['ctc'] > r['pool'][0]['ctc'] + margin
                hyps.append(best['text'] if take else r['baseline'])
                changes += int(take)
            err = edits(refs[name],' '.join(hyps).split())
            row['sets'][name] = dict(errors=err,wer=err/len(refs[name]),changed_segments=changes)
        row['errors'] = sum(v['errors'] for v in row['sets'].values())
        report['grid'].append(row)
    report['baseline_errors'] = sum(v['baseline_errors'] for v in report['sets'].values())
    report['best_development_errors'] = min(r['errors'] for r in report['grid'])
    report['promoted'] = False
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
