import itertools
import json

import numpy as np
import pytest

from phase0.analysis.calibration_review import decision, load_review
from phase0.analysis.context_spotter import spot
from phase0.analysis.text_contract import LEGACY_TOKENS, LITERAL_TOKENS, encode, literal, metrics, legacy_training_text, edits


def test_literal_symbols_and_metrics():
    text = 'get2germany für x = 2; _ Ä'
    assert ''.join(LITERAL_TOKENS[i] for i in encode(text)) == text
    assert len(set(LITERAL_TOKENS)) == len(LITERAL_TOKENS)
    assert LITERAL_TOKENS[:29] == LEGACY_TOKENS
    assert literal('fu\u0308r') == 'für'
    assert metrics('für 2', 'fr')['char_edits'] == 3
    with pytest.raises(ValueError):
        legacy_training_text('get2germany')
    with pytest.raises(ValueError):
        encode('emoji🙂')


def test_review_not_model_confidence():
    good = dict(status='ok', margin_per_char=0, greedy_cer=0)
    assert not decision(dict(status='pending', text='hello'), good, -1.5, .8)['accepted']
    assert not decision(dict(status='rejected', text='hello'), good, -1.5, .8)['accepted']
    bad = dict(status='ok', margin_per_char=-20, greedy_cer=1)
    assert decision(dict(status='confirmed', text='hello'), bad, -1.5, .8)['accepted']
    assert not decision(dict(status='confirmed', text='hello'), dict(good, cut_at_end=True), -1.5, .8)['accepted']
    assert not decision(dict(status='confirmed', text='für'), good, -1.5, .8)['accepted']


def test_review_bound_to_prompts(tmp_path):
    source = tmp_path / 'phrases.jsonl'
    source.write_text('original')
    assert load_review(tmp_path, [(0, 0, 1, 'hello')])[0]['status'] == 'pending'
    source.write_text('changed')
    with pytest.raises(ValueError):
        load_review(tmp_path, [(0, 0, 1, 'hello')])


def test_spotter_repeated_letters_require_blank():
    m = np.full((4,28), -30.)
    m[np.arange(4), [0,1,0,1]] = 0
    hit = spot(m, 'aa', max_cost=.01)[0]
    assert (hit['start'], hit['end'], hit['cost']) == (1,4,0)
    assert not spot(m[:2], 'aa', max_cost=.01)


def test_spotter_matches_exhaustive_subsequence_viterbi():
    rng = np.random.default_rng(7)
    m = np.full((5,28), -100.)
    m[:,:3] = rng.normal(size=(5,3))
    m -= np.logaddexp.reduce(m, axis=1, keepdims=True)
    relative = m - m.max(1, keepdims=True)
    target = (1,2)
    best = -np.inf
    for a in range(5):
        for b in range(a+1,6):
            for path in itertools.product(range(3), repeat=b-a):
                collapsed = tuple(c for i,c in enumerate(path) if c and (not i or c != path[i-1]))
                if collapsed == target:
                    best = max(best, sum(relative[t,c] for t,c in zip(range(a,b),path)))
    assert spot(m,'ab',max_cost=100,topk=1)[0]['cost'] == pytest.approx(-best/2)


def test_head_migration_preserves_existing_symbols():
    import torch
    from torch import nn
    from phase0.analysis.text_contract import expand_head
    model = nn.Module()
    model.head = nn.Linear(4,29)
    model.kw = {'nsym':29}
    before = model.head.weight.detach().clone()
    expand_head(model)
    assert torch.equal(before,model.head.weight[:29])
    assert model.kw['nsym'] == len(LITERAL_TOKENS)
    assert torch.all(model.head.bias[29:]==-12)


def test_ingestion_requires_review_and_retains_hard_confirmed_labels(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from phase0.analysis import calib_ingest as I
    session = tmp_path/'session'
    session.mkdir()
    (session/'phrases.jsonl').write_text('prompts')
    monkeypatch.setattr(I,'ensure_stream',lambda p:SimpleNamespace(t=np.arange(20.)))
    monkeypatch.setattr(I,'prompt_rows',lambda p:[(0,0,10,'hello')])
    monkeypatch.setattr(I.V,'load_data',lambda p:{})
    monkeypatch.setattr(I.V,'load_manifest_models',lambda *a:([],None))
    monkeypatch.setattr(I.V,'ensemble_cont',lambda *a:np.zeros((10,29)))
    monkeypatch.setattr(I,'score_window',lambda *a:dict(status='ok',t0=0,t1=10,
                                                    margin_per_char=-10,greedy_cer=1))
    a = SimpleNamespace(out_root=tmp_path/'out',base_root=tmp_path/'base',align_manifest='',
                        align_group='desk',sessions=[session],min_margin=-1.5,max_cer=.8,
                        version='test',no_train=True)
    assert I.cmd_ingest(a)==2
    assert not (a.out_root/'desk_data.json').exists()
    path = session/'calibration_review.json'
    review = json.loads(path.read_text())
    review['entries'][0]['status']='confirmed'
    path.write_text(json.dumps(review))
    assert I.cmd_ingest(a)==0
    data = json.loads((a.out_root/'desk_data.json').read_text())
    assert data['sessions']['session']['windows'][0][2]=='hello'
    report = json.loads((a.out_root/'ingest/session.json').read_text())
    assert report['rows'][0]['reason']=='confirmed_model_disagrees'


def test_pool_oracle_matches_exhaustive_choices():
    from phase0.analysis.nextgen_report import oracle_errors
    pools=[['a b','a'],['c','b c'],['','d']]
    ref='a b c'.split()
    expected=min(edits(ref,' '.join(c).split()) for c in itertools.product(*pools))
    assert oracle_errors(ref,pools)==expected
