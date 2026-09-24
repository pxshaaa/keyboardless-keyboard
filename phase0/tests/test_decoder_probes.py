import numpy as np
from phase0.analysis.consensus_probe import decode
from phase0.analysis.span_repair import parse


def test_mbr_uses_expected_word_errors():
    texts=['a b','a c','d c']
    p=np.array([.4,.35,.25])
    assert decode(texts,np.log(p),1.,'map')=='a b'
    assert decode(texts,np.log(p),1.,'mbr')=='a c'


def test_consensus_can_combine_slots():
    texts=['a b','a c','d c']
    assert decode(texts,np.log([.4,.35,.25]),1.,'consensus')=='a c'


def test_generation_parser_accepts_strings_and_records():
    assert parse('["one", "two"]')==['one','two']
    assert parse('```json\n[{"transcription":"one"},{"text":"two"}]\n```')==['one','two']
    assert parse('not json')==[]
    assert parse('[{"unrelated":"no"},null,42]')==[]


def test_repeat_fusion_keeps_identical_evidence():
    from phase0.analysis.repeat_consensus import fuse
    p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    np.testing.assert_allclose(fuse(np.log(p),np.log(p)),np.log(p),atol=1e-7)


def test_cloze_composes_independent_repairs_and_respects_margin():
    from phase0.analysis.cloze_probe import compose,slots,parse
    texts=['a bad test','a good test','a bad case','a very bad test']
    assert compose(texts[0],texts,[0,3,2,1],0)=='a very good case'
    assert compose(texts[0],texts,[0,3,2,1],2)=='a good test'
    assert len(slots('a test'))==5
    assert parse('```json\n{"0":["a"]}\n```')=={'0':['a']}


def test_lexical_proposals_match_scalar_bigram_scores(tmp_path):
    from phase0.analysis.lexical_cloze import Cloze
    from phase0.analysis.seqctc2 import BigramScorer
    path=tmp_path/'bigram.npz'
    np.savez(path,vocab=np.array(['a','b','c']),uni=[10,3,2,1],codes=[1,2,4,6,9],cnt=[4,2,2,1,3],ctx=[6,3,3,0],types=[2,2,1,0])
    obj=Cloze.__new__(Cloze);obj.lm=BigramScorer(path);obj.cache={}
    for previous in ['', 'a','b','c']:
        for following in ['', 'a','b','c']:
            expected=sorted(obj.lm.vocab,key=lambda w:-(obj.lm.score([((previous,) if previous else (),w)])[0]+(obj.lm.score([((w,),following)])[0] if following else 0)))
            assert obj.suggest(previous,following)==expected
