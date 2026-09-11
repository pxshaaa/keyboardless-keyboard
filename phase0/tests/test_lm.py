import math

import numpy as np
import pytest

from phase0.analysis.decode import ALPHABET, A_INDEX, NA, CharLM
from phase0.analysis.lm import (
    BOS,
    DecWeights,
    MixCharLM,
    WordBigramLM,
    align_score,
    beam_nbest,
    bootstrap_ci,
    char_logppl,
    contamination,
    decontaminate,
    normalize,
    paired_delta_ci,
    split_corpus,
    tune_mix,
)

TEXT = ("the meeting is on tuesday and the report is late " * 200
        + "please send me the document when you can " * 200)


@pytest.fixture(scope="module")
def clm():
    return CharLM.train(TEXT, 6)


def _sharp_obs(text: str, p: float = 0.9) -> np.ndarray:
    o = np.full((len(text), NA), (1 - p) / (NA - 1))
    for i, c in enumerate(text):
        o[i, A_INDEX[c]] = p
    return np.log(o)


def test_normalize_keeps_only_lowercase_and_space():
    assert normalize("Hello, World! 42\nOK") == "hello world ok"


def test_decontaminate_drops_sentences_carrying_a_test_phrase():
    keep = decontaminate(["i will send the report later today", "unrelated line"], "t")
    assert keep == ["unrelated line"]


def test_contamination_counts_exact_hits_and_ngrams():
    rep = contamination("aa i will send the report later today bb", ["i will send the report later today"])
    assert sum(rep["exact"].values()) == 1
    hit, tot = next(iter(rep["ngram"].values()))
    assert hit == tot > 0


def test_split_corpus_is_disjoint_and_covers_everything():
    tr, dv = split_corpus("x" * 500_000, dev_frac=0.1)
    assert len(tr) + len(dv) == 500_000 and len(dv) > 0


def test_mix_is_a_normalised_distribution_between_its_components(clm):
    other = CharLM.train("abc " * 5000, 6)
    mix = MixCharLM([clm, other], [0.5, 0.5])
    lp = mix.logprobs(" th")
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-6)
    a, b = np.exp(clm.logprobs(" th")), np.exp(other.logprobs(" th"))
    assert np.allclose(np.exp(lp), 0.5 * a + 0.5 * b, atol=1e-9)


def test_mix_of_one_model_with_itself_is_that_model(clm):
    assert np.allclose(MixCharLM([clm, clm], [0.3, 0.7]).logprobs(" the"), clm.logprobs(" the"))


def test_tune_mix_picks_the_component_matching_the_dev_text(clm):
    off = CharLM.train("zqx " * 20_000, 6)
    w, _ = tune_mix([clm, off], "the meeting is on tuesday " * 40)
    assert w[0] > w[1]


def test_char_logppl_is_lower_for_in_domain_text(clm):
    assert char_logppl(clm, "the report is late ") < char_logppl(clm, "zqxjvk zqxjvk ")


def test_word_bigram_prefers_the_seen_continuation():
    wl = WordBigramLM.train(TEXT, min_bi=1)
    assert wl.word_score("meeting", "the") > wl.word_score("meeting", "send")
    assert wl.is_prefix("meet") and not wl.is_prefix("qxz")


def test_word_bigram_backs_off_to_unigram_for_unseen_context():
    wl = WordBigramLM.train(TEXT, min_bi=1)
    assert wl.word_score("the", "zzz") == pytest.approx(math.log(wl.alpha) + wl.uni["the"])


def test_beam_recovers_clean_text_from_sharp_observations(clm):
    ref = "the meeting is on tuesday"
    out = beam_nbest(_sharp_obs(ref), clm, None, DecWeights(), beam=30, nbest=1)
    assert out[0][0] == ref


def test_nbest_is_sorted_deduplicated_and_capped(clm):
    out = beam_nbest(_sharp_obs("the report is late"), clm, None, DecWeights(), 30, nbest=8)
    assert len(out) == 8 == len({t for t, _ in out})
    assert all(out[i][1] >= out[i + 1][1] for i in range(len(out) - 1))


def test_length_bonus_makes_output_no_shorter(clm):
    obs = _sharp_obs("the report", p=0.3)
    short = beam_nbest(obs, clm, None, DecWeights(), 30)[0][0]
    long = beam_nbest(obs, clm, None, DecWeights(length_bonus=4.0), 30)[0][0]
    assert len(long) >= len(short)


def test_zero_taps_decodes_to_empty(clm):
    assert beam_nbest(np.zeros((0, NA)), clm, None, DecWeights(), 30)[0][0] == ""


def test_align_score_beats_a_wrong_string_on_matching_observations():
    obs = _sharp_obs("the report")
    w = DecWeights()
    assert align_score(obs, "the report", w) > align_score(obs, "zzz zzzzzz", w)


def test_align_score_matches_the_diagonal_when_lengths_agree():
    ref = "abcdef"
    obs = _sharp_obs(ref)
    diag = sum(obs[i, A_INDEX[c]] for i, c in enumerate(ref))
    assert align_score(obs, ref, DecWeights()) == pytest.approx(diag)


def test_bootstrap_ci_brackets_the_point_estimate():
    rows = [("hello world", "hello word"), ("good morning", "goo morning")]
    c, lo, hi = bootstrap_ci(rows, B=500)
    assert lo <= c <= hi and 0 < c < 0.2


def test_bootstrap_ci_is_zero_width_on_a_perfect_decode():
    assert bootstrap_ci([("abc", "abc")] * 5, B=200) == (0.0, 0.0, 0.0)


def test_paired_delta_ci_is_zero_when_both_systems_agree():
    rows = [("hello world", "hello word"), ("abc", "abd")]
    d, lo, hi = paired_delta_ci(rows, rows, B=500)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_delta_ci_is_negative_when_the_second_system_is_better():
    a = [("hello world", "xxxxx xxxxx")] * 4
    b = [("hello world", "hello world")] * 4
    d, lo, hi = paired_delta_ci(a, b, B=500)
    assert d < 0 and hi <= 0
