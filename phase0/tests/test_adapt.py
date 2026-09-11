import math

import numpy as np
import pytest

from phase0.analysis import adapt as AD
from phase0.analysis.decode import A_INDEX, NA


def _obs(text, acc=0.9, drop=None, spurious=None, seed=0):
    """Observation matrix for a tap sequence that types `text`, optionally with a missed
    character (drop) or an extra detected tap (spurious)."""
    rng = np.random.RandomState(seed)
    codes = [A_INDEX[c] for c in text]
    if drop is not None:
        codes = codes[:drop] + codes[drop + 1:]
    rows = []
    for c in codes:
        r = np.full(NA, (1 - acc) / (NA - 1))
        r[c] = acc
        rows.append(r)
    if spurious is not None:
        rows.insert(spurious, rng.dirichlet(np.ones(NA)))
    return np.log(np.array(rows))


def test_forward_backward_recovers_one_to_one_alignment():
    text = "the meeting moved"
    ll, q, q_ins = AD.forward_backward(_obs(text), text, AD.AlignParams())
    assert math.isfinite(ll)
    assert q.shape == (len(text), NA)
    assert [int(v) for v in q.argmax(1)] == [A_INDEX[c] for c in text]
    assert q_ins.max() < 0.2


def test_posteriors_are_normalised():
    text = "hello world"
    _, q, q_ins = AD.forward_backward(_obs(text, acc=0.3), text, AD.AlignParams())
    assert np.allclose(q.sum(1), 1.0)
    assert ((q_ins >= 0) & (q_ins <= 1)).all()


def test_missing_tap_is_absorbed_as_a_skip():
    text = "please review the"
    ll, q, _ = AD.forward_backward(_obs(text, drop=5), text, AD.AlignParams())
    want = [A_INDEX[c] for c in text]
    del want[5]
    assert [int(v) for v in q.argmax(1)] == want
    assert math.isfinite(ll)


def test_spurious_tap_gets_insertion_mass():
    text = "ready on friday"
    _, _, q_ins = AD.forward_backward(_obs(text, spurious=4), text, AD.AlignParams())
    assert q_ins[4] > q_ins[np.arange(len(q_ins)) != 4].mean()


def test_true_text_scores_above_a_wrong_text():
    text = "the weather has been nice"
    obs = _obs(text)
    par = AD.AlignParams()
    good, _, _ = AD.forward_backward(obs, text, par)
    bad, _, _ = AD.forward_backward(obs, "quizzical jumpy foxes vex", par)
    assert good > bad + 5.0


def test_uninformative_observations_leave_posteriors_diffuse():
    text = "can you let me know"
    flat = np.log(np.full((len(text), NA), 1.0 / NA))
    _, q, _ = AD.forward_backward(flat, text, AD.AlignParams())
    ent = -(q * np.log(np.maximum(q, 1e-12))).sum(1).mean()
    assert ent > 0.8


def test_from_counts_keeps_the_character_budget_consistent():
    par = AD.AlignParams.from_counts(n_taps=591, n_chars=676, p_ins=0.08)
    assert 0.0 < par.p_skip < 0.5
    assert abs((1 - par.p_ins) * 591 / (1 - par.p_skip) - 676) < 1.0


def test_from_counts_clips_when_taps_outnumber_characters():
    par = AD.AlignParams.from_counts(n_taps=900, n_chars=100)
    assert par.p_skip == pytest.approx(0.01)


def test_align_all_ignores_empty_segments():
    text = "thanks for getting back"
    obs = np.vstack([_obs(text), np.log(np.full((3, NA), 1.0 / NA))])
    segs = [(text, np.arange(len(text))), ("unused", np.array([], int))]
    q, q_ins, ll_per_char, nchar = AD.align_all(obs, segs, AD.AlignParams())
    assert nchar == len(text)
    assert ll_per_char < 0
    assert np.allclose(q[len(text):], 1.0 / NA)  # taps outside any segment stay uninformed


def test_softmax_head_fits_soft_targets():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 6)
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    Q = np.full((300, NA), 0.02 / NA)
    Q[np.arange(300), y] += 0.98
    Q /= Q.sum(1, keepdims=True)
    head = AD.SoftmaxHead(l2=0.05).fit(X, Q)
    assert (head.proba(X).argmax(1) == y).mean() > 0.9


def test_softmax_head_honours_sample_weights():
    X = np.vstack([np.zeros((50, 2)), np.ones((50, 2))])
    Q = np.zeros((100, NA))
    Q[:50, 0] = 1.0
    Q[50:, 1] = 1.0
    w = np.concatenate([np.ones(50), np.zeros(50)])
    p = AD.SoftmaxHead(l2=0.01).fit(X, Q, w).proba(np.ones((1, 2)))
    assert p[0, 0] > p[0, 1]  # the down-weighted half must not decide the ones-region


def test_mix_interpolates_and_renormalises():
    a = np.full((4, NA), 1.0 / NA)
    b = np.zeros((4, NA))
    b[:, 3] = 1.0
    m = AD.mix(a, b, 0.5)
    assert np.allclose(m.sum(1), 1.0)
    assert m[0, 3] > a[0, 3]


def test_diagnostics_flag_a_collapsed_posterior():
    q = np.full((20, NA), 1e-6)
    q[:, 7] = 1.0
    q /= q.sum(1, keepdims=True)
    d = AD.diagnostics(q, np.zeros(20), [("x" * 20, np.arange(20))])
    assert d["conf90"] == 1.0
    assert d["topmass"] == 1.0
    assert d["ent"] < 0.01
