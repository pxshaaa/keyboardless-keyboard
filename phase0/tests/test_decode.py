import json
import math

import numpy as np
import pytest

from phase0.analysis import decode as D


def one_hot(text: str, conf: float = 0.999) -> np.ndarray:
    o = np.full((len(text), D.NA), (1 - conf) / (D.NA - 1))
    for i, c in enumerate(text):
        o[i, D.A_INDEX[c]] = conf
    return np.log(o)


@pytest.fixture(scope="module")
def lm():
    text = ("the cat sat on the mat and the dog ran to the park " * 200
            + "she sells sea shells and he sells nothing at all " * 200)
    return D.CharLM.train(text, order=5), D.WordLM.train(text)


def test_edit_distance_and_rates():
    assert D.edit_distance("kitten", "sitting") == 3
    assert D.cer("abc", "abc") == 0.0
    assert D.cer("abc", "") == 1.0
    assert D.wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_alphabet_is_lowercase_plus_space():
    assert D.NA == 27
    assert D.A_INDEX[" "] == 26
    assert D.KEY_ALIAS["space"] == " "


def test_charlm_prefers_seen_continuations(lm):
    clm, _ = lm
    p = clm.logprobs("the ca")
    assert p.argmax() == D.A_INDEX["t"]
    assert np.exp(p).sum() == pytest.approx(1.0, abs=1e-9)


def test_charlm_backs_off_to_something_finite(lm):
    clm, _ = lm
    p = clm.logprobs("zqxjk")
    assert np.isfinite(p).all()
    assert np.exp(p).sum() == pytest.approx(1.0, abs=1e-9)


def test_wordlm_prefix_trie(lm):
    _, wlm = lm
    assert wlm.is_prefix("she") and wlm.is_prefix("sh") and wlm.is_prefix("")
    assert not wlm.is_prefix("zqx")
    assert wlm.word_score("the") > wlm.word_score("zqxjk")


def test_beam_decode_recovers_clean_observations(lm):
    clm, wlm = lm
    assert D.beam_decode(one_hot("the cat sat"), clm, wlm, beam=20) == "the cat sat"


def test_beam_decode_drops_a_spurious_tap(lm):
    clm, wlm = lm
    obs = one_hot("the caat sat")
    out = D.beam_decode(obs, clm, wlm, D.Weights(insertion=-2.0), beam=40)
    assert out == "the cat sat"


def test_beam_decode_inserts_a_missing_character(lm):
    clm, wlm = lm
    obs = one_hot("the ct sat")
    out = D.beam_decode(obs, clm, wlm, D.Weights(deletion=-2.0), beam=40)
    assert out == "the cat sat"


def test_uniform_spatial_is_uninformative():
    lp = D.UniformSpatial().logp(None, [{}] * 3)
    assert lp.shape == (3, D.NA)
    assert np.allclose(lp, -math.log(D.NA))


def test_gaussian_spatial_separates_well_separated_keys():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal((0, 0), 1, (40, 2)), rng.normal((100, 100), 1, (40, 2))])
    m = D.GaussianSpatial.fit(xy, ["a"] * 40 + ["b"] * 40)
    lp = m.logp(None, [{"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 100.0}])
    assert lp.argmax(1).tolist() == [D.A_INDEX["a"], D.A_INDEX["b"]]
    assert np.allclose(np.exp(lp).sum(1), 1.0)


def test_gaussian_spatial_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    xy = rng.normal(0, 5, (60, 2))
    m = D.GaussianSpatial.fit(xy, ["a"] * 30 + ["b"] * 30)
    p = tmp_path / "s.npz"
    m.save(p)
    back = D.load_spatial(p)
    taps = [{"x": 1.0, "y": 2.0}]
    assert np.allclose(m.logp(None, taps), back.logp(None, taps))


def test_desk_segments_uses_shown_done_windows(tmp_path):
    rows = [{"t": 0.0, "event": "shown", "phrase": "one", "idx": 0},
            {"t": 10.0, "event": "done", "phrase": "one", "idx": 0},
            {"t": 10.0, "event": "shown", "phrase": "two", "idx": 1},
            {"t": 20.0, "event": "done", "phrase": "two", "idx": 1}]
    (tmp_path / "phrases.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    taps = [{"t": 1.0}, {"t": 5.0}, {"t": 11.0}, {"t": 99.0}]
    segs = D.desk_segments(tmp_path, taps)
    assert [t for t, _ in segs] == ["one", "two"]
    assert [len(s) for _, s in segs] == [2, 1]


def test_kbd_segments_splits_on_pauses_and_maps_space(tmp_path):
    keys = ([{"t": i * 0.2, "event": "down", "key": "a"} for i in range(10)]
            + [{"t": 2.0, "event": "down", "key": "space"}]
            + [{"t": 2.2 + i * 0.2, "event": "down", "key": "b"} for i in range(10)]
            + [{"t": 90.0 + i * 0.2, "event": "down", "key": "c"} for i in range(20)])
    (tmp_path / "keys.jsonl").write_text("\n".join(json.dumps(r) for r in keys))
    segs = D.kbd_segments(tmp_path, [])
    assert len(segs) == 2
    assert segs[0][0] == "aaaaaaaaaa bbbbbbbbbb"
    assert set(segs[1][0]) == {"c"}


# --- keyprobs spatial model -----------------------------------------------
def test_keyprobs_reads_the_tap_distribution():
    taps = [{"key_probs": {"a": 0.7, " ": 0.2}}, {"key_probs": {" ": 0.99}}]
    lp = D.KeyProbsSpatial().logp(None, taps)
    assert lp.shape == (2, D.NA)
    assert np.allclose(np.exp(lp).sum(1), 1.0)
    assert int(lp[0].argmax()) == D.A_INDEX["a"]
    assert int(lp[1].argmax()) == D.SPACE


def test_keyprobs_spreads_the_unlisted_tail_instead_of_zeroing_it():
    lp = D.KeyProbsSpatial().logp(None, [{"key_probs": {"a": 0.5}}])
    p = np.exp(lp[0])
    assert p[D.A_INDEX["z"]] > 0.0
    assert p[D.A_INDEX["z"]] < p[D.A_INDEX["a"]]
    assert p[D.A_INDEX["z"]] == pytest.approx(p[D.A_INDEX["b"]])


def test_keyprobs_leaves_a_tap_without_a_distribution_uniform():
    lp = D.KeyProbsSpatial().logp(None, [{"key_probs": {"a": 0.9}}, {}])
    assert np.allclose(np.exp(lp[1]), 1.0 / D.NA)


def test_keyprobs_refuses_taps_that_never_ran_tap_pos():
    with pytest.raises(SystemExit):
        D.KeyProbsSpatial().logp(None, [{"t": 1.0}, {"t": 2.0}])
