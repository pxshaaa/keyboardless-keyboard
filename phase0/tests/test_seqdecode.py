import math

import numpy as np
import pytest

from phase0.analysis.decode import ALPHABET, A_INDEX, NA, Weights
from phase0.analysis import seqdecode as sd


needs_data = pytest.mark.skipif(not sd.sess_path(sd.HELDOUT_SESSION).exists(), reason="private recordings not present")


def test_char_class_map_covers_alphabet():
    assert (sd.CHAR_CLS >= 0).all()
    assert sd.CHAR_CLS[A_INDEX[" "]] == sd.THUMB
    assert sd.CHAR_HAND[A_INDEX[" "]] == -1
    # 'f' and 'j' are the two index fingers on opposite hands
    assert sd.CHAR_HAND[A_INDEX["f"]] != sd.CHAR_HAND[A_INDEX["j"]]


@pytest.mark.parametrize("a,b,kind", [
    (0, 3, "thumb"), (3, 0, "thumb"), (1, 1, "same_finger"),
    (1, 2, "same_hand"), (1, 5, "alt_hand"), (5, 8, "same_hand"),
])
def test_trans_kind(a, b, kind):
    assert sd.trans_kind(a, b) == kind
    assert sd.KIND_MAT[a, b] == sd.KIND_ID[kind]


def _stats():
    logtrans = np.log(np.full((sd.NC, sd.NC), 1.0 / sd.NC))
    prior = np.log(np.full(sd.NC, 1.0 / sd.NC))
    mu = np.array([math.log(0.20), math.log(0.10), math.log(0.08), math.log(0.14)])
    sd_ = np.full(4, 0.4)
    return sd.MotorStats(logtrans, mu, sd_, prior, {}, {})


def test_ratio_is_zero_without_evidence():
    ms = _stats()
    assert np.allclose(ms.ratio(1, None, 1.0, 1.0), 0.0)
    assert np.allclose(ms.ratio(1, 0.1, 1.0, 0.0), 0.0)


def test_short_interval_penalises_same_finger():
    ms = _stats()
    r = ms.ratio(1, 0.06, 0.0, 1.0)
    same = r[1]                      # class 1 repeated == same finger
    alt = r[5]                       # opposite hand
    assert alt > same
    # a normalized ratio cannot uniformly inflate or deflate every class
    assert r.min() < 0.0 < r.max()


def test_long_interval_flips_the_preference():
    ms = _stats()
    r = ms.ratio(1, 0.40, 0.0, 1.0)
    assert r[1] > r[5]


def test_motor_off_matches_decode_beam():
    from phase0.analysis.decode import beam_decode

    rng = np.random.default_rng(0)
    obs = np.log(rng.dirichlet(np.ones(NA) * 0.3, size=6))
    t = np.arange(6) * 0.12
    a, _ = sd.joint_beam_decode(obs, t, None, None, None, Weights(), sd.MotorWeights(), beam=8)
    assert a == beam_decode(obs, None, None, Weights(), beam=8).strip()


def test_joint_decode_returns_a_trace_within_range():
    ms = _stats()
    rng = np.random.default_rng(1)
    obs = np.log(rng.dirichlet(np.ones(NA) * 0.3, size=8))
    t = np.arange(8) * 0.1
    text, trace = sd.joint_beam_decode(obs, t, None, None, ms, Weights(),
                                       sd.MotorWeights(1.0, 1.0), beam=8)
    idx = [i for i, _ in trace]
    assert idx == sorted(set(idx)) and all(0 <= i < 8 for i in idx)
    assert all(0 <= c < NA for _, c in trace)
    assert "".join(ALPHABET[c] for _, c in trace).strip() == text


def test_per_tap_scores_counts_uncovered_as_error():
    ref = ["a", "b", "c", "d"]
    pred = [A_INDEX["a"], A_INDEX["b"], -1, -1]
    r = sd.per_tap_scores([], ref, pred)
    assert r["n"] == 4 and r["covered"] == 0.5
    assert r["key"] == 0.5 and r["key_cov"] == 1.0


@needs_data
def test_finger_oracle_only_penalises_the_wrong_classes():
    s = sd.sess_path(sd.HELDOUT_SESSION)
    o = sd.FingerOracle(s, "taps_pos.jsonl")
    taps = [{"t": t} for t in list(o.cls)[:5]]
    b = o.bonus(taps)
    assert b.shape == (5, NA)
    assert (b <= 0).all() and (b == 0).any(1).all()


@needs_data
def test_timing_power_beats_the_kind_prior():
    ms = sd.fit_motor(list(sd.TRAIN_SESSIONS))
    tp = sd.timing_power(ms, list(sd.TRAIN_SESSIONS))
    assert tp["bayes_acc"] > tp["prior_acc"]
    # the interval is real but weak evidence: well under a fifth of the kind entropy
    assert 0.0 < tp["info_nats"] < 0.2 * tp["H_prior_nats"]


@needs_data
def test_fit_motor_on_a_real_session_is_ordered():
    ms = sd.fit_motor(list(sd.TRAIN_SESSIONS))
    assert ms.counts["pairs"] > 1000
    assert ms.iki["same_finger"]["median_ms"] > ms.iki["alt_hand"]["median_ms"]
    assert np.allclose(np.exp(ms.logtrans).sum(1), 1.0)
