import numpy as np
import pytest

from phase0.analysis.desk_tune import (
    DESK_CFG,
    MISSING_HAND_FRAC,
    _time_refractory,
    mark_aborted,
    pick,
    ratios,
    run_length,
    score_ratios,
)


def test_run_length_matches_taps_gb():
    from phase0.analysis.taps_gb import _run_length

    rng = np.random.default_rng(0)
    m = rng.random(500) > 0.6
    assert np.array_equal(run_length(m), _run_length(m))


def test_time_refractory_uses_wall_clock_not_frame_index():
    # frames 1..3 arrive in a burst 2 ms apart; a frame-count guard would let them all through
    t = np.array([0.0, 0.100, 0.102, 0.104, 0.400])
    idx = np.array([0, 1, 2, 3, 4])
    assert list(_time_refractory(idx, t, 80.0)) == [0, 1, 4]
    assert list(_time_refractory(idx, t, 0.0)) == [0, 1, 2, 3, 4]


def test_pick_applies_threshold_and_time_refractory():
    t = np.arange(10) / 60.0
    p = np.array([0.0, 0.9, 0.0, 0.8, 0.0, 0.2, 0.0, 0.95, 0.0, 0.0])
    cfg = {"thr": 0.5, "refractory": 1, "n_consec": 1, "refractory_ms": 0.0}
    assert list(pick(p, t, cfg)) == [1, 3, 7]
    assert list(pick(p, t, dict(cfg, refractory_ms=60.0))) == [1, 7]


def test_mark_aborted_flags_windows_with_a_missing_hand():
    t = np.linspace(0, 2, 121)
    P = np.zeros((121, 2, 21, 7))
    P[60:, 1, :, :] = np.nan  # right hand leaves the frame halfway through
    W = [{"t0": 0.0, "t1": 1.0, "n_chars": 10}, {"t0": 1.0, "t1": 2.0, "n_chars": 10}]
    mark_aborted(W, {"t": t, "P": P})
    assert W[0]["aborted"] is False
    assert W[1]["aborted"] is True
    assert W[1]["frac_two_hands"] < 1.0 - MISSING_HAND_FRAC


def test_score_ratios_ignores_aborted_windows():
    W = [{"aborted": False}, {"aborted": False}, {"aborted": True}]
    npass, err, live = score_ratios(W, [1.0, 2.0, 0.05])
    assert (npass, live) == (1, 2)
    assert err == pytest.approx(np.log(2.0))


def test_ratios_counts_taps_per_window():
    W = [{"t0": 0.0, "t1": 1.0, "n_chars": 2}, {"t0": 1.0, "t1": 2.0, "n_chars": 4}]
    assert ratios(W, np.array([0.1, 0.2, 0.5, 1.5])) == [1.5, 0.25]


def test_desk_cfg_is_a_complete_pick_config():
    assert set(DESK_CFG) == {"smooth", "gate_thr", "refractory", "n_consec", "thr", "refractory_ms"}
