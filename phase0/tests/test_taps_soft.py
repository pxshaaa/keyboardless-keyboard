import numpy as np
import pytest

from phase0.analysis.taps_soft import (
    GATE,
    TARGETS,
    Target,
    apply_events,
    breakdown,
    clip_taps,
    dist_to_key,
    fbeta,
    make_target,
    rollover_mask,
    tune_events,
)

FPS = 60.0


def grid(n=600):
    return np.arange(n) / FPS


def test_dist_to_key_empty_is_large():
    assert (dist_to_key(grid(10), np.array([])) > 100).all()


def test_hard_target_is_binary_and_width_matches():
    t = grid(120)
    kt = np.array([1.0])
    y, w = make_target(t, kt, Target("h", "cls", half=0.033))
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert y.sum() == ((np.abs(t - 1.0) <= 0.033).sum())
    assert np.allclose(w[y > 0], 8.0) and np.allclose(w[y == 0], 1.0)


def test_gauss_target_peaks_at_one_and_decays():
    t = grid(240)
    kt = np.array([2.0])
    y, w = make_target(t, kt, Target("g", "xent", sigma=0.030))
    assert y.max() == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= y.min() < 0.01
    k = int(np.argmin(np.abs(t - 2.0)))
    assert y[k] > y[k + 3] > y[k + 6]
    assert w[k] > w[k + 6]


def test_weighted_hard_target_decays_weight_not_label():
    t = grid(120)
    kt = np.array([1.0])
    y, w = make_target(t, kt, Target("w", "cls", half=0.067, sigma=0.033, weighted=True))
    pos = np.where(y > 0)[0]
    k = int(np.argmin(np.abs(t - 1.0)))
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert w[k] > w[pos[-1]]
    assert w[pos].max() <= 8.0 + 1e-9


def test_target_registry_has_the_baseline_shape():
    hard = [s for s in TARGETS if s.name == "hard33"][0]
    assert (hard.kind, hard.half, hard.pos_weight) == ("cls", 0.033, 8.0)
    assert GATE.half == pytest.approx(0.300)


def test_rollover_mask_first_key_never_rollover():
    kt = np.array([0.0, 0.05, 0.5, 0.55, 0.60])
    m = rollover_mask(kt, 0.080)
    assert m.tolist() == [False, True, False, True, True]
    assert rollover_mask(np.array([1.0])).tolist() == [False]


def test_breakdown_perfect_detection():
    kt = np.array([0.0, 0.05, 1.0])
    s = breakdown(kt, kt.copy())
    assert s["recall"] == 1.0 and s["precision"] == 1.0
    assert s["recall_rollover"] == 1.0 and s["recall_nonrollover"] == 1.0
    assert s["n_rollover"] == 1 and s["n_nonrollover"] == 2


def test_breakdown_isolates_a_missed_rollover_key():
    kt = np.array([0.0, 0.05, 1.0, 2.0])
    s = breakdown(kt, np.array([0.0, 1.0, 2.0]))
    assert s["recall"] == pytest.approx(0.75)
    assert s["recall_rollover"] == 0.0
    assert s["recall_nonrollover"] == 1.0


def test_breakdown_counts_still_fp():
    s = breakdown(np.array([0.0]), np.array([0.0, 5.0]))
    assert s["still_fp"] == 1 and s["precision"] == pytest.approx(0.5)


def test_fbeta_weights_recall_more_as_beta_grows():
    s = {"recall": 0.9, "precision": 0.6}
    assert fbeta(s, 1.0) == pytest.approx(2 * 0.9 * 0.6 / 1.5)
    assert fbeta(s, 2.0) > fbeta(s, 1.0)
    lowr = {"recall": 0.6, "precision": 0.9}
    assert fbeta(lowr, 2.0) < fbeta(s, 2.0)
    assert fbeta({"recall": 0.0, "precision": 0.0}, 1.5) == 0.0


def test_tune_events_with_larger_beta_never_loses_recall_on_a_ramp():
    t = grid(1800)
    kt = t[np.arange(60, 1740, 90)]
    rng = np.random.default_rng(0)
    p = np.clip(np.exp(-0.5 * (dist_to_key(t, kt) / 0.03) ** 2)
                * rng.uniform(0.35, 1.0, len(t)) + 0.02 * rng.random(len(t)), 0, 1)
    mask = np.ones(len(t), bool)
    c1, _ = tune_events(p, t, kt, mask, None, 1.0)
    c2, _ = tune_events(p, t, kt, mask, None, 2.0)
    r1 = breakdown(kt, t[apply_events(p, c1)])["recall"]
    r2 = breakdown(kt, t[apply_events(p, c2)])["recall"]
    assert r2 >= r1 - 1e-9
    assert c2["thr"] <= c1["thr"] + 1e-9


def test_apply_events_respects_gate_and_refractory():
    p = np.zeros(200)
    p[50], p[53], p[120] = 0.9, 0.7, 0.95
    gate = np.ones(200)
    gate[110:130] = 0.0
    ev = apply_events(p, {"thr": 0.5, "smooth": 1, "refractory": 10, "n_consec": 1,
                          "gate_thr": 0.5}, gate)
    assert ev.tolist() == [50]


def test_clip_taps_drops_events_outside_the_keystroke_span():
    kt = np.array([1.0, 2.0])
    tt = np.array([0.5, 0.95, 1.5, 2.05, 3.0])
    assert clip_taps(tt, kt).tolist() == [0.95, 1.5, 2.05]
