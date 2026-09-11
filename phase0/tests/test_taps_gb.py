"""Unit tests for the gradient-boosted tap detector's feature and event plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phase0.analysis import taps_gb as G


def _synth(F: int = 400, seed: int = 0) -> np.ndarray:
    """[F,2,21,7] plausible hands: still for the first half, one tap dip in the second."""
    rng = np.random.default_rng(seed)
    P = np.zeros((F, 2, 21, 7))
    for s in range(2):
        base_x = 300 + 400 * s
        for j in range(21):
            P[:, s, j, 0] = base_x + 10 * (j % 5)
            P[:, s, j, 1] = 300 + 8 * (j // 5)
            P[:, s, j, 2] = 0.9
            P[:, s, j, 3] = 2.0 * (j % 5)
            P[:, s, j, 4] = 0.01 * (j % 5)
            P[:, s, j, 5] = 0.01 * (j // 5)
            P[:, s, j, 6] = 0.002 * j
        P[:, s, 9, 0] = base_x + 60
        P[:, s, 9, 1] = 380
    P += rng.normal(0, 0.05, P.shape)
    dip = np.exp(-0.5 * ((np.arange(F) - 300) / 3.0) ** 2)
    P[:, 0, 8, 1] += 25 * dip
    return P


def test_build_groups_shapes_and_finiteness():
    P = _synth()
    g = G.build_groups(P)
    assert set(g) == set(G.GROUPS)
    for name, block in g.items():
        assert block.shape[0] == len(P), name
        assert block.shape[1] > 0, name
        # only the +/-lag edges may be NaN; the interior must be usable
        assert np.isfinite(block[20:-20]).all(), name


def test_assemble_respects_group_order_and_subset():
    P = _synth(200)
    g = G.build_groups(P)
    full = G.assemble(g, G.GROUPS)
    assert full.shape[1] == sum(b.shape[1] for b in g.values())
    sub = G.assemble(g, ("pose", "still"))
    assert sub.shape[1] == g["pose"].shape[1] + g["still"].shape[1]
    assert np.allclose(sub[:, : g["pose"].shape[1]], g["pose"], equal_nan=True)
    assert len(G.group_names(g, ("pose", "still"))) == sub.shape[1]


def test_rest_features_are_near_zero_when_pose_is_static():
    P = _synth(400)
    g = G.build_groups(P)
    rest = g["rest"]
    still_half = rest[50:200]
    moving = np.abs(g["rest"][295:305]).max()
    assert np.abs(still_half).max() < moving


def test_still_energy_rises_during_the_tap():
    P = _synth(400)
    g = G.build_groups(P)
    e = g["still"]
    assert np.nanmax(e[290:315]) > np.nanmax(e[50:200])


def test_labels_window():
    t = np.arange(0, 1, 1 / 60)
    kt = np.array([0.5])
    y = G.labels(t, kt, 0.033)
    assert y.sum() == int((np.abs(t - 0.5) <= 0.033).sum()) >= 3
    assert np.all(np.abs(t[y] - 0.5) <= 0.033 + 1e-9)
    assert G.labels(t, np.array([]), 0.033).sum() == 0


def test_run_length():
    m = np.array([0, 1, 1, 1, 0, 1, 0], bool)
    assert G._run_length(m).tolist() == [0, 3, 3, 3, 0, 1, 0]


def test_pick_events_refractory_and_consecutive():
    p = np.zeros(100)
    p[10] = 0.9
    p[12] = 0.8  # inside the refractory window of the peak at 10
    p[50] = 0.9
    assert G.pick_events(p, 0.5, refractory=5).tolist() == [10, 50]
    # a one-frame spike cannot satisfy a 3-frame run requirement
    assert len(G.pick_events(p, 0.5, refractory=5, n_consec=3)) == 0
    q = np.zeros(100)
    q[20:26] = 0.9
    assert len(G.pick_events(q, 0.5, refractory=5, n_consec=3)) == 1


def test_post_hoc_height_filter_matches_find_peaks():
    rng = np.random.default_rng(1)
    p = np.abs(rng.normal(0, 0.3, 2000))
    for thr in (0.2, 0.4, 0.6):
        a = G.pick_events(p, thr, refractory=7)
        from scipy.signal import find_peaks
        idx, pr = find_peaks(p, height=0.05, distance=7)
        b = idx[pr["peak_heights"] >= thr]
        assert a.tolist() == b.tolist()


def test_runs_and_score_masked_micro_average():
    t = np.arange(0, 10, 1 / 60)
    mask = np.zeros(len(t), bool)
    mask[100:200] = True
    mask[400:500] = True
    assert G._runs(mask) == [(100, 199), (400, 499)]
    kt = np.array([t[150], t[450]])
    ev = np.array([150, 450])
    s = G.score_masked(kt, t, mask, ev)
    assert s["hits"] == 2 and s["f1"] == pytest.approx(1.0)


def test_score_masked_ignores_events_outside_mask():
    t = np.arange(0, 10, 1 / 60)
    mask = np.zeros(len(t), bool)
    mask[100:200] = True
    kt = np.array([t[150]])
    s = G.score_masked(kt, t, mask, np.array([150, 450]))
    assert s["taps"] == 1 and s["hits"] == 1


def test_block_groups_are_contiguous_per_session():
    t = np.concatenate([np.arange(0, 60, 1 / 60), np.arange(100, 160, 1 / 60)])
    sid = np.concatenate([np.zeros(3600, int), np.ones(3600, int)])
    g = G.block_groups(t, sid, block_s=20.0)
    assert len(np.unique(g)) == 6
    for b in np.unique(g):
        m = g == b
        assert len(np.unique(sid[m])) == 1
        assert np.all(np.diff(np.where(m)[0]) == 1)


def test_cv_folds_session_mode_is_leave_one_session_out():
    t = np.arange(1000) / 60
    sid = np.array([0] * 400 + [1] * 600)
    folds = G.cv_folds("session", t, sid)
    assert len(folds) == 2
    assert folds[0].sum() == 400 and folds[1].sum() == 600
    assert not (folds[0] & folds[1]).any()


def test_purge_removes_training_frames_next_to_the_test_fold():
    t = np.arange(1200) / 60
    te = np.zeros(1200, bool)
    te[600:700] = True
    keep = G.purge(~te, te, t, embargo=0.5)
    assert not keep[te].any()
    assert not keep[570]  # within 0.5 s of the fold start
    assert keep[500]


def test_subsample_keys_is_sorted_and_sized():
    kt = np.sort(np.random.default_rng(0).uniform(0, 100, 200))
    s = G._subsample_keys(kt, 0.25, 0)
    assert len(s) == 50
    assert np.all(np.diff(s) >= 0)
    assert set(s.tolist()) <= set(kt.tolist())


def test_tune_events_recovers_a_clean_spike_train():
    t = np.arange(1200) / 60
    p = np.zeros(1200)
    peaks = np.arange(60, 1200, 60)
    for k in peaks:
        p[k - 1:k + 2] = [0.6, 0.95, 0.6]
    kt = t[peaks]
    cfg, f1 = G.tune_events(p, t, kt, np.ones(1200, bool))
    assert f1 > 0.95
    assert set(cfg) == {"thr", "smooth", "refractory", "n_consec", "gate_thr"}


def test_gate_suppresses_events_where_typing_probability_is_low():
    p = np.zeros(600)
    p[100] = p[400] = 0.9
    gate = np.zeros(600)
    gate[50:150] = 1.0
    cfg = {"thr": 0.5, "smooth": 1, "refractory": 5, "n_consec": 1, "gate_thr": 0.5}
    assert G.apply_events(p, cfg, gate).tolist() == [100]
    assert G.apply_events(p, dict(cfg, gate_thr=0.0), gate).tolist() == [100, 400]


def test_reliability_reports_perfect_calibration_as_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200000)
    y = rng.uniform(size=p.size) < p
    out = G.reliability(y, p)
    ece = float(out.rsplit("ECE=", 1)[1])
    assert ece < 0.01


def test_flex_velocity_peaks_on_the_moving_finger():
    P = _synth(400)
    fv = G.flex_velocity(P)
    assert fv.shape == (400, 2, 5)
    k = int(np.nanargmax(np.abs(np.nan_to_num(fv[:, 0, 1]))))
    assert 280 < k < 320


def test_score_prep_matches_eval_taps_score_on_one_window():
    from phase0.analysis.eval_taps import score as ref
    rng = np.random.default_rng(3)
    t = np.arange(6000) / 60
    kt = np.sort(rng.uniform(1, 90, 300))
    ev = np.sort(rng.choice(6000, 400, replace=False))
    mask = np.ones(6000, bool)
    got = G.score_masked(kt, t, mask, ev)
    exp = ref(kt, t[ev], t[0], t[-1] + 1e-6)
    for k in ("keydowns", "taps", "hits", "still_fp"):
        assert got[k] == exp[k], k
    assert got["f1"] == pytest.approx(exp["f1"])


# --------------------------------------------------------------- real-data regression
SESSION_DIRS = [
    Path("data/sessions/20260910-131629-kbd"),
    Path("data/sessions/20260910-021315-kbd"),
    Path("data/sessions/20260910-015948-kbd"),
    Path("data/sessions/20260910-015217-kbd"),
]
MAX_HAND_SPAN_PX = 200.0  # a real hand spans ~110 px at this framing; a two-hand chimera ~520


@pytest.mark.parametrize("session", SESSION_DIRS, ids=lambda p: p.name)
def test_load_frames_slots_hold_one_hand_each(session):
    """Regression: per-row dedup used to mix both hands into one slot (span 518 px, not 110)."""
    if not (session / "landmarks.parquet").exists():
        pytest.skip(f"{session} not present")
    _, _, P = G.load_frames(session)
    for slot in range(2):
        span = np.linalg.norm(P[:, slot, 9, :2] - P[:, slot, 0, :2], axis=1)
        med = float(np.nanmedian(span))
        assert np.isfinite(med), f"slot {slot} never populated"
        assert med < MAX_HAND_SPAN_PX, f"slot {slot} median wrist->middle-MCP {med:.1f} px"


@pytest.mark.parametrize("session", SESSION_DIRS, ids=lambda p: p.name)
def test_load_frames_matches_tap_pos_slotting(session):
    """taps_gb and tap_pos must not diverge on which hand lands in which slot."""
    if not (session / "landmarks.parquet").exists():
        pytest.skip(f"{session} not present")
    from phase0.analysis import tap_pos

    _, _, A = G.load_frames(session)
    _, _, B = tap_pos.load_frames(session)
    assert A.shape == B.shape
    assert np.allclose(A, B, equal_nan=True)
