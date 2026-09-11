import numpy as np
import torch

from phase0.analysis.taps_nn import (TCN, Config, _resolve_sides, build_features, extract, flip_P,
                                     hand_features, make_targets, peaks_subframe, receptive_field,
                                     warp_P)

BASE = {0: (0, 0), 1: (20, -30), 2: (50, -50), 3: (70, -60), 4: (85, -65),
        5: (100, -20), 6: (150, -25), 7: (185, -28), 8: (215, -30),
        9: (105, 0), 10: (160, 0), 11: (200, 0), 12: (235, 0),
        13: (100, 20), 14: (150, 22), 15: (185, 24), 16: (215, 25),
        17: (90, 40), 18: (130, 45), 19: (160, 48), 20: (185, 50)}


def _hand(n=120, drift=0.0, scale=1.0, offset=(0.0, 0.0)):
    H = np.zeros((n, 21, 7))
    t = np.arange(n) / 60.0
    for j, (x, y) in BASE.items():
        H[:, j, 0] = scale * (x + drift * t) + offset[0]
        H[:, j, 1] = scale * (y + drift * t) + offset[1]
        H[:, j, 2] = 1.0
        H[:, j, 3] = scale * 0.1 * x
        H[:, j, 4:7] = np.array([x, y, 0.0]) * 1e-3
    return H


def test_features_invariant_to_translation_and_scale():
    a = hand_features(_hand(), depth=True)
    b = hand_features(_hand(scale=2.0, offset=(500.0, -300.0)), depth=True)
    # log(span) is deliberately scale-dependent; every other column must match
    diff = np.nanmax(np.abs(a - b), axis=0)
    changed = np.flatnonzero(diff > 1e-6)
    assert len(changed) == 1 and np.isclose(np.nanmedian(a[:, changed[0]] - b[:, changed[0]]), -np.log(2.0))


def test_features_have_no_columns_that_are_all_nan():
    X, fv = build_features(np.stack([_hand(), _hand(drift=50.0)], 1))
    assert X.shape[1] == 646 and fv.shape[1:] == (2, 5)
    assert not np.isnan(X[2:-2]).all(0).any()


def test_whole_hand_drift_leaves_pose_features_still():
    still = hand_features(_hand(), depth=True)
    moving = hand_features(_hand(drift=400.0), depth=True)
    rel = slice(0, 40)
    assert np.nanmax(np.abs(still[:, rel] - moving[:, rel])) < 1e-6


def test_resolve_sides_follows_handedness_label_not_slot():
    i = np.array([0, 0, 1, 1])
    hand = np.array([0, 1, 0, 1])
    lab = np.array(["Left", "Right", "Right", "Left"])
    y0 = np.array([10.0, 600.0, 600.0, 10.0])
    assert _resolve_sides(i, hand, lab, y0).tolist() == [0, 1, 1, 0]


def test_resolve_sides_falls_back_to_position_when_both_slots_claim_one_label():
    i = np.array([0, 0, 1, 1])
    hand = np.array([0, 1, 0, 1])
    lab = np.array(["Left", "Right", "Right", "Right"])
    y0 = np.array([10.0, 600.0, 620.0, 5.0])
    assert _resolve_sides(i, hand, lab, y0).tolist() == [0, 1, 1, 0]


def test_targets_mark_the_label_window_and_mask_ambiguous_hands():
    t = np.arange(0, 1.0, 1 / 60)
    kt = np.array([0.5])
    Y, W = make_targets(t, kt, np.array([-1]), 0.033, 3)
    assert 3 <= Y[:, 0].sum() <= 5
    assert W[Y[:, 0] > 0, 1:].max() == 0.0 and W[Y[:, 0] == 0, 1:].min() == 1.0
    Y2, W2 = make_targets(t, kt, np.array([1]), 0.033, 3)
    assert Y2[:, 2].sum() == Y2[:, 0].sum() and Y2[:, 1].sum() == 0


def test_peak_picking_honours_the_refractory_and_refines_sub_frame():
    t = np.arange(200) / 60.0
    p = np.zeros(200)
    for c in (50, 52, 120):
        p += np.exp(-0.5 * ((np.arange(200) - c) / 1.2) ** 2)
    idx, times = peaks_subframe(p, t, 0.4, refr=5)
    assert len(idx) == 2
    p2 = np.exp(-0.5 * ((np.arange(200) - 100.4) / 1.5) ** 2)
    _, ref = peaks_subframe(p2, t, 0.4, refr=3)
    assert abs(ref[0] - 100.4 / 60.0) < abs(t[100] - 100.4 / 60.0)


def test_extract_hands_mode_can_resolve_a_rollover_the_any_channel_merges():
    t = np.arange(200) / 60.0
    g = lambda c: np.exp(-0.5 * ((np.arange(200) - c) / 1.2) ** 2)
    prob = np.stack([g(50) + g(52), g(50), g(52)], 1)
    assert len(extract(prob, t, 0.4, 1, 5, "any")[1]) == 1
    assert len(extract(prob, t, 0.4, 1, 5, "hands")[1]) == 2


def test_flip_swaps_hands_and_mirrors_the_separating_axis():
    P = np.stack([_hand(), _hand(offset=(0.0, 600.0))], 1)
    Q = flip_P(P)
    assert np.allclose(Q[:, 0, :, 0], P[:, 1, :, 0])
    assert np.allclose(Q[:, 0, :, 1], -P[:, 1, :, 1])


def test_time_warp_keeps_length_and_compresses_the_trajectory():
    P = np.stack([_hand(drift=100.0), _hand()], 1)
    t = np.arange(len(P)) / 60.0
    Q, tw = warp_P(P, t, 2.0)
    assert Q.shape == P.shape and len(tw) == len(t)
    assert np.allclose(Q[10, 0, 0, 0], P[20, 0, 0, 0], atol=1e-6)


def test_model_is_shift_equivariant_and_covers_the_stated_window():
    cfg = Config()
    assert receptive_field(cfg.dilations) == 31
    m = TCN(8, 6, (1, 2), 3, 0.0).eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        a = m(x)
        b = m(torch.roll(x, 5, dims=2))
    assert torch.allclose(a[0, :, 20:30], torch.roll(b, -5, dims=2)[0, :, 20:30], atol=1e-5)
