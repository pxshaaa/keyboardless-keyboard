"""Unit tests for the sequence tap detector: targets, decoding, model plumbing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phase0.analysis import taps_seq as S


def _t(n: int = 600) -> np.ndarray:
    return np.arange(n) / S.FPS


def test_gauss_target_peaks_at_keydowns_and_decays():
    t = _t()
    kt = np.array([1.0, 5.0])
    y = S.gauss_target(t, kt, 0.030)
    for k in kt:
        i = int(np.argmin(np.abs(t - k)))
        assert y[i] > 0.99
        assert y[i] >= y.max() - 1e-6
    assert y[int(0.5 * S.FPS)] < 1e-3
    assert (y >= 0).all() and (y <= 1).all()


def test_gauss_target_empty_keys():
    assert S.gauss_target(_t(50), np.array([]), 0.03).sum() == 0.0


def test_ctc_collapse_emits_one_event_per_run_at_its_peak():
    p = np.zeros(60)
    p[10:15] = [0.6, 0.9, 0.7, 0.6, 0.55]
    p[30:32] = [0.95, 0.5]
    ev = S.ctc_collapse(p, 0.5)
    assert ev.tolist() == [11, 30]


def test_ctc_collapse_min_run_drops_short_bursts():
    p = np.zeros(40)
    p[5] = 0.9
    p[20:24] = 0.9
    assert S.ctc_collapse(p, 0.5, min_run=1).tolist() == [5, 20]
    assert S.ctc_collapse(p, 0.5, min_run=3).tolist() == [20]


def test_ctc_collapse_below_threshold_emits_nothing():
    assert len(S.ctc_collapse(np.full(30, 0.2), 0.5)) == 0


def test_tune_ctc_recovers_a_clean_signal():
    t = _t(600)
    kt = np.array([1.0, 3.0, 6.0])
    p = S.gauss_target(t, kt, 0.030)
    cfg, f1 = S.tune_ctc(p, t, kt, np.ones(len(t), bool))
    assert f1 == pytest.approx(1.0)
    assert cfg["mode"] == "ctc"
    assert len(S.decode(p, cfg)) == 3


def test_gate_channel_is_absent_when_disabled():
    assert S.n_channels(S.Cfg(head="heatmap", gate=False)) == 1
    assert S.n_channels(S.Cfg(head="heatmap")) == 2
    assert S.n_channels(S.Cfg(head="ctc")) == 3


def test_gate_threshold_suppresses_events_outside_typing():
    t = _t(600)
    kt = np.array([2.0, 8.0])
    p = S.gauss_target(t, kt, 0.030)
    gate = (t < 5.0).astype(float)
    cfg = {"mode": "ctc", "thr": 0.5, "min_run": 1, "gate_thr": 0.5}
    assert len(S.decode(p, cfg, gate)) == 1
    assert len(S.decode(p, dict(cfg, gate_thr=0.0), gate)) == 2


def test_decode_dispatches_on_mode():
    t = _t(600)
    kt = np.array([2.0, 4.0])
    p = S.gauss_target(t, kt, 0.030)
    peak = S.decode(p, {"thr": 0.5, "smooth": 1, "refractory": 5, "n_consec": 1, "gate_thr": 0.0})
    assert len(peak) == 2
    assert len(S.decode(p, {"mode": "ctc", "thr": 0.5, "min_run": 1})) == 2


def test_receptive_field_covers_a_few_hundred_ms():
    rf = S.receptive_field(S.Cfg())
    assert rf == 1 + 4 + 8 + 16 + 32
    assert 0.2 < rf / S.FPS < 2.0


def test_tcn_is_shape_preserving_and_small():
    cfg = S.Cfg(width=16, blocks=3)
    m = S.TCN(40, cfg, 1)
    y = m(torch.zeros(2, 40, 300))
    assert y.shape == (2, 1, 300)
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_tcn_ctc_head_has_two_channels():
    y = S.TCN(40, S.Cfg(width=8, blocks=2), 2)(torch.zeros(1, 40, 128))
    assert y.shape == (1, 2, 128)


def test_focal_heatmap_penalises_a_missed_peak():
    y = torch.zeros(1, 1, 100)
    y[0, 0, 50] = 1.0
    good = torch.full((1, 1, 100), -6.0)
    good[0, 0, 50] = 6.0
    bad = torch.full((1, 1, 100), -6.0)
    assert S.focal_heatmap(good, y).item() < S.focal_heatmap(bad, y).item()


def test_clip_taps_drops_events_outside_the_typing_span():
    kt = np.array([10.0, 12.0])
    tt = np.array([0.0, 9.95, 11.0, 12.05, 30.0])
    assert S.clip_taps(kt, tt).tolist() == [9.95, 11.0, 12.05]


def test_rollover_recall_splits_fast_and_slow_keydowns():
    kt = np.array([1.0, 1.05, 3.0, 5.0])  # only the second key is a fast follower
    tt = np.array([1.0, 3.0, 5.0])        # and it is the one we miss
    r = S.rollover_recall(kt, tt)
    assert r["n"] == 1
    assert r["recall"] == 0.0
    assert r["recall_slow"] == 1.0


def test_rollover_recall_handles_empty_inputs():
    assert S.rollover_recall(np.array([]), np.array([1.0]))["n"] == 0


def test_norm_standardises_and_clips():
    rng = np.random.default_rng(0)

    class Fake:
        pass

    a, b = Fake(), Fake()
    a.X = rng.normal(3.0, 2.0, (500, 4)).astype(np.float32)
    b.X = rng.normal(3.0, 2.0, (500, 4)).astype(np.float32)
    n = S.Norm([a, b])
    z = n(a.X)
    assert abs(z.mean()) < 0.2 and abs(z.std() - 1.0) < 0.2
    assert np.abs(n(np.full((3, 4), 1e6, np.float32))).max() <= 8.0


def test_norm_survives_a_constant_column():
    class Fake:
        pass

    f = Fake()
    f.X = np.hstack([np.ones((50, 1)), np.arange(50)[:, None]]).astype(np.float32)
    assert np.isfinite(S.Norm([f])(f.X)).all()


@pytest.mark.slow
def test_training_learns_a_synthetic_tap_pattern(tmp_path):
    """One channel spikes at each keydown; the model must score those frames highest."""
    rng = np.random.default_rng(0)
    n = 3000
    t = np.arange(n) / S.FPS
    kt = t[np.arange(60, n - 60, 90)]
    X = rng.normal(0, 1, (n, 6)).astype(np.float32)
    X[:, 0] += 6.0 * S.gauss_target(t, kt, 0.030)

    class Fake:
        pass

    s = Fake()
    s.X, s.t, s.kt = X, t, kt
    b = S.train_model([s], S.Cfg(width=12, blocks=3, crop=256, batch=8, steps=120), seed=0)
    p, g = S.frame_scores(b, s)
    assert p.shape == (n,) and g.shape == (n,)
    assert (p >= 0).all() and (p <= 1).all()
    hit = np.array([p[max(0, i - 2):i + 3].max() for i in np.searchsorted(t, kt)])
    assert hit.mean() > 2 * p.mean()
