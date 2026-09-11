import json

import numpy as np
import pytest

from phase0.analysis import pretrain as PT
from phase0.analysis import taps_gb as G


def _synthetic(n=200, fps=29.5):
    t = np.arange(n) / fps
    P = np.full((n, 2, 21, len(G.CHANNELS)), np.nan)
    for s in range(2):
        for j in range(21):
            P[:, s, j, 0] = 100 * s + j + 5 * np.sin(2 * np.pi * t)
            P[:, s, j, 1] = 50 + j + 5 * np.cos(2 * np.pi * t)
            P[:, s, j, 2] = 0.9
            P[:, s, j, 3] = 0.0
            P[:, s, j, 4:7] = 0.01 * j
    return t, P


def test_resample_hits_60_fps_and_preserves_values():
    t, P = _synthetic()
    tn, Pn = PT.resample_60(t, P)
    assert np.allclose(np.diff(tn), 1 / 60.0)
    assert tn[0] == pytest.approx(t[0])
    assert tn[-1] <= t[-1]
    # cubic interpolation must reproduce the analytic signal at the new grid times
    truth = 8 + 5 * np.sin(2 * np.pi * tn[60:120])
    assert np.abs(Pn[60:120, 0, 8, 0] - truth).max() < 0.01


def test_resample_marks_tracking_dropouts_nan():
    t, P = _synthetic()
    P[80:110, 1] = np.nan
    tn, Pn = PT.resample_60(t, P)
    inside = (tn > t[85]) & (tn < t[105])
    assert np.isnan(Pn[inside, 1, 0, 0]).all()
    assert np.isfinite(Pn[tn < t[70], 1, 0, 0]).all()


def test_resample_leaves_other_hand_alone():
    t, P = _synthetic()
    P[:, 1] = np.nan
    tn, Pn = PT.resample_60(t, P)
    assert np.isnan(Pn[:, 1]).all()
    assert np.isfinite(Pn[:, 0, 0, 0]).all()


def test_resample_too_short_is_empty():
    t, P = _synthetic(n=5)
    tn, Pn = PT.resample_60(t, P)
    assert len(tn) == 0 and len(Pn) == 0


def test_time_subset_keeps_one_contiguous_window_per_session():
    t = np.concatenate([np.arange(0, 120, 1 / 60.0), np.arange(0, 60, 1 / 60.0)])
    sid = np.array([0] * 7200 + [1] * 3600)
    m = PT.time_subset(t, sid, 0.5, seed=0)
    assert m.mean() == pytest.approx(0.5, abs=0.01)
    # one run per session: a scattered mask would embargo whole folds away
    assert len(G._runs(m)) <= 2
    assert m[sid == 0].sum() == 3600 and m[sid == 1].sum() == 1800


def test_time_subset_full_fraction_keeps_everything():
    t = np.arange(0, 60, 1 / 60.0)
    assert PT.time_subset(t, np.zeros(len(t), int), 1.0, 0).all()


def test_rotation_is_the_verified_one():
    from phase0.analysis.extract_landmarks import ROTATION_NAMES

    assert PT.ROTATION in ROTATION_NAMES


def test_public_keystroke_times_start_at_zero(tmp_path):
    """Clips are trimmed so frame 0 is the first keydown; timestamps are ms from there."""
    cs = PT.clips("test")
    first = [c["keystrokes"][0]["timestamp_ms"] for c in cs if c["keystrokes"]]
    assert first and max(first) == 0


def test_public_metadata_fps_differs_from_container():
    fps = np.array([c["actual_fps"] for c in PT.all_clips()])
    assert fps.min() < 30.0 < fps.max()
    # using the container's 30 fps would misalign labels by hundreds of ms
    worst = max(abs(c["duration_sec"] * (30.0 - c["actual_fps"]) / c["actual_fps"])
                for c in PT.all_clips())
    assert worst > 0.3


def test_row_formats_missing_timing_error():
    s = {"recall": 0.5, "precision": 0.5, "f1": 0.5, "still_fp": 0, "taps": 0,
         "median_abs_err_ms": None}
    assert "-ms" in PT.row("x", s)
