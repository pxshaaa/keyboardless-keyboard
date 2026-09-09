"""Synthetic-trajectory proof for the tap detector.
Ground truth is constructed, so a miss or a false fire is unambiguous."""

# y increases DOWNWARD (image coords): a tap is a bump toward +y.
# Every generator below obeys that; the lift test deliberately inverts it.

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase0.analysis.detect_taps import (  # noqa: E402
    Params,
    Rejections,
    apply_refractory,
    detect_taps_series,
    split_segments,
)

FPS = 60.0
BASE_Y = 500.0
BASE_X = 300.0
TAP_AMPLITUDE = 25.0  # px of vertical excursion, typical keystroke at 1080p
TAP_DURATION = 0.120  # s, full down-and-up
NOISE_PX = 0.8  # per-axis landmark jitter, ~1 px is realistic for MediaPipe


def timebase(duration: float, fps: float = FPS, jitter: float = 0.0, seed: int = 0):
    """Frame timestamps. `jitter` (s, uniform) models real capture timing."""
    n = int(duration * fps)
    t = np.arange(n) / fps
    if jitter:
        t = t + np.random.default_rng(seed).uniform(-jitter, jitter, n)
        t = np.maximum.accumulate(t)  # timestamps must stay monotonic
    return t


def tap_bump(t, t_center, amplitude=TAP_AMPLITUDE, duration=TAP_DURATION):
    """Raised cosine centred on t_center: one clean descent-contact-rebound.
    Its single dy/dt +->- crossing is exactly at t_center, so ground truth is exact."""
    half = duration / 2.0
    within = np.abs(t - t_center) <= half
    phase = np.pi * (t - t_center) / half
    return np.where(within, amplitude * 0.5 * (1.0 + np.cos(phase)), 0.0)


def make_series(tap_times, duration=None, amplitude=TAP_AMPLITUDE,
                tap_duration=TAP_DURATION, noise=NOISE_PX, seed=1, jitter=0.0):
    """Baseline plus one bump per tap time, plus gaussian landmark noise."""
    if duration is None:
        duration = (max(tap_times) if tap_times else 1.0) + 1.0
    t = timebase(duration, jitter=jitter, seed=seed)
    y = np.full_like(t, BASE_Y)
    for tc in tap_times:
        y = y + tap_bump(t, tc, amplitude, tap_duration)
    x = np.full_like(t, BASE_X)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0.0, noise, len(t))
    x = x + rng.normal(0.0, noise, len(t))
    return t, x, y


def match(detected, expected, tol=0.020):
    """Greedy nearest matching. Returns (pairs, missed, spurious)."""
    remaining = list(detected)
    pairs, missed = [], []
    for exp in expected:
        best = min(remaining, key=lambda z: abs(z.t - exp), default=None)
        if best is not None and abs(best.t - exp) <= tol:
            remaining.remove(best)
            pairs.append((exp, best))
        else:
            missed.append(exp)
    return pairs, missed, remaining


def test_single_tap_is_recovered_at_the_right_time_and_place():
    expected = [0.500]
    t, x, y = make_series(expected)
    taps = detect_taps_series(t, x, y)

    assert len(taps) == 1, f"expected 1 tap, got {len(taps)}"
    tap = taps[0]
    assert abs(tap.t - 0.500) < 0.020, f"tap time off by {tap.t - 0.500:.4f} s"

    # Contact y is the DEEPEST point, i.e. baseline + full amplitude. The tight
    # tolerance is the point: smoothing must not flatten the peak we measure.
    assert tap.y == pytest.approx(BASE_Y + TAP_AMPLITUDE, abs=2.0)
    assert tap.x == pytest.approx(BASE_X, abs=2.0)
    assert tap.finger == 8 and tap.hand == 0


def test_many_taps_recovered_within_20ms():
    expected = [0.30 + 0.25 * k for k in range(12)]
    t, x, y = make_series(expected)
    taps = detect_taps_series(t, x, y)

    pairs, missed, spurious = match(taps, expected)
    assert not missed, f"missed taps at {missed}"
    assert not spurious, f"spurious taps at {[round(z.t, 3) for z in spurious]}"
    assert len(pairs) == len(expected)

    errors = np.array([abs(d.t - e) for e, d in pairs])
    assert errors.max() < 0.020, f"worst timing error {errors.max() * 1000:.1f} ms"
    # Sub-frame interpolation must beat the 16.7 ms frame period on average.
    assert errors.mean() < 0.008, f"mean timing error {errors.mean() * 1000:.1f} ms"


def test_contact_position_is_the_lowest_point_not_the_baseline():
    """The x,y written to taps.jsonl is what the whole experiment measures."""
    expected = [0.4, 0.9, 1.4]
    t, x, y = make_series(expected, noise=0.5)
    # Give x a real gradient so a mis-timed contact reports a wrong x too.
    x = x + 40.0 * t
    taps = detect_taps_series(t, x, y)

    assert len(taps) == len(expected)
    for tc, tap in zip(expected, taps):
        assert tap.y == pytest.approx(BASE_Y + TAP_AMPLITUDE, abs=2.0)
        assert tap.x == pytest.approx(BASE_X + 40.0 * tc, abs=2.0)


def test_timing_jitter_does_not_break_detection():
    """Capture timestamps are not perfectly uniform; SavGol assumes they are."""
    expected = [0.3 + 0.2 * k for k in range(10)]
    t, x, y = make_series(expected, jitter=0.004, seed=7)
    taps = detect_taps_series(t, x, y)

    pairs, missed, spurious = match(taps, expected, tol=0.025)
    assert not missed and not spurious
    assert len(pairs) == len(expected)


def test_light_taps_still_detected_heavier_noise():
    expected = [0.4, 0.8, 1.2, 1.6]
    t, x, y = make_series(expected, amplitude=12.0, noise=1.2, seed=11)
    taps = detect_taps_series(t, x, y)

    _, missed, spurious = match(taps, expected, tol=0.025)
    assert not missed, f"missed light taps at {missed}"
    assert not spurious, f"spurious: {[round(z.t, 3) for z in spurious]}"


# ---- negative cases: the detector must stay silent -------------------------


def test_slow_drift_produces_no_taps():
    """Hand repositioning: monotonic descent, never reverses."""
    t = timebase(3.0)
    y = BASE_Y + 30.0 * t + np.random.default_rng(3).normal(0, NOISE_PX, len(t))
    x = np.full_like(t, BASE_X)
    assert detect_taps_series(t, x, y) == []


def test_hover_produces_no_taps():
    """Held above the desk: many dy/dt zero crossings, no momentum behind any."""
    t = timebase(4.0)
    y = BASE_Y + 1.5 * np.sin(2 * np.pi * 1.2 * t)
    y = y + np.random.default_rng(5).normal(0, NOISE_PX, len(t))
    x = np.full_like(t, BASE_X)

    rej = Rejections()
    taps = detect_taps_series(t, x, y, rej=rej)
    assert taps == [], f"hover produced {len(taps)} phantom taps"
    assert rej.crossings > 5, "test is vacuous unless hover actually crosses zero"


def test_a_lift_is_not_a_tap():
    """SIGN GUARD: an upward excursion (-y) must not fire. Inverting the sign
    convention would make this pass as a tap and corrupt everything downstream."""
    t, x, y = make_series([])
    y = y - tap_bump(t, 0.5)  # minus: finger rises off the desk, then returns
    assert detect_taps_series(t, x, y) == []


def test_hold_without_rebound_is_not_a_tap():
    """Finger descends onto a key and stays: contact, but no release."""
    t = timebase(2.0)
    y = np.where(t < 0.5, BASE_Y, BASE_Y + TAP_AMPLITUDE)
    ramp = (t >= 0.5) & (t < 0.56)
    y[ramp] = BASE_Y + TAP_AMPLITUDE * (t[ramp] - 0.5) / 0.06
    y = y + np.random.default_rng(9).normal(0, NOISE_PX, len(t))
    x = np.full_like(t, BASE_X)
    assert detect_taps_series(t, x, y) == []


def test_stationary_noise_only_produces_no_taps():
    t = timebase(5.0)
    y = BASE_Y + np.random.default_rng(13).normal(0, NOISE_PX, len(t))
    x = BASE_X + np.random.default_rng(14).normal(0, NOISE_PX, len(t))
    assert detect_taps_series(t, x, y) == []


# ---- gating behaviour ------------------------------------------------------


def test_refractory_collapses_a_double_crossing():
    """Two crossings 40 ms apart are one physical strike; keep the stronger."""
    rej = Rejections()
    p = Params()
    t, x, y = make_series([0.50])
    strong = detect_taps_series(t, x, y)[0]
    weak = type(strong)(t=0.54, hand=0, finger=8, x=1.0, y=1.0, conf=1.0, i=0,
                        descent_vel=90.0, rebound_vel=50.0, rebound_px=3.0)
    assert weak.score < strong.score

    kept = apply_refractory([weak, strong], p.refractory, rej)
    assert len(kept) == 1 and kept[0] is strong
    assert rej.refractory == 1


def test_fast_alternating_fingers_are_not_suppressed():
    """Refractory is per finger: rollover taps 40 ms apart on DIFFERENT fingers
    must both survive, or fast typing loses half its keystrokes."""
    rej = Rejections()
    t, x, y = make_series([0.5])
    a = detect_taps_series(t, x, y, finger=8)[0]
    b = detect_taps_series(t, x, y, finger=12)[0]
    b.t += 0.040
    per_finger = apply_refractory([a], 0.080, rej) + apply_refractory([b], 0.080, rej)
    assert len(per_finger) == 2


def test_low_confidence_samples_are_ignored():
    expected = [0.4, 1.0]
    t, x, y = make_series(expected)
    conf = np.ones_like(t)
    conf[(t > 0.30) & (t < 0.55)] = 0.1  # tracker lost the finger over tap 1

    taps = detect_taps_series(t, x, y, conf=conf)
    assert len(taps) == 1
    assert taps[0].t == pytest.approx(1.0, abs=0.020)


def test_missing_frames_do_not_fabricate_a_tap():
    """A dropout must cut the series, not be smoothed across into a crossing."""
    t = timebase(2.0)
    keep = ~((t > 0.8) & (t < 1.2))
    t2 = t[keep]
    # Descending before the gap, ascending after: interpolating across the hole
    # would look exactly like a tap.
    y2 = np.where(t2 < 1.0, BASE_Y + 60.0 * t2, BASE_Y + 60.0 * (2.0 - t2))
    x2 = np.full_like(t2, BASE_X)

    segs = split_segments(t2, np.ones_like(t2), Params().max_gap, Params().min_conf)
    assert len(segs) == 2, "the dropout must split the series"
    assert detect_taps_series(t2, x2, y2) == []


def test_thresholds_are_actually_tunable():
    """A tap below the descent gate is dropped, and raising the gate drops it."""
    expected = [0.5]
    t, x, y = make_series(expected, amplitude=8.0, noise=0.3)
    assert len(detect_taps_series(t, x, y)) == 1

    strict = Params(min_descent_vel=1000.0)
    rej = Rejections()
    assert detect_taps_series(t, x, y, params=strict, rej=rej) == []
    assert rej.weak_descent >= 1


def test_record_matches_the_frozen_contract():
    t, x, y = make_series([0.5])
    frames = np.arange(len(t)) + 100
    tap = detect_taps_series(t, x, y, frame_i=frames, hand=1, finger=20)[0]
    rec = tap.to_record()

    assert list(rec) == ["t", "hand", "finger", "x", "y", "conf", "i"]
    assert rec["hand"] == 1 and rec["finger"] == 20
    assert isinstance(rec["i"], int) and rec["i"] >= 100
    assert all(isinstance(rec[k], float) for k in ("t", "x", "y", "conf"))


def test_realistic_typing_burst_recovers_every_keystroke():
    """End-to-end: 5 s of two-handed typing at ~5 keys/s across all fingertips.
    This is the number that must roughly equal the keystroke count on real data."""
    rng = np.random.default_rng(42)
    fingers = [(h, j) for h in (0, 1) for j in (4, 8, 12, 16, 20)]
    total_expected = 0
    total_found = 0
    total_missed = 0
    total_spurious = 0

    for hand, finger in fingers:
        times = np.sort(rng.uniform(0.35, 4.65, 6))
        times = [times[0]] + [b for a, b in zip(times, times[1:]) if b - a > 0.16]
        t, x, y = make_series(times, duration=5.0,
                              amplitude=float(rng.uniform(15, 35)),
                              seed=int(rng.integers(1e6)))
        taps = detect_taps_series(t, x, y, hand=hand, finger=finger)
        _, missed, spurious = match(taps, times, tol=0.025)
        total_expected += len(times)
        total_found += len(taps)
        total_missed += len(missed)
        total_spurious += len(spurious)

    assert total_expected >= 40, "burst is too small to be meaningful"
    assert total_missed == 0, f"{total_missed}/{total_expected} taps missed"
    assert total_spurious == 0, f"{total_spurious} spurious taps"
    ratio = total_found / total_expected
    assert 0.95 <= ratio <= 1.05, f"taps/keystroke ratio {ratio:.2f}"


def test_not_seed_luck_across_many_realisations():
    """40 independent noise realisations; the defaults must hold, not one seed."""
    expected = [0.35 + 0.22 * k for k in range(15)]
    missed = spurious = 0
    errors = []
    for seed in range(40):
        t, x, y = make_series(expected, seed=seed, jitter=0.003)
        taps = detect_taps_series(t, x, y)
        pairs, m, s = match(taps, expected, tol=0.025)
        missed += len(m)
        spurious += len(s)
        errors += [abs(d.t - e) for e, d in pairs]

    total = 40 * len(expected)
    assert missed == 0, f"{missed}/{total} missed"
    assert spurious == 0, f"{spurious} spurious"
    assert np.median(errors) < 0.005, f"median timing error {np.median(errors) * 1000:.1f} ms"
