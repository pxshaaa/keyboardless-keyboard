import numpy as np

from phase0.analysis.taps_flexion import FINGERTIP_JOINTS, Params, Tap, detect, hand_features, nms


def _synthetic_hand(press_times, n=600, fps=60.0, drift_px=200.0):
    """Hand translating steadily (whole-hand travel) with index-finger flexion pulses at press_times."""
    t = np.arange(n) / fps
    P = np.zeros((n, 21, 2))
    base = {0: (0, 0), 1: (20, -30), 2: (50, -50), 3: (70, -60), 4: (85, -65),
            5: (100, -20), 6: (150, -25), 7: (185, -28), 8: (215, -30),
            9: (105, 0), 10: (160, 0), 11: (200, 0), 12: (235, 0),
            13: (100, 20), 14: (150, 22), 15: (185, 24), 16: (215, 25),
            17: (90, 40), 18: (130, 45), 19: (160, 48), 20: (185, 50)}
    for j, (x, y) in base.items():
        P[:, j, 0] = x + drift_px * t / t[-1]
        P[:, j, 1] = y + 0.3 * drift_px * t / t[-1]
    for tp in press_times:
        pulse = np.exp(-0.5 * ((t - tp) / 0.035) ** 2)
        P[:, 8, 0] -= 25 * pulse
        P[:, 7, 0] -= 12 * pulse
    return {"t": t, "i": np.arange(n), "P": P, "conf": np.ones((n, 21))}


def test_flexion_pulses_found_despite_hand_drift():
    presses = [1.0, 2.5, 4.0, 6.0]
    h = _synthetic_hand(presses)
    feats = {0: hand_features(h)}
    for feature in ("dmcp", "resspeed"):
        p = Params(feature=feature, smooth_window=5, min_vel=1.0, min_amp=0.02)
        taps = [z for z in detect({0: h}, feats, p) if z.finger == 8]
        assert len(taps) == len(presses), (feature, [z.t for z in taps])
        assert all(min(abs(z.t - tp) for tp in presses) < 0.04 for z in taps)


def test_pure_hand_drift_is_silent():
    h = _synthetic_hand([])
    feats = {0: hand_features(h)}
    assert detect({0: h}, feats, Params(feature="dmcp", min_vel=1.0, min_amp=0.02)) == []


def test_nms_keeps_strongest_within_hand():
    taps = [Tap(1.00, 0, f, 0, 0, 1, 0, strength=s) for f, s in zip(FINGERTIP_JOINTS, (1, 5, 2, 3, 4))]
    taps.append(Tap(1.02, 1, 8, 0, 0, 1, 0, strength=1))
    kept = nms(taps, 0.040, same_finger=False)
    assert [(z.hand, z.finger) for z in kept] == [(0, 8), (1, 8)]
