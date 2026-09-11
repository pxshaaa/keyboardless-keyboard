import numpy as np
import pytest

from phase0.analysis import contact as C


class FakeSess:
    """Two hands, 21 joints, 7 channels; the tip we care about sits at a known offset."""

    def __init__(self, n=40):
        self.P = np.zeros((n, 2, 21, 7))
        self.P[:, :, :, :2] = 1.0
        self.P[:, 0, 0, :2] = [100.0, 200.0]          # left wrist
        self.P[:, 0, 9, :2] = [100.0, 100.0]          # left middle MCP -> span 100
        self.P[:, 0, 8, :2] = [130.0, 160.0]          # left index tip
        self.P[:, 1, 0, :2] = [500.0, 200.0]
        self.P[:, 1, 9, :2] = [500.0, 100.0]
        self.P[:, 1, 8, :2] = [530.0, 160.0]
        for j in (5, 13, 17):  # joint 9 must keep its own value: it defines the hand span
            self.P[:, :, j, :2] = self.P[:, :, 0, :2]
        self.anchor = np.array([[100.0, 200.0], [500.0, 200.0]])
        self.span = np.array([100.0, 100.0])


def _k(n=4):
    return np.arange(n)


def test_gt_finger_uses_touch_typing_map():
    s = FakeSess()
    hand, tip = C.gt_finger(s, _k(3), ["a", "l", "zz"])
    assert (hand[0], tip[0]) == (0, 20)               # left pinky
    assert (hand[1], tip[1]) == (1, 16)               # right ring
    assert hand[2] == -1 and tip[2] == -1             # unmapped key


def test_gt_finger_space_picks_the_lower_thumb():
    s = FakeSess()
    s.P[:, 0, 4, :2] = [110.0, 300.0]
    s.P[:, 1, 4, :2] = [510.0, 250.0]
    hand, tip = C.gt_finger(s, _k(1), [" "])
    assert (hand[0], tip[0]) == (0, 4)                # larger y = lower in the image


def test_embed_abs_returns_the_raw_fingertip():
    s = FakeSess()
    xy = C.embed(s, _k(2), np.zeros(2, int), np.full(2, 8), "abs")
    assert np.allclose(xy, [[130.0, 160.0]] * 2)


@pytest.mark.parametrize("frame,expect", [
    ("wrist", [30.0, -40.0]),
    ("wristn", [0.3, -0.4]),
    ("local", [0.4, 0.3]),                            # e1 = wrist->MCP9 = (0,-1)
])
def test_embed_relative_frames(frame, expect):
    s = FakeSess()
    xy = C.embed(s, _k(1), np.zeros(1, int), np.full(1, 8), frame)
    assert np.allclose(xy[0], expect)


def test_embed_marks_undefined_fingers_nan():
    s = FakeSess()
    xy = C.embed(s, _k(2), np.array([-1, 0]), np.array([-1, 8]), "abs")
    assert np.isnan(xy[0]).all() and np.isfinite(xy[1]).all()


def test_embed_offset_shifts_the_frame_and_clips_at_the_ends():
    s = FakeSess(n=10)
    s.P[5, 0, 8, :2] = [999.0, 999.0]
    assert np.allclose(C.embed(s, np.array([2]), np.zeros(1, int), np.array([8]), "abs", 3)[0],
                       [999.0, 999.0])
    assert np.isfinite(C.embed(s, np.array([9]), np.zeros(1, int), np.array([8]), "abs", 5)).all()


def test_layout_stats_recovers_a_synthetic_grid():
    rng = np.random.default_rng(0)
    keys, pts = [], []
    for c in "asdfghjkl":
        mu = np.array([C.NOMINAL[c][0] * 50.0, 0.0])
        for _ in range(30):
            keys.append(c)
            pts.append(mu + rng.normal(0, 5.0, 2))
    st = C.layout_stats(np.array(pts), keys)
    assert st["home_pitch"] == pytest.approx(50.0, rel=0.05)
    assert st["home_ratio"] < 0.25
    assert C.pair_spread(st) == pytest.approx(1.0, abs=0.15)


def test_layout_stats_drops_keys_below_min_n():
    xy = np.vstack([np.zeros((20, 2)), np.ones((3, 2))])
    st = C.layout_stats(xy, ["a"] * 20 + ["s"] * 3, min_n=12)
    assert st["keys"] == 1


def test_key_fingers_cover_the_alphabet_and_space():
    from phase0.analysis.decode import ALPHABET, A_INDEX
    assert set(C.KEY_FINGERS) == {A_INDEX[c] for c in ALPHABET}
    assert len(C.KEY_FINGERS[A_INDEX[" "]]) == 2      # either thumb


def test_nominal_layout_is_a_staggered_grid():
    assert C.NOMINAL["a"][1] == C.NOMINAL["l"][1] == 1.0
    assert C.NOMINAL["s"][0] - C.NOMINAL["a"][0] == pytest.approx(1.0)
    assert C.NOMINAL["q"][0] < C.NOMINAL["a"][0] < C.NOMINAL["z"][0]
