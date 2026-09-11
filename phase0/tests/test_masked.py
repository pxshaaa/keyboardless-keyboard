import numpy as np
import pytest

from phase0.analysis import masked as MK


class FakeSess:
    def __init__(self, n=8):
        self.P = np.full((n, 2, 21, 7), np.nan)
        for h in (0, 1):
            self.P[:, h, :, 0] = np.linspace(300.0, 340.0, 21)[None] + 300.0 * h
            self.P[:, h, :, 1] = np.linspace(200.0, 240.0, 21)[None]
        self.anchor = np.array([[10.0, 10.0], [20.0, 20.0]])
        self.span = np.array([100.0, 100.0])
        self.frames = np.arange(n)

    def rows(self, taps):
        return np.clip(np.searchsorted(self.frames, [t["i"] for t in taps]), 0, len(self.frames) - 1)


def _d(n=4, seed=0):
    rng = np.random.default_rng(seed)
    sh = (n, len(MK.TIP_OFFSETS), 2, 5, MK.TIP_CROP, MK.TIP_CROP)
    return {"tips": rng.integers(0, 256, sh, np.uint8),
            "masks": (rng.random(sh) < 0.6).astype(np.uint8),
            "bank_rows": np.arange(n)}


def test_frame_mask_is_binary_and_covers_the_hand():
    cv2 = pytest.importorskip("cv2")
    s = FakeSess()
    m = MK._frame_mask(cv2, s.P[0, :, :, :2], s.span, (720, 1280))
    assert set(np.unique(m)).issubset({0, 1})
    assert m[220, 320] == 1 and m[600, 100] == 0


def test_frame_mask_skips_undetected_hand():
    cv2 = pytest.importorskip("cv2")
    s = FakeSess()
    P = s.P[0, :, :, :2].copy()
    P[1] = np.nan
    m = MK._frame_mask(cv2, P, s.span, (720, 1280))
    assert m[220, 320] == 1 and m[220, 620] == 0


def test_tip_geometry_freezes_centres_at_the_tap_frame():
    s = FakeSess()
    P, side, base = MK._tip_geometry(s, [{"i": 0}, {"i": 5}])
    assert P.shape == (2, 2, 5, 2) and np.isfinite(P).all()
    assert np.allclose(side, MK.TIP_SPANS * s.span)
    assert list(base) == [0, 5]


def test_hand_and_bg_crops_partition_the_unmasked_crop():
    d = _d()
    a = MK.tip_crops(d, 4, "none", "single")
    h = MK.tip_crops(d, 4, "hand", "single")
    b = MK.tip_crops(d, 4, "bg", "single")
    assert a.shape == (4, 10, 1, MK.TIP_CROP, MK.TIP_CROP)
    assert np.allclose(h + b, a)
    assert (h * b == 0).all()


def test_diff_rep_adds_a_second_channel_masked_per_frame():
    d = _d()
    x = MK.tip_crops(d, 4, "hand", "diff")
    assert x.shape[2] == 2
    j0 = MK.TIP_OFFSETS.index(min(MK.TIP_OFFSETS))
    ref = (d["tips"][:, j0].astype(np.float32) * d["masks"][:, j0]).reshape(
        4, 10, 1, MK.TIP_CROP, MK.TIP_CROP) / 255.0
    assert np.allclose(x[:, :, 1:], x[:, :, :1] - ref)


def test_tip_crops_honours_an_explicit_row_selection():
    d = _d(n=6)
    sel = np.array([4, 1])
    assert np.allclose(MK.tip_crops(d, 4, "none", "single", rows=sel),
                       MK.tip_crops(d, 4, "none", "single")[sel])


def test_boot_ci_brackets_the_point_estimate():
    lo, hi = MK.boot_ci(np.r_[np.ones(60), np.zeros(40)], n_boot=500)
    assert lo < 0.6 < hi and 0.0 <= lo < hi <= 1.0


def test_correct_drops_unlabelled_rows():
    p = np.eye(4)[[0, 1, 2, 3]]
    y = np.array([0, 2, -1, 3])
    assert list(MK._correct(p, y)) == [1.0, 0.0, 1.0]
