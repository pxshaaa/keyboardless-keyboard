import numpy as np
import pytest

from phase0.analysis import keypre as KP
from phase0.analysis.decode import A_INDEX, NA


def test_pub_class_maps_by_physical_position():
    assert KP.pub_class("y") == A_INDEX["z"]
    assert KP.pub_class("Z") == A_INDEX["y"]
    assert KP.pub_class("space") == A_INDEX[" "]
    assert KP.pub_class("A") == A_INDEX["a"]
    for k in ("period", "Shift_L", "1", "quotedbl"):
        assert KP.pub_class(k) is None


def test_subset_is_one_contiguous_window_of_the_right_size():
    for seed in range(5):
        r = KP.subset(1000, 0.25, seed, "s")
        assert len(r) == 250 and (np.diff(r) == 1).all() and r[0] >= 0 and r[-1] < 1000
    assert (KP.subset(10, 1.0, 0, "s") == np.arange(10)).all()


def test_subset_differs_across_seeds_and_is_deterministic():
    a = [KP.subset(1000, 0.5, s, "s")[0] for s in range(6)]
    assert len(set(a)) > 1
    assert (KP.subset(1000, 0.5, 3, "s") == KP.subset(1000, 0.5, 3, "s")).all()


def test_pub_rep_is_contact_and_contact_minus_tap_in_row_order():
    rng = np.random.default_rng(0)
    tips = rng.integers(0, 256, (6, 2, 2, 5, 32, 32), dtype=np.uint8)
    rows = np.array([4, 1, 3])
    x = KP.pub_rep(tips, rows)
    assert x.shape == (3, 10, 2, 32, 32)
    base = tips[rows, 1].reshape(3, 10, 32, 32) / 255.0
    ref = tips[rows, 0].reshape(3, 10, 32, 32) / 255.0
    assert np.allclose(x[:, :, 0], base, atol=1e-6)
    assert np.allclose(x[:, :, 1], base - ref, atol=1e-6)


def test_geosess_rows_and_palm_width_span():
    tn = 1.0 + np.arange(120) / 60.0
    P = np.zeros((120, 2, 21, 7))
    P[:, :, 5, 1] = 50.0
    s = KP.GeoSess(P, 0.75, tn)
    assert np.allclose(s.span, 37.5) and np.allclose(s.anchor, 0.0)
    assert (s.rows_at(np.array([0.5, 1.0, 1.5, 9.0])) == [0, 0, 30, 119]).all()


def test_equiv_taps_interpolates_and_flags_extrapolation():
    n, acc = [100, 200, 400], [0.30, 0.40, 0.50]
    ne, tag = KP.equiv_taps(n, acc, 0.45)
    assert tag == "" and ne == pytest.approx(np.sqrt(200 * 400))
    ne, tag = KP.equiv_taps(n, acc, 0.60)
    assert tag == ">" and ne == pytest.approx(800)
    _, tag = KP.equiv_taps(n, acc, 0.20)
    assert tag == "<"


def test_nested_alpha_never_reads_the_held_fold(monkeypatch):
    rng = np.random.default_rng(1)
    ys = {s: rng.integers(0, NA, 50) for s in KP.KBD}
    pix = {s: rng.random((50, NA)) for s in KP.KBD}
    pose = {s: rng.random((50, NA)) for s in KP.KBD}
    held = KP.KBD[0]
    a = KP.nested_alpha(pix, pose, ys, held)
    pix2 = dict(pix, **{held: np.eye(NA)[ys[held]]})
    assert KP.nested_alpha(pix2, pose, ys, held) == a


def test_gmean_endpoints():
    rng = np.random.default_rng(2)
    a, b = rng.random((4, NA)), rng.random((4, NA))
    a /= a.sum(1, keepdims=True)
    b /= b.sum(1, keepdims=True)
    assert np.allclose(KP.gmean(a, b, 1.0), a)
    assert np.allclose(KP.gmean(a, b, 0.0), b)
