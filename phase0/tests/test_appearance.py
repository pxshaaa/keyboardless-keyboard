import numpy as np
import pytest

from phase0.analysis import appearance as AP


class FakeSess:
    def __init__(self, n=6):
        self.P = np.full((n, 2, 21, 7), np.nan)
        for h in (0, 1):
            self.P[:, h, :, 0] = 300.0 + 200.0 * h
            self.P[:, h, :, 1] = 200.0
        self.anchor = np.array([[10.0, 10.0], [20.0, 20.0]])
        self.span = np.array([100.0, 120.0])


def _bank_dict(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return {"bank": rng.integers(0, 255, (n, len(AP.BANK_OFFSETS), 2, AP.CROP, AP.CROP), np.uint8),
            "bank_rows": np.arange(n)}


def test_centres_uses_landmarks_and_span():
    s = FakeSess()
    c, side = AP._centres(s, np.arange(3))
    assert c.shape == (3, 2, 2) and side.shape == (3, 2)
    assert np.allclose(c[:, 0], [300.0, 200.0]) and np.allclose(c[:, 1], [500.0, 200.0])
    assert np.allclose(side[0], [AP.BOX_SPANS * 100.0, AP.BOX_SPANS * 120.0])


def test_centres_falls_back_to_anchor_when_hand_missing():
    s = FakeSess()
    s.P[:, 1, :, :2] = np.nan
    c, _ = AP._centres(s, np.arange(2))
    assert np.allclose(c[:, 1], s.anchor[1])


@pytest.mark.parametrize("rep,c", [("single", 1), ("stack3", 3), ("diff", 2),
                                   ("stack3diff", 4), ("diffonly", 1)])
def test_crops_shapes(rep, c):
    x = AP.crops(_bank_dict(), 3, rep)
    assert x.shape == (4, c, AP.CROP, 2 * AP.CROP)
    assert x.dtype == np.float32 and np.isfinite(x).all()


def test_crops_lays_hands_side_by_side():
    d = _bank_dict()
    x = AP.crops(d, 0, "single")
    j = AP.BANK_OFFSETS.index(0)
    assert np.allclose(x[0, 0, :, :AP.CROP], d["bank"][0, j, 0] / 255.0)
    assert np.allclose(x[0, 0, :, AP.CROP:], d["bank"][0, j, 1] / 255.0)


def test_diff_channel_is_a_true_difference():
    d = _bank_dict()
    x = AP.crops(d, 3, "diff")
    base = AP.crops(d, 3, "single")[:, 0]
    pre = AP.crops(d, 0, "single")[:, 0]
    assert np.allclose(x[:, 1], base - pre, atol=1e-6)


def test_crops_rejects_unknown_rep():
    with pytest.raises(SystemExit):
        AP.crops(_bank_dict(), 0, "nonsense")


def test_finger_labels_follow_the_touch_typing_map():
    s = FakeSess()
    keys = ["a", "f", "l", "q"]
    y = AP.finger_labels(s, np.zeros(len(keys), int), keys)
    assert list(y) == [0 * 5 + 4, 0 * 5 + 1, 1 * 5 + 3, 0 * 5 + 4]
    assert set(y).issubset(range(AP.N_FINGER))


def test_finger_labels_mark_unmapped_keys_as_missing():
    s = FakeSess()
    y = AP.finger_labels(s, np.zeros(2, int), ["/", "a"])
    assert y[0] == -1 and y[1] >= 0


def test_fuse_endpoints_and_normalisation():
    rng = np.random.default_rng(1)
    p = rng.random((7, 5)) + 1e-3
    q = rng.random((7, 5)) + 1e-3
    p, q = p / p.sum(1, keepdims=True), q / q.sum(1, keepdims=True)
    assert np.allclose(AP.fuse(p, q, 1.0), p)
    assert np.allclose(AP.fuse(p, q, 0.0), q)
    assert np.allclose(AP.fuse(p, q, 0.5).sum(1), 1.0)


def test_score_ignores_missing_labels():
    p = np.eye(4)[[0, 1, 2, 3]] + 1e-6
    y = np.array([0, 1, -1, 2])
    r = AP.score(p, y)
    assert r["n"] == 3 and r["top1"] == pytest.approx(2 / 3)
    assert r["top5"] == pytest.approx(1.0)


def test_load_bank_missing_is_a_clear_error():
    with pytest.raises(SystemExit):
        AP.load_bank("no-such-session-id")
