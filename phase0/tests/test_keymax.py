import numpy as np

from phase0.analysis import combine as CB
from phase0.analysis import keymax as km


def _bank(n=4):
    rng = np.random.default_rng(0)
    tips = rng.integers(0, 255, (n, len(km.TIP_OFFSETS), 2, 5, km.TIP_CROP, km.TIP_CROP),
                        dtype=np.uint8)
    masks = (tips > 128).astype(np.uint8)
    return {"tips": tips, "masks": masks, "bank_rows": np.arange(n)}


def test_chans_channel_count_matches_the_rep_spec():
    d = _bank()
    for name, spec in km.REPS.items():
        x = km.chans(d, spec, "none")
        assert x.shape == (4, 10, len(spec), km.TIP_CROP, km.TIP_CROP), name


def test_chans_difference_channel_is_the_actual_frame_difference():
    d = _bank()
    x = km.chans(d, (("abs", 4), ("diff", 4, 0)), "none", dtype=float)
    j4 = km.TIP_OFFSETS.index(4)
    j0 = km.TIP_OFFSETS.index(0)
    a = d["tips"][:, j4].reshape(4, 10, 32, 32) / 255.0
    b = d["tips"][:, j0].reshape(4, 10, 32, 32) / 255.0
    assert np.allclose(x[:, :, 0], a)
    assert np.allclose(x[:, :, 1], a - b)


def test_chans_silh_mode_is_the_mask_alone_and_zero_blanks_one_hand():
    d = _bank()
    x = km.chans(d, (("abs", 4),), "silh")
    assert set(np.unique(x)) <= {0.0, 1.0}
    y = km.chans(d, (("abs", 4),), "none", zero=0)
    assert (y[:, :5] == 0).all() and (y[:, 5:] != 0).any()


def test_gmean_matches_combine_fuse_for_two_experts():
    rng = np.random.default_rng(1)
    p = [x / x.sum(1, keepdims=True) for x in rng.random((2, 6, km.NA))]
    got = km.gmean(p, [0.75, 0.25])
    assert np.allclose(got.sum(1), 1.0)
    assert np.allclose(got, CB.fuse(p[0], p[1], 0.75))


def test_gmean_with_a_single_weight_returns_that_expert():
    rng = np.random.default_rng(2)
    p = [x / x.sum(1, keepdims=True) for x in rng.random((2, 5, km.NA))]
    assert np.allclose(km.gmean(p, [1.0, 0.0]), p[0], atol=1e-9)


def test_fuse_grid_is_a_simplex():
    g = km.fuse_grid(["a", "b", "c"], 0.25)
    assert all(abs(sum(w) - 1.0) < 1e-9 for w in g)
    assert (1.0, 0.0, 0.0) in g and (0.25, 0.5, 0.25) in g


class _Sess:
    def __init__(self, f=60):
        rng = np.random.default_rng(3)
        self.P = rng.random((f, 2, 21, 7)) * 100
        self.span = np.array([100.0, 100.0])


def test_seq_feats_is_finite_and_zero_relative_at_the_tap_frame():
    s = _Sess()
    k = np.array([10, 30, 50])
    x = km.seq_feats(s, k, half=8, stride=2)
    assert x.shape == (3, 9, 2 * 21 * 2 * 2 + 2)
    assert np.isfinite(x).all()
    mid = x.shape[1] // 2
    assert np.allclose(x[:, mid, 84:168], 0.0)


def test_seq_feats_clips_at_the_session_edges():
    s = _Sess(f=20)
    x = km.seq_feats(s, np.array([0, 19]), half=40, stride=5)
    assert np.isfinite(x).all()


def test_topk_is_monotone_in_k():
    rng = np.random.default_rng(4)
    p = rng.random((50, km.NA))
    y = rng.integers(0, km.NA, 50)
    assert km.topk(p, y, 1) <= km.topk(p, y, 5) <= km.topk(p, y, km.NA)
    assert km.topk(p, y, km.NA) == 1.0


def test_boot_acc_brackets_the_point_estimate():
    c = np.array([1.0] * 30 + [0.0] * 70)
    lo, hi = km.boot_acc(c, n=2000)
    assert lo < c.mean() < hi


def test_variant_specs_are_well_formed():
    for name, (kind, spec) in km.VARIANTS.items():
        assert kind in ("lgbm", "seq", "pix"), name
        if kind == "pix":
            assert spec["rep"] in km.REPS and spec["mode"] in ("none", "hand", "silh")
        if kind == "lgbm":
            assert 0 in spec["offsets"]
