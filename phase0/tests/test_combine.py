import numpy as np

from phase0.analysis import combine as CB


def _rows():
    """Two candidates: the first is better on phrase 0, the second on phrases 1-2."""
    return [{"cfg": {"thr": 0.3}, "alpha": 0.0, "obs": 1.0, "deletion": -3.0, "a_silh": 0.0,
             "mode": "hand", "n_taps": 10, "per": [(1, 10), (9, 10), (9, 10)]},
            {"cfg": {"thr": 0.4}, "alpha": 0.5, "obs": 2.0, "deletion": -2.0, "a_silh": 0.0,
             "mode": "hand", "n_taps": 12, "per": [(9, 10), (1, 10), (1, 10)]}]


def test_select_uses_only_the_training_phrases():
    r = _rows()
    assert CB.select(r, {1, 2}) is r[1]
    assert CB.select(r, {0, 1}) is r[0]      # 10/20 vs 10/20 -> first wins on ties
    assert CB.select(r, {0}) is r[0]


def test_pooled_is_character_weighted():
    assert CB.pooled(_rows()[0]) == 19 / 30


def test_fuse_is_a_log_linear_mix_that_normalises():
    rng = np.random.default_rng(0)
    pix = rng.random((5, CB.NA))
    pix /= pix.sum(1, keepdims=True)
    pose = rng.random((5, CB.NA))
    pose /= pose.sum(1, keepdims=True)
    assert np.allclose(CB.fuse(pix, pose, 0.0), pose)
    assert np.allclose(CB.fuse(pix, pose, 1.0), pix, atol=1e-9)
    q = CB.fuse(pix, pose, 0.5)
    assert np.allclose(q.sum(1), 1.0)
    assert np.allclose(q, CB.fuse3(pix, pose, pose, 0.5, 0.0))


def test_fuse3_with_zero_pose_weight_matches_a_two_way_mix():
    rng = np.random.default_rng(1)
    p = [x / x.sum(1, keepdims=True) for x in rng.random((3, 4, CB.NA))]
    assert np.allclose(CB.fuse3(p[0], p[1], p[2], 0.5, 0.5), CB.fuse(p[0], p[1], 0.5))


def test_subsample_is_ordered_by_tap_count_and_keeps_the_extremes():
    cands = [({"i": i}, np.arange(n)) for i, n in enumerate([30, 5, 20, 12, 7])]
    got = CB.subsample(cands, 3)
    sizes = sorted(len(ev) for _, ev in got)
    assert len(got) == 3 and sizes[0] == 5 and sizes[-1] == 30
    assert CB.subsample(cands, 9) is cands


def test_cfg_weights_carry_the_swept_observation_and_deletion_costs():
    w = CB.Cfg(obs=2.5, deletion=-2.0).weights()
    assert (w.obs, w.deletion, w.max_deletions) == (2.5, -2.0, 3)
