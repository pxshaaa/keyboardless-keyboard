import numpy as np

from phase0.analysis.ceiling import _inner_folds, max_matched, sat


def test_max_matched_unconstrained_when_keys_are_slow():
    kt = np.arange(10) * 0.5
    assert max_matched(kt, 0.080) == 10


def test_refractory_binds_only_beyond_the_pairing_slack():
    # two keys 20 ms apart: an 80 ms refractory still fits both inside +/-80 ms, 200 ms does not
    kt = np.array([1.0, 1.02])
    assert max_matched(kt, 0.080) == 2
    assert max_matched(kt, 0.200) == 1


def test_max_matched_respects_a_frame_grid():
    kt = np.array([1.0, 1.02])
    coarse = np.arange(0, 2, 0.5)  # only t=1.0 is inside either key's window
    assert max_matched(kt, 0.080, grid=coarse) == 1


def test_inner_folds_partition_each_subwindow():
    t = np.concatenate([np.linspace(0, 9, 90), np.linspace(20, 29, 90)])
    sid = np.repeat([0, 1], 90)
    folds = _inner_folds(t, sid)
    assert sum(f.sum() for f in folds) == len(t)
    for f in folds:
        assert f[sid == 0].any() and f[sid == 1].any()


def test_sat_is_increasing_and_asymptotes():
    assert sat(100, 0.9, 2.0, 0.5) < sat(1000, 0.9, 2.0, 0.5) < 0.9
