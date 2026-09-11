"""What each calibrator does to the argmax, and that it actually calibrates."""

import numpy as np
import pytest

from phase0.analysis import calib as cb


def _synthetic(n=600, k=8, seed=0):
    """Overconfident probabilities with a sound ranking: the defect this module targets."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    z = rng.normal(0, 1.0, (n, k))
    z[np.arange(n), y] += 1.6
    p = np.exp(z * 3.0)  # the overconfidence: logits inflated before the softmax
    p /= p.sum(1, keepdims=True)
    F = {"hand": rng.integers(0, 2, n).astype(float),
         "finger": rng.choice(cb.FINGERS, n).astype(float),
         "energy": rng.normal(1, 0.3, n), "x": rng.random(n), "y": rng.random(n),
         "maxp": p.max(1), "entropy": -(p * np.log(p)).sum(1)}
    return p, y, F


def test_norm_rows_sum_to_one():
    lp = cb._norm(np.log(np.array([[0.2, 0.3, 0.5], [1e-9, 0.5, 0.5]])))
    assert np.allclose(np.exp(lp).sum(1), 1.0)


@pytest.mark.parametrize("name", ["temp", "condtemp"])
def test_temperature_family_preserves_argmax_exactly(name):
    p, y, F = _synthetic()
    lp = cb.methods()[name].fit(p, y, F).apply(p, F)
    assert (lp.argmax(1) == p.argmax(1)).all()


@pytest.mark.parametrize("name", ["vector", "isotonic", "beta"])
def test_per_class_methods_do_move_the_argmax(name):
    """Per-class calibration is NOT rank-preserving: it reweights classes against each other."""
    p, y, F = _synthetic()
    agree = (cb.methods()[name].fit(p, y, F).apply(p, F).argmax(1) == p.argmax(1)).mean()
    assert 0.80 <= agree < 1.0


@pytest.mark.parametrize("name", ["temp", "vector", "isotonic", "beta", "condtemp"])
def test_calibration_improves(name):
    p, y, F = _synthetic()
    tr, te = slice(0, 400), slice(400, None)
    Ftr = {k: v[tr] for k, v in F.items()}
    Fte = {k: v[te] for k, v in F.items()}
    m = cb.methods()[name].fit(p[tr], y[tr], Ftr)
    before = cb.clf_metrics(np.log(p[te]), y[te])
    after = cb.clf_metrics(m.apply(p[te], Fte), y[te])
    assert after["ece"] < before["ece"]
    assert after["nll"] < before["nll"]


def test_temperature_softens_an_overconfident_model():
    p, y, F = _synthetic()
    assert cb.methods()["temp"].fit(p, y, F).T > 1.0


def test_ece_perfect_and_worst():
    c = np.full(100, 0.95)
    assert cb.ece(c, np.ones(100)) == pytest.approx(0.05, abs=1e-9)
    assert cb.ece(c, np.zeros(100)) == pytest.approx(0.95, abs=1e-9)


def test_feature_matrix_shape_and_constant():
    _, _, F = _synthetic(n=50)
    D = cb.feature_matrix(F)
    assert D.shape == (50, 2 + len(cb.FINGERS) - 1 + 3)
    assert np.allclose(D[:, 0], 1.0)


def test_motion_energy_is_finite_with_nan_landmarks():
    class S:
        P = np.full((20, 2, 21, 7), np.nan)
    S.P[:, :, :, :2] = np.arange(20)[:, None, None, None]
    assert np.all(np.isfinite(cb.motion_energy(S, np.array([0, 10, 19]))))
