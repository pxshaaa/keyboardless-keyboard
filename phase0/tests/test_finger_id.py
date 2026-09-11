import json

import numpy as np
import pytest

from phase0.analysis import finger_id as fid


def test_qwertz_y_and_z_are_swapped_vs_qwerty():
    assert fid.KEY_LABEL["y"] == ("Left", 4)
    assert fid.KEY_LABEL["z"] == ("Right", 1)


def test_every_letter_row_maps_to_exactly_one_class():
    assert fid.KEY_LABEL["a"] == ("Left", 4)
    assert fid.KEY_LABEL["space"][0] == "Either"
    seen = [k for k in "qwertzuiopasdfghjklyxcvbnm" if k in fid.KEY_LABEL]
    assert len(seen) == 26
    assert set(fid.KEY_LABEL.values()) - {("Either", 0)} == set(fid.CLASSES8)


def test_modifiers_and_unknown_are_dropped():
    assert "unknown" in fid.DROP_KEYS and "unknown" not in fid.KEY_LABEL
    assert "shift" not in fid.KEY_LABEL


def _fake_hand(n=200, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 21, 3))
    for j in range(21):
        P[:, j] = [j * 3.0, (j % 5) * 2.0, -j * 1.0]
    P += rng.normal(0, 0.01, P.shape)
    W = P / 1000.0
    return dict(t=np.arange(n) / 60.0, P=P, W=W, conf=np.ones((n, 21)))


def test_frame_axes_are_orthonormal():
    R = fid._frame_axes(_fake_hand()["P"])
    g = np.einsum("nij,nkj->nik", R, R)
    assert np.allclose(g, np.eye(3)[None], atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_signals_cover_all_declared_groups():
    sig = fid.hand_signals(_fake_hand())
    assert set(sig) == set(fid.SIGNAL_KEYS)
    for v in sig.values():
        assert v.shape[1] == 5


def test_summarize_recovers_a_known_dip():
    W = fid.HALF_WINDOW
    q = np.zeros((2 * W + 1, 5))
    q[W, 2] = -1.0  # only the middle finger dips, exactly at the tap frame
    S = fid._summarize(q, np.zeros_like(q), np.zeros_like(q), np.zeros(5))
    assert S[fid._STATS.index("d_at")].argmin() == 2
    assert S[fid._STATS.index("d_min")][2] == pytest.approx(-1.0)
    assert S[fid._STATS.index("d_max")][2] == pytest.approx(0.0)


def test_featurize_is_fixed_length_with_a_missing_hand():
    hands = {"Left": _fake_hand(), "Right": _fake_hand(seed=1)}
    for h in hands.values():
        h["sig"] = fid.hand_signals(h)
        h["rest"] = {k: np.median(v, 0) for k, v in h["sig"].items()}
        h["vel"] = {k: np.gradient(v, axis=0) * 60 for k, v in h["sig"].items()}
        h["acc"] = {k: np.gradient(v, axis=0) * 60 for k, v in h["vel"].items()}
    both = fid.featurize_tap(hands, 1.0)
    one = fid.featurize_tap({"Left": hands["Left"]}, 1.0)
    assert both is not None and one is not None
    assert len(both[0]) == len(one[0]) == len(both[1])
    assert both[1] == one[1]
    assert np.isnan(one[0]).any()


def test_ablation_masks_partition_the_depth_question():
    hands = {"Left": _fake_hand()}
    h = hands["Left"]
    h["sig"] = fid.hand_signals(h)
    h["rest"] = {k: np.median(v, 0) for k, v in h["sig"].items()}
    h["vel"] = {k: np.gradient(v, axis=0) * 60 for k, v in h["sig"].items()}
    h["acc"] = {k: np.gradient(v, axis=0) * 60 for k, v in h["vel"].items()}
    names = fid.featurize_tap(hands, 1.0)[1]
    nd = fid.ablation_mask(names, "no_depth")
    assert not any("imgz" in n or "world" in n or n.endswith("_z")
                   for n, keep in zip(names, nd) if keep)
    assert nd.sum() > 0
    assert fid.ablation_mask(names, "all").all()


def test_topk_and_confusion():
    classes = ["a", "b", "c"]
    proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    assert fid.topk_acc(proba, classes, ["a", "b"], 1) == pytest.approx(0.5)
    assert fid.topk_acc(proba, classes, ["a", "b"], 2) == pytest.approx(1.0)
    M = fid.confusion(classes, ["a", "b"], ["a", "c"])
    assert M[0, 0] == 1 and M[1, 2] == 1


def test_load_keydowns_filters_key_ups(tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    (s / "keys.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"t": 1.0, "event": "down", "key": "a"},
        {"t": 1.1, "event": "up", "key": "a"},
    ]))
    kd = fid.load_keydowns(s)
    assert [k["key"] for k in kd] == ["a"]
