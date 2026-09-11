import json

import numpy as np
import pytest

from phase0.analysis import tap_pos as TP


def _write_session(root, n_frames=40, swap_hand_col=True):
    """Synthetic session whose `hand` column flips every frame while `handedness` is stable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {k: [] for k in ("i", "t", "hand", "handedness", "joint",
                            "x", "y", "conf", "z", "wx", "wy", "wz")}
    for i in range(n_frames):
        for s, hd in enumerate(("Left", "Right")):
            for j in range(21):
                rows["i"].append(i)
                rows["t"].append(1000.0 + i / 60.0)
                rows["hand"].append((s + i) % 2 if swap_hand_col else s)
                rows["handedness"].append(hd)
                rows["joint"].append(j)
                rows["x"].append(100.0 + 4.0 * j + 30.0 * s)
                rows["y"].append(50.0 + 3.0 * j + 500.0 * s)
                rows["conf"].append(0.9)
                rows["z"].append(0.01 * j)
                rows["wx"].append(0.001 * j)
                rows["wy"].append(0.002 * j)
                rows["wz"].append(0.003 * j)
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), root / "landmarks.parquet")
    return root


def test_load_frames_slots_whole_hand_by_handedness(tmp_path):
    d = _write_session(tmp_path / "s")
    frames, t, P = TP.load_frames(d)
    assert P.shape == (40, 2, 21, 7)
    # every joint of a hand must land in that hand's slot, so the |j9-j0| span stays hand-sized
    span = np.linalg.norm(P[:, :, 9, :2] - P[:, :, 0, :2], axis=2)
    assert np.allclose(span, np.hypot(36.0, 27.0))
    assert np.all(P[:, 1, 0, 1] - P[:, 0, 0, 1] == 500.0)


def test_taps_gb_loader_matches_tap_pos(tmp_path):
    """Regression: taps_gb.load_frames once slotted joints by the flipping `hand` index,
    mixing both hands into one slot. It must now agree with tap_pos exactly."""
    from phase0.analysis.taps_gb import load_frames as gb_load

    d = _write_session(tmp_path / "s")
    _, _, P = gb_load(d)
    span = np.linalg.norm(P[:, :, 9, :2] - P[:, :, 0, :2], axis=2)
    assert np.allclose(span, np.hypot(36.0, 27.0))
    _, _, P_ref = TP.load_frames(d)
    assert np.allclose(np.nan_to_num(P), np.nan_to_num(P_ref))


def test_features_rel_is_translation_invariant(tmp_path):
    a = TP.Sess(_write_session(tmp_path / "a"))
    b = TP.Sess(_write_session(tmp_path / "b"))
    b.P[:, :, :, :2] += 137.0
    b.anchor = b.anchor + 137.0
    k = np.array([5, 10, 20])
    fa = TP.features(a, k, "rel", TP.JOINTS_TIPS)
    fb = TP.features(b, k, "rel", TP.JOINTS_TIPS)
    assert np.allclose(fa, fb, atol=1e-4)
    assert not np.allclose(TP.features(a, k, "abs", TP.JOINTS_TIPS),
                           TP.features(b, k, "abs", TP.JOINTS_TIPS))


def test_features_finite_and_shaped(tmp_path):
    s = TP.Sess(_write_session(tmp_path / "s"))
    k = np.array([0, 39])  # clipped at both ends of the session
    X = TP.features(s, k, "abs")
    assert X.shape[0] == 2 and np.isfinite(X).all()


def test_topk_and_space_recall():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    p = np.hstack([p, np.zeros((2, TP.NA - 3))])
    assert TP.topk(p, np.array([0, 1]), 1) == 1.0
    assert TP.topk(p, np.array([2, 2]), 1) == 0.0
    assert TP.topk(p, np.array([2, 2]), 3) == 1.0
    sp = np.zeros((2, TP.NA))
    sp[0, TP.A_INDEX[" "]] = 1.0
    sp[1, TP.A_INDEX["a"]] = 1.0
    assert TP.space_recall(sp, np.array([TP.A_INDEX[" "]] * 2)) == 0.5


def test_decode_cls():
    assert TP._decode_cls("thumb") == (None, 0)
    assert TP._decode_cls("L-index") == (0, 1)
    assert TP._decode_cls("R-pinky") == (1, 4)


def test_gauss_logp_normalised():
    from phase0.analysis.decode import GaussianSpatial

    rng = np.random.default_rng(0)
    xy = rng.normal(size=(300, 2)) * 20 + [400, 300]
    labels = [("a" if i % 2 else "b") for i in range(300)]
    g = GaussianSpatial.fit(xy, labels)
    lp = TP._gauss_logp(g, xy[:5])
    assert np.allclose(np.exp(lp).sum(1), 1.0)


class _Stub:
    classes = np.array([TP.A_INDEX["a"], TP.A_INDEX[" "]])

    def proba(self, sess, k):
        p = np.full((len(k), TP.NA), 1e-6)
        p[:, TP.A_INDEX["a"]] = 0.6
        p[:, TP.A_INDEX[" "]] = 0.4
        return p / p.sum(1, keepdims=True)


def test_apply_writes_key_probs(tmp_path, monkeypatch):
    d = _write_session(tmp_path / "sess")
    (d / "taps.jsonl").write_text("".join(
        json.dumps({"t": 1000.0 + i / 60.0, "hand": 0, "finger": 8, "x": 1.0, "y": 2.0, "i": i})
        + "\n" for i in (3, 9, 15)))

    import pickle
    mp = tmp_path / "m.pkl"
    mp.write_bytes(pickle.dumps({"approach": "stub", "model": _Stub()}))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(TP, "CACHE", tmp_path / "cache")
    rc = TP.main(["apply", str(d), "--model", str(mp), "--out", str(out), "--topk", "2"])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3
    for r in rows:
        assert set(r["key_probs"]) == {"a", " "}
        assert r["key_probs"]["a"] == pytest.approx(0.6, abs=1e-3)
        assert r["t"] and r["i"] in (3, 9, 15)
