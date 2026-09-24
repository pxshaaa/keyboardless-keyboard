"""Tests for trackpad-contact -> tap-label conversion (no hardware, synthetic sessions)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase0.analysis import pad_labels as pl  # noqa: E402
from phase0.analysis.eval_taps import keydown_times  # noqa: E402

W, H = 124.8, 76.8
# pad-normalized -> landmark px ground truth used to synthesise sessions
H_TRUE = np.array([[300.0, 20.0, 400.0], [-15.0, -250.0, 560.0], [0.0001, 0.0002, 1.0]])


def trow(t, event, pid, x, y, **kw):
    r = {"t": t, "event": event, "id": pid, "x_norm": x, "y_norm": y,
         "x_mm": x * W, "y_mm": (1 - y) * H, "density": 0.1, "major": 8.0}
    r.update(kw)
    return r


def contact(t, dur, pid, x, y):
    return [trow(t, "down", pid, x, y), trow(t + dur / 2, "move", pid, x, y), trow(t + dur, "up", pid, x, y)]


# --- geometry ---------------------------------------------------------------------

def test_fit_homography_recovers_exact_mapping():
    src = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [0.3, 0.6], [0.8, 0.2]], float)
    dst = pl.project(H_TRUE, src)
    Hf = pl.fit_homography(src, dst)
    q = np.random.default_rng(0).random((20, 2))
    assert np.allclose(pl.project(Hf, q), pl.project(H_TRUE, q), atol=1e-6)


def test_fit_homography_rejects_degenerate_points():
    with pytest.raises(pl.CalibError):
        pl.fit_homography(np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))


def test_manual_calibration_parses_and_validates():
    cal = pl.manual_calibration("100,100; 500,110; 510,400; 90,390")
    assert np.allclose(pl.project(cal["H"], [(0, 1)])[0], (100, 100), atol=1e-6)
    with pytest.raises(pl.CalibError):
        pl.manual_calibration("1,2;3,4")


# --- contacts ---------------------------------------------------------------------

def test_contacts_pair_by_id_and_handle_reuse():
    rows = contact(1.0, 0.1, 5, 0.5, 0.5) + contact(2.0, 0.1, 5, 0.2, 0.2)
    cs = pl.contacts_from_touches(rows)
    assert [(c.id, c.t_down, round(c.dur, 3)) for c in cs] == [(5, 1.0, 0.1), (5, 2.0, 0.1)]


def test_unclosed_contact_is_truncated_at_last_row():
    cs = pl.contacts_from_touches([trow(1.0, "down", 1, 0.5, 0.5), trow(1.2, "move", 1, 0.5, 0.5)])
    assert cs[0].truncated and cs[0].t_up == 1.2


def test_classify_tap_rest_slide():
    tap = pl.contacts_from_touches(contact(1.0, 0.09, 1, 0.5, 0.5))[0]
    rest = pl.contacts_from_touches(contact(1.0, 0.8, 1, 0.5, 0.5))[0]
    slide_rows = [trow(1.0, "down", 1, 0.2, 0.5), trow(1.05, "move", 1, 0.4, 0.5), trow(1.1, "up", 1, 0.4, 0.5)]
    slide = pl.contacts_from_touches(slide_rows)[0]
    assert (pl.classify(tap), pl.classify(rest), pl.classify(slide)) == ("tap", "rest", "slide")


def test_offset_clock_rows_are_remapped():
    rows = contact(1.0, 0.1, 1, 0.5, 0.5)
    for r in rows:
        r["t_dev"], r["t_rx"] = r["t"] - 500.0, r["t"] + 0.004
    cs = pl.contacts_from_touches(rows)
    assert cs[0].t_down == pytest.approx(1.004)


# --- frames and assignment -------------------------------------------------------

def test_nearest_frame_edges():
    ft = np.array([0.0, 1 / 60, 2 / 60])
    assert pl.nearest_frame(ft, -1) == 0 and pl.nearest_frame(ft, 5) == 2
    assert pl.nearest_frame(ft, 0.009) == 1 and pl.nearest_frame(ft, 0.007) == 0


def test_tips_array_places_rows_and_ignores_non_tips():
    frame_i = np.array([10, 11])
    tips = pl.tips_array(frame_i, [11, 11, 10], [1, 1, 0], [8, 7, 20], [5.0, 9.0, 1.0], [6.0, 9.0, 2.0])
    assert tuple(tips[1, 1, 1]) == (5.0, 6.0) and tuple(tips[0, 0, 4]) == (1.0, 2.0)
    assert np.isnan(tips[0, 1]).all()


def _hands_at(pad_xy_by_tip):
    """tips[2,5,2] with the given (side, tip_index) -> pad xy placed via H_TRUE, others far away."""
    tips = np.full((2, 5, 2), np.nan)
    for s in range(2):
        for f in range(5):
            tips[s, f] = (2000 + 50 * s, 2000 + 50 * f)
    for (s, f), xy in pad_xy_by_tip.items():
        tips[s, f] = pl.project(H_TRUE, [xy])[0]
    return tips


def test_assign_picks_nearest_fingertip_and_reports_margin():
    tips = _hands_at({(0, 2): (0.3, 0.5), (1, 1): (0.7, 0.5)})[None]
    c = pl.contacts_from_touches(contact(0.0, 0.1, 1, 0.7, 0.5))[0]
    r = pl.assign(c, H_TRUE, tips, np.array([0.0]), np.array([42]))
    assert (r["hand"], r["finger"], r["i"], r["handedness"]) == (1, 8, 42, "Right")
    assert r["d_px"] == pytest.approx(0, abs=1e-6) and r["conf"] == 1.0 and r["d2_px"] > 50
    assert set(("t", "hand", "finger", "x", "y", "conf", "i")) <= set(r)


def test_assign_without_hands_returns_none():
    c = pl.contacts_from_touches(contact(0.0, 0.1, 1, 0.5, 0.5))[0]
    assert pl.assign(c, H_TRUE, np.full((1, 2, 5, 2), np.nan), np.array([0.0]), np.array([0])) is None


def test_pad_coverage():
    tips = _hands_at({(1, 1): (0.5, 0.5)})[None]
    assert pl.pad_coverage(H_TRUE, tips) == pytest.approx(0.1)


# --- end to end ------------------------------------------------------------------

def _write_session(root: Path, lag_frames: int = 0) -> Path:
    rng = np.random.default_rng(1)
    sess = root / "20260913-100000-pad"
    sess.mkdir(parents=True)
    fps, n = 60.0, 60 * 40
    ft = 100.0 + np.arange(n) / fps
    with open(sess / "frames.jsonl", "w") as fh:
        for i, t in enumerate(ft):
            fh.write(json.dumps({"i": i, "t": float(t)}) + "\n")

    # right index rests at a base position; each event moves one fingertip to its pad xy
    tip_pad = np.zeros((n, 2, 5, 2))
    for s in range(2):
        for f in range(5):
            tip_pad[:, s, f] = (0.15 + 0.35 * s + 0.07 * f, 0.4)
    touches, phrases = [], []
    corners = [(0.03, 0.97), (0.97, 0.97), (0.97, 0.03), (0.03, 0.03)]
    phrases.append({"t": 100.5, "event": "shown", "phrase": "CALIB tap and hold corners", "idx": 0})
    for k, xy in enumerate(corners):
        t0 = 101.0 + 2.0 * k
        a, b = int((t0 - 100) * fps), int((t0 + 1.0 - 100) * fps)
        tip_pad[a:b + 2, 1, 1] = xy
        touches += contact(t0, 1.0, 10 + k, *xy)
    phrases.append({"t": 109.5, "event": "done", "phrase": "CALIB tap and hold corners", "idx": 0})
    phrases.append({"t": 110.0, "event": "shown", "phrase": "the meeting moved", "idx": 1})
    taps = []
    for k in range(40):
        t0 = 110.5 + 0.6 * k
        s, f = k % 2, 1 + k % 4
        xy = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        a = int(round((t0 - 100) * fps)) + lag_frames
        tip_pad[a - 2:a + 8, s, f] = xy
        touches += contact(t0, 0.1, 30 + k % 3, *xy)
        taps.append((t0, s, pl.TIPS[f]))
    phrases.append({"t": 135.0, "event": "done", "phrase": "the meeting moved", "idx": 1})
    touches.sort(key=lambda r: r["t"])
    for name, rows in (("touches.jsonl", touches), ("phrases.jsonl", phrases)):
        with open(sess / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    (sess / "meta.json").write_text(json.dumps({"session_id": sess.name, "condition": "pad"}))

    px = pl.project(H_TRUE, tip_pad.reshape(-1, 2)).reshape(n, 2, 5, 2)
    cols = {"i": [], "t": [], "hand": [], "handedness": [], "joint": [], "x": [], "y": [], "conf": []}
    for i in range(n):
        for s in range(2):
            for j in range(21):
                f = pl.TIPS.index(j) if j in pl.TIPS else None
                x, y = px[i, s, f] if f is not None else (0.0, 0.0)
                for key, v in (("i", i), ("t", ft[i]), ("hand", s), ("handedness", pl.SIDES[s]),
                               ("joint", j), ("x", x), ("y", y), ("conf", 0.9)):
                    cols[key].append(v)
    tb = pa.table({"i": pa.array(cols["i"], pa.int32()), "t": pa.array(cols["t"], pa.float64()),
                   "hand": pa.array(cols["hand"], pa.int8()), "handedness": pa.array(cols["handedness"]),
                   "joint": pa.array(cols["joint"], pa.int8()), "x": pa.array(cols["x"], pa.float32()),
                   "y": pa.array(cols["y"], pa.float32()), "conf": pa.array(cols["conf"], pa.float32())})
    pq.write_table(tb, sess / "landmarks.parquet")
    (sess / "truth.json").write_text(json.dumps(taps))
    return sess


def test_end_to_end_corner_calibration_labels_every_tap(tmp_path):
    sess = _write_session(tmp_path)
    rep = pl.run(sess)
    truth = json.loads((sess / "truth.json").read_text())
    rows = [json.loads(s) for s in (sess / "taps_pad.jsonl").read_text().splitlines()]
    assert rep["calibration"]["method"] == "corner_taps"
    assert rep["kinds"] == {"tap": 40, "rest": 4, "slide": 0}
    assert rep["taps_by_block"] == {"typing": 40}
    assert len(rows) == 40
    got = [(r["hand"], r["finger"]) for r in rows]
    assert got == [(s, f) for _, s, f in truth]
    assert rep["d_px_median"] < 3.0

    view = Path(rep["view"])
    assert view == tmp_path / "20260913-100000-pad-padtrain"
    assert (view / "landmarks.parquet").is_symlink() and (view / "landmarks.parquet").exists()
    kt = keydown_times(view)
    assert len(kt) == 40 and kt[0] == pytest.approx(110.5)


def test_end_to_end_is_rerunnable_and_manual_corners_work(tmp_path):
    sess = _write_session(tmp_path)
    corners = ";".join(f"{x},{y}" for x, y in pl.project(H_TRUE, pl.CORNERS_NORM))
    pl.run(sess)
    rep = pl.run(sess, corners_px=corners)
    assert rep["calibration"]["method"] == "manual" and rep["taps_assigned"] == 40


def test_calibration_error_when_corner_missing(tmp_path):
    sess = _write_session(tmp_path)
    rows = [json.loads(s) for s in (sess / "touches.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["id"] != 12]  # drop the near-right hold
    (sess / "touches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(pl.CalibError, match="near-right"):
        pl.run(sess)


def test_refine_does_not_degrade_exact_calibration(tmp_path):
    sess = _write_session(tmp_path)
    rep = pl.run(sess, refine=True)
    assert rep["taps_assigned"] == 40 and rep["d_px_median"] < 3.0
    assert all(math.isfinite(v) for v in rep["calibration"]["refine_median_d_px"])
