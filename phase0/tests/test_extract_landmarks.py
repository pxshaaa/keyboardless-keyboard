"""Tests for phase0.analysis.extract_landmarks (synthetic video, no real recording)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from phase0.analysis import extract_landmarks as el


class FakeLM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def write_session(tmp_path: Path, n_video: int, n_json: int, w=64, h=48, fps=30.0) -> Path:
    """Build a session dir with a tiny synthetic mp4 and a frames.jsonl."""
    sess = tmp_path / "20260101-000000-kbd"
    sess.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(
        str(sess / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    assert vw.isOpened(), "cv2 could not open an mp4 writer"
    rng = np.random.default_rng(0)
    for _ in range(n_video):
        vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release()
    with open(sess / "frames.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n_json):
            fh.write(json.dumps({"i": i, "t": 1000.0 + i / fps}) + "\n")
    return sess


# --- normalized -> pixel conversion ---
def test_normalized_to_pixels_scales_by_resolution():
    lms = [FakeLM(0.0, 0.0), FakeLM(0.5, 0.5), FakeLM(1.0, 1.0), FakeLM(0.25, 0.75)]
    x, y = el.normalized_to_pixels(lms, 1920, 1080)
    assert x.dtype == np.float32 and y.dtype == np.float32
    np.testing.assert_allclose(x, [0.0, 960.0, 1920.0, 480.0], rtol=1e-6)
    np.testing.assert_allclose(y, [0.0, 540.0, 1080.0, 810.0], rtol=1e-6)


def test_normalized_to_pixels_is_not_clipped_and_uses_x_for_width():
    x, y = el.normalized_to_pixels([FakeLM(-0.1, 1.2)], 100, 200)
    # x scales with width, y with height (a swap would give -20/120 here)
    np.testing.assert_allclose(x, [-10.0], rtol=1e-6)
    np.testing.assert_allclose(y, [240.0], rtol=1e-6)


def test_normalized_to_pixels_rejects_bad_resolution():
    with pytest.raises(ValueError):
        el.normalized_to_pixels([FakeLM(0.5, 0.5)], 0, 480)


# --- parquet schema ---
def test_writer_emits_contract_schema_and_rows(tmp_path):
    out = tmp_path / "landmarks.parquet"
    with el.LandmarkBatchWriter(out, batch_frames=1) as w:
        for i in range(3):
            for hand, label in ((0, "Left"), (1, "Right")):
                x = np.arange(21, dtype=np.float32)
                w.add_hand(i, 1000.0 + i, hand, label, x, x * 2, 0.93)

    table = pq.read_table(out)
    assert table.schema.names == ["i", "t", "hand", "handedness", "joint", "x", "y", "conf"]
    assert table.schema == el.SCHEMA
    assert table.num_rows == 3 * 2 * 21  # up to 42 rows per frame
    d = table.to_pydict()
    assert sorted(set(d["joint"])) == list(range(21))
    assert set(d["hand"]) == {0, 1}
    assert set(d["handedness"]) == {"Left", "Right"}
    # conf is the hand-level score replicated across every joint of that hand
    assert set(np.round(d["conf"], 5)) == {0.93}
    assert d["x"][:3] == [0.0, 1.0, 2.0] and d["y"][:3] == [0.0, 2.0, 4.0]


def test_writer_emits_empty_but_valid_file_when_no_hands(tmp_path):
    out = tmp_path / "landmarks.parquet"
    el.LandmarkBatchWriter(out).close()
    table = pq.read_table(out)
    assert table.num_rows == 0
    assert table.schema == el.SCHEMA


# --- frames.jsonl / video length mismatch ---
def test_check_frame_alignment_ok():
    frames = [el.FrameRow(i, float(i)) for i in range(10)]
    el.check_frame_alignment(10, frames)  # no raise


@pytest.mark.parametrize("n_video", [9, 11, 0])
def test_check_frame_alignment_raises_on_mismatch(n_video):
    frames = [el.FrameRow(i, float(i)) for i in range(10)]
    with pytest.raises(el.FrameAlignmentError):
        el.check_frame_alignment(n_video, frames)
    el.check_frame_alignment(n_video, frames, strict=False)  # opt-out still warns only


def test_read_frames_jsonl_rejects_non_contiguous(tmp_path):
    p = tmp_path / "frames.jsonl"
    p.write_text('{"i": 0, "t": 1.0}\n{"i": 2, "t": 2.0}\n')
    with pytest.raises(el.FrameAlignmentError):
        el.read_frames_jsonl(p)


def test_read_frames_jsonl_roundtrip(tmp_path):
    sess = write_session(tmp_path, n_video=4, n_json=4)
    rows = el.read_frames_jsonl(sess / "frames.jsonl")
    assert [r.i for r in rows] == [0, 1, 2, 3]
    assert rows[0].t == pytest.approx(1000.0)


def test_extract_aborts_on_video_frames_jsonl_mismatch(tmp_path):
    """The critical guard: a length disagreement must abort, not silently misalign."""
    sess = write_session(tmp_path, n_video=5, n_json=8)
    with pytest.raises(el.FrameAlignmentError):
        el.extract(sess, model_path=tmp_path / "nonexistent.task")
    assert not (sess / "landmarks.parquet").exists()


# end-to-end: exercises real mediapipe if the model is available
def test_extract_end_to_end(tmp_path):
    mp = pytest.importorskip("mediapipe")
    assert mp is not None
    sess = write_session(tmp_path, n_video=6, n_json=6)
    try:
        model = el.ensure_model(el.DEFAULT_MODEL_PATH)
    except Exception as exc:  # no network / no model on disk
        pytest.skip(f"hand landmarker model unavailable: {exc}")

    out = el.extract(sess, limit=3, model_path=model)
    table = pq.read_table(out)
    assert table.schema == el.SCHEMA
    # random noise frames -> no hands expected, but the file must be valid
    assert set(table.to_pydict()["i"]) <= {0, 1, 2}
