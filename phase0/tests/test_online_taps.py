"""Online tap detector: hand keying, warm-up, causal event picking, and offline/online equivalence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from phase0.analysis.eval_taps import keydown_times, score
from phase0.analysis.online_taps import (
    AHEAD,
    BACK,
    RUN_FRAC,
    CausalPicker,
    N_CHANNELS,
    N_JOINTS,
    OnlineTapModel,
    assign_sides,
    pack_frame,
    replay_session,
)
from phase0.analysis.taps_gb import (
    apply_events,
    assemble,
    build_groups,
    load_frames,
    pick_events,
)

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "taps_gb.joblib"
HELDOUT = REPO / "data" / "sessions" / "20260910-015948-kbd"
CFG = {"thr": 0.3, "smooth": 1, "refractory": 5, "n_consec": 4, "gate_thr": 0.75}

needs_model = pytest.mark.skipif(not MODEL.exists(), reason="models/taps_gb.joblib missing")
needs_data = pytest.mark.skipif(
    not (HELDOUT / "landmarks.parquet").exists(), reason="held-out session missing")


def _hand(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.full((N_JOINTS, N_CHANNELS), np.nan)
    arr[:, 0] = rng.uniform(100, 900, N_JOINTS)
    arr[:, 1] = rng.uniform(100, 700, N_JOINTS)
    arr[:, 2] = 0.9
    arr[:, 3] = rng.uniform(-20, 20, N_JOINTS)
    arr[:, 4:7] = rng.uniform(-0.1, 0.1, (N_JOINTS, 3))
    return arr


def _run_causal(p, gate=None, cfg=CFG) -> np.ndarray:
    pk = CausalPicker(cfg)
    out: list[int] = []
    for n, v in enumerate(p):
        out += pk.push(float(v), 1.0 if gate is None else float(gate[n]))
    return np.array(out, dtype=int)


def _triangles(centres, width=9, height=0.9, n=200):
    p = np.zeros(n)
    for c in centres:
        for d in range(-(width // 2), width // 2 + 1):
            p[c + d] = max(p[c + d], height * (1 - abs(d) / (width / 2 + 1)))
    return p


# ---------------------------------------------------------------- hand keying
def test_sides_follow_handedness_not_slot():
    assert assign_sides(["Left", "Right"]) == [0, 1]
    assert assign_sides(["Right", "Left"]) == [1, 0]


def test_duplicate_handedness_falls_back_to_slot():
    assert assign_sides(["Left", "Left"]) == [0, 1]
    assert assign_sides(["Right", "Right"]) == [1, 1]  # load_frames lets the later row win


def test_pack_frame_places_hands_by_handedness():
    a, b = _hand(1), _hand(2)
    P = pack_frame([("Right", a), ("Left", b)])
    assert np.allclose(P[1], a, equal_nan=True)
    assert np.allclose(P[0], b, equal_nan=True)
    assert np.isnan(pack_frame([("Left", a)])[1]).all()


# ---------------------------------------------------------------- causal picking
def test_causal_picker_matches_find_peaks_on_well_separated_peaks():
    p = _triangles([30, 80, 140])
    assert list(_run_causal(p)) == list(pick_events(p, CFG["thr"], CFG["refractory"],
                                                    CFG["n_consec"], RUN_FRAC))


def test_causal_picker_drops_peaks_whose_run_is_too_short():
    p = np.zeros(60)
    p[30] = 0.9  # a single-frame spike: above thr, but its run is 1 < n_consec
    assert len(_run_causal(p)) == 0
    assert len(pick_events(p, CFG["thr"], CFG["refractory"], CFG["n_consec"], RUN_FRAC)) == 0


def test_causal_picker_respects_the_gate_and_the_refractory():
    p = _triangles([30, 80])
    assert list(_run_causal(p, gate=np.where(np.arange(200) < 60, 0.9, 0.1))) == [30]
    assert len(_run_causal(_triangles([30, 33], width=5))) == 1


@pytest.mark.parametrize("seed", range(6))
def test_every_causal_event_is_a_valid_offline_peak(seed):
    rng = np.random.default_rng(seed)
    p = np.convolve(rng.random(400), np.ones(5) / 5, mode="same")
    ev = _run_causal(p)
    assert set(ev) <= set(pick_events(p, CFG["thr"], 1, CFG["n_consec"], RUN_FRAC))
    assert all(np.diff(ev) >= CFG["refractory"])


def test_causal_picker_never_looks_ahead():
    """Truncating the stream cannot retract an event the longer stream had already emitted."""
    rng = np.random.default_rng(7)
    p = np.convolve(rng.random(400), np.ones(5) / 5, mode="same")
    full = list(_run_causal(p))
    for cut in (120, 250, 380):
        early = list(_run_causal(p[:cut]))
        assert early == full[:len(early)]


# ---------------------------------------------------------------- warm-up
@needs_model
def test_no_score_before_the_window_is_full():
    det = OnlineTapModel(MODEL, back=8, ahead=4)
    for i in range(12):
        assert det.push(i, i / 60, pack_frame([("Left", _hand(i))])) == []
        assert det.n_score == 0
    det.push(12, 0.2, pack_frame([("Left", _hand(99))]))
    assert det.n_score == 1 and 0.0 <= det.score <= 1.0


# ---------------------------------------------------------------- equivalence
@needs_model
@needs_data
@pytest.mark.slow
def test_online_scores_track_the_offline_scores():
    """A finite window truncates taps_gb's centred rolling filters; the drift must stay small."""
    b = joblib.load(MODEL)
    frames, t, P = load_frames(HELDOUT)
    with threadpool_limits(limits=1, user_api="openmp"):
        p_off = b["model"].predict_proba(assemble(build_groups(P), b["groups"]))[:, 1]

    det = OnlineTapModel(MODEL, back=BACK, ahead=AHEAD)
    on = []
    for k in range(len(frames)):
        det.push(int(frames[k]), float(t[k]), P[k])
        if det.n_score > len(on):
            on.append((det.rows[-1]["i"], det.rows[-1]["p"]))
    pos = {int(f): j for j, f in enumerate(frames)}
    err = np.abs(np.array([p for _, p in on]) - np.array([p_off[pos[i]] for i, _ in on]))
    assert err.mean() < 0.05


@needs_model
@needs_data
@pytest.mark.slow
def test_online_f1_is_close_to_the_offline_f1():
    b = joblib.load(MODEL)
    frames, t, P = load_frames(HELDOUT)
    X = assemble(build_groups(P), b["groups"])
    with threadpool_limits(limits=1, user_api="openmp"):
        p, g = b["model"].predict_proba(X)[:, 1], b["gate"].predict_proba(X)[:, 1]
    kt = keydown_times(HELDOUT)
    lo, hi = kt.min() - 0.1, kt.max() + 0.1

    def clipped(tt):
        tt = np.asarray(tt)
        return score(kt, tt[(tt >= lo) & (tt <= hi)])["f1"]

    offline = clipped(t[apply_events(p, b["cfg"], g)])
    recs, _, _ = replay_session(HELDOUT, MODEL)
    online = clipped([r["t"] for r in recs])
    assert online >= offline - 0.05, f"online F1 {online:.3f} vs offline {offline:.3f}"


@needs_model
@needs_data
@pytest.mark.slow
def test_events_are_emitted_within_the_advertised_latency():
    frames, t, P = load_frames(HELDOUT)
    det = OnlineTapModel(MODEL)
    lags, emitted = [], []
    for k in range(len(frames)):
        for rec in det.push(int(frames[k]), float(t[k]), P[k]):
            lags.append(int(frames[k]) - rec["i"])
            emitted.append(rec["i"])
    assert lags and det.latency_frames <= min(lags) and max(lags) <= det.max_latency_frames
    assert min(np.diff(emitted)) >= det.picker.refractory


# ---------------------------------------------------------------- preview wiring
def test_preview_falls_back_to_the_heuristic_when_the_model_is_missing(tmp_path, capsys):
    from phase0.capture.preview import resolve_detector

    name, model = resolve_detector("model", tmp_path / "nope.joblib")
    assert (name, model) == ("heuristic", None)
    assert "falling back to the crude heuristic" in capsys.readouterr().err


@needs_model
def test_preview_uses_the_model_when_it_is_there():
    from phase0.capture.preview import resolve_detector

    name, model = resolve_detector("model", MODEL)
    assert name == "model" and isinstance(model, OnlineTapModel)
    assert resolve_detector("heuristic", MODEL) == ("heuristic", None)


@needs_model
@needs_data
@pytest.mark.slow
def test_replay_cli_writes_scoreable_jsonl(tmp_path):
    out = tmp_path / "taps.jsonl"
    subprocess.run([sys.executable, "-m", "phase0.analysis.online_taps", "replay", str(HELDOUT),
                    "--out", str(out)], cwd=REPO, check=True,
                   env=os.environ | {"PYTHONPATH": str(REPO)})
    recs = [json.loads(line) for line in open(out)]
    assert recs and all({"t", "hand", "finger", "i", "p"} <= set(r) for r in recs)
