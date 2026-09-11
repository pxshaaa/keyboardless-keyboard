"""Causal streaming wrapper around the trained taps_gb model, for the live preview.
Run: python -m phase0.analysis.online_taps replay <session> [--out O] [--model M] [--ahead N]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from phase0.analysis.taps_gb import (
    DEFAULT_MODEL,
    GROUPS,
    SIDES,
    assemble,
    build_groups,
    flex_velocity,
    load_frames,
)
from phase0.analysis.taps_ml import events_to_records

N_JOINTS = 21
N_CHANNELS = 7  # x, y, conf, z, wx, wy, wz -- taps_gb.CHANNELS
BACK = 90  # frames; measured optimum -- taps_gb's rolling filters stop mattering past this
AHEAD = 8  # forward frames of the centred filters, i.e. the detector's own latency floor
RUN_FRAC = 0.6  # taps_gb.pick_events' run-length gate, relative to thr


def assign_sides(labels: list[str]) -> list[int]:
    """Handedness labels in slot order -> side per slot; slot order is unstable, handedness is not.
    Mirrors taps_gb.load_frames, incl. its fallback to the raw slot when two hands claim one label."""
    out, used = [], set()
    for slot, label in enumerate(labels[:2]):
        s = SIDES.index(label) if label in SIDES else slot
        s = slot if s in used else s
        used.add(s)
        out.append(s)
    return out


def pack_frame(hands: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """[(handedness, arr[21,7])] in slot order -> P[2,21,7], NaN where a hand is absent."""
    P = np.full((2, N_JOINTS, N_CHANNELS), np.nan)
    for slot, s in enumerate(assign_sides([h[0] for h in hands])):
        P[s] = hands[slot][1]
    return P


class CausalPicker:
    """Causal form of taps_gb.pick_events: a peak is emitted once it has fallen and its
    >=thr*RUN_FRAC run has reached n_consec frames. Refractory is first-come, not height-ordered."""

    def __init__(self, cfg: dict):
        self.thr = float(cfg["thr"])
        self.gate_thr = float(cfg.get("gate_thr", 0.0))
        self.smooth = int(cfg.get("smooth", 1))
        self.refractory = int(cfg.get("refractory", 5))
        self.n_consec = int(cfg.get("n_consec", 1))
        self.low = self.thr * RUN_FRAC
        self.pad = self.smooth // 2
        self.lag = self.pad + 1
        self.max_lag = self.lag + self.n_consec - 1
        self.raw: deque = deque(maxlen=max(1, self.smooth))
        self.ps: deque = deque(maxlen=3)
        self.n = -1
        self.run_len = 0
        self.pending: list[int] = []
        self.last_cand = -10**9

    def push(self, p: float, gate: float = 1.0) -> list[int]:
        """Feed one score -> absolute indices of the samples now confirmed as taps."""
        self.n += 1
        self.raw.append(0.0 if self.gate_thr > 0 and gate < self.gate_thr else p)
        j = self.n - self.pad
        if j < 0:
            return []
        self.ps.append(sum(self.raw) / self.smooth if self.smooth > 1 else self.raw[-1])
        prev, mask = self.run_len, self.ps[-1] >= self.low
        self.run_len = prev + 1 if mask else 0
        if len(self.ps) == 3:
            p0, p1, p2 = self.ps
            if p1 > p0 and p1 > p2 and p1 >= self.thr and j - 1 - self.last_cand >= self.refractory:
                self.last_cand = j - 1
                self.pending.append(j - 1)
        if (self.run_len if mask else prev) >= self.n_consec:
            out, self.pending = self.pending, []
            return out
        if not mask:
            self.pending = []
        return []


class OnlineTapModel:
    """Frame-by-frame tap detector; feed it packed frames, it hands back confirmed taps."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, back: int = BACK, ahead: int = AHEAD):
        b = joblib.load(model_path)
        self.model, self.gate_model = b["model"], b.get("gate")
        self.cfg, self.groups = b["cfg"], b.get("groups", GROUPS)
        self.thr, self.gate_thr = float(self.cfg["thr"]), float(self.cfg.get("gate_thr", 0.0))
        self.back, self.ahead = int(back), int(ahead)
        self.picker = CausalPicker(self.cfg)
        self.latency_frames = self.ahead + self.picker.lag
        self.max_latency_frames = self.ahead + self.picker.max_lag
        self.buf: deque = deque(maxlen=self.back + self.ahead + 1)
        self.rows: deque = deque(maxlen=64)
        self.score = 0.0  # newest raw model score; NOT calibrated (scale_pos_weight=8)
        self.n_score = 0
        self.sum_score = 0.0

    def push(self, i: int, t: float, P_row: np.ndarray) -> list[dict]:
        """Feed one frame -> taps confirmed by it (usually none). See self.score for the meter."""
        self.buf.append((i, t, P_row))
        if len(self.buf) < self.buf.maxlen:
            return []
        W = np.stack([f[2] for f in self.buf])
        X = assemble(build_groups(W), self.groups)[self.back:self.back + 1]
        # single-row LightGBM costs ~1.5 ms on one thread and far more under OpenMP contention
        with threadpool_limits(limits=1, user_api="openmp"):
            self.score = float(self.model.predict_proba(X)[0, 1])
            g = float(self.gate_model.predict_proba(X)[0, 1]) if self.gate_model is not None else 1.0
        self.sum_score += self.score
        self.n_score += 1
        ci, ct, _ = self.buf[self.back]
        self.rows.append({"i": ci, "t": ct, "P": W[self.back], "fv": flex_velocity(W)[self.back],
                          "p": self.score, "gate": g})
        base = self.n_score - len(self.rows)
        return [self._record(self.rows[k - base]) for k in self.picker.push(self.score, g)
                if k - base >= 0]

    def _record(self, r: dict) -> dict:
        rec = events_to_records([0], np.array([r["i"]]), np.array([r["t"]]),
                                r["P"][None], r["fv"][None])[0]
        rec["p"] = float(r["p"])
        return rec

    def push_hands(self, i: int, t: float, hands: list[tuple[str, np.ndarray]]) -> list[dict]:
        return self.push(i, t, pack_frame(hands))

    @property
    def mean_score(self) -> float:
        return self.sum_score / self.n_score if self.n_score else 0.0


# ---------------------------------------------------------------- replay (equivalence test)
def replay_session(session: Path, model_path: Path = DEFAULT_MODEL, back: int = BACK,
                   ahead: int = AHEAD) -> tuple[list[dict], OnlineTapModel, float]:
    """Push a recorded landmarks.parquet through the wrapper one frame at a time."""
    frames, t, P = load_frames(session)
    det = OnlineTapModel(model_path, back=back, ahead=ahead)
    recs: list[dict] = []
    t0 = time.perf_counter()
    for k in range(len(frames)):
        recs.extend(det.push(int(frames[k]), float(t[k]), P[k]))
    return recs, det, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.online_taps",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("replay")
    s.add_argument("session", type=Path)
    s.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--back", type=int, default=BACK)
    s.add_argument("--ahead", type=int, default=AHEAD)
    a = ap.parse_args(argv)

    recs, det, secs = replay_session(a.session, a.model, a.back, a.ahead)
    out = a.out or a.session / "taps_online.jsonl"
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    dt = float(np.median(np.diff([r["t"] for r in det.rows]))) if len(det.rows) > 1 else 1 / 60
    print(f"wrote {len(recs)} events -> {out}")
    print(f"frames={det.n_score} wall={secs:.1f}s ({det.n_score / max(secs, 1e-9):.1f} fps)  "
          f"latency={det.latency_frames}-{det.max_latency_frames} frames "
          f"({1000 * det.latency_frames * dt:.0f}-{1000 * det.max_latency_frames * dt:.0f} ms)  "
          f"mean_score={det.mean_score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
