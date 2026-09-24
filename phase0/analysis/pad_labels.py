"""Trackpad contacts (touches.jsonl) -> taps_pad.jsonl (taps.jsonl schema) + a `<sid>-padtrain` view with keys.jsonl.
Run: python -m phase0.analysis.pad_labels data/sessions/<sid> [--corners-px "x,y;x,y;x,y;x,y"] [--refine]"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from phase0.capture.touchpad import map_clock

TIPS = (4, 8, 12, 16, 20)
SIDES = ("Left", "Right")  # hand index == side, as in taps_gb output
# pad-normalized corners, y_norm=1 is the far edge: far-left, far-right, near-right, near-left
CORNERS_NORM = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
CORNER_NAMES = ("far-left", "far-right", "near-right", "near-left")
CORNER_RADIUS = 0.25
CALIB_PREFIX = "CALIB"
CALIB_FALLBACK_S = 30.0
CALIB_SIDE, CALIB_TIP = 1, 8  # right index fingertip
TAP_MAX_S = 0.30
TAP_MAX_TRAVEL_MM = 4.0
SIGMA_PX = 25.0
VIEW_LINKS = ("landmarks.parquet", "frames.jsonl", "video.mp4", "meta.json", "phrases.jsonl")


class CalibError(RuntimeError):
    pass


@dataclass
class Contact:
    id: int
    t_down: float
    t_up: float
    x_norm: float
    y_norm: float
    x_mm: float
    y_mm: float
    travel_mm: float = 0.0
    max_density: float = 0.0
    max_major: float = 0.0
    truncated: bool = False
    kind: str = ""

    @property
    def dur(self) -> float:
        return self.t_up - self.t_down


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


# ------------------------------------------------------------------ contacts
def contacts_from_touches(rows: list[dict[str, Any]]) -> list[Contact]:
    """Pair down..up per path id (ids are reused after lift); unclosed contacts end at their last row."""
    if rows and all("t_dev" in r and "t_rx" in r for r in rows):
        for r, t in zip(rows, map_clock([r["t_dev"] for r in rows], [r["t_rx"] for r in rows])):
            r["t"] = t
    open_: dict[int, Contact] = {}
    last_t: dict[int, float] = {}
    out: list[Contact] = []
    for r in rows:
        pid, ev = r["id"], r["event"]
        if ev == "down":
            if pid in open_:
                c = open_.pop(pid)
                c.t_up, c.truncated = last_t[pid], True
                out.append(c)
            open_[pid] = Contact(pid, r["t"], r["t"], r["x_norm"], r["y_norm"], r["x_mm"], r["y_mm"],
                                 max_density=r.get("density", 0.0), max_major=r.get("major", 0.0))
        elif pid in open_:
            c = open_[pid]
            c.travel_mm = max(c.travel_mm, math.hypot(r["x_mm"] - c.x_mm, r["y_mm"] - c.y_mm))
            c.max_density = max(c.max_density, r.get("density", 0.0))
            c.max_major = max(c.max_major, r.get("major", 0.0))
            if ev == "up":
                c.t_up = r["t"]
                c.truncated = bool(r.get("truncated", False))
                out.append(open_.pop(pid))
        last_t[pid] = r["t"]
    for pid, c in open_.items():
        c.t_up, c.truncated = last_t[pid], True
        out.append(c)
    out.sort(key=lambda c: c.t_down)
    return out


def classify(c: Contact, tap_max_s: float = TAP_MAX_S, max_travel_mm: float = TAP_MAX_TRAVEL_MM) -> str:
    if c.travel_mm > max_travel_mm:
        return "slide"
    if c.dur > tap_max_s:
        return "rest"
    return "tap"


# ------------------------------------------------------------------ video side
def load_frame_times(session: Path) -> tuple[np.ndarray, np.ndarray]:
    fr = load_jsonl(session / "frames.jsonl")
    return np.array([r["i"] for r in fr], dtype=np.int64), np.array([r["t"] for r in fr], dtype=float)


def load_tips(session: Path, frame_i: np.ndarray) -> np.ndarray:
    """-> tips[F, 2 sides, 5 fingertips, 2] px aligned to frames.jsonl rows; NaN where no hand."""
    import pyarrow.parquet as pq

    tb = pq.read_table(session / "landmarks.parquet", columns=["i", "handedness", "joint", "x", "y"])
    i = tb.column("i").to_numpy()
    joint = tb.column("joint").to_numpy()
    side = (np.asarray(tb.column("handedness").to_pylist()) != "Left").astype(int)
    x, y = tb.column("x").to_numpy(), tb.column("y").to_numpy()
    return tips_array(frame_i, i, side, joint, x, y)


def tips_array(frame_i, i, side, joint, x, y) -> np.ndarray:
    tips = np.full((len(frame_i), 2, len(TIPS), 2), np.nan)
    pos = {int(v): k for k, v in enumerate(frame_i)}
    jmap = {j: k for k, j in enumerate(TIPS)}
    for a, s, j, xx, yy in zip(i, side, joint, x, y):
        f = jmap.get(int(j))
        k = pos.get(int(a))
        if f is not None and k is not None:
            tips[k, s, f] = (xx, yy)
    return tips


def nearest_frame(frame_t: np.ndarray, t: float) -> int:
    k = int(np.searchsorted(frame_t, t))
    if k <= 0:
        return 0
    if k >= len(frame_t):
        return len(frame_t) - 1
    return k if frame_t[k] - t < t - frame_t[k - 1] else k - 1


# ------------------------------------------------------------------ geometry
def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT, least squares for N > 4."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise CalibError("need >= 4 point pairs")

    def norm(p):
        c = p.mean(0)
        s = math.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = norm(src), norm(dst)
    ps = (Ts @ np.c_[src, np.ones(len(src))].T).T
    pd = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (x, y, _), (u, v, _) in zip(ps, pd):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, sv, vt = np.linalg.svd(np.asarray(A))
    if sv[-2] < 1e-9:
        raise CalibError("degenerate calibration points (collinear or repeated)")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def project(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.asarray(pts).reshape(-1, 2)))]
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


# ------------------------------------------------------------------ calibration
def calib_window(session: Path, contacts: list[Contact]) -> tuple[float, float]:
    for r in load_jsonl(session / "phrases.jsonl"):
        if r["event"] == "shown" and r["phrase"].upper().startswith(CALIB_PREFIX):
            done = [d["t"] for d in load_jsonl(session / "phrases.jsonl")
                    if d["event"] == "done" and d["idx"] == r["idx"]]
            return r["t"], (done[0] if done else r["t"] + CALIB_FALLBACK_S)
    t0 = contacts[0].t_down if contacts else 0.0
    return t0, t0 + CALIB_FALLBACK_S


def corner_calibration(contacts: list[Contact], tips: np.ndarray, frame_t: np.ndarray,
                       window: tuple[float, float], side: int = CALIB_SIDE, tip: int = CALIB_TIP
                       ) -> dict[str, Any]:
    """Longest hold near each pad corner in the window, paired with the calib fingertip's median px."""
    f = TIPS.index(tip)
    src, dst, used = [], [], []
    for name, corner in zip(CORNER_NAMES, CORNERS_NORM):
        cands = [c for c in contacts if window[0] <= c.t_down <= window[1]
                 and math.hypot(c.x_norm - corner[0], c.y_norm - corner[1]) <= CORNER_RADIUS]
        if not cands:
            raise CalibError(f"no touch near the {name} corner in the calibration window "
                             f"{window[0]:.1f}-{window[1]:.1f}s; pass --corners-px")
        c = max(cands, key=lambda c: c.dur)
        i0 = nearest_frame(frame_t, c.t_down + min(0.1, c.dur / 2))
        i1 = max(i0, nearest_frame(frame_t, c.t_up))
        seg = tips[i0:i1 + 1, side, f]
        seg = seg[np.isfinite(seg).all(1)]
        if not len(seg):
            raise CalibError(f"{SIDES[side]} fingertip {tip} not tracked during the {name} corner hold")
        src.append((c.x_norm, c.y_norm))
        dst.append(np.median(seg, 0))
        used.append({"corner": name, "t_down": c.t_down, "dur": round(c.dur, 3),
                     "x_norm": c.x_norm, "y_norm": c.y_norm, "px": [float(v) for v in dst[-1]]})
    return {"H": fit_homography(np.array(src), np.array(dst)), "method": "corner_taps", "corners": used}


def manual_calibration(corners_px: str) -> dict[str, Any]:
    pts = [tuple(float(v) for v in s.split(",")) for s in corners_px.split(";") if s.strip()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise CalibError('--corners-px needs 4 "x,y" pairs: far-left;far-right;near-right;near-left')
    return {"H": fit_homography(np.array(CORNERS_NORM), np.array(pts)), "method": "manual",
            "corners": [{"corner": n, "px": list(p)} for n, p in zip(CORNER_NAMES, pts)]}


# ------------------------------------------------------------------ assignment
def assign(c: Contact, H: np.ndarray, tips: np.ndarray, frame_t: np.ndarray,
           frame_i: np.ndarray, sigma: float = SIGMA_PX, lag_s: float = 0.0) -> Optional[dict[str, Any]]:
    k = nearest_frame(frame_t, c.t_down + lag_s)
    p = project(H, [(c.x_norm, c.y_norm)])[0]
    d = np.linalg.norm(tips[k] - p, axis=-1)
    d = np.where(np.isfinite(d), d, np.inf)
    if not np.isfinite(d).any():
        return None
    order = np.argsort(d, axis=None)
    s, f = np.unravel_index(order[0], d.shape)
    d1 = float(d[s, f])
    d2 = float(d.flat[order[1]]) if np.isfinite(d.flat[order[1]]) else None
    return {
        "t": c.t_down, "hand": int(s), "finger": TIPS[f],
        "x": float(tips[k, s, f, 0]), "y": float(tips[k, s, f, 1]),
        "conf": round(math.exp(-0.5 * (d1 / sigma) ** 2), 4), "i": int(frame_i[k]),
        "src": "pad", "handedness": SIDES[s], "contact_x": round(float(p[0]), 2),
        "contact_y": round(float(p[1]), 2), "d_px": round(d1, 2),
        "d2_px": None if d2 is None else round(d2, 2), "t_up": c.t_up, "dur": round(c.dur, 4),
        "pad_id": c.id, "x_mm": c.x_mm, "y_mm": c.y_mm, "density": c.max_density,
    }


def refine_homography(taps: list[Contact], H: np.ndarray, tips, frame_t, frame_i,
                      iters: int = 3, keep: float = 0.7) -> tuple[np.ndarray, list[float]]:
    """Refit on (pad xy, nearest fingertip px) for the best `keep` fraction; stops when the median stops falling."""
    med: list[float] = []
    H_prev = H
    for _ in range(iters + 1):
        rows = [(c, assign(c, H, tips, frame_t, frame_i)) for c in taps]
        rows = [(c, r) for c, r in rows if r is not None]
        if len(rows) < 8:
            break
        dist = np.array([r["d_px"] for _, r in rows])
        med.append(float(np.median(dist)))
        if len(med) > 1 and med[-1] >= med[-2]:
            H = H_prev
            med.pop()
            break
        best = np.argsort(dist)[: max(8, int(keep * len(rows)))]
        H_prev = H
        H = fit_homography(np.array([(rows[j][0].x_norm, rows[j][0].y_norm) for j in best]),
                           np.array([(rows[j][1]["x"], rows[j][1]["y"]) for j in best]))
    return H, med


# ------------------------------------------------------------------ outputs
def write_view(session: Path, taps: list[Contact], view: Path) -> Path:
    """Sibling session dir whose keys.jsonl holds one down/up per pad tap, so taps_gb/touch tools train unchanged."""
    view.mkdir(parents=True, exist_ok=True)
    for name in VIEW_LINKS:
        src, dst = session / name, view / name
        if src.exists():
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src.resolve(), view.resolve()))
    ev = sorted([(c.t_down, "down") for c in taps] + [(c.t_up, "up") for c in taps])
    with open(view / "keys.jsonl", "w") as fh:
        for t, e in ev:
            fh.write(json.dumps({"t": t, "event": e, "key": "unknown"}) + "\n")
    return view


def pad_coverage(H: np.ndarray, tips: np.ndarray, margin: float = 0.05) -> Optional[float]:
    """Fraction of tracked fingertip samples that project onto the glass; low = taps landed off-pad, unlabelled."""
    pts = tips.reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    if not len(pts):
        return None
    q = project(np.linalg.inv(H), pts)
    inside = (q > -margin).all(1) & (q < 1 + margin).all(1)
    return round(float(inside.mean()), 3)


def block_of(t: float, windows: list[tuple[float, float, str]]) -> str:
    for lo, hi, phrase in windows:
        if lo <= t <= hi:
            head = phrase.split()[0].upper() if phrase.split() else ""
            return head if head in ("CALIB", "DRILL", "NEG") else "typing"
    return "between"


def phrase_windows(session: Path) -> list[tuple[float, float, str]]:
    rows = load_jsonl(session / "phrases.jsonl")
    done = {r["idx"]: r["t"] for r in rows if r["event"] == "done"}
    return [(r["t"], done.get(r["idx"], math.inf), r["phrase"]) for r in rows if r["event"] == "shown"]


def run(session: Path, corners_px: Optional[str] = None, refine: bool = False,
        tap_max_s: float = TAP_MAX_S, sigma: float = SIGMA_PX, view: Optional[Path] = None,
        lag_ms: float = 0.0) -> dict[str, Any]:
    session = Path(session)
    touches = load_jsonl(session / "touches.jsonl")
    if not touches:
        raise SystemExit(f"{session}/touches.jsonl is missing or empty -- record with --touchpad")
    contacts = contacts_from_touches(touches)
    for c in contacts:
        c.kind = classify(c, tap_max_s)
    frame_i, frame_t = load_frame_times(session)
    tips = load_tips(session, frame_i)

    win = calib_window(session, contacts)
    cal = manual_calibration(corners_px) if corners_px else corner_calibration(contacts, tips, frame_t, win)
    H = cal["H"]
    taps = [c for c in contacts if c.kind == "tap"]
    lag_s = lag_ms / 1000.0
    refine_med: list[float] = []
    if refine:
        H, refine_med = refine_homography(taps, H, tips, frame_t, frame_i)

    rows = [r for r in (assign(c, H, tips, frame_t, frame_i, sigma, lag_s) for c in taps) if r is not None]
    with open(session / "taps_pad.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    view = view or session.parent / f"{session.name}-padtrain"
    write_view(session, taps, view)

    kinds = {k: sum(c.kind == k for c in contacts) for k in ("tap", "rest", "slide")}
    d = np.array([r["d_px"] for r in rows]) if rows else np.array([])
    span = (frame_t[-1] - frame_t[0]) if len(frame_t) > 1 else 0.0
    wins = phrase_windows(session)
    blocks: dict[str, int] = {}
    for c in taps:
        b = block_of(c.t_down, wins)
        blocks[b] = blocks.get(b, 0) + 1
    report = {
        "session": session.name, "contacts": len(contacts), "kinds": kinds, "taps_by_block": blocks,
        "lag_ms": lag_ms, "pad_coverage": pad_coverage(H, tips),
        "calib_window": win, "calibration": {**{k: v for k, v in cal.items() if k != "H"},
                                             "H": np.round(H, 6).tolist(), "refine_median_d_px": refine_med},
        "taps_labelled": len(taps), "taps_assigned": len(rows), "unassigned_no_hands": len(taps) - len(rows),
        "taps_per_min": round(60 * len(taps) / span, 1) if span else None,
        "tap_dur_ms_median": round(1000 * float(np.median([c.dur for c in taps])), 1) if taps else None,
        "d_px_median": round(float(np.median(d)), 1) if len(d) else None,
        "frac_d_over_2sigma": round(float((d > 2 * sigma).mean()), 3) if len(d) else None,
        "per_finger": {f"{SIDES[s]}{t}": sum(r["hand"] == s and r["finger"] == t for r in rows)
                       for s in (0, 1) for t in TIPS},
        "view": str(view),
    }
    (session / "pad_labels.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.pad_labels", description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--corners-px", default=None,
                    help='landmark-pixel corners "x,y;x,y;x,y;x,y" far-left;far-right;near-right;near-left')
    ap.add_argument("--refine", action="store_true", help="refit the homography on all taps (reported, opt-in)")
    ap.add_argument("--tap-max-s", type=float, default=TAP_MAX_S)
    ap.add_argument("--sigma-px", type=float, default=SIGMA_PX)
    ap.add_argument("--view", type=Path, default=None)
    ap.add_argument("--lag-ms", type=float, default=0.0, help="added to pad time before picking the frame")
    a = ap.parse_args(argv)
    try:
        rep = run(a.session, a.corners_px, a.refine, a.tap_max_s, a.sigma_px, a.view, a.lag_ms)
    except CalibError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in rep.items() if k != "calibration"}, indent=2))
    print(f"calibration: {rep['calibration']['method']}; wrote {a.session}/taps_pad.jsonl, pad_labels.json, {rep['view']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
