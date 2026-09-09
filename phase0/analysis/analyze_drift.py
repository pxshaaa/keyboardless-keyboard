"""Phase 0 verdict: do per-key landing positions survive removing the keyboard?
Two labelling paths per CONTRACT amendment 1 - kbd timestamp pairing, desk phrase alignment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

# --- contract constants ---------------------------------------------------
PAIR_WINDOW_S = 0.080
MIN_SAMPLES_PER_KEY = 15
SHIFT_THRESHOLD_PITCH = 0.5
VAR_RATIO_THRESHOLD = 2.0
MIN_QUALIFYING_KEYS = 8

HOME_ROW_ADJACENT = [("a", "s"), ("s", "d"), ("d", "f"), ("j", "k"), ("k", "l")]
FINGER_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
THUMB_JOINT = 4
MODIFIERS = {"shift", "ctrl", "alt", "cmd", "esc"}

# alignment weights: rank-position mismatch dominates, thumb/space agreement is the
# only real observational signal available on a bare desk.
W_RANK = 4.0
W_THUMB = 0.6
GAP_DELETE = 0.5  # a typed character with no detected tap
GAP_INSERT = 0.5  # a detected tap with no character
CONFIDENT_MATCH_COST = 0.5
MIN_FRAC_CONFIDENT = 0.70
MAX_NORM_COST = 0.60
TAP_COUNT_RATIO_RANGE = (0.60, 1.60)

# Fallback pitch: 19 mm Apple pitch at the contract's oblique framing is ~55 px at
# 1920 px wide. Only used when calibration is impossible; results are labelled ASSUMED.
FALLBACK_PITCH_FRACTION_OF_WIDTH = 55.0 / 1920.0
FALLBACK_PITCH_PX_DEFAULT = 55.0


class AnalysisError(RuntimeError):
    """Input is missing or too degenerate to produce a verdict."""


# --- io -------------------------------------------------------------------
def read_jsonl(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise AnalysisError(f"missing required file: {path}")
        return []
    rows: list[dict] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{lineno}: malformed JSON ({exc})") from exc
    return rows


@dataclass
class Session:
    dir: Path
    condition: str
    keys: list[dict]
    taps: list[dict]
    phrases: list[dict]
    meta: dict
    records: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        ts = [r["t"] for r in self.taps] + [r["t"] for r in self.keys]
        return max(ts) - min(ts) if ts else 0.0


def load_session(session_dir: Path, condition: str) -> Session:
    if not session_dir.is_dir():
        raise AnalysisError(f"session dir does not exist: {session_dir}")
    taps = read_jsonl(session_dir / "taps.jsonl")
    if not taps:
        raise AnalysisError(f"{session_dir/'taps.jsonl'}: no taps - run detect_taps first")
    for f in ("t", "x", "y", "finger"):
        if f not in taps[0]:
            raise AnalysisError(f"{session_dir/'taps.jsonl'}: rows lack required field {f!r}")

    keys = [r for r in read_jsonl(session_dir / "keys.jsonl", required=(condition == "kbd")) if r.get("event") == "down"]
    if condition == "kbd" and not keys:
        raise AnalysisError(f"{session_dir/'keys.jsonl'}: no keydown events in the kbd session")
    phrases = read_jsonl(session_dir / "phrases.jsonl", required=(condition == "desk"))
    if condition == "desk" and not phrases and not keys:
        raise AnalysisError(
            f"{session_dir}: desk session has neither phrases.jsonl nor keys.jsonl - nothing to label taps with"
        )

    meta = {}
    if (session_dir / "meta.json").exists():
        try:
            meta = json.loads((session_dir / "meta.json").read_text())
        except json.JSONDecodeError:
            meta = {}
    if meta.get("condition") and meta["condition"] != condition:
        raise AnalysisError(
            f"{session_dir}: meta.json says condition={meta['condition']!r} but it was passed as --{condition}"
        )
    keys.sort(key=lambda r: r["t"])
    taps.sort(key=lambda r: r["t"])
    return Session(dir=session_dir, condition=condition, keys=keys, taps=taps, phrases=phrases, meta=meta)


# --- labelling path 1: kbd timestamp pairing ------------------------------
def pair_events(
    key_times: Sequence[float], tap_times: Sequence[float], window: float = PAIR_WINDOW_S
) -> list[tuple[int, int]]:
    """1:1 min-total-|dt| match within +/-window; greedy nearest would double-assign one
    tap to two keystrokes typed 40 ms apart, so each feasibility component is solved globally."""
    kt = np.asarray(key_times, dtype=float)
    tt = np.asarray(tap_times, dtype=float)
    if kt.size == 0 or tt.size == 0:
        return []
    if not np.all(np.diff(kt) >= 0) or not np.all(np.diff(tt) >= 0):
        raise AnalysisError("pair_events expects time-sorted inputs")

    lo = np.searchsorted(tt, kt - window, side="left")
    hi = np.searchsorted(tt, kt + window, side="right")
    pairs: list[tuple[int, int]] = []
    i, n = 0, kt.size
    while i < n:
        if lo[i] >= hi[i]:
            i += 1
            continue
        j, comp_hi = i, hi[i]
        while j + 1 < n and lo[j + 1] < comp_hi and lo[j + 1] < hi[j + 1]:
            j += 1
            comp_hi = max(comp_hi, hi[j])
        k_idx = [k for k in range(i, j + 1) if lo[k] < hi[k]]
        pairs.extend(_solve_component(kt, tt, k_idx, int(min(lo[k] for k in k_idx)), int(comp_hi), window))
        i = j + 1
    pairs.sort()
    return pairs


def _solve_component(kt, tt, k_idx, t_lo, t_hi, window) -> list[tuple[int, int]]:
    t_idx = list(range(t_lo, t_hi))
    if not t_idx:
        return []
    cost = np.abs(kt[k_idx][:, None] - tt[t_idx][None, :])
    infeasible = cost > window
    if not HAVE_SCIPY or cost.size > 4_000_000:
        return _greedy_component(cost, infeasible, k_idx, t_idx)
    big = float(window * 10 * (cost.size + 1))
    rows, cols = linear_sum_assignment(np.where(infeasible, big, cost))
    return [(k_idx[r], t_idx[c]) for r, c in zip(rows, cols) if not infeasible[r, c]]


def _greedy_component(cost, infeasible, k_idx, t_idx) -> list[tuple[int, int]]:
    # fallback only: globally greedy by |dt|, never reuses a key or a tap.
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    used_k, used_t, out = set(), set(), []
    for r, c in order:
        if infeasible[r, c] or r in used_k or c in used_t:
            continue
        used_k.add(r)
        used_t.add(c)
        out.append((k_idx[r], t_idx[c]))
    return out


def label_by_pairing(session: Session) -> None:
    usable = [i for i, r in enumerate(session.keys) if r.get("key") not in MODIFIERS and r.get("key") != "unknown"]
    sub = pair_events([session.keys[i]["t"] for i in usable], [r["t"] for r in session.taps])
    pairs = [(usable[a], b) for a, b in sub]
    session.records = [
        {
            "key": session.keys[ki]["key"],
            "t": session.keys[ki]["t"],
            "x": session.taps[ti]["x"],
            "y": session.taps[ti]["y"],
            "finger": session.taps[ti]["finger"],
            "hand": session.taps[ti].get("hand"),
        }
        for ki, ti in pairs
    ]
    session.stats = {
        "method": "timestamp pairing (+/-80 ms, global assignment)",
        "keydowns": len(session.keys),
        "excluded_keys": len(session.keys) - len(usable),
        "paired": len(pairs),
        "unpaired_keys": len(usable) - len(pairs),
        "unpaired_taps": len(session.taps) - len(pairs),
        "rate_keys": len(pairs) / len(usable) if usable else 0.0,
        "rate_taps": len(pairs) / len(session.taps) if session.taps else 0.0,
    }


# --- labelling path 2: desk phrase alignment ------------------------------
@dataclass
class Alignment:
    matches: list[tuple[int, int, float]]  # (char_idx, tap_idx, match cost)
    total_cost: float
    norm_cost: float
    frac_confident: float
    n_chars: int
    n_taps: int


def normalize_char(ch: str) -> str | None:
    if ch == " ":
        return "space"
    if ch.isalnum() and len(ch) == 1:
        return ch.lower()
    return None


def align_phrase(chars: Sequence[str], taps: Sequence[dict]) -> Alignment:
    """Monotonic Needleman-Wunsch of typed characters onto detected taps; cost is rank-position
    disagreement plus thumb/space agreement, gaps model missed and hallucinated taps (no 1:1)."""
    n, m = len(chars), len(taps)
    if n == 0 or m == 0:
        return Alignment([], float(n) * GAP_DELETE + float(m) * GAP_INSERT, float("inf"), 0.0, n, m)

    # tap position is normalized ELAPSED TIME, not rank: a missed tap then leaves a real gap,
    # which is what tells the alignment which character was dropped.
    pos_c = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    ts = np.array([t["t"] for t in taps], dtype=float)
    span = float(ts[-1] - ts[0])
    pos_t = (ts - ts[0]) / span if span > 1e-9 else (np.linspace(0.0, 1.0, m) if m > 1 else np.zeros(1))
    is_space = np.array([c == "space" for c in chars])
    is_thumb = np.array([int(t.get("finger", 0)) == THUMB_JOINT for t in taps])
    match = W_RANK * np.abs(pos_c[:, None] - pos_t[None, :]) + W_THUMB * (is_space[:, None] != is_thumb[None, :])

    D = np.empty((n + 1, m + 1))
    D[0, :] = np.arange(m + 1) * GAP_INSERT
    D[:, 0] = np.arange(n + 1) * GAP_DELETE
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    ptr[0, 1:] = 2
    ptr[1:, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = (D[i - 1, j - 1] + match[i - 1, j - 1], D[i - 1, j] + GAP_DELETE, D[i, j - 1] + GAP_INSERT)
            k = int(np.argmin(cands))
            D[i, j] = cands[k]
            ptr[i, j] = k

    matches: list[tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        k = ptr[i, j]
        if k == 0 and i > 0 and j > 0:
            matches.append((i - 1, j - 1, float(match[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif k == 1 and i > 0:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    confident = [mm for mm in matches if mm[2] < CONFIDENT_MATCH_COST]
    total = float(D[n, m])
    return Alignment(matches, total, total / n, len(confident) / n, n, m)


def label_by_phrases(session: Session) -> None:
    windows = _phrase_windows(session.phrases)
    tap_t = np.array([t["t"] for t in session.taps])
    records, used_phrases, dropped, norm_costs, fracs, taps_in_windows = [], 0, [], [], [], 0

    for w in windows:
        chars = [c for c in (normalize_char(ch) for ch in w["phrase"]) if c]
        lo, hi = np.searchsorted(tap_t, w["t0"], "left"), np.searchsorted(tap_t, w["t1"], "right")
        taps = session.taps[lo:hi]
        taps_in_windows += len(taps)
        if not chars or not taps:
            dropped.append((w["idx"], "no chars or no taps in window"))
            continue
        ratio = len(taps) / len(chars)
        if not (TAP_COUNT_RATIO_RANGE[0] <= ratio <= TAP_COUNT_RATIO_RANGE[1]):
            dropped.append((w["idx"], f"tap/char ratio {ratio:.2f} outside {TAP_COUNT_RATIO_RANGE} (likely mistyped)"))
            continue
        al = align_phrase(chars, taps)
        norm_costs.append(al.norm_cost)
        fracs.append(al.frac_confident)
        if al.frac_confident < MIN_FRAC_CONFIDENT or al.norm_cost > MAX_NORM_COST:
            dropped.append(
                (w["idx"], f"poor alignment (cost {al.norm_cost:.2f}, {al.frac_confident:.0%} confident)")
            )
            continue
        used_phrases += 1
        for ci, ti, cost in al.matches:
            if cost >= CONFIDENT_MATCH_COST:
                continue
            tap = taps[ti]
            records.append(
                {
                    "key": chars[ci],
                    "t": tap["t"],
                    "x": tap["x"],
                    "y": tap["y"],
                    "finger": tap["finger"],
                    "hand": tap.get("hand"),
                }
            )
    records.sort(key=lambda r: r["t"])
    session.records = records
    session.stats = {
        "method": "phrase alignment (monotonic needleman-wunsch)",
        "phrases_total": len(windows),
        "phrases_used": used_phrases,
        "phrases_dropped": dropped,
        "median_norm_cost": statistics.median(norm_costs) if norm_costs else float("nan"),
        "median_frac_confident": statistics.median(fracs) if fracs else float("nan"),
        "taps_total": len(session.taps),
        "taps_in_windows": taps_in_windows,
        "taps_labelled": len(records),
        "rate_taps": len(records) / len(session.taps) if session.taps else 0.0,
        "rate_keys": statistics.median(fracs) if fracs else 0.0,
    }


def _phrase_windows(phrases: list[dict]) -> list[dict]:
    shown: dict[int, dict] = {}
    out = []
    for r in sorted(phrases, key=lambda r: r["t"]):
        if r.get("event") == "shown":
            shown[r["idx"]] = r
        elif r.get("event") == "done" and r["idx"] in shown:
            s = shown.pop(r["idx"])
            out.append({"idx": r["idx"], "phrase": s.get("phrase", r.get("phrase", "")), "t0": s["t"], "t1": r["t"]})
    return out


def label_session(session: Session) -> None:
    if session.condition == "kbd":
        label_by_pairing(session)
    elif session.phrases:
        label_by_phrases(session)
    else:
        label_by_pairing(session)
        session.stats["method"] += " [FALLBACK: desk session had no phrases.jsonl but did have key events]"


# --- per-key aggregation --------------------------------------------------
@dataclass
class KeyStats:
    key: str
    n: int
    cx: float
    cy: float
    var_px2: float  # mean squared radial distance to centroid = trace of covariance
    spread_px: float
    det_cov: float
    finger: str
    finger_share: float


def key_points(session: Session) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in session.records:
        out[r["key"]].append(r)
    return out


def summarise_key(key: str, pts: list[dict]) -> KeyStats:
    xy = np.array([[p["x"], p["y"]] for p in pts], dtype=float)
    c = xy.mean(axis=0)
    var = float(((xy - c) ** 2).sum(axis=1).mean())
    cov = np.cov(xy.T) if len(xy) > 1 else np.zeros((2, 2))
    fingers = Counter(FINGER_NAMES.get(int(p["finger"]), f"j{p['finger']}") for p in pts)
    top, cnt = fingers.most_common(1)[0]
    return KeyStats(key, len(pts), float(c[0]), float(c[1]), var, math.sqrt(var), float(np.linalg.det(cov)), top, cnt / len(pts))


# --- key pitch ------------------------------------------------------------
def calibrate_pitch(kbd_stats: dict[str, KeyStats], frame_width: int | None) -> tuple[float, str]:
    dists, used = [], []
    for a, b in HOME_ROW_ADJACENT:
        if a in kbd_stats and b in kbd_stats:
            d = math.dist((kbd_stats[a].cx, kbd_stats[a].cy), (kbd_stats[b].cx, kbd_stats[b].cy))
            dists.append(d)
            used.append(f"{a}-{b}={d:.1f}px")
    if len(dists) >= 2:
        return statistics.median(dists), f"home-row pairs ({len(dists)}/5): " + ", ".join(used)
    if len(dists) == 1:
        return dists[0], f"WEAK: only one home-row pair had data ({used[0]})"
    cents = [(s.cx, s.cy) for s in kbd_stats.values()]
    if len(cents) >= 4:
        nn = [min(math.dist(p, q) for j, q in enumerate(cents) if j != i) for i, p in enumerate(cents)]
        return statistics.median(nn), (
            f"FALLBACK: no home-row pair had data; median nearest-neighbour distance between "
            f"{len(cents)} key centroids"
        )
    px = frame_width * FALLBACK_PITCH_FRACTION_OF_WIDTH if frame_width else FALLBACK_PITCH_PX_DEFAULT
    return px, (
        f"ASSUMED: too few key centroids to calibrate; assuming {px:.1f} px (19 mm pitch at the "
        f"contract framing). Pitch-unit numbers are NOT measured."
    )


# --- within-session temporal drift ---------------------------------------
def drift_over_time(session: Session, min_per_third: int = 5) -> dict:
    pts = sorted(session.records, key=lambda r: r["t"])
    if len(pts) < 3 * min_per_third:
        return {"ok": False, "reason": f"only {len(pts)} labelled samples"}
    n = len(pts)
    thirds = [pts[: n // 3], pts[n // 3 : 2 * n // 3], pts[2 * n // 3 :]]
    per_key = {}
    for key in {p["key"] for p in pts}:
        cs = []
        for th in thirds:
            sel = [p for p in th if p["key"] == key]
            if len(sel) < min_per_third:
                cs = []
                break
            cs.append(
                (
                    float(np.mean([s["x"] for s in sel])),
                    float(np.mean([s["y"] for s in sel])),
                    float(np.mean([s["t"] for s in sel])),
                )
            )
        if cs:
            per_key[key] = cs
    if not per_key:
        return {"ok": False, "reason": f"no key has >={min_per_third} samples in all three thirds"}

    rates, monotonic, rows = [], 0, []
    for key, cs in sorted(per_key.items()):
        v1 = (cs[1][0] - cs[0][0], cs[1][1] - cs[0][1])
        v2 = (cs[2][0] - cs[1][0], cs[2][1] - cs[1][1])
        mono = (v1[0] * v2[0] + v1[1] * v2[1]) > 0
        monotonic += int(mono)
        total = math.dist(cs[0][:2], cs[2][:2])
        minutes = (cs[2][2] - cs[0][2]) / 60.0
        rate = total / minutes if minutes > 1e-9 else float("nan")
        rates.append(rate)
        rows.append({"key": key, "rate_px_min": rate, "monotonic": mono, "total_px": total})
    finite = [r for r in rates if math.isfinite(r)]
    return {
        "ok": True,
        "keys": len(per_key),
        "median_rate_px_min": statistics.median(finite) if finite else float("nan"),
        "monotonic_keys": monotonic,
        "rows": rows,
        "span_min": (pts[-1]["t"] - pts[0]["t"]) / 60.0,
    }


# --- plot -----------------------------------------------------------------
def write_plot(path: Path, kbd_pts, desk_pts, keys: list[str], meta: dict) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return f"plot skipped ({exc})"
    fig, ax = plt.subplots(figsize=(11, 7))
    for key in keys:
        for pts, color in ((kbd_pts.get(key, []), "tab:blue"), (desk_pts.get(key, []), "tab:orange")):
            if pts:
                ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=8, alpha=0.35, c=color, linewidths=0)
        if kbd_pts.get(key) and desk_pts.get(key):
            k = np.mean([[p["x"], p["y"]] for p in kbd_pts[key]], axis=0)
            d = np.mean([[p["x"], p["y"]] for p in desk_pts[key]], axis=0)
            ax.annotate("", xy=d, xytext=k, arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
            ax.text(k[0], k[1], key, fontsize=11, weight="bold", ha="center", va="center")
    ax.scatter([], [], c="tab:blue", label="kbd")
    ax.scatter([], [], c="tab:orange", label="desk")
    ax.legend(loc="upper right")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Landing positions per key: keyboard vs bare desk (arrow = centroid shift)")
    if meta.get("width") and meta.get("height"):
        ax.set_xlim(0, meta["width"])
        ax.set_ylim(meta["height"], 0)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return None


# --- main -----------------------------------------------------------------
def analyze(kbd_dir: Path, desk_dir: Path) -> dict:
    kbd = load_session(kbd_dir, "kbd")
    desk = load_session(desk_dir, "desk")
    label_session(kbd)
    label_session(desk)

    kbd_pts, desk_pts = key_points(kbd), key_points(desk)
    kbd_stats = {k: summarise_key(k, v) for k, v in kbd_pts.items() if len(v) >= 2}
    desk_stats = {k: summarise_key(k, v) for k, v in desk_pts.items() if len(v) >= 2}
    pitch, pitch_note = calibrate_pitch(kbd_stats, kbd.meta.get("width"))

    qualifying = sorted(
        k
        for k in set(kbd_stats) & set(desk_stats)
        if kbd_stats[k].n >= MIN_SAMPLES_PER_KEY and desk_stats[k].n >= MIN_SAMPLES_PER_KEY
    )
    rows = []
    for k in qualifying:
        a, b = kbd_stats[k], desk_stats[k]
        shift = math.dist((a.cx, a.cy), (b.cx, b.cy))
        rows.append(
            {
                "key": k,
                "n_kbd": a.n,
                "n_desk": b.n,
                "shift_px": shift,
                "shift_pitch": shift / pitch if pitch else float("nan"),
                "var_kbd": a.var_px2,
                "var_desk": b.var_px2,
                "var_ratio": b.var_px2 / a.var_px2 if a.var_px2 > 0 else float("inf"),
                "spread_kbd": a.spread_px,
                "spread_desk": b.spread_px,
                "finger_kbd": a.finger,
                "finger_desk": b.finger,
                "finger_changed": a.finger != b.finger,
                "finger_share_kbd": a.finger_share,
                "finger_share_desk": b.finger_share,
            }
        )

    med_shift = statistics.median([r["shift_pitch"] for r in rows]) if rows else float("nan")
    finite = [r["var_ratio"] for r in rows if math.isfinite(r["var_ratio"])]
    med_ratio = statistics.median(finite) if finite else float("nan")
    inconclusive = len(rows) < MIN_QUALIFYING_KEYS
    passed = (
        not inconclusive
        and math.isfinite(med_shift)
        and math.isfinite(med_ratio)
        and med_shift < SHIFT_THRESHOLD_PITCH
        and med_ratio < VAR_RATIO_THRESHOLD
    )
    return {
        "kbd": kbd,
        "desk": desk,
        "kbd_pts": kbd_pts,
        "desk_pts": desk_pts,
        "pitch_px": pitch,
        "pitch_note": pitch_note,
        "rows": rows,
        "qualifying": qualifying,
        "median_shift_pitch": med_shift,
        "median_var_ratio": med_ratio,
        "verdict": "INCONCLUSIVE" if inconclusive else ("PASS" if passed else "FAIL"),
        "inconclusive": inconclusive,
        "drift_desk": drift_over_time(desk),
        "drift_kbd": drift_over_time(kbd),
    }


def _warnings(res: dict) -> list[str]:
    w = []
    k = res["kbd"].stats
    if k.get("rate_keys", 1) < 0.60:
        w.append(
            f"kbd: only {k['rate_keys']:.0%} of keystrokes paired to a tap "
            f"({k['paired']}/{k['keydowns']-k['excluded_keys']}) - tap detector or clock is suspect"
        )
    if k.get("rate_taps", 1) < 0.40:
        w.append(f"kbd: only {k['rate_taps']:.0%} of taps paired to a keystroke - detector firing on non-keystroke motion")
    d = res["desk"].stats
    if "phrases_total" in d:
        if d["phrases_used"] < 0.6 * max(d["phrases_total"], 1):
            w.append(
                f"desk: only {d['phrases_used']}/{d['phrases_total']} phrases aligned well enough to use - "
                f"the rest were dropped, not force-fitted"
            )
        if d["median_frac_confident"] < MIN_FRAC_CONFIDENT:
            w.append(f"desk: median only {d['median_frac_confident']:.0%} of characters confidently aligned")
    elif d.get("rate_keys", 1) < 0.60:
        w.append(f"desk: only {d['rate_keys']:.0%} of keystrokes paired (fallback pairing path)")
    if res["inconclusive"]:
        w.append(
            f"only {len(res['rows'])} keys reached >={MIN_SAMPLES_PER_KEY} samples in BOTH conditions "
            f"(need >={MIN_QUALIFYING_KEYS}); the medians below are not a result"
        )
    if res["pitch_note"].startswith(("ASSUMED", "FALLBACK", "WEAK")):
        w.append("key pitch: " + res["pitch_note"])
    return w


def print_report(res: dict) -> None:
    p = print
    p("=" * 78)
    p("PHASE 0 DRIFT ANALYSIS")
    p("=" * 78)
    for s in (res["kbd"], res["desk"]):
        st = s.stats
        p(f"[{s.condition:4}] {s.dir.name}  {s.duration_s/60:.1f} min, {len(s.taps)} taps")
        p(f"        labelling: {st['method']}")
        if "paired" in st:
            p(
                f"        keydowns {st['keydowns']} ({st['excluded_keys']} excluded) | paired {st['paired']} | "
                f"unpaired keys {st['unpaired_keys']} | unpaired taps {st['unpaired_taps']} | "
                f"key rate {st['rate_keys']:.0%}, tap rate {st['rate_taps']:.0%}"
            )
        else:
            p(
                f"        phrases {st['phrases_used']}/{st['phrases_total']} used | "
                f"median align cost {st['median_norm_cost']:.3f} | "
                f"median chars confidently aligned {st['median_frac_confident']:.0%} | "
                f"taps labelled {st['taps_labelled']}/{st['taps_total']}"
            )
            for idx, why in st["phrases_dropped"]:
                p(f"          dropped phrase {idx}: {why}")

    warn = _warnings(res)
    if warn:
        p("")
        p("!" * 78)
        for x in warn:
            p(f"!! WARNING: {x}")
        p("!" * 78)

    p("")
    p(f"Key pitch: {res['pitch_px']:.1f} px   [{res['pitch_note']}]")
    p("")
    if res["rows"]:
        p(f"PER-KEY (>= {MIN_SAMPLES_PER_KEY} labelled samples in both conditions)")
        p(
            f"{'key':>6} {'n_kbd':>6} {'n_desk':>7} {'shift_px':>9} {'shift_pitch':>12} "
            f"{'var_kbd':>9} {'var_desk':>9} {'ratio':>7}  finger kbd->desk"
        )
        for r in res["rows"]:
            flag = "  <== FINGER CHANGED" if r["finger_changed"] else ""
            p(
                f"{r['key']:>6} {r['n_kbd']:>6} {r['n_desk']:>7} {r['shift_px']:>9.1f} {r['shift_pitch']:>12.2f} "
                f"{r['var_kbd']:>9.1f} {r['var_desk']:>9.1f} {r['var_ratio']:>7.2f}  "
                f"{r['finger_kbd']}->{r['finger_desk']}{flag}"
            )
        changed = [r["key"] for r in res["rows"] if r["finger_changed"]]
        if changed:
            p(f"\nDominant finger changed on {len(changed)}/{len(res['rows'])} keys: {', '.join(changed)}")
    else:
        p("PER-KEY: no key qualified.")

    p("")
    p("WITHIN-SESSION DRIFT (session split into thirds)")
    for label, d in (("desk", res["drift_desk"]), ("kbd (control)", res["drift_kbd"])):
        if d["ok"]:
            p(
                f"  {label}: {d['keys']} keys over {d['span_min']:.1f} min, median "
                f"{d['median_rate_px_min']:.2f} px/min ({d['median_rate_px_min']/res['pitch_px']:.3f} pitch/min), "
                f"monotonic in {d['monotonic_keys']}/{d['keys']} keys"
            )
        else:
            p(f"  {label}: not measurable ({d['reason']})")

    p("")
    p("#" * 78)
    p("#  VERDICT - this is the decision criterion for whether Phase 2 proceeds")
    p("#" * 78)
    p(f"#  keys qualifying        : {len(res['rows'])} (need >= {MIN_QUALIFYING_KEYS})")
    p(f"#  median centroid shift  : {res['median_shift_pitch']:.3f} key pitch  (threshold < {SHIFT_THRESHOLD_PITCH})")
    p(f"#  median variance ratio  : {res['median_var_ratio']:.3f} desk/kbd     (threshold < {VAR_RATIO_THRESHOLD})")
    p("#")
    if res["verdict"] == "PASS":
        p("#  >>> PASS - per-key landing positions survive removing the keyboard. Phase 2 proceeds.")
    elif res["verdict"] == "FAIL":
        p("#  >>> FAIL - landing positions do NOT survive removing the keyboard. Phase 2 does NOT proceed.")
    else:
        p("#  >>> INCONCLUSIVE - too few qualifying keys. This is NOT a PASS and NOT a FAIL.")
        p("#      Collect a longer session or fix labelling, then re-run. Do not decide on this.")
    p("#" * 78)


def write_markdown(path: Path, res: dict) -> None:
    L: list[str] = []
    a = L.append
    a("# Phase 0 drift report\n")
    a(f"- kbd session: `{res['kbd'].dir}` - {res['kbd'].stats['method']}")
    a(f"- desk session: `{res['desk'].dir}` - {res['desk'].stats['method']}")
    a(f"- assignment solver: {'scipy linear_sum_assignment' if HAVE_SCIPY else 'greedy fallback (scipy missing)'}\n")
    a("## Verdict\n")
    a(f"**{res['verdict']}**\n")
    a("| metric | value | threshold |")
    a("|---|---|---|")
    a(f"| qualifying keys | {len(res['rows'])} | >= {MIN_QUALIFYING_KEYS} |")
    a(f"| median centroid shift | {res['median_shift_pitch']:.3f} pitch | < {SHIFT_THRESHOLD_PITCH} |")
    a(f"| median variance ratio (desk/kbd) | {res['median_var_ratio']:.3f} | < {VAR_RATIO_THRESHOLD} |")
    a("\nThis is the decision criterion for whether Phase 2 proceeds.\n")
    w = _warnings(res)
    if w:
        a("## Warnings\n")
        for x in w:
            a(f"- **{x}**")
        a("")
    a("## Labelling\n")
    ks = res["kbd"].stats
    a(
        f"- kbd: {ks['keydowns']} keydowns ({ks['excluded_keys']} excluded), {len(res['kbd'].taps)} taps, "
        f"{ks['paired']} paired, {ks['unpaired_keys']} unpaired keys, {ks['unpaired_taps']} unpaired taps"
    )
    ds = res["desk"].stats
    if "phrases_total" in ds:
        a(
            f"- desk: {ds['phrases_used']}/{ds['phrases_total']} phrases used, median alignment cost "
            f"{ds['median_norm_cost']:.3f}, median {ds['median_frac_confident']:.0%} of characters confidently "
            f"aligned, {ds['taps_labelled']}/{ds['taps_total']} taps labelled"
        )
        for idx, why in ds["phrases_dropped"]:
            a(f"  - dropped phrase {idx}: {why}")
    else:
        a(f"- desk: {ds['paired']} paired via fallback timestamp pairing")
    a(f"\n## Key pitch\n\n{res['pitch_px']:.1f} px - {res['pitch_note']}\n")
    a("## Per-key\n")
    a("| key | n kbd | n desk | shift px | shift (pitch) | var kbd | var desk | ratio | spread kbd | spread desk | finger kbd | finger desk | finger changed |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        a(
            f"| {r['key']} | {r['n_kbd']} | {r['n_desk']} | {r['shift_px']:.1f} | {r['shift_pitch']:.2f} | "
            f"{r['var_kbd']:.1f} | {r['var_desk']:.1f} | {r['var_ratio']:.2f} | {r['spread_kbd']:.1f} | "
            f"{r['spread_desk']:.1f} | {r['finger_kbd']} ({r['finger_share_kbd']:.0%}) | "
            f"{r['finger_desk']} ({r['finger_share_desk']:.0%}) | {'YES' if r['finger_changed'] else ''} |"
        )
    a("\n## Within-session drift\n")
    for label, d in (("desk", res["drift_desk"]), ("kbd (control)", res["drift_kbd"])):
        if d["ok"]:
            a(
                f"- **{label}**: {d['keys']} keys over {d['span_min']:.1f} min, median "
                f"{d['median_rate_px_min']:.2f} px/min ({d['median_rate_px_min']/res['pitch_px']:.3f} pitch/min), "
                f"monotonic in {d['monotonic_keys']}/{d['keys']} keys"
            )
        else:
            a(f"- **{label}**: not measurable ({d['reason']})")
    a("\nPublished invisible-keyboard work reports ~1.32 mm/min drift vs ~0.25 mm/min with a visible keyboard; one key pitch = 19 mm, so convert with the calibrated pitch above.\n")
    a("![kbd vs desk landing positions](drift_scatter.png)\n")
    path.write_text("\n".join(L))


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0 keyboard-vs-desk landing drift analysis")
    ap.add_argument("--kbd", required=True, type=Path)
    ap.add_argument("--desk", required=True, type=Path)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        res = analyze(args.kbd, args.desk)
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Refusing to produce a verdict from unusable input.", file=sys.stderr)
        return 2
    print_report(res)
    write_markdown(args.desk / "drift_report.md", res)
    if not args.no_plot:
        err = write_plot(
            args.desk / "drift_scatter.png",
            res["kbd_pts"],
            res["desk_pts"],
            res["qualifying"] or sorted(set(res["kbd_pts"]) & set(res["desk_pts"])),
            res["desk"].meta,
        )
        if err:
            print(err, file=sys.stderr)
    print(f"\nwrote {args.desk/'drift_report.md'}")
    if not args.no_plot:
        print(f"wrote {args.desk/'drift_scatter.png'}")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
