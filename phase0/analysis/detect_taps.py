"""Detect keystroke taps from 2D fingertip trajectories -- see phase0/CONTRACT.md.
Run: python -m phase0.analysis.detect_taps data/sessions/<session_id>"""

# COORDINATE CONVENTION: image coords, origin top-left, y increases DOWNWARD.
# So descending toward the desk is dy/dt > 0; lifting off is dy/dt < 0.

# A tap is therefore a POSITIVE-TO-NEGATIVE zero crossing of dy/dt (local MAX of y).
# Inverting this sign detects LIFTS, still gives a plausible rate, and is silently wrong.

# test_detect_taps.py pins the sign with an explicit "a lift is not a tap" case.
# No depth: keystroke z-excursion is ~1-3 mm and LiDAR noise ~cm, so kinematics only.

# SMOOTHING: Savitzky-Golay over One Euro, because a local polynomial fit has an
# analytic derivative -- differentiating a separately-smoothed signal re-adds noise.

# SavGol also preserves the TIME LOCATION of extrema, and the tap instant IS an
# extremum, so extremum-location bias is this experiment's measurement error.

# SavGol is zero-phase; One Euro is causal with speed-dependent lag that would smear
# tap times, and its low-latency advantage is moot for offline analysis.

# TRADEOFF: a tap's down-up excursion is ~100-150 ms = 6-9 samples at 60 Hz.
# Window too long -> descent and rebound cancel, the crossing drifts and fast taps merge.

# Window too short -> jitter survives, dy/dt rings around zero, spurious crossings.
# Defaults 7 samples / poly 2, tunable via --smooth-window/--smooth-poly; sweep vs --plot.

# Explicit per-finger loops, not one vectorized pass: silent wrongness is easy here,
# so every rejection reason stays countable and is printed in the summary.

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

FINGERTIP_JOINTS = (4, 8, 12, 16, 20)  # MediaPipe thumb/index/middle/ring/pinky

DEFAULT_SMOOTH_WINDOW = 7  # samples (~117 ms @60 Hz), odd, must fit inside one tap
DEFAULT_SMOOTH_POLY = 2  # quadratic; higher orders re-admit jitter into dy/dt

DEFAULT_POS_WINDOW = 5
# The DERIVATIVE needs more smoothing than the POSITION does, so x,y at contact
# get their own shorter window: 7/2 attenuates the peak 2.4 px, 5/2 only 0.7 px.

# px/s in the VIDEO'S OWN RESOLUTION: re-tune if frame size or camera distance
# changes. Values below assume 1080p, ~40 cm above desk, ~15-40 px per keystroke.

DEFAULT_MIN_DESCENT_VEL = 60.0
# The finger must have had downward momentum. Rejects resting fingers, slow
# repositioning and hover wobble, which cross zero constantly. Most important gate.

DEFAULT_MIN_REBOUND_VEL = 40.0
# It must come back up: rejects a hold (settling onto a key and staying).
# Lower than descent on purpose -- release is less forceful than the strike.

DEFAULT_MIN_REBOUND_PX = 1.5
# ...and cover real distance, not a one-sample negative blip from noise.

DEFAULT_VEL_WINDOW = 0.060
# s, half-width searched for peak descent/rebound. ~half a tap: long enough to
# contain the velocity extremum, short enough not to reach the neighbouring tap.

DEFAULT_REFRACTORY = 0.080
# s. One finger cannot strike twice this fast (12.5 strokes/s); anything closer
# is one physical tap detected twice. Per (hand, finger) only -- rollover is real.

DEFAULT_MIN_CONF = 0.50
# Below this MediaPipe is guessing (usually self-occlusion from top-down) and
# its "velocity" is interpolation artifact, not motion.

DEFAULT_MAX_GAP = 0.050
# s. A larger sample gap means the hand was missing. We cut the series rather
# than interpolate: never invent a velocity across a dropout and call it a key.

MIN_SEGMENT_SAMPLES = 5  # shorter cannot support a window plus a derivative


@dataclass
class Params:
    smooth_window: int = DEFAULT_SMOOTH_WINDOW
    smooth_poly: int = DEFAULT_SMOOTH_POLY
    pos_window: int = DEFAULT_POS_WINDOW
    min_descent_vel: float = DEFAULT_MIN_DESCENT_VEL
    min_rebound_vel: float = DEFAULT_MIN_REBOUND_VEL
    min_rebound_px: float = DEFAULT_MIN_REBOUND_PX
    vel_window: float = DEFAULT_VEL_WINDOW
    refractory: float = DEFAULT_REFRACTORY
    min_conf: float = DEFAULT_MIN_CONF
    max_gap: float = DEFAULT_MAX_GAP


@dataclass
class Tap:
    t: float
    hand: int
    finger: int
    x: float
    y: float
    conf: float
    i: int
    # diagnostics only; taps.jsonl is frozen and must not carry them
    descent_vel: float = 0.0
    rebound_vel: float = 0.0
    rebound_px: float = 0.0

    @property
    def score(self) -> float:
        # NMS strength: a real strike is strong on BOTH sides, so a one-sided
        # event loses to the true crossing beside it.
        return min(self.descent_vel, self.rebound_vel)

    def to_record(self) -> dict:
        return {
            "t": float(self.t),
            "hand": int(self.hand),
            "finger": int(self.finger),
            "x": float(self.x),
            "y": float(self.y),
            "conf": float(self.conf),
            "i": int(self.i),
        }


@dataclass
class Rejections:
    """Every zero crossing lands in exactly one bucket, so a wrong tap count
    points straight at the threshold to move."""

    too_short_segment: int = 0
    weak_descent: int = 0
    weak_rebound_vel: int = 0
    weak_rebound_px: int = 0
    low_conf: int = 0
    refractory: int = 0
    accepted: int = 0
    crossings: int = 0

    def as_dict(self) -> dict:
        return {
            "crossings_examined": self.crossings,
            "accepted": self.accepted,
            "rejected_low_conf": self.low_conf,
            "rejected_weak_descent": self.weak_descent,
            "rejected_weak_rebound_vel": self.weak_rebound_vel,
            "rejected_weak_rebound_px": self.weak_rebound_px,
            "rejected_refractory": self.refractory,
            "skipped_short_segments": self.too_short_segment,
        }


def _effective_window(n: int, window: int, poly: int) -> int | None:
    """Largest usable odd window <= `window` fitting n samples; None if none fits.
    Shrinking rather than failing matters for short inter-dropout segments."""
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return None
    return w


def smooth_and_differentiate(
    t: np.ndarray, v: np.ndarray, window: int, poly: int
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed v and dv/dt for ONE contiguous segment, sampled at t."""
    # Frame timing jitters, so the segment's MEDIAN dt is SavGol's `delta`;
    # resampling to a uniform grid would bias the very turnaround we measure.
    n = len(v)
    dts = np.diff(t)
    dt = float(np.median(dts)) if len(dts) else 1.0
    if dt <= 0 or not math.isfinite(dt):
        dt = 1.0

    w = _effective_window(n, window, poly)
    if w is None:
        # Too short to filter: raw + finite differences. The kinematic gates
        # then reject nearly all of it, which is the conservative outcome.
        smooth = v.astype(np.float64, copy=True)
        vel = np.gradient(smooth, t) if n >= 2 else np.zeros(n)
        return smooth, vel

    smooth = savgol_filter(v.astype(np.float64), w, poly, deriv=0, mode="interp")
    vel = savgol_filter(v.astype(np.float64), w, poly, deriv=1, delta=dt, mode="interp")
    return smooth, vel


def split_segments(
    t: np.ndarray, conf: np.ndarray, max_gap: float, min_conf: float
) -> list[tuple[int, int]]:
    """Contiguous [start, stop) ranges of trustworthy tracking; breaks on a
    time gap > max_gap (hand missing) or conf < min_conf (tracker guessing)."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for k in range(len(t)):
        ok = conf[k] >= min_conf
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            segments.append((start, k))
            start = None
        elif ok and start is not None and k > start:
            if (t[k] - t[k - 1]) > max_gap:
                segments.append((start, k))
                start = k
    if start is not None:
        segments.append((start, len(t)))
    return segments


def _window_indices(t: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Half-open index range of samples with lo <= t < hi; t must be sorted."""
    return int(np.searchsorted(t, lo, "left")), int(np.searchsorted(t, hi, "right"))


def find_candidates(
    t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    conf: np.ndarray,
    frame_i: np.ndarray,
    vel: np.ndarray,
    hand: int,
    finger: int,
    p: Params,
    rej: Rejections,
) -> list[Tap]:
    """Candidate taps in one segment of one (hand, finger) series.
    vel is dy/dt in image coords, so vel > 0 means moving DOWN (see module doc)."""
    out: list[Tap] = []
    n = len(t)

    for k in range(n - 1):
        # the crossing itself: descending, then not
        if not (vel[k] > 0.0 and vel[k + 1] <= 0.0):
            continue
        rej.crossings += 1

        # Sub-sample contact instant: frame quantisation is 16.7 ms against a
        # +/-80 ms pairing window in the contract, so this is not cosmetic.
        denom = vel[k] - vel[k + 1]
        frac = float(vel[k] / denom) if denom > 0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        t_c = float(t[k] + frac * (t[k + 1] - t[k]))

        # x,y AT CONTACT is what the whole experiment measures: interpolated at
        # the crossing, and smoothed, rather than snapped to the nearest frame.
        x_c = float(xs[k] + frac * (xs[k + 1] - xs[k]))
        y_c = float(ys[k] + frac * (ys[k + 1] - ys[k]))
        conf_c = float(conf[k] + frac * (conf[k + 1] - conf[k]))
        i_c = int(frame_i[k] if frac < 0.5 else frame_i[k + 1])

        # redundant with split_segments, but makes the failure mode countable
        if conf_c < p.min_conf:
            rej.low_conf += 1
            continue

        # gate 1: was it actually coming down?
        lo, hi = _window_indices(t, t_c - p.vel_window, t_c)
        pre = vel[lo : max(hi, k + 1)]
        descent = float(pre.max()) if pre.size else float(vel[k])
        if descent < p.min_descent_vel:
            rej.weak_descent += 1
            continue

        # gate 2: did it bounce back?
        lo2, hi2 = _window_indices(t, t_c, t_c + p.vel_window)
        post = vel[min(lo2, k + 1) : hi2]
        rebound = float(-post.min()) if post.size else float(-vel[k + 1])
        if rebound < p.min_rebound_vel:
            rej.weak_rebound_vel += 1
            continue

        # gate 3: over real distance. Rising means y DECREASES, hence y_c - min.
        post_y = ys[min(lo2, k + 1) : hi2]
        rebound_px = float(y_c - post_y.min()) if post_y.size else 0.0
        if rebound_px < p.min_rebound_px:
            rej.weak_rebound_px += 1
            continue

        out.append(
            Tap(
                t=t_c,
                hand=hand,
                finger=finger,
                x=x_c,
                y=y_c,
                conf=conf_c,
                i=i_c,
                descent_vel=descent,
                rebound_vel=rebound,
                rebound_px=rebound_px,
            )
        )
    return out


def apply_refractory(taps: list[Tap], refractory: float, rej: Rejections) -> list[Tap]:
    """Greedy NMS by score, not "keep the earliest": velocity can dip through
    zero twice at the turnaround, and keeping the first biases contacts early."""
    accepted: list[Tap] = []
    for tap in sorted(taps, key=lambda z: (-z.score, z.t)):
        if any(abs(tap.t - a.t) < refractory for a in accepted):
            rej.refractory += 1
            continue
        accepted.append(tap)
    accepted.sort(key=lambda z: z.t)
    return accepted


def detect_taps_series(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conf: np.ndarray | None = None,
    frame_i: np.ndarray | None = None,
    hand: int = 0,
    finger: int = 8,
    params: Params | None = None,
    rej: Rejections | None = None,
) -> list[Tap]:
    """Taps in one (hand, finger) series, sorted by time.
    t ascending; y in image coords (increases downward)."""
    p = params or Params()
    rej = rej if rej is not None else Rejections()

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    conf = np.ones_like(t) if conf is None else np.asarray(conf, dtype=np.float64)
    frame_i = np.arange(len(t)) if frame_i is None else np.asarray(frame_i)

    found: list[Tap] = []
    for start, stop in split_segments(t, conf, p.max_gap, p.min_conf):
        if stop - start < MIN_SEGMENT_SAMPLES:
            rej.too_short_segment += 1
            continue
        ts = t[start:stop]
        # vy (detection) comes from the wider window; xs/ys (the reported contact
        # point, the experiment's actual measurement) from the narrower one.
        _, vy = smooth_and_differentiate(ts, y[start:stop], p.smooth_window, p.smooth_poly)
        ys, _ = smooth_and_differentiate(ts, y[start:stop], p.pos_window, p.smooth_poly)
        xs, _ = smooth_and_differentiate(ts, x[start:stop], p.pos_window, p.smooth_poly)
        found.extend(
            find_candidates(
                ts, xs, ys, conf[start:stop], frame_i[start:stop], vy,
                hand, finger, p, rej,
            )
        )

    kept = apply_refractory(found, p.refractory, rej)
    rej.accepted += len(kept)
    return kept


def load_landmarks(session: Path):
    import pyarrow.parquet as pq

    path = session / "landmarks.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run extract_landmarks first")
    return pq.read_table(path)


def detect_session(session: Path, p: Params) -> tuple[list[Tap], Rejections]:
    table = load_landmarks(session)
    cols = {name: np.asarray(table.column(name)) for name in
            ("i", "t", "hand", "joint", "x", "y", "conf")}

    rej = Rejections()
    taps: list[Tap] = []
    for hand in sorted({int(h) for h in cols["hand"]}):
        for finger in FINGERTIP_JOINTS:
            m = (cols["hand"] == hand) & (cols["joint"] == finger)
            if not m.any():
                continue
            order = np.argsort(cols["t"][m], kind="stable")
            taps.extend(
                detect_taps_series(
                    t=cols["t"][m][order],
                    x=cols["x"][m][order],
                    y=cols["y"][m][order],
                    conf=cols["conf"][m][order],
                    frame_i=cols["i"][m][order],
                    hand=hand,
                    finger=finger,
                    params=p,
                    rej=rej,
                )
            )
    taps.sort(key=lambda z: (z.t, z.hand, z.finger))
    return taps, rej


def write_taps(session: Path, taps: list[Tap]) -> Path:
    out = session / "taps.jsonl"
    with out.open("w") as fh:
        for tap in taps:
            fh.write(json.dumps(tap.to_record()) + "\n")
    return out


def count_keystrokes(session: Path) -> int | None:
    """Key-DOWN events only: a press plus its release is one tap, not two."""
    path = session / "keys.jsonl"
    if not path.exists():
        return None
    n = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("event") == "down":
                    n += 1
            except json.JSONDecodeError:
                continue
    return n


FINGER_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}


def print_summary(session: Path, taps: list[Tap], rej: Rejections) -> None:
    print(f"session      : {session}")
    print(f"taps         : {len(taps)}")

    if taps:
        span = taps[-1].t - taps[0].t
        rate = len(taps) / span if span > 0 else float("nan")
        print(f"span         : {span:.2f} s")
        print(f"taps/sec     : {rate:.2f}")
    else:
        print("taps/sec     : n/a")

    print("\nper finger (hand, joint):")
    per: dict[tuple[int, int], int] = {}
    for tap in taps:
        per[(tap.hand, tap.finger)] = per.get((tap.hand, tap.finger), 0) + 1
    if not per:
        print("  (none)")
    for (hand, finger), n in sorted(per.items()):
        print(f"  hand {hand}  {finger:>2} {FINGER_NAMES.get(finger, '?'):<6} {n:>6}")

    print("\ncrossing accounting:")
    for key, val in rej.as_dict().items():
        print(f"  {key:<28} {val:>6}")

    keys = count_keystrokes(session)
    if keys is None:
        print("\nkeys.jsonl   : absent (no keystroke cross-check available)")
        return
    print(f"\nkeystrokes   : {keys}")
    if keys == 0:
        print("ratio        : n/a (no keystrokes)")
        return

    # Best sanity signal available: one keystroke should produce ~one tap.
    # Far below 1 = gates too strict or sign inverted; far above 1 = jitter.
    ratio = len(taps) / keys
    print(f"taps/keystroke: {ratio:.2f}", end="  ")
    if 0.8 <= ratio <= 1.25:
        print("[healthy]")
    elif ratio < 0.8:
        print("[LOW -- missing taps: lower --min-descent-vel/--min-rebound-vel"
              " or shorten --smooth-window]")
    else:
        print("[HIGH -- spurious taps: raise the velocity gates"
              " or lengthen --smooth-window]")


def plot_finger(session: Path, taps: list[Tap], hand: int, finger: int, p: Params) -> Path:
    """y(t) plus detected taps for one finger, for eyeballing sanity.
    The y axis is INVERTED so "toward the desk" is down on the page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = load_landmarks(session)
    cols = {n: np.asarray(table.column(n)) for n in ("t", "hand", "joint", "y", "conf")}
    m = (cols["hand"] == hand) & (cols["joint"] == finger)
    if not m.any():
        raise SystemExit(f"no samples for hand={hand} joint={finger}")
    order = np.argsort(cols["t"][m], kind="stable")
    t, y, conf = cols["t"][m][order], cols["y"][m][order], cols["conf"][m][order]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t, y, lw=0.6, color="0.7", label="raw y")

    for start, stop in split_segments(t, conf, p.max_gap, p.min_conf):
        if stop - start < MIN_SEGMENT_SAMPLES:
            continue
        _, vy = smooth_and_differentiate(t[start:stop], y[start:stop],
                                         p.smooth_window, p.smooth_poly)
        ys, _ = smooth_and_differentiate(t[start:stop], y[start:stop],
                                         p.pos_window, p.smooth_poly)
        ax.plot(t[start:stop], ys, lw=1.2, color="C0")
        ax2.plot(t[start:stop], vy, lw=0.9, color="C0")

    sel = [z for z in taps if z.hand == hand and z.finger == finger]
    if sel:
        ax.plot([z.t for z in sel], [z.y for z in sel], "v", color="C3",
                ms=7, ls="none", label=f"taps (n={len(sel)})")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhline(p.min_descent_vel, color="C2", lw=0.6, ls="--", label="descent gate")
    ax2.axhline(-p.min_rebound_vel, color="C1", lw=0.6, ls="--", label="rebound gate")

    ax.invert_yaxis()
    ax.set_ylabel("y (px, image coords -- DOWN is down)")
    ax.set_title(f"{session.name}  hand={hand} joint={finger} "
                 f"({FINGER_NAMES.get(finger, '?')})  tap = dy/dt crosses + -> -")
    ax.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel("dy/dt (px/s)\n+ = descending")
    ax2.set_xlabel("t (s, monotonic)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = session / f"taps_plot_hand{hand}_joint{finger}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m phase0.analysis.detect_taps",
        description="Detect fingertip taps (dy/dt zero crossing, descent -> rebound) "
                    "from landmarks.parquet. y increases DOWNWARD (image coords).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("session", type=Path, help="data/sessions/<session_id>")
    ap.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                    help="SavGol window in SAMPLES (odd); must fit inside one tap")
    ap.add_argument("--smooth-poly", type=int, default=DEFAULT_SMOOTH_POLY,
                    help="SavGol polynomial order for both filters")
    ap.add_argument("--pos-window", type=int, default=DEFAULT_POS_WINDOW,
                    help="SavGol window in SAMPLES (odd) for the reported x,y at contact")
    ap.add_argument("--min-descent-vel", type=float, default=DEFAULT_MIN_DESCENT_VEL,
                    help="px/s downward required before contact")
    ap.add_argument("--min-rebound-vel", type=float, default=DEFAULT_MIN_REBOUND_VEL,
                    help="px/s upward required after contact")
    ap.add_argument("--min-rebound-px", type=float, default=DEFAULT_MIN_REBOUND_PX,
                    help="px the fingertip must rise after contact")
    ap.add_argument("--vel-window", type=float, default=DEFAULT_VEL_WINDOW,
                    help="s; half-width searched for peak descent/rebound velocity")
    ap.add_argument("--refractory", type=float, default=DEFAULT_REFRACTORY,
                    help="s; minimum spacing between taps of the SAME finger")
    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF,
                    help="reject landmarks below this tracker confidence")
    ap.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP,
                    help="s; a larger sample gap means the hand was missing")
    ap.add_argument("--plot", action="store_true", help="save a PNG for one finger")
    ap.add_argument("--plot-hand", type=int, default=0)
    ap.add_argument("--plot-finger", type=int, default=8, choices=FINGERTIP_JOINTS,
                    help="fingertip joint to plot (8 = index)")
    ap.add_argument("--dry-run", action="store_true",
                    help="summarise without writing taps.jsonl")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for name, w in (("--smooth-window", args.smooth_window), ("--pos-window", args.pos_window)):
        if w % 2 == 0:
            raise SystemExit(f"{name} must be odd (SavGol requires it)")
        if w <= args.smooth_poly:
            raise SystemExit(f"{name} must exceed --smooth-poly")

    session = args.session
    if not session.is_dir():
        raise SystemExit(f"not a session directory: {session}")

    p = Params(
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        pos_window=args.pos_window,
        min_descent_vel=args.min_descent_vel,
        min_rebound_vel=args.min_rebound_vel,
        min_rebound_px=args.min_rebound_px,
        vel_window=args.vel_window,
        refractory=args.refractory,
        min_conf=args.min_conf,
        max_gap=args.max_gap,
    )

    taps, rej = detect_session(session, p)
    if not args.dry_run:
        print(f"wrote        : {write_taps(session, taps)}")
    print_summary(session, taps, rej)
    if args.plot:
        out = plot_finger(session, taps, args.plot_hand, args.plot_finger, p)
        print(f"\nplot         : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
