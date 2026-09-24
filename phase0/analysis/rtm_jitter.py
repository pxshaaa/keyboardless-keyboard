"""Still-hand jitter, hand width and coverage: landmarks.parquet (MediaPipe) vs landmarks_rtm.parquet (RTMPose).
Still = no key event within +-300 ms; jitter = fingertip frame-to-frame displacement (px) on consecutive still frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TIPS = (4, 8, 12, 16, 20)
STILL_S = 0.3


def hands_by_frame(parquet: Path) -> dict[tuple[int, str], np.ndarray]:
    """{(frame, handedness): [21, 2]}; frames where one label occurs twice are dropped for that label."""
    tb = pq.read_table(parquet, columns=["i", "handedness", "joint", "x", "y"])
    i = tb.column("i").to_numpy()
    lab = np.asarray(tb.column("handedness").to_pylist(), dtype=object)
    j = tb.column("joint").to_numpy().astype(int)
    x, y = tb.column("x").to_numpy(), tb.column("y").to_numpy()
    out, dup = {}, set()
    for k in range(0, len(i), 21):
        key = (int(i[k]), str(lab[k]))
        if not (j[k:k + 21] == np.arange(21)).all():
            continue
        if key in out:
            dup.add(key)
        out[key] = np.stack([x[k:k + 21], y[k:k + 21]], -1).astype(np.float64)
    for key in dup:
        out.pop(key, None)
    return out


def summary(d: np.ndarray) -> dict:
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "rms": float(np.sqrt(np.mean(d ** 2))), "mean": float(d.mean())}


def jitter(hands: dict, still: np.ndarray, frames_common: set | None = None) -> dict:
    d_all, per_tip, width = [], {t: [] for t in TIPS}, []
    for (fi, lab), P in hands.items():
        width.append(np.hypot(*(P[5] - P[17])))
        if not (still[fi] and fi + 1 < len(still) and still[fi + 1]):
            continue
        if frames_common is not None and (fi, lab) not in frames_common:
            continue
        Q = hands.get((fi + 1, lab))
        if Q is None:
            continue
        d = np.hypot(*(Q[TIPS, :] - P[TIPS, :]).T)
        d_all.append(d)
        for t, v in zip(TIPS, d):
            per_tip[t].append(v)
    d_all = np.concatenate(d_all) if d_all else np.zeros(0)
    return {"tips_all": summary(d_all), "per_tip": {str(t): summary(np.array(v)) for t, v in per_tip.items()},
            "hand_width_5_17_px": summary(np.array(width))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = a.session_dir
    frames = [json.loads(l) for l in open(s / "frames.jsonl")]
    ft = np.array([r["t"] for r in frames])
    kt = np.array([json.loads(l)["t"] for l in open(s / "keys.jsonl")]) if (s / "keys.jsonl").exists() else np.zeros(0)
    if len(kt):
        pos = np.searchsorted(kt, ft)
        near = np.minimum(np.abs(ft - kt[np.clip(pos, 0, len(kt) - 1)]), np.abs(ft - kt[np.clip(pos - 1, 0, len(kt) - 1)]))
        still = near > STILL_S
    else:
        still = np.ones(len(ft), bool)
    src = {"mediapipe": hands_by_frame(s / "landmarks.parquet"), "rtmpose": hands_by_frame(s / "landmarks_rtm.parquet")}
    common = set(src["mediapipe"]) & set(src["rtmpose"])
    common = {k for k in common if (k[0] + 1, k[1]) in common}
    res = {"session": s.name, "frames": len(ft), "still_frames": int(still.sum()), "key_events": int(len(kt))}
    for name, hands in src.items():
        r = jitter(hands, still)
        r["common_pairs"] = jitter(hands, still, common)["tips_all"]
        r["coverage_hands_per_frame"] = len(hands) / len(ft)
        r["frames_with_2_hands"] = float(np.mean(np.bincount([k[0] for k in hands], minlength=len(ft)) >= 2))
        res[name] = r
    st = s / "landmarks_rtm_stats.json"
    if st.exists():
        res["rtmpose"]["extraction"] = json.loads(st.read_text())
    for name in src:
        r = res[name]
        print(f"{name:<10} tips median {r['tips_all']['median']:.2f} p90 {r['tips_all']['p90']:.2f} rms {r['tips_all']['rms']:.2f} px "
              f"(n {r['tips_all']['n']}) | common median {r['common_pairs']['median']:.2f} rms {r['common_pairs']['rms']:.2f} "
              f"| width {r['hand_width_5_17_px']['median']:.1f} px | hands/frame {r['coverage_hands_per_frame']:.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
