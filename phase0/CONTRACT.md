# Phase 0 data contracts — FROZEN. Do not change without updating every module.

Goal: measure whether per-key fingertip landing positions survive removing the keyboard.

## Clock rule (critical)
ONE process captures video AND logs keystrokes, stamping both with `time.monotonic()`.
Never correlate across processes. `t` everywhere below is that float, seconds.

## Session layout
data/sessions/<session_id>/          session_id = "<YYYYmmdd-HHMMSS>-<kbd|desk>"
  meta.json
  video.mp4                          H.264, as captured
  frames.jsonl                       one row per written frame
  keys.jsonl                         one row per key event
  phrases.jsonl                      desk sessions only
  landmarks.parquet                  derived
  taps.jsonl                         derived

## meta.json
{"session_id": str, "condition": "kbd"|"desk", "started_at_iso": str,
 "t0_monotonic": float, "camera_name": str, "width": int, "height": int,
 "fps_requested": float, "fps_actual": float, "notes": str}

## frames.jsonl   {"i": int, "t": float}
`i` is 0-based and matches the frame index in video.mp4 exactly.

## keys.jsonl     {"t": float, "event": "down"|"up", "key": str}
`key` is a normalized name: single lowercase char for letters/digits,
else one of: space enter tab backspace shift ctrl alt cmd esc unknown.
PRIVACY: only keys in the allowlist are recorded; everything else -> "unknown".

## phrases.jsonl  {"t": float, "event": "shown"|"done", "phrase": str, "idx": int}

## landmarks.parquet
columns: i(int32) t(float64) hand(int8: 0|1) handedness(str "Left"|"Right")
         joint(int8 0-20) x(float32 px) y(float32 px) conf(float32)
Origin top-left, pixels, in the video's own resolution.
MediaPipe joint numbering. Fingertips are joints 4,8,12,16,20.

## taps.jsonl     {"t": float, "hand": int, "finger": int, "x": float, "y": float,
                   "conf": float, "i": int}
`finger` is the fingertip joint id (4|8|12|16|20). x,y in px.
A tap = fingertip vertical velocity zero-crossing (descent -> ascent) with rebound.

## Pairing (analysis)
A keystroke at t_k pairs with the tap minimizing |t_tap - t_k| within +/-80 ms.
Unpaired keystrokes and unpaired taps are both reported as counts.

## The output that decides Phase 0
For each key present in BOTH conditions with >= 15 paired samples:
  centroid shift (px and in key-pitch units), and variance ratio desk/kbd.
DECISION: idea is alive if median centroid shift < 0.5 key pitch
          AND median variance ratio < 2.0.
Key pitch is calibrated from the kbd session: median distance between the
centroids of horizontally adjacent home-row keys.

## AMENDMENT 1 — labelling the desk session (added after review)
PROBLEM: in the desk condition there is no keyboard, so NO key events are
produced. `keys.jsonl` for a desk session is empty or near-empty. The
timestamp-pairing rule above therefore applies to the **kbd session only**.

DESK SESSION GROUND TRUTH:
The operator transcribes phrases from `phase0/phrases.txt`, presented one at a
time by the prompter, which appends to `phrases.jsonl`. The intended character
sequence for a phrase is therefore known a priori.

LABELLING RULE (desk): within each phrase window (from its "shown" t to its
"done" t), take the detected taps in time order and align them to the phrase's
character sequence with a MONOTONIC alignment (needleman-wunsch / DTW with
insert+delete penalties). Do NOT assume a 1:1 match — tap detection will miss
and hallucinate. Report the alignment cost and the fraction of characters
confidently aligned; a poor alignment invalidates that phrase and it must be
dropped, not force-fitted.

Spaces count as taps (thumb). Backspaces cannot be observed on a desk, so
instruct the operator NOT to correct errors; a phrase typed with a visible
mistake is dropped at analysis time.

CONSEQUENCE FOR analyze_drift: it needs TWO labelling paths — timestamp pairing
for kbd, sequence alignment for desk — producing the same (key, x, y) records
that the per-key statistics consume.
