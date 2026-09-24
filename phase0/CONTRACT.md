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

## AMENDMENT 2 — depth columns in landmarks.parquet (additive)
Appended columns: `z` (MediaPipe image-space depth, px-scaled like x, wrist=0),
`wx wy wz` (metric world landmarks, metres, hand-centred). NaN when unavailable.
Existing readers that select columns by name are unaffected.

## AMENDMENT 3 — desk alignment cost model (analyze_drift only; thresholds untouched)

The PASS thresholds (0.5 key pitch, 2.0 variance ratio, >= 8 keys, >= 15 samples) and
`MIN_FRAC_CONFIDENT = 0.50` are UNCHANGED. What changed is how `align_phrase` scores an
alignment, and it was chosen on **keyboard** sessions only, where `keys.jsonl` gives the true
character<->tap map by timestamp pairing. The desk session was never consulted.

MEASURED PROBLEM. On desk-length segments (25-60 chars, cut on word boundaries) of
`20260910-131629-kbd`, the confident set contained **4 non-space characters out of 1270**:
`GAP_DELETE + GAP_INSERT = 1.0` was cheaper than almost any imperfect match (a positionally
correct match with disagreeing key evidence costs up to `W_KEY = 2.0`), so the DP deleted
characters instead of aligning them, and `_mark_confident`'s gap-free-word requirement then
had no gap-free words to find. The rule was reported as "unreachable at 73-91% detector
recall"; the binding constraint was actually the cost model, not the recall.

CHANGES
1. `GAP_DELETE = GAP_INSERT = 1.5` (was 0.5). Invariant: a gap pair must cost more than the
   most expensive observational disagreement, so evidence decides matches and position
   decides gaps.
2. `MAX_NORM_COST` is now `1.2 * GAP_DELETE` instead of the literal `0.60`. 0.60 was exactly
   1.2 gap-units under the old gap cost, so this preserves the gate's meaning rather than
   loosening it. NOTE: on 28 kbd segments this gate's correlation with per-segment alignment
   accuracy is +0.23 (i.e. none). It is not a quality filter; do not rely on it.
3. Tap position in the match cost is normalized RANK, not normalized ELAPSED TIME. Elapsed
   time assumes a uniform typing rate, so ordinary rhythm read as a missing tap.
4. `_mark_confident` is UNCHANGED. Deletion-tolerant variants (allow k missed taps per word;
   local gap-free window; key-posterior margin) were all measured and all sit BELOW the
   gap-free-word rule on the precision/coverage frontier — they buy coverage by admitting
   wrong labels. No relaxation was adopted.

GROUND-TRUTH JUSTIFICATION (confident-set precision = fraction of confident labels that are
truly correct; coverage = confident labels / characters):

| session | taps | before (cov/prec) | after (cov/prec) |
|---|---|---|---|
| 20260910-131629-kbd | taps_pos | 0.177 / 0.569 | 0.363 / 0.813 |
| 20260910-015948-kbd | taps_pos | 0.161 / 0.714 | 0.218 / 0.816 |
| 20260910-021315-kbd | taps_gb  | 0.080 / 0.327 | 0.122 / 0.506 |

Strictly better on BOTH axes in all three sessions. That dominance, not any desk-session
outcome, is why the change was kept.

## AMENDMENT 4 — what the detector must reach for the desk path to be usable

Measured by synthesising a detector at a controlled recall from `20260910-131629-kbd` ground
truth (one ideal tap per character, kept with probability r, plus hallucinated taps drawn
from the session's real unpaired taps at rate f per character), then running the alignment
and confidence rule above. Current detector on that session: r = 0.837, f = 0.148.

Median `frac_confident` / confident-set precision:

| r \ f | 0.00 | 0.02 | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| 0.85 | 0.37/0.92 | 0.39/0.90 | 0.38/0.85 | 0.35/0.81 | 0.36/0.72 |
| 0.93 | 0.57/0.98 | 0.57/0.96 | 0.53/0.93 | 0.48/0.84 | 0.42/0.81 |
| 0.97 | 0.70/0.99 | 0.68/0.98 | 0.60/0.95 | 0.50/0.91 | 0.42/0.86 |
| 1.00 | 0.82/1.00 | 0.76/0.99 | 0.68/0.97 | 0.57/0.94 | 0.47/0.91 |

TARGET: **recall >= 0.93 AND false positives <= 0.05 taps/character.** Below that, no setting
clears `MIN_FRAC_CONFIDENT = 0.50` at >= 0.90 confident precision. False positives are the
harder half: at f = 0.15 even a perfect-recall detector fails the gate, because a hallucinated
tap breaks a gap-free word exactly as a missed one does.

## AMENDMENT 5 — a prior blocker that outranks all of the above

With PERFECT labels (keylogger timestamp pairing, no alignment involved) the per-key landing
scatter on the keyboard sessions is 2.0-2.4x the calibrated key pitch:

| session | keys with >= 15 samples | pitch px | median spread px | spread/pitch |
|---|---|---|---|---|
| 20260910-131629-kbd | 20 | 64.4 | 148.4 | 2.31 |
| 20260910-015948-kbd | 3 | 55.0 (assumed) | 130.7 | 2.38 |
| 20260910-021315-kbd | 15 | 91.4 | 186.1 | 2.04 |

Home-row calibration is itself incoherent (a-s = 22 px, s-d = 64 px, k-l = 120 px on 131629),
i.e. the tap x,y do not lay out like a keyboard. Consequently, injecting label noise into the
kbd session at rates up to 40% moves the median variance ratio only from 1.00 to 1.11 — the
frozen 2.0 threshold cannot detect contamination, because the clean baseline is already
keyboard-wide. Phase 0 cannot be answered from these taps by ANY labeller, correct or not.
Fix the landing-position estimate before spending more effort on desk labelling.

## AMENDMENT 6 — amendment 5's blocker is NOT overturned; the "fix" was label leakage

`contact.py` correctly identified WHY the 174 px scatter exists (98.8% of the variance is the
wrong fingertip). It does NOT supply a fix that works without the key label, and the two
headline numbers that suggested it did are both artifacts.

ARTIFACT 1 — the oracle finger is a function of the label. `contact.gt_finger` reads the
touch-typing finger out of `KEY_LABEL[key]`. So "GT finger, abs px" is not a landing-position
estimate, it is `position of finger f(key)`. On a desk session the key is exactly what is
unknown, so this estimator does not exist there.

ARTIFACT 2 — `taps_contact.jsonl` was written by a `FingerClf` fitted on `TRAIN`, which
CONTAINS the reference session `20260910-131629-kbd`. Its finger predictions on that session
are in-sample. Refitting the identical model on the other three kbd sessions only:

| within-key scatter on 20260910-131629-kbd | home pitch | scatter/pitch |
|---|---|---|
| oracle finger (label-derived, not obtainable on desk) | 74.3 px | **0.22** |
| predicted finger, fit INCLUDING this session (`taps_contact.jsonl`) | 74.0 px | **0.26** |
| predicted finger, fit EXCLUDING this session (`taps_fix4.jsonl`) | 28.2 px | **4.34** |
| reported `taps.jsonl` (argmax flexion velocity) | 22.6 px | 7.73 |

The "real ~74 px pitch / a-s 65, s-d 74, d-f 74, k-l 91" home row is likewise only reachable
with the oracle or in-sample finger; out of sample it collapses to 17/39/17/63 px.

Cross-session 10-way fingertip accuracy (LOSO over the four kbd sessions) is 0.39-0.65, hand
alone 0.69-0.81. That is the whole gap. AMENDMENT 5 STANDS: with the finger attribution
actually available on a desk session, per-key scatter is 2.1-4.4 key pitch and the frozen
2.0 variance-ratio threshold cannot discriminate.

RESIDUAL POSITIVE RESULT (keep it): with the oracle finger, keys that SHARE a finger still
separate — median same-finger centroid separation 0.52-0.54 pitch against 0.20-0.22 pitch
within-key scatter. The hand geometry does carry per-key information. The missing piece is a
label-free finger attribution, not the geometry.

## AMENDMENT 7 — desk labelling coverage is the binding constraint, and it is not close

On `20260910-202149-desk` (676 transcribed characters, 591 taps), with the frozen gates:

| desk taps | phrases used | taps labelled | median frac_confident | keys >= 15 both |
|---|---|---|---|---|
| taps_fix3/4/5 (offset +3/+4/+5) | 2/20 | 34/591 | 0.16 | **0** |
| taps_pos | 1/20 | 20/591 | 0.17 | **0** |
| taps_contact | 1/20 | 18/591 | 0.21 | **0** |
| taps.jsonl / taps_gb (argmax flexion) | 0/20 | 0/591 | 0.00 | **0** |

The desk TEXT is not the limit: 13 keys occur >= 15 times in it, and >= 50% label coverage
would clear the >= 8 key rule. Achieved coverage is 3-5%. Same measurement on desk-length,
word-boundary segments of `20260910-131629-kbd` with keylogger truth: 3-7 of 42 segments pass
the gates, confident-set coverage 0.038-0.087 at precision 0.91-0.93, and exactly ONE key
(`space`) reaches 15 confident samples. The gates are honest — precision stays above 0.90 —
they simply admit far too little.

Removing the confidence gate to force a verdict was measured and NOT adopted: it yields 10-11
qualifying keys at median centroid shift 1.24-2.73 pitch (variance ratio 0.68-58), i.e. FAIL,
never PASS. There is no gate setting that manufactures a PASS.

Also recorded, because it bounds any absolute-pixel comparison: between the kbd reference and
the desk session the resting hands are not in the same place — left index tip median moves
~96 px (1.3 pitch), right index ~208 px (2.8 pitch). The two hands move by different amounts,
so it is not a pure camera translation and cannot be calibrated out from the landmarks alone.
A fixed scene fiducial in shot is required before absolute pixels can be compared ACROSS
sessions, however well they work within one.

## AMENDMENT 8 — `pad` condition: built-in trackpad as touch ground truth (additive)

WHY. Desk sessions have no touch ground truth; keyboard sessions have key travel. The MacBook's
Force Touch trackpad is flat glass, in shot, and macOS's private MultitouchSupport framework reports
every contact. Recording "typing" on it gives exact press/lift times on a desk-like surface.
Protocol: `phase0/PAD_PROTOCOL.md`.

CONDITION. `meta.json.condition` may be `"pad"`; session_id `<YYYYmmdd-HHMMSS>-pad`.
`recorder --touchpad` (implied by `--condition pad`) adds touch logging to any condition; the
default (no flag) is unchanged. Same process, same `time.monotonic()` clock rule.

`meta.json` gains an optional `"touchpad"` object (device size, frames, downs, clock mode).
Readers that ignore unknown keys are unaffected.

### touches.jsonl  (one row per contact event; only glass contact, never hover)
{"t": float, "t_dev": float, "t_rx": float, "frame": int, "event": "down"|"move"|"up",
 "id": int, "finger": int, "hand": int, "state": int, "x_norm": float, "y_norm": float,
 "x_mm": float, "y_mm": float, "vx": float, "vy": float, "major": float, "minor": float,
 "angle": float, "size": float, "density": float, ["truncated": true]}
- `t`: monotonic seconds. `t_dev` is the framework timestamp, `t_rx` the callback arrival time.
  MEASURED on Mac15,6 / Darwin 25.2 with real taps:
  - `t_dev` has its own epoch: 13,953.7 s ahead of monotonic, and not `mach_continuous_time`.
  - Its clock runs **0.88% slow** against the host: slope 1.0067-1.0092 in every contact segment.
  - Around an affine fit the residual is p95 0.55 ms, max 2.0 ms. A constant offset leaves 25.9 ms
    over 3 s, i.e. about 5 s of drift over a 10-minute session.

  So the online `t` is `t_rx` (`clock_mode: "arrival"`), or `t_dev` when the two agree within 2 s
  (`"shared"`). `pad_labels` re-derives `t` offline from `t_dev` with a lower-envelope affine fit
  to `t_rx` (`touchpad.map_clock`), which removes both drift and arrival jitter.
  `meta.json.touchpad.host_per_dev_second` records the measured rate.
- `id` is the contact path id: stable from down to up, reused after lift. `finger` and `hand` are
  the framework's own guesses and are not ground truth.
- `x_norm, y_norm` are in 0..1 with `y_norm = 1` at the far edge. `x_mm, y_mm` have their origin at
  the far-left corner as the user sits, with y growing toward the user. Measured sensor size on
  Mac15,6 is 124.8 x 76.8 mm.
- `down` is the first frame in state make_touch/touching. `up` is the first frame out of it, or
  the frame where the contact vanishes. `truncated` marks contacts still down at shutdown.
- `density` is capacitance density, a pressure proxy only. Force Touch force is not exposed here.

### Derived (phase0/analysis/pad_labels.py)
- `taps_pad.jsonl` follows the taps.jsonl schema: `t` is the pad down time, `i` is the nearest
  frame, and hand/finger come from the fingertip landmark nearest the contact point. That contact
  point is projected through a pad-to-landmark-px homography fitted from four corner holds.
  Additive keys: `src`, `handedness`, `contact_x/contact_y`, `d_px`, `d2_px`, `t_up`, `dur`,
  `pad_id`, `x_mm/y_mm`, `density`.
- Only contacts of <= 0.30 s with <= 4 mm travel are taps. Longer contacts are `rest`, and larger
  travel is `slide`. Both are reported but not labelled.
- `<session_id>-padtrain/` is a sibling folder of symlinks to the session's video, frames,
  landmarks, meta and phrases, plus `keys.jsonl` with one `down`/`up` pair (key `unknown`) per pad
  tap. Every tool that reads `keys.jsonl` down times (`taps_gb train`, `eval_taps`,
  `touch_common.kt`) trains or scores on it unchanged.
- LIMIT: hand/finger is a nearest-landmark assignment, NOT independent finger ground truth; only
  the time and the pad position are measured.
