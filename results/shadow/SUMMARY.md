# Raking-light shadow on REAL footage — SUMMARY (2026-09-21, Mac mini)

## Verdict: not detectable in the existing recordings; the lamp is UNTESTED, not refuted.
On three keyboard sessions with exact keylogger truth, features built around each fingertip to find its cast
shadow carry no information beyond where the fingertip is. Fitted with the same LightGBM harness as the
PressureVision probe, the shadow features score 0.895 / 0.806 / 0.912 — and the static-average-image control
(same features read from the session's mean image at the same tip positions, i.e. position only, no per-frame
appearance) scores 0.885 / 0.812 / 0.897. Added on top of the kinematic features they gain +0.002 / +0.004 /
+0.000. That is the PressureVision result again, for the same reason: the signal is fingertip position.

The reason is the footage, not the idea. The room light in these sessions is high and diffuse (estimated
elevation ~50–70°, extended source behind the typist's left shoulder; at night, the screen glow and the
keyboard backlight), and both the keyboard keys and the desk are black. A 7 mm hover under a 60° light is a
4 mm shadow gap (~8 px), softer than its own penumbra, cast onto black keys. Nothing in this data can show a
raking-light shadow, so the geometry result in results/angle/SUMMARY.md §3 stands untested.

Reference points from results/pressure/SUMMARY.md (session 20260911-164237-kbd): raw stop-motion 0.525,
PressureVision raw 0.660, static-image control 0.688, kinematic LightGBM 0.928, fingertip-crop CNN 0.904.

## Numbers

Label = a key-down within ±40 ms of the frame; AUC with 10 s block-bootstrap 95% CI; LightGBM out-of-fold
over 5 contiguous time folds with 1 s purge; controls as in pv_probe.

| score | 164237 (afternoon, 40,049 fr, 1,734 keys) | 021315 (night, 14,384 fr, 943 keys) | 131629 (midday, 35,511 fr, 1,693 keys) |
|---|---|---|---|
| shadow "depth", raw max over tips | 0.609 [0.573, 0.649] | 0.566 [0.486, 0.650] | 0.536 [0.506, 0.569] |
| shadow "gap", raw min over tips | 0.500 [0.461, 0.543] | 0.410 [0.373, 0.445] | 0.382 [0.338, 0.425] |
| control: static image, depth raw | 0.567 [0.526, 0.605] | 0.525 [0.488, 0.563] | 0.301 [0.258, 0.345] |
| our stop-motion cue, raw | 0.525 [0.510, 0.540] | 0.497 [0.468, 0.531] | 0.501 [0.491, 0.512] |
| **shadow features → LightGBM** | **0.895 [0.873, 0.915]** | **0.806 [0.749, 0.856]** | **0.912 [0.886, 0.934]** |
| **control: static image → LightGBM** | **0.885 [0.860, 0.908]** | **0.812 [0.760, 0.856]** | **0.897 [0.867, 0.923]** |
| kinematic → LightGBM | 0.928 [0.913, 0.943] | 0.869 [0.827, 0.906] | 0.941 [0.924, 0.956] |
| shadow + kinematic → LightGBM | 0.930 [0.915, 0.944] | 0.872 [0.832, 0.908] | 0.941 [0.924, 0.956] |
| control: time-shuffled shadow LightGBM | 0.503 [0.433, 0.574] | 0.539 [0.456, 0.625] | 0.537 [0.458, 0.614] |

Paired deltas (block bootstrap, same frames):

| delta | 164237 | 021315 | 131629 |
|---|---|---|---|
| shadow LGBM − static-image LGBM | +0.010 [+0.003, +0.017] | −0.006 [−0.028, +0.020] | +0.015 [+0.009, +0.023] |
| (shadow + kin) − kin | +0.002 [−0.000, +0.004] | +0.004 [−0.001, +0.009] | +0.000 [−0.001, +0.002] |
| shadow LGBM − kin LGBM | −0.034 [−0.043, −0.025] | −0.063 [−0.083, −0.043] | −0.030 [−0.042, −0.019] |

Per-frame appearance around the tip is worth at most 0.015 AUC over position alone, and that residual is
just as likely key-cap travel or finger flattening as a shadow. The per-finger rows look more exciting and are
a trap: the right-hand fingers' "depth" AUCs reach 0.82–0.90 against static-control values of 0.44–0.68, and
their measured "gap" is 65–95 px at hover vs 21–39 px at key-down — the direction a real shadow gap would
show. Looking at the frames (montage_right_index_keydown_vs_hover_164237.jpg in this folder) explains it: the ">150 ms from any key"
negatives for the right hand are mostly frames where that hand is *resting on the silver palm-rest*, so the
"darkest point along the ray" is the keyboard bezel 60–110 px away, while at key-down the tip is over the key
field where the nearest dark key gap is ~20 px away. That is surface and position, which the mean image only
partly captures because it blends deck and keys at each location. The left-hand fingers, which stay over the
keys, show nothing (0.32–0.53, at or below their controls). There is no resolvable fingertip shadow here.

## What the existing lighting actually is

- **164237 and 131629 (daytime, same room and rig):** the whole hand casts one soft shadow on the silver
  palm-rest, offset 22–53 px (11–26 mm at 2 px/mm) from the hand edge toward image down-right, i.e. toward the
  typist's right and away from him; with the hand 25–40 mm above the deck that is a light elevation of roughly
  **50–70°** from behind the typist's left shoulder, and the soft edge says an extended source (ceiling light
  or window), not a lamp. The darkest ray around fingertips over the bright deck points down (38% / 30%) or
  right (26% / 27%) in the image, consistent with that. Under the fingertips the surface is the black key
  field 60% of the time (median gray 90 of 255; over the deck 40%).
- **021315 (night):** dim (median gray under the tip 58), lit by the screen from the front and by the keyboard
  backlight; no frame has a fingertip over a bright surface. No shadow geometry is measurable at all.
- The bare-desk sessions (20260912/13-desk) are on a black desk too, and have no per-frame truth; not scored.

## What Pasha should do: the shortest confirming recording

Requirements the simulation cannot supply and this footage lacks — a **low, small light** and a **light-coloured
surface under the fingers**. Everything else (camera, keylogger, harness) already exists.

1. Surface: a **white or light-grey silicone keyboard cover** on the MacBook (or a light external keyboard).
   The keys must be light, because the fingertip's shadow lands on the neighbouring keys, and the keylogger
   keeps per-frame truth so the result is scored by exactly this harness. A light desk mat under the laptop
   for the palm-rest area.
2. Light: a single small bright lamp (a phone LED or a bare-bulb desk lamp; the source should be < 2 cm
   across so the penumbra stays under the gap), at **25–30° elevation** measured from the key plane, placed
   to the typist's **left or right at ~0.5 m**, aimed at the hands. Room lights **off** or dimmed (any fill
   light washes the shadow out — that is what this footage shows).
3. Record with the existing top-down rig and keylogger: **3 min of natural typing with the lamp at ~25°,
   then 2 min with the lamp at ~40°, then 2 min lamp off** (same cover, same session — the lamp-off block is
   the control). Roughly 7 minutes, no new tooling.
4. Score with `python -m phase0.analysis.shadow_probe extract/probe <sid>` (radial search 3–120 px with a
   3–8 px bin, so a 7–11 px contact gap is resolvable — the first version of this probe floored at 12 px and
   was fixed before the numbers above were produced): the decisive numbers are (shadow LGBM − static-image
   LGBM) and (shadow + kin − kin) with the lamp on vs off, and the per-finger gap at key-down vs hover, which
   must drop by tens of pixels *on frames where the hand is over the keys* (compare like with like — see the
   resting-hand trap above). Anything under +0.03 over the static control with the lamp on means the lamp
   does not work in practice.
5. Only if that passes: one bare-desk phrase session on the light mat with the lamp, to see the same gap on
   the surface the product actually targets.

## Protocol

- `phase0/analysis/shadow_probe.py`. For each landmark frame (MediaPipe tips, both hands) and each fingertip:
  grayscale intensity along 8 rays from 3 to 120 px in 2 px steps; per ray the mean in six radial bins
  (3–8, 8–14, 14–24, 24–36, 36–50, 50–80 px); ring means; pad intensity (8 px disc); contrast = darkest ray −
  ring; darkest direction (over the bins to 36 px); "gap" = radius of the intensity minimum along the darkest
  ray; "depth" = (ring mean − that minimum)/ring mean; far-ring surface brightness. 61 values × 10 tips, plus
  lags (±2, ±4 frames) and a 3-frame difference of gap/depth/contrast → 760 features. Frames with no tip in
  view are dropped (40,049 of 40,315 in 164237). A first pass with a 12–64 px search (results/shadow/
  v1_probe_floor12px/) gave the same aggregate picture (0.900/0.814/0.914 vs controls 0.893/0.821/0.900).
- Static-average-image control: identical features read from the session's mean grayscale frame at the same
  per-frame tip positions, so it carries position and surface but no per-frame appearance. Time-shuffle
  control: scores rolled by 60 s. Kinematic reference: taps_gb groups + touch_feats (2,122 features), as in
  pv_probe.
- Light-direction estimate: (i) histogram of the darkest ray over tips that sit on the bright deck (mean image
  > 110 gray) vs on the keys; (ii) whole-hand shadow on the deck in 16 frames with the right hand fully over
  it: pixels darker than the mean image by > 20 within 60 px of the hand and outside the dilated landmark
  hull, distance to the hull edge (median 22–53 px in the 5 clean frames; the other 11 had the other hand or
  the screen in the mask and were discarded by eye). Elevation from that offset assumes a 25–40 mm hand
  height; it is a blob statistic, not a matched-point displacement, so treat it as ±15° at best.

## Caveats

- The feature design is a fixed ray/ring probe, not a shadow segmenter. A dedicated detector could conceivably
  find a shadow this probe misses, but the raw and fitted numbers, the static control and the gap direction
  all say there is nothing resolvable at fingertip scale under this lighting on these surfaces.
- "Key-down within ±40 ms" labels on a physical keyboard include ~1–2 mm of key travel; a shadow at contact
  would be measured against the key top, not the desk.
- The +0.007 / +0.014 residuals over the static control are statistically nonzero on two sessions but tiny,
  and cannot be attributed to shadow rather than any other per-frame appearance change.
- Light elevation is a rough geometric estimate from a soft shadow; treat 50–70° as "high and diffuse", not a
  measurement to the degree.
