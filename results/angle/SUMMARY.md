# Camera angle for reading bare-desk typing — SUMMARY (2026-09-21, Mac mini)

Simulated geometry: EgoPressure 3D MANO hands re-projected through a virtual camera matched to Pasha's rig.
EgoPressure is CC BY-NC-SA — this is a research measurement only; nothing derived from it ships.
How We Type mocap (section 6) is CC-BY-NC-4.0, likewise research only.

## Recommendation, in plain terms

1. **Do not tilt the camera to see contact.** Moving from straight-down (90°) to a low front view buys at
   most +0.08 AUC (0.65 → 0.74 at 0–10° elevation) and a low side view at most +0.13 (→ 0.79 at 0°). Both
   are far short of the 0.92 that true fingertip height would give, adjacent 10° steps are inside the
   landmark-noise spread (±0.02–0.03), and every low angle pays in key identity: planar localisation at
   contact goes from 1 mm (90°) to 2.5 mm (30°), 3.8 mm (20°), 6–15 mm (≤10°) against a 19 mm key pitch,
   and side views occlude 37–47% of fingertips below 40°.
2. **Add a raking light instead and keep the camera where it is.** A lamp at 20–30° elevation makes the
   tip-to-shadow gap on the desk a direct height read-out: 53–84 px at a typical hover, 7–11 px at contact
   (2 px/mm on this rig). In geometry it reaches the true-height ceiling (0.92), and still 0.83 / 0.75 if 30% /
   50% of shadows cannot be measured. That beats every camera angle. 30° is the safer choice than 10°: the
   shadow stays compact enough to attribute to the right finger.
   **This is unvalidated on real footage** — it assumes a shadow is detectable and attributable per fingertip.
   See results/shadow/SUMMARY.md for the test on existing recordings.
3. **If a second view is ever added, put it at the side (azimuth 90°), low.** Side 0° = 0.79 vs front 0° = 0.74;
   a top-down + side-20° pair scores 0.75, no better than the side view alone.

**What re-recording costs.** This model has never transferred across camera positions. Changing the camera
angle invalidates the three labelled desk sessions (20260910-202149, 20260912-174542, 20260913-125733) and
every LOSO baseline built on them; all would have to be re-recorded and re-labelled. The angle gain above
(+0.05–0.13 contact AUC, within noise between neighbouring angles) does not justify that. A lamp does not
move the camera, so it invalidates nothing — the existing top-down data stays valid and a lamp can be tested
with one short recording. **There is no measured mapping from contact AUC to word error rate**; the honest
statement is that today's 2D signal sits at ~0.65 and perfect height at ~0.92, and only a pilot recording
can say how much of that gap turns into words.

## 1. Contact: press vs hover ROC AUC by virtual camera (EgoPressure index-press events)

2,486 fingertip descent bottoms, 365 contacts, 21 participants; participant-grouped 7-fold LightGBM, mean of 3
seeds; 95% participant-cluster bootstrap CI. Occlusion and localisation columns come from the typing-like
gestures (section 2), not from these index-press events (whose other four fingers are curled into a fist).

| view | elevation | contact AUC [95% CI] | occlusion, typing gestures | planar error at contact, median / p90 |
|---|---|---|---|---|
| front | 0° | 0.737 [0.710, 0.763] | 0.31 | 15.2 / 67.9 mm |
| front | 10° | 0.739 [0.706, 0.771] | 0.30 | 6.3 / 16.2 mm |
| front | 20° | 0.715 [0.679, 0.749] | 0.29 | 3.8 / 8.9 mm |
| front | 30° | 0.679 [0.640, 0.719] | 0.29 | 2.5 / 5.8 mm |
| front | 40° | 0.699 [0.666, 0.733] | 0.29 | 1.7 / 3.9 mm |
| front | 50° | 0.673 [0.635, 0.710] | 0.29 | 1.2 / 2.8 mm |
| front | 60° | 0.650 [0.614, 0.684] | 0.29 | 0.7 / 2.0 mm |
| front | 70° | 0.659 [0.616, 0.700] | 0.29 | 0.7 / 1.5 mm |
| front | 80° | 0.676 [0.640, 0.708] | 0.30 | 0.8 / 1.7 mm |
| **top-down** | **90°** | **0.652 [0.614, 0.690]** | 0.32 | 1.1 / 2.3 mm |
| side | 0° | **0.786 [0.752, 0.818]** | 0.43 | 22.8 / 95.1 mm |
| side | 20° | 0.759 [0.732, 0.784] | 0.44 | 5.7 / 14.2 mm |
| side | 40° | 0.721 [0.686, 0.753] | 0.37 | 2.9 / 7.1 mm |
| side | 60° | 0.674 [0.637, 0.707] | 0.32 | 1.6 / 4.1 mm |
| side | 90° | 0.666 [0.633, 0.693] | 0.32 | 1.0 / 2.4 mm |
| two views: top-down + front 10° | — | 0.746 [0.714, 0.775] | | |
| two views: top-down + side 20° (mirror) | — | 0.753 [0.723, 0.780] | | |
| EgoPressure's real overhead camera (ref_cam4, reproduction anchor) | — | 0.669 [0.617, 0.717] | | |
| top-down through contact_ego's own normaliser (el90_front_postjit) | — | 0.653 [0.619, 0.686] | | |
| **true fingertip height (ceiling)** | — | **0.918 [0.889, 0.945]** | | |

Anchors: the published top-down number in results/contact was 0.615 (2D) / 0.78 (MANO relative depth) /
0.92 (true height). This pipeline reproduces the same event set exactly (2486 / 365) and lands at 0.652–0.669
top-down; the difference from 0.615 is the landmark-noise draw (next paragraph), not the pipeline.

**Landmark jitter dominates small differences** (results/angle/jitter_draws.json; 5 draws of the 0.8 px
MediaPipe-like noise, one seed each): top-down mean 0.677, sd 0.022, range 0.646–0.700; front 20° mean 0.708,
sd 0.012, range 0.693–0.725. A 10° step in the table is smaller than one noise draw. Read the curve, not the
points: flat and noisy from 90° down to ~30°, rising only below 20°, and the side azimuth above the front at
every elevation below 60°.

## 2. Occlusion and key identity on typing-like gestures (EgoPressure type_ipad / press_fingers / press_flat)

210 sequences, 5,054 descent bottoms, 2,551 contacts, ~1,000 per finger. Occlusion = a fingertip hidden behind
any bone of the same hand not touching it (9 mm capsule); forearm, other hand and sleeve are not modelled, so
these are lower bounds. Planar error = back-project the (jittered) tip pixel to the desk plane and compare with
the true position; "at contact" is what becomes key errors.

| elevation | occlusion front / side | planar error at contact, front / side (median) |
|---|---|---|
| 0° | 0.31 / 0.43 | 15.2 / 22.8 mm |
| 10° | 0.30 / 0.47 | 6.3 / 9.1 mm |
| 20° | 0.29 / 0.44 | 3.8 / 5.7 mm |
| 30° | 0.29 / 0.41 | 2.5 / 3.9 mm |
| 40° | 0.29 / 0.37 | 1.7 / 2.9 mm |
| 60° | 0.29 / 0.32 | 0.7 / 1.6 mm |
| 90° | 0.32 / 0.32 | 1.1 / 1.0 mm |

Front-azimuth occlusion is flat in elevation here (0.29–0.32) — section 6, on real typing posture, finds it
rising toward top-down instead; side-azimuth occlusion climbs below 40° in both as fingers line up behind
each other. With a 19 mm key pitch, front views down to ~30° keep key identity (≤2.5 mm), 20° is
marginal, ≤10° costs keys. The top-down 1.1 mm is parallax at 227 mm working distance, not noise.

## 3. Raking light with the existing top-down camera (results/angle/shadow.json)

Pure geometry, no rendering: gap on the desk between a fingertip and its cast shadow = height / tan(light
elevation), projected at 2.02 px/mm (538 px focal length over 267 mm to the desk).

| light elevation | median gap at hover | median gap at contact | AUC of the gap (with 1.7–3 px noise, 2 px floor) |
|---|---|---|---|
| 10° | 174 px | 23 px | 0.918–0.919 |
| 20° | 84 px | 11 px | 0.918–0.919 |
| 30° | 53 px | 7 px | 0.915–0.919 |

The gap is a monotone function of true height, so its ideal AUC *is* the height ceiling — that number is not
the finding. The finding is the gap size: tens of pixels, so pixel noise is irrelevant. The contact gap is
non-zero because a MANO joint centre sits ~2 mm above the pad. Degradation arms at 20°: 30% of shadows
unmeasurable → 0.828 [0.799, 0.857]; 50% → 0.749 [0.719, 0.778]; gap capped at 30 px (shadow leaves the clean
desk zone) → 0.895 [0.862, 0.928]. Even the pessimistic arms beat every camera angle in section 1.

## 4. Protocol

- Events: `phase0/analysis/contact_angle.py events` (run on the MacBook, output `.cache/contact_angle/events.npz`,
  identical bytes on both machines). EgoPressure index_press_{high,low,no-contact} sequences; MANO joints
  resampled to 60 Hz; descent bottoms = fingertip-height local minima ≥ 8 mm prominence within 0–60 mm;
  contact = ≥ 150 force counts within 12 mm of the tip within ±4 frames (contact_ego.CONTACT_THR).
- Virtual camera: 227 mm from the median palm centroid, focal length set so a median EgoPressure hand
  (knuckle width 57.3 mm, joints 5–17) spans 136 px as on the real rig, 1280×720, principal point centred
  (f = 538.3 px, horizontal FOV 100°). Elevation 0° = along the desk, 90° = straight down; azimuth front = beyond
  the fingertips looking back, side = 90° from that. Isotropic 0.8 px Gaussian landmark noise.
- Features: the project's own tap-detector stack (taps_gb.build_groups, all groups) on the projected
  landmarks in our image convention; the similarity transform into that convention is derived once from the
  top-down view and reused at every angle so foreshortening survives (HANDOFF gotcha 1).
- Scoring: LightGBM (300 trees), participant-grouped 7-fold out-of-fold, AUC = mean over 3 seeds (never the
  seed ensemble, HANDOFF gotcha 3); 2,000-resample participant-cluster bootstrap CI.
- Sweep run on the mini (`.cache/contact_angle/run_sweep.sh`, log `sweep.log`) → `angle_sweep.json`;
  occlusion_typing.json and jitter_draws.json were produced on the MacBook (they need the raw EgoPressure
  sequences, which are not on the mini); shadow.json from `contact_angle shadow`.

## 5. Caveats — what is simulated and must be confirmed

- Different dataset, different task, other people's hands: EgoPressure is deliberate single-finger presses on
  a pressure pad by 21 participants; hover heights there (median 15 mm) are generous compared with touch
  typing (median 7 mm in How We Type, section 6).
- Perfect 3D joints plus isotropic noise; real MediaPipe error is larger, anisotropic and pose-dependent, and
  the replicate study shows the metric is sensitive to exactly that.
- No rendering: no blur, rolling shutter, exposure, desk texture; for the light, no penumbra, no ambient fill,
  no second shadow from room lighting, and no test of whether a shadow can be *found* and attributed to a
  finger. That is the whole open question for recommendation 2.
- taps_gb.build_groups normalises by the per-frame wrist→MCP span, which is foreshortened at grazing angles;
  low-elevation AUCs are probably pessimistic for that reason (HANDOFF gotcha 2).
- Occlusion is a 9 mm capsule test on bones; no forearm, no second hand, no sleeve: lower bounds.
- Nothing here maps to word error rate. The only measured relationship in this project is that words are
  lost where the posteriors never propose the right characters; contact AUC is the upstream signal, not WER.

## 6. Ten-finger typing check: How We Type mocap through the same virtual cameras (added on the mini)

The EgoPressure sequences above are single-finger presses. To see what an oblique camera does to a hand that is
actually touch-typing, the same virtual cameras, occlusion test and feature stack were run on the How We Type
mocap (Feit et al., CHI'16, CC-BY-NC-4.0; 30 participants typing on a physical keyboard at 240 Hz with a
keylogger). Code `phase0/analysis/angle_hwt.py`; raw `results/angle/angle_sweep_hwt.json` (contact AUC, 9
cameras, 2 seeds × 5 participant folds) and `angle_geometry_hwt.json` (occlusion + localisation, all 20 cameras).

Protocol differences, stated up front: the mocap streams are the project's normalised streams (knuckle-width
units, wrist-relative height), so metric scale is set to EgoPressure's 57.3 mm knuckle width and each hand's
wrist height is estimated from its own key-downs ("the pressing tip is on the key", median 10–25 mm); the two
hands are placed 140 mm apart because normalisation drops their relative position. Events = fingertip descent
bottoms (30,942; 15,439 contacts = bottom within ±50 ms of a key-down by the lowest tip; 15,503 hovers = no
key-down within ±100 ms; ambiguous bottoms dropped; hovers capped at 2× contacts per session). Hover heights
are realistic for typing: median 7 mm (EgoPressure: 15 mm). True-height AUC on these labels is 0.858
[0.835, 0.879] — the labels are noisier than pressure-pad truth, so treat it as the anchor, not a ceiling.

| view | elevation | key-down-timing AUC [95% CI] (see note) | occlusion at descent bottoms, other finger/palm/other hand, 9 mm / 6 mm capsule | of which ring / pinky (9 mm) | planar error at contact, median / p90 |
|---|---|---|---|---|---|
| front | 0° | 0.898 [0.872, 0.920] | 0.14 / 0.08 | 0.12 / 0.26 | 11.2 / 29.0 mm |
| front | 20° | 0.873 [0.846, 0.897] | 0.23 / 0.15 | 0.29 / 0.37 | 3.1 / 8.5 mm |
| front | 40° | 0.880 [0.851, 0.905] | 0.28 / 0.17 | 0.44 / 0.45 | 1.4 / 4.1 mm |
| front | 60° | 0.891 [0.858, 0.919] | 0.33 / 0.22 | 0.55 / 0.53 | 0.7 / 2.0 mm |
| **top-down** | **90°** | **0.887 [0.856, 0.914]** | **0.41 / 0.28** | 0.68 / 0.68 | 1.0 / 2.4 mm |
| side | 0° | 0.878 [0.834, 0.917] | 0.62 / 0.47 | 0.68 / 0.51 | 16.7 / 41.9 mm |
| side | 20° | 0.881 [0.843, 0.916] | 0.77 / 0.62 | 0.72 / 0.51 | 4.8 / 12.4 mm |
| side | 40° | 0.886 [0.856, 0.914] | 0.61 / 0.44 | 0.51 / 0.51 | 2.5 / 6.5 mm |

Note on the AUC column: the feature vector is the project's frame-level stack over both hands and is
finger-agnostic by construction, while the events are per-finger descent bottoms. The only label such a
vector can learn is "a key-down is happening at this frame", not "this finger is in contact" (checked on the
first session: same-frame bottoms never carry conflicting labels, so nothing is contradictory, but nothing
identifies the finger either). It is therefore **not comparable** to section 1's numbers (different event
set, 50% vs 15% positives, smaller height separation), and should not be read against the 0.858 true-height
anchor.

What this adds to sections 1–2:
* **Whole-hand key-down timing is equally visible from every camera**: 0.87–0.90 at all nine views, every
  CI overlapping, top-down 0.887. Tilting the camera does not buy a better view of the typing rhythm; whether
  it buys per-finger contact in typing is not answered here (section 1 says: a little, for single presses).
  The +0.08–0.13 gains in section 1 should not be assumed to carry over to ten-finger typing.
* **Occlusion from above is higher here than in section 2** (0.41 at descent bottoms vs 0.32 for the EgoPressure
  typing-like gestures, same 9 mm capsule) and, unlike section 2's flat front curve, it *falls* toward a low
  front view (0.14 at 0°): in real typing posture the curled ring and pinky tips sit under their taller
  neighbours when seen from a camera centred between the hands (0.68 each at 90°), while from the front the
  fingertips are the nearest points. Section 2's hands are posed pad gestures; these are 30 people actually
  typing, so where the two disagree, trust this one — but note the sensitivity: at a 6 mm capsule the top-down
  figure drops to 0.28 (0.08 at 0° front), and the debug frame that flagged a ring tip did so at 8.5 mm from
  the middle finger's phalanx, i.e. a marginal hit. The ordering (side worst, front-low best, top-down in
  between) is stable; the absolute level is not.
  This does not change the recommendation: it argues for a *front* tilt, whose cost is localisation, and the
  ranking of side views as worst holds in both datasets.
* Localisation is the same story as section 2: ≤1.4 mm down to 40° front, 3 mm at 20°, 11 mm at 0°.

Net: a front view around 30–45° would trade nothing in contact AUC, halve ring/pinky self-occlusion and
cost ~1–2 mm of key localisation — a defensible geometry *if* one were re-recording anyway — but it does not
solve the contact problem, which the raking light is still the only candidate for. The recommendation at the
top stands.
