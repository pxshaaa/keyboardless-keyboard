# A mirror as a second viewpoint for the top-down phone camera — SUMMARY (2026-09-21, Mac mini)

Simulated geometry. Height/AUC numbers: EgoPressure descent-bottom events (CC BY-NC-SA, research measurement
only, nothing derived from it ships) — the same 2,486 events, 365 contacts, 21 participants and the same
LightGBM / participant-fold / cluster-bootstrap protocol as results/angle/SUMMARY.md, so every number here is
comparable to that table. Visibility, mirrored-hand resolution and frame footprint: How We Type mocap
(CC-BY-NC-4.0, research only), 12 real ten-finger typists, same capsule occlusion test as results/angle §6.

## Recommendation, in plain terms

**Yes — a small mirror standing on the desk beyond the fingertips gets contact visibility to the height
ceiling for the fingertips it can see, and it keeps key identity because the phone stays top-down.**

* Geometry: a mirror 16–24 cm beyond the fingertips (between the hands and where a screen would be), glass
  centre 6 cm above the desk, leaning back 18–27°, gives the camera a second virtual viewpoint from 5–7°
  above the desk on the far side of the hands, with a 450–680 mm stereo baseline at 83–85° vergence. From the
  two views a fingertip's height triangulates to **0.2–0.4 mm median error (0.6–0.9 mm p90)** with the same
  0.8 px landmark noise as before; hover-vs-contact separation is 7–15 mm, so the press-vs-hover AUC from
  triangulated height alone is **0.917–0.918 — the true-height ceiling (0.918)** — against 0.652 for today's
  top-down 2D features and 0.753 for the earlier arm that only concatenated two 2D projections. Triangulation
  is exactly what that arm was missing.
* What it costs, honestly:
  - **Visibility.** A grazing rear view sees 85–87% of typing fingertips (better than the top-down view's own
    67% in this geometric model), but height needs both views: **54–60% of fingertip descent bottoms are
    visible in both** (59–65% of typing frames), 70% for the index, 65% middle, ~40% ring, pinky and thumb.
    When the rest fall back to top-down-only features the AUC is **0.87–0.88** (CI ≈ ±0.03). Treat that as an
    **upper bound conditional on the tracker**: the triangulated height here is ground truth plus 0.3 mm,
    and the hidden 30% are drawn at random rather than being the actually-occluded events (ring/pinky, which
    are also the fingers with the weakest kinematic signal). Real tracker error on the mirror view only moves
    it down.
  - **Resolution.** The mirrored hand appears at 0.72–0.85× the direct scale (perspective depth, not path
    length): a 136 px knuckle becomes ~100–115 px. Scaling the landmark noise by that ratio changes the height
    error by 0.05–0.1 mm and the AUC by nothing. MediaPipe on a small, mirror-flipped, grazing-angle hand is
    the thing this simulation cannot vouch for.
  - **Frame budget.** At the current 227 mm working distance the mirror image needs 8–10% of the frame at the
    far end of the long (1280 px) axis and fits without overlapping the hands only for the nearer placement
    (16 cm); the 24 cm placements need the phone at ~320 mm, where the mirror uses 4–9% and everything fits,
    at the price of the direct view shrinking from 136 to ~100 px per knuckle and height error rising to
    0.3–0.4 mm (still fine). Side and 45° mirrors either overlap the hand region or fall outside the frame at
    227 mm.
  - **Hardware.** A ~22×8 cm first-surface or thin (≤2 mm) glass mirror (sized for both hands, 250 mm across;
    21×7 cm is the computed minimum at 16 cm, 22×8 at 320 mm phone height), bottom edge on the desk, leaning
    back ~20–25°, centred on the hands' axis 16–24 cm beyond the fingertips, held at a fixed angle relative
    to the phone (a bracket, or a one-off calibration of the mirror plane from the desk plane and two
    markers; triangulation needs the plane to ~1°). A phone-clip prism does **not** do the job: it reflects
    from a point a centimetre from the lens, so the virtual camera sits next to the real one (baseline ≈ 0,
    vergence ≈ 0) and carries no height information. The baseline comes from the mirror being far from the
    phone, near the hands.
  - **Data.** The phone does not move, so the existing top-down desk recordings stay valid for the direct
    view; a new recording is still needed to train and validate anything that uses the mirror view.
* **Side mirrors are out** (both-view visibility 20–30%, fallback AUC 0.72–0.77, fingers line up behind each
  other); a **45° mirror is a compromise** (47–52% both-visible, 0.80–0.85) that also overlaps the hand
  region in the frame.
* **Best no-mirror compromise, for comparison:** front view at 40° elevation — contact AUC 0.699 [0.666, 0.733]
  (+0.05 over top-down, inside the ±0.02–0.03 landmark-noise spread), occlusion 0.29 (EgoPressure typing-like)
  / 0.28 (How We Type, 9 mm; 0.17 at 6 mm), key localisation 1.7 mm median / 3.9 mm p90 at contact — and it
  invalidates every existing desk recording because the camera moves. It does not solve contact; the mirror
  does, for the fingers it can see.

## 1. Placements, geometry and triangulated height (EgoPressure events, 0.8 px noise on both views)

"vel" = elevation of the virtual (mirror) viewpoint above the desk; "scale" = mirrored-hand px per direct px;
"fit" = mirror image inside the 1280×720 frame / hands inside / the two overlap. Height error = |triangulated −
true| at the descent bottom. AUC = mean over 3 seeds of the participant-grouped LightGBM (or the raw scalar).

| placement (azimuth, distance, glass-centre height) @ phone height | baseline mm | vergence | vel | tilt back | mirror size | scale | fit (mirror/hands/overlap) | height err mm med / p90 | AUC height only [95% CI] | AUC top-down 2D + height |
|---|---|---|---|---|---|---|---|---|---|---|
| rear 16 cm, 6 cm @ 227 | 454 | 83° | 7° | 23° | 19×7 cm (21×7 for both hands), 1–8 cm up | 0.81 | ✓ / ✓ / no | 0.25 / 0.62 | 0.917 [0.887, 0.944] | 0.925 |
| rear 24 cm, 6 cm @ 227 | 584 | 85° | 5° | 18° | 15×5 cm | 0.72 | ✗ / ✓ / no | 0.24 / 0.58 | 0.918 [0.888, 0.944] | 0.927 |
| rear 24 cm, 12 cm @ 227 | 510 | 72° | 18° | 7° | 14×7 cm, 7–14 cm up | 0.87 | ✗ / ✓ / no | 0.19 / 0.47 | 0.919 [0.889, 0.945] | 0.928 |
| **rear 16 cm, 6 cm @ 320** | 560 | 83° | 7° | 27° | 20×8 cm | 0.85 | ✓ / ✓ / no | 0.36 / 0.89 | 0.918 [0.889, 0.944] | 0.928 |
| **rear 24 cm, 6 cm @ 320** | 678 | 85° | 5° | 23° | 16×6 cm | 0.75 | ✓ / ✓ / no | 0.37 / 0.90 | 0.916 [0.886, 0.943] | 0.926 |
| rear 24 cm, 12 cm @ 320 | 577 | 72° | 18° | 13° | 16×8 cm | 0.83 | ✓ / ✓ / no | 0.34 / 0.84 | 0.916 [0.886, 0.942] | 0.924 |
| side 16 cm, 6 cm @ 227 | 454 | 83° | 7° | 23° | 16×7 cm | 0.72 | ✗ / ✓ / **yes** | 0.30 / 0.74 | 0.916 | 0.926 |
| side 24 cm, 6 cm @ 320 | 678 | 85° | 5° | 23° | 14×6 cm | 0.69 | ✗ / ✓ / no | 0.42 / 1.00 | 0.916 | 0.926 |
| side 24 cm, 12 cm @ 320 | 577 | 72° | 18° | 13° | 13×9 cm | 0.78 | ✗ / ✓ / no | 0.37 / 0.88 | 0.918 | 0.926 |
| 45° 20 cm, 6 cm @ 227 | 518 | 84° | 6° | 20° | 17×7 cm | 0.74 | ✗ / ✓ / **yes** | 0.26 / 0.63 | 0.918 | 0.928 |
| 45° 20 cm, 12 cm @ 320 | 506 | 68° | 22° | 14° | 17×14 cm | 0.85 | ✗ / ✓ / **yes** | 0.40 / 0.98 | 0.918 | 0.924 |
| rear, any @ 420 | 670–790 | 72–85° | 5–18° | 18–31° | 17–20 cm wide | 0.78–0.87 | ✓ / ✓ / no | 0.5 / 1.2–1.3 | 0.917–0.918 | 0.923–0.929 |
| side / 45°, any @ 420 | 600–790 | 68–85° | 5–22° | 18–31° | 14–20 cm wide | 0.72–0.86 | mostly ✗ or overlap | 0.5 / 1.2–1.3 | 0.914–0.920 | 0.921–0.924 |
| reference: true height | | | | | | | | 0 | 0.918 [0.889, 0.945] | |
| reference: top-down 2D features (today) | | | | | | | | | 0.652 [0.614, 0.690] | |
| reference: "concatenate 2D" mirror arm, top-down + side 20° / + front 10° (results/angle) | | | | | | | | | 0.753 [0.723, 0.780] / 0.746 [0.714, 0.775] | |

Noise scaled to the mirrored hand's size (σ = 0.8 px × direct/mirror scale) moves the median height error to
0.3–0.8 mm and the AUC by ≤ 0.005 — resolution is not the limit. Full arrays: results/mirror/mirror_ego.json.

## 2. Costs: visibility, fallback AUC, resolution, frame (How We Type, 12 typists, 9 mm capsule)

"both" = fingertip not hidden by another finger, the palm or the other hand in *either* view (typing frames =
within 300 ms of a key-down); "index" = index-finger descent bottoms visible in both views, which is the rate
applied to the EgoPressure index-press events for the fallback AUC.

| placement @ phone | direct view visible | mirror view visible | both, all tips | both, per finger th/ix/mi/ri/pi | knuckle px direct → mirror | mirror frame share | **fallback AUC** (height where both-visible, else top-down 2D) [95% CI] |
|---|---|---|---|---|---|---|---|
| rear 16 cm, 6 cm @ 227 | 0.67 | 0.85 | 0.59 | 0.40 / 0.70 / 0.65 / 0.37 / 0.40 | 116 → 106 | 10% | **0.875 [0.848, 0.899]** |
| rear 24 cm, 6 cm @ 227 | 0.67 | 0.87 | 0.59 | | 116 → 104 | 8% (out of frame) | 0.878 [0.843, 0.909] |
| rear 16 cm, 6 cm @ 320 | 0.70 | 0.86 | 0.63 | | 86 → 79 | 5% | **0.883 [0.847, 0.915]** |
| rear 24 cm, 6 cm @ 320 | 0.70 | 0.87 | 0.63 | 0.43 / 0.72 / 0.69 / 0.41 / 0.49 | 86 → 74 | 4% | 0.871 [0.834, 0.902] |
| rear 24 cm, 12 cm @ 320 | 0.70 | 0.85 | 0.63 | 0.36 / 0.69 / 0.69 / 0.40 / 0.48 | 86 → 84 | 9% | 0.875 [0.843, 0.904] |
| rear, any @ 420 | 0.72 | 0.86–0.87 | 0.65 | | 67 → 58–63 | 2–4% | 0.866–0.880 |
| 45° 20 cm, 6 cm @ 320 | 0.70 | 0.67 | 0.50 | 0.25 / 0.59 / 0.68 / 0.36 / 0.40 | 86 → 69 | 11% | 0.837 [0.797, 0.871] |
| 45° 20 cm, 12 cm @ 320 | 0.70 | 0.67 | 0.52 | | 86 → 85 | 22% | 0.801 [0.770, 0.828] |
| side 24 cm, 6 cm @ 320 | 0.70 | 0.33 | 0.24 | 0.21 / 0.32 / 0.29 / 0.24 / 0.38 | 86 → 47 | 3% | 0.772 [0.740, 0.804] |
| side 16 cm, 6 cm @ 227 | 0.67 | 0.29 | 0.20 | | 116 → 76 | 9% | 0.722 [0.684, 0.757] |

Two things to read from it. First, most of the both-view loss is the *top-down* view's own self-occlusion in
this capsule model (at descent bottoms it sees 0.59 of tips — ring and pinky tucked under their neighbours,
results/angle §6 — and the rear mirror takes that to 0.54 both-visible, i.e. the mirror itself costs ~5
points), whereas the mirror alone sees 0.85–0.87. A grazing rear view reads height almost directly from the
image vertical, so a mirror-only height for tips the top-down view loses is a possible upside; it was not
modelled and is not in the 0.875. Second, at a 6 mm capsule the top-down visibility rises to ~0.72 and the
both-view rate with it; the ordering rear ≫ 45° ≫ side does not change.

Knuckle px above are How We Type hands (57 mm knuckle at 116 px @ 227); on the real rig's 136 px the mirrored
hand is ~100–115 px at 227 mm and ~75–95 px at 320 mm.

## 3. Frame layout at the recommended placement (rear 16 cm, 6 cm, phone at 227 mm)

Both hands (250 mm across, 180 mm deep, 0–50 mm high) project to x 417–863, y 50–670 of 1280×720; the mirror
image sits at x 940–1111, y 97–623 — the far 13% of the long axis, no overlap. At 320 mm the hands occupy x
484–796, y 143–577 and the 24 cm mirror x 998–1106. Both the direct view and the mirror view fit at 227 mm
with the near placement; there is no need to raise the phone unless the mirror is pushed to 24 cm.

## 4. Protocol

- Real camera: pinhole, f = 538.3 px at 1280×720 (136 px knuckle at 227 mm, results/angle rig), straight down,
  long axis along the typing direction as on the real rig; also evaluated at 320 and 420 mm.
- Mirror: plane through a centre point `dist` from the hand centre along the azimuth (rear = beyond the
  fingertips, side = 90°, 45°), `hc` above the desk, normal = bisector of the directions to the camera and to
  the hand centre (so the camera's ray off the centre lands on the hands). Virtual camera = real camera
  reflected through the plane; baseline and vergence from that. Mirrored pixel = real-camera projection of the
  reflected point (this is what fixes scale, footprint and frame fit). Required mirror = extent of the plane
  hit by rays to a ±110 × ±90 mm, 0–60 mm high box around the hand centre.
- Triangulation: least-squares closest point of the direct pixel ray and the mirrored ray unfolded through the
  plane; independent 0.8 px Gaussian noise on both pixel observations (and a variant with the mirror-view
  noise scaled by direct/mirror px ratio). The projected point is the ground-truth fingertip, so the
  triangulated height is truth + geometric noise — a measurement of the geometry, not of a tracker. Mirror
  sizes in the table are for a ±110 × ±90 mm hand box; the recommendation uses ±125 mm (both hands). Per EgoPressure sequence the hand centre is the median fingertip
  position + 60 mm toward the wrist at palm height (the raw joints are not on this machine, so cameras are
  placed nominally per sequence, not from the palm centroid as in results/angle).
- AUC: `contact_angle.auc_ci` / `_oof` — LightGBM 300 trees, 7 participant folds, mean of 3 seeds, 2,000
  participant-cluster bootstraps; "height only" is the raw −height scalar (as for the true-height ceiling).
  Fallback: the HWT index-finger both-visible rate for the placement is applied at random to the events; hidden
  events get NaN height (LightGBM handles it) alongside the top-down 2D features.
- Visibility: `angle_hwt.world_joints` + `contact_angle.occluded` (other finger / palm) + `occluded_cross`
  (other hand), 9 mm capsule, from the real camera centre and from the reflected camera centre. Scripts:
  `phase0/analysis/mirror_sim.py hwt` (→ hwt_visibility.json) then `ego` (→ mirror_ego.json).

## 5. Caveats — what must be confirmed on real footage

- **MediaPipe on the mirror view is the whole risk.** The mirrored hand is left-right flipped (handedness will
  come out wrong and must be swapped), 0.7–0.85× smaller, seen from 5–18° above the desk — a viewpoint the
  tracker was never validated on in this project. The 0.8 px noise model is an assumption; real landmark error
  on a grazing view could be several px, and a 3 px error at 0.7× scale is still only ~2 mm of height, but
  systematic bias (the tracker "seeing" the tip on the nail rather than the pad) would matter more.
- Visibility is a capsule-on-joints proxy (9 mm; halve-ish at 6 mm), on other people's hands; forearm, sleeve
  and the mirror's own frame are not modelled. The mirror also sees the far side of the hand, so fingers of the
  other hand can occlude in ways this test only partly captures (cross-hand is included; wrist and forearm are not).
- The real phone lens is an ultra-wide with distortion at the frame edge, exactly where the mirror image lands;
  the pinhole frame budget is optimistic by a few percent, and the calibration must include distortion.
- Triangulation needs the mirror plane relative to the camera to ~1°; a bracket or a per-session calibration
  is required. Any bump to the mirror invalidates the height until re-calibrated.
- The bare-desk product scenario has no laptop between the hands and the mirror; with a laptop the rear
  placement is blocked and the 45° placement is the fallback (0.80–0.85).
- EgoPressure hovers are single-finger presses at generous heights (median 15 mm); in typing hovers are lower
  (7 mm), which still leaves >10× the height error. The same per-finger visibility numbers as §2 would set the
  ceiling for ten-finger use.
- No rendering: no motion blur, exposure difference between the direct and mirrored views, or dust/smudges.

**Confirming recording (no new tooling beyond calibration):** the existing top-down rig plus a 22×8 cm mirror
on a small stand 16 cm beyond the fingertips, glass centre 6 cm up, leaning back ~23°; 3 minutes of keyboard
typing with the keylogger. Check three things: (1) MediaPipe finds the mirrored hand and its tips in ≥80% of
frames; (2) the triangulated index-tip height at key-down vs 150 ms before it differs by ≥5 mm with a per-event
spread under 2 mm; (3) the pv_probe/shadow_probe harness with the triangulated height added beats the kinematic
0.928 and the static-image control by more than +0.03. If (1) fails, the idea fails on tracking, not geometry.
