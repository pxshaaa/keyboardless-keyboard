# Is a 45° camera worth it for bare-desk typing? — SUMMARY (2026-09-25, Mac mini)

## Verdict: NO for 45° specifically. Do not re-record at 45°.

Three independent measurements, two of them new and one on real footage, agree: tilting the camera to 45°
changes contact detection and end-to-end accuracy by **nothing measurable**, and the one thing it clearly
improves — fingertip self-occlusion, which halves — does not turn into text accuracy.

| what | top-down (90°) | 45° | paired difference | verdict |
|---|---|---|---|---|
| key identification, top-1, real ten-finger typing (29,772 keystrokes, 30 people) | 0.317 | 0.338 | **+0.020 ± 0.016 SE**, 17/30 people better | not significant |
| keystroke-vs-hover AUC, same data | 0.917 | 0.912 | **−0.004 ± 0.002 SE**, 10/30 better | slightly worse |
| fingertip occlusion (all tips / pressing tip) | 0.309 / 0.437 | 0.193 / 0.330 | **−0.12 / −0.11** | clearly better |
| planar key-localisation error at contact (median / p90) | 0.9 / 2.6 mm | 1.1 / 3.6 mm | +0.2 / +1.0 mm | slightly worse |
| CTC character error rate, pretrained at that angle, 6 held-out participants, 2 seeds | 0.362 | 0.355 | −0.007 (seed spread 0.005–0.012) | within noise |
| contact AUC on **real** oblique footage (PressureVisionDB, 3 people, same frames) | 0.852 | 0.895 at 40° | **+0.070**, 3/3 better | positive but n=3 |
| MediaPipe landmark yield on that real footage | 0.844 | 0.837 at 40° | −0.007 | **the tracker does not care about the angle** |

**The one genuinely new fact:** going lower than 45° does help key identification — at **30°** the paired
gain is **+0.054 ± 0.018 SE** (20/30 people better), three standard errors, the only statistically solid
angle effect in this study. But at 30° keystroke-vs-hover falls by 0.014 ± 0.002 (5/30 better) and
localisation error more than doubles (2.0 mm median, 6.1 mm p90 against a 19 mm key pitch). It is a trade,
not a win, and the CTC CER at 30° (0.350) is still inside seed noise of the top-down 0.362.

**Recommendation.** Keep the camera top-down. If I spend one recording on this anyway, it should be
at **30–40°, not 45°** — that is where the measurable effects are — and use the protocol in
`docs/ANGLE45_TRIAL_PROTOCOL.md`, which scores the new session against a top-down session with identical
code. Re-recording the labelled desk corpus is only justified if that trial shows a CER improvement larger
than 0.03; nothing measured here predicts that it will.

## 1. Real ten-finger typing through virtual cameras (the decisive simulated answer)

How We Type mocap (30 participants, 240 Hz finger markers + keylogger) re-projected through a virtual
camera at 90/60/45/30° front elevation, matched to our rig (f = 538 px, 1280×720, 227 mm to the hand
centre) with the same 0.8 px landmark noise as `results/angle/`. Code `phase0/analysis/hwt45.py`; raw
`results/angle45/hwt_angle_metrics.json`.

| elevation | key-ID top-1 (25 classes) | keystroke-vs-hover AUC | occlusion, all tips | occlusion, pressing tip | localisation at contact, median / p90 |
|---|---|---|---|---|---|
| 90° (today) | 0.317 | 0.917 | 0.309 | 0.437 | 0.9 / 2.6 mm |
| 60° | 0.326 | 0.919 | 0.225 | 0.362 | 0.7 / 2.0 mm |
| **45°** | **0.338** | **0.912** | **0.193** | **0.330** | **1.1 / 3.6 mm** |
| 30° | 0.372 | 0.903 | 0.171 | 0.288 | 2.0 / 6.1 mm |

Paired per-participant differences against top-down (same people, same keystrokes — the right statistic,
since the spread across people is ±0.11 in key-ID and dwarfs every angle effect):

| vs 90° | Δ key-ID | Δ keystroke-vs-hover |
|---|---|---|
| 60° | +0.008 ± 0.010 SE (16/30 better) | +0.001 ± 0.001 (14/30) |
| 45° | +0.020 ± 0.016 SE (17/30 better) | −0.004 ± 0.002 (10/30) |
| 30° | **+0.054 ± 0.018 SE (20/30 better)** | **−0.014 ± 0.002 (5/30)** |

Why key-ID improves while localisation worsens: they are different quantities. Back-projecting a noisy
pixel onto the desk plane degrades at grazing angles (localisation), but an oblique view maps finger
extension and keyboard-row depth into the image, which is what a key classifier uses. Key-ID is from the
whole two-hand pose at the key-down frame; it tops out at 0.317–0.372 against a 25-class problem at any
angle, which is the same "landing positions do not localise keys" ceiling already in REPORT.md.

## 2. Real footage: the same hands from four angles at once (PressureVisionDB, MIT)

PressureVisionDB records every sequence with four calibrated cameras and a per-frame pressure image. From
each sequence's own `metadata.json`, taking camera_2's axis as the surface normal, the cameras sit at
**90°, 51°, 40° and 39°** above the surface (the papers say "~45°"). The `type_sentence_5x_both` sequences
are two-handed typing. MediaPipe was run on all four views of the same frames, then our own feature stack
(static pose groups only — the published small set is 3 fps, so velocity features are meaningless) and a
leave-one-participant-out contact probe. 3 participants, 595 frames, 396 in contact. Code
`phase0/analysis/pvdb45.py`; raw `results/angle45/pvdb_contact_probe.json`, `pvdb_landmark_yield.json`.

| view | elevation | contact AUC | per participant | paired Δ vs 90° | landmark rate |
|---|---|---|---|---|---|
| camera_2 | 90° | 0.852 | 0.719 / 0.952 / 0.852 | — | 0.844 |
| camera_3 | 40° | **0.895** | 0.889 / 0.976 / 0.870 | **+0.070 (3/3 better)** | 0.837 |
| camera_0 | 39° | 0.863 | 0.833 / 0.941 / 0.895 | +0.048 (2/3) | 0.840 |
| camera_1 | 51° | 0.859 | 0.827 / 0.960 / 0.850 | +0.038 (2/3) | 0.844 |
| time-shuffled control | — | 0.465–0.492 | | | |

Read it carefully: every oblique view beats the near-normal one, which is the first real-footage evidence
that the angle helps at all. But **two views at almost the same elevation (39° and 40°) differ by 0.032**,
which is as large as the gap to top-down — they differ in azimuth, not just elevation, so part of what looks
like an elevation effect is azimuth and rig geometry. With n = 3 participants this is suggestive, not
decisive. The solid finding here is the control one: **MediaPipe's detection rate is flat across all four
angles (0.837–0.844)**, so the tracker is not the obstacle to tilting the camera.

## 3. The 45° pipeline (built and evaluated)

The HWT pretraining corpus — which is what the shipped model is initialised from — was previously an
effectively top-down orthographic projection with no camera parameter. `phase0/analysis/hwt45.py build`
adds one: it reconstructs metric 3D from the mocap, projects through a virtual camera at any elevation,
adds landmark noise, and writes drop-in `seqctc` streams to `.cache/hwt/streams_el{90,60,45,30}/`.
Sanity anchor: at 90° the projection reproduces the existing shipped streams (per-joint correlation
0.994–0.998, median difference 0.03–0.09 hand-scale units), so the 90° arm is a true control rather than a
different pipeline.

The shipped recipe (`seqctc` transformer, 1500 steps, 24 train participants, 6 held out) was then trained
at each angle and scored on held-out participants **at the same angle** — i.e. "if the whole world were
filmed at this angle, how well would the model read typing?":

| pretraining angle | CER, 6 held-out participants (seed 0 / seed 1) | mean |
|---|---|---|
| 90° | 0.360 / 0.365 | 0.362 |
| 45° | 0.357 / 0.353 | **0.355** |
| 30° | 0.344 / 0.356 | 0.350 |

The 45° model is 0.007 better than the 90° model and the seed-to-seed spread is 0.005–0.012. **There is no
angle effect here that a second seed could not produce.** Checkpoints are in `.cache/hwt/pre45/` and are
what `angle45_trial` loads for a real 45° session.

## 4. Protocol and one-command trial

* `docs/ANGLE45_TRIAL_PROTOCOL.md` — where to put the phone (45° elevation, 23 cm to the hand centre =
  16 cm above the desk and 16 cm beyond the fingertips, same lens and 1280×720 @ 60 fps), how to check
  framing, and the two recorder commands (12 min keyboard with keylogger labels + 3 min bare-desk phrases).
* `python -m phase0.analysis.angle45_trial run --session <new-id> --baseline 20260911-164237-kbd` —
  extracts landmarks, then runs the *same* two measurements on the new 45° session and on a top-down
  session and prints the difference: key-down detection AUC (pv_probe protocol, time-blocked CV, 10 s block
  bootstrap, time-shuffled control) and CTC CER on the held-out last quarter, fine-tuned from the matching
  pretrained checkpoint plus a from-scratch control.
* Verified on an existing session: it reproduces that session's known key-down AUC of 0.869
  (`results/shadow/SUMMARY.md`) exactly. The two metrics run in separate processes on purpose — LightGBM's
  OpenMP runtime and torch/MPS segfault in one process on this machine.

## 5. Resources

`docs/ANGLE45_RESOURCES.md` surveys what exists at 30–60°, with licences. Two things worth repeating here:
PressureVisionDB is **MIT** (commercially usable) and is the only public corpus with simultaneous
near-normal and oblique views of hands touching a surface; and **TypingHands26**, the only public
flat-surface typing video corpus and the closest prior system to ours, is announced but its repository
contains a 110-byte README and no data. A 533 MB subset of PressureVisionDB was obtained without the
19.4 GB download by reading the zip's central directory over HTTP range requests
(`phase0/tools/zipsub.py`), well inside the 12 GB budget; total new data this session ≈ 1.7 GB.

## 6. Caveats

* How We Type is **CC BY-NC 4.0** and EgoPressure **CC BY-NC-SA**: every number in sections 1 and 3, and
  the `.cache/hwt/streams_el*` corpus and `pre45` checkpoints derived from them, are **research
  measurements only and must never ship**. PressureVisionDB (section 2) is MIT and carries no such limit.
* Section 1 is mocap re-projected through an ideal pinhole with isotropic 0.8 px noise: no motion blur, no
  rolling shutter, no MediaPipe failure modes, and hand-scale normalisation that a real tracker would not
  reproduce exactly. The occlusion figures are a 9 mm capsule test on joints with no forearm or sleeve.
* Section 2 is 3 participants, 595 frames, 3 fps, static pose features only, and a frame-level "any finger
  in contact" label rather than per-key. Its views differ in azimuth as well as elevation.
* Section 3 trains and tests at the *same* angle. It does not measure transfer from a 90°-pretrained model
  to 45° footage, which is what would matter if I switched angle without re-pretraining — that arm
  needs the real trial session.
* Nothing here measures the product condition directly: all of it is keyboard typing (HWT, PVDB), not a
  bare desk. The bare-desk block in the protocol is what closes that gap.
