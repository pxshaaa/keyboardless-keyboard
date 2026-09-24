# PressureVision++ contact signal vs our kinematic cues — SUMMARY (2026-09-20, Mac mini)

## Verdict: NO. Stop here.
PressureVision++ (PV) does not predict real key-downs better than the cues we already have, and adds
nothing usable on top of them. Its raw contact score looks informative (AUC 0.66), but a static-image
control — PV run once on the session's mean image and read out at the same fingertip positions — scores
the same (0.69). The signal is fingertip geometry, not image evidence of contact.

Session 20260911-164237-kbd (keyboard present, exact keylogger truth), 40,315 frames with landmarks,
1,734 key-downs, positive = frame within +-40 ms of a key-down (7,571 positives). AUC, 95% block-bootstrap CI:

| score                                   | AUC   | 95% CI        |
|-----------------------------------------|-------|---------------|
| PV contact, raw max over 10 tips        | 0.660 | 0.618-0.706   |
| PV pressure, raw max                    | 0.645 | 0.601-0.694   |
| PV "finger head" bottleneck, raw max    | 0.693 | 0.651-0.733   |
| **control: static mean image, raw**    | 0.688 | 0.648-0.731   |
| our stop-motion cue, raw max            | 0.525 | 0.510-0.540   |
| PV features -> LightGBM (out-of-fold)   | 0.898 | 0.877-0.918   |
| **our kinematic features -> LightGBM**  | **0.928** | 0.913-0.943 |
| PV + kinematic -> LightGBM              | 0.930 | 0.915-0.945   |
| control: static-image feats -> LightGBM | 0.871 | 0.846-0.895   |
| control: PV LightGBM, time-shuffled     | 0.504 | 0.430-0.583   |
| control: PV raw, time-shuffled          | 0.497 | 0.445-0.554   |

Paired deltas (same frames, block bootstrap):
- PV LightGBM - kinematic LightGBM: **-0.030 [-0.039, -0.023]** (PV alone is worse)
- (PV + kinematic) - kinematic: **+0.002 [+0.000, +0.004]** (no usable gain)
- PV LightGBM - static-image LightGBM: +0.027 [+0.018, +0.038] (PV carries ~0.03 AUC of real image
  information; all of it is already in the kinematic features)
- PV raw - stop-motion raw: +0.135 [+0.097, +0.176] (explained by the static-image control above)

Cost for that nothing: 73,398 hand crops at 17.6 crops/s on the mini's GPU = 70 min for a 25-min session.

Second check, PV vs our own fingertip-crop CNN on identical frames (results/pressure/step1_vspix_*.json;
CNN trained on the other 4 keyboard sessions, 3 seeds; frames = the pixel bank for this session,
key-down-anchored positives plus sampled near/far negatives, so absolute AUCs are not comparable with the
dense-frame table above — only the ranking is):

| score                       | AUC   | 95% CI      |
|-----------------------------|-------|-------------|
| **our fingertip CNN (mean of 3 seeds)** | **0.904** | 0.874-0.927 (seeds 0.908 / 0.888 / 0.903) |
| PV contact, raw max         | 0.569 | 0.531-0.608 |
| PV pressure, raw max        | 0.535 | 0.498-0.570 |
| stop-motion cue, raw max    | 0.503 | 0.488-0.520 |
PV - our CNN: **-0.335 [-0.380, -0.285]**. A small CNN trained on our own footage sees far more contact
evidence in the same crops than PV does. Whatever PV learned on its pressure-pad data does not transfer.

## Per finger (positives = key-downs of that finger; negatives = frames >150 ms from any key)
| finger | keys | AUC PV | AUC static-image ctrl | AUC stop-motion raw |
|--------|------|--------|-----------------------|---------------------|
| L-ix   | 233  | 0.704  | 0.628 | 0.602 |
| L-mi   | 192  | 0.672  | 0.669 | 0.502 |
| L-ri   | 125  | 0.630  | 0.624 | 0.510 |
| L-pi   | 114  | 0.705  | 0.467 | 0.476 |
| R-ix   | 186  | 0.753  | 0.797 | 0.458 |
| R-mi   |  85  | 0.764  | 0.439 | 0.450 |
| R-ri   | 159  | 0.598  | 0.543 | 0.453 |
| R-pi   |  31  | 0.780  | 0.465 | 0.449 |
| thumb (space, either) | 264 | 0.658 | - | 0.477 |
Per finger PV beats the static control on some fingers (L-pi, R-mi, R-pi) and not others (L-mi, L-ri,
R-ix). No finger gets past ~0.78, and none of this survives once the kinematic model is in the mix.
Raw file: results/pressure/step1_perfinger_staticimg_ctrl.json.

## Protocol
- Model: vendored PressureVision++ (MIT), checkpoint pv2/data/model/paper_29.pth, preprocessing copied
  from pv2/prediction/pred_util.py (448 px crop, hand bbox from MediaPipe landmarks x1.5, ImageNet norm).
  Run on MPS in batches of 12, both hands, every frame (stride 1). Extraction log
  .cache/pressurevision/logs/extract_164237.log, features .cache/pressurevision/feats/<sid>.npz.
- Per-fingertip read-out: max/mean of the 9-class contact posterior (1 - P[no contact]) and expected
  pressure inside a disc of radius tip_r (median 25 px at 448) around each MediaPipe tip; plus per-crop
  globals and the 7-d bottleneck logits.
- Probe: phase0/analysis/pv_probe.py. Labels from keys.jsonl key-down timestamps. LightGBM out-of-fold
  over 5 contiguous time folds with a 1 s purge either side of each fold; block bootstrap over 10 s
  blocks, 2000 resamples, for every CI and paired delta. Kinematic features = the existing taps_gb groups
  + touch_feats per-hand/pooled features (2,122 dims) — the same cues the deployed tap detector uses.
- Controls: (1) static image — PV run on the per-hand mean image of the whole session, sampled at each
  frame's tip positions, so it can only carry fingertip-position information; (2) time shuffle — PV
  scores rolled by 60 s against the same labels (both at chance, so no leakage); (3) the paired deltas
  are on identical frames. The random-label control from the brief was not run: it guards against a
  spurious *positive*, and the result here is negative with the positive-looking raw AUC already
  explained by the static-image control.
- pv_vspix (PV vs our fingertip-crop CNN): phase0/analysis/pv_vspix.py, CNN = touch_pix recipe retrained
  without the probe session, 3 seeds on MPS under gpulock. I had stopped the original run_after.sh chain
  so this GPU job would not queue ahead of the RTM training; a watchdog script left by the previous
  session (.cache/pressurevision/watchdog.sh) relaunched it at 23:36, after the RTM fine-tunes had
  released the lock, and it finished at 23:39. The lock serialised everything; no two GPU jobs ran
  together.

## Caveats
- One session, keyboard present. On a bare desk the visual contact cue is if anything weaker (no key
  travel, no key-cap deformation), so this is the favourable case for PV and it still fails.
- PV++ was trained on hands pressing a flat pressure pad from cameras at oblique angles; our view is
  top-down over a keyboard. The contact posterior floor is 0.37 (the network never gives >63% to
  "no contact"), and the mean-image control shows the heatmap largely follows hand shape. Domain
  mismatch is the likely reason; fine-tuning PV on our data would need contact labels we do not have.
- Left hand missing from MediaPipe in 17% of frames (right hand 0.7%); those frames have no PV score for
  the left tips and are excluded from per-finger rows for that hand.
- The earlier results/pressure/step1_*.json (22:08) was a 6,000-frame partial run and has been
  overwritten by the full run (23:09-23:29).
