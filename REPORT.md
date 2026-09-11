# Computer-vision keyboard project — status

_Last updated: 2026-09-10, overnight session._

## Where this stands

**Goal A (gesture shortcuts):** harness built, not started — deliberately parked behind Goal B.
**Goal B (camera reads typing):** capture is solved; tap detection is the open problem.

## What works

### Capture — solved
The blocking problem all evening was field of view. Resolved by building our own iOS app.

| Route | Ultra-wide | fps | Verdict |
|---|---|---|---|
| Continuity Camera | no (1x only, hard floor) | 30 | Ruled out — measured, not assumed |
| Apple Desk View | ultra-wide *sensor*, cropped FOV | 30 | Wrong crop; aims at the desk in front of a laptop |
| Camo Pro (paid) | yes | 60 | Viable fallback, ~$5/mo |
| **WideCam (ours)** | **yes, true 0.5x** | **60** | **In use** |

`widecam/` — SwiftUI app streaming MJPEG over WiFi with Bonjour discovery, per-frame
phone-side timestamps, exposure lock. `phase0/capture/netsource.py` receives it.
Measured: **1280x720 @ 60.07 fps, 0 dropped, 0 duplicate frames, ~45 Mbit/s.**

Verified negatives worth not re-testing:
- Continuity Camera exposes no exposure control at all (every mode unsupported, bias range 0.0-0.0).
- `minAvailableVideoZoomFactor` is 1.0 — you cannot zoom out to the ultra-wide.
- Center Stage *narrows* the frame (~2x crop), never widens it.
- `videoFieldOfView` reports 0.0 for every format; no wider format is hiding in the list.
- OpenCV's AVFoundation backend sees only a prefix of the device list — it cannot open Desk View.
- A camera can read at 56 fps while returning byte-identical frames (closed lid). The recorder
  now refuses to record that; nothing downstream can detect it after the fact.

### Data
Three sessions, ground truth from a local keylogger (allowlisted keys only; everything else
logged as `unknown` with its timestamp preserved).

| Session | Length | Keydowns | Role |
|---|---|---|---|
| 20260910-015217-kbd | 45 s | 107 | train |
| 20260910-021315-kbd | 240 s | 943 | train |
| 20260910-015948-kbd | 90 s | 221 | **held-out test** |

Landmarks now include depth: `z` (image-space) and `wx wy wz` (metric world coordinates).

## The open problem: tap detection

Scored by `phase0/analysis/eval_taps.py` — 1:1 assignment within ±80 ms, plus false
positives during still periods. All numbers below are the **unseen test session**, clipped
to the typing span.

| Detector | F1 | Recall | Precision |
|---|---|---|---|
| `detect_taps` (image-y velocity zero-crossing) | 43.4% | 71.9% | 31.1% |
| `taps_flexion` (palm-residual speed pulse) | 39.7% | 33.0% | 49.7% |
| `taps_mc` (motion-compensated + cross-finger NMS) | 51.2% | 55.2% | 47.7% |
| `taps_ml` supervised, 107 labels | 59.4% | 68.3% | 52.6% |
| `taps_ml` supervised, 1,050 labels (no-depth variant, model not saved) | 70.3% | 75.6% | 65.7% |
| `taps_ml` as saved on disk (depth variant) | 65.6% | 65.6% | 65.6% |
| `taps_gb` LightGBM, 1,050 labels | 76.3% | 75.1% | 77.6% |
| `taps_gb`, 2,743 labels | 75.7% | 72.4% | 79.2% |
| **`taps_gb`, 2,743 labels + widened refractory grid** | **77.5%** | **82.8%** | 72.9% |

Still-period false positives — the "it fires while my hands rest" bug — fell from
**159 taps (31%) to 11 (5%)** across that progression. Median timing error 20 ms.

### What moved the needle
1. **More labels.** 107 → 1,050 training keypresses took F1 from 59.4% to 70.3%. Nothing else
   came close to that effect size. Each minute of ordinary typing yields ~230 free labels.
2. **Dynamics only, no static pose.** Including absolute posture lets the model memorise
   "typing posture" and precision collapses (a hold-out produced 231 taps for 31 keypresses).
   Features are differences and velocities only.
3. **Threshold tuned on out-of-fold probabilities.** Tuning on in-sample probabilities picked a
   badly over-fit operating point.
4. **Motion-energy and resting-pose features inside the classifier**, rather than a separate
   "is the user typing" gate. An explicit second-stage gate turned out nearly redundant
   (12 -> 11 still-FPs); putting the same information in as features is what took the rate
   from 31% to 5%. Requiring 4 consecutive frames above threshold was worth more than the gate.
5. **Depth bought nothing measurable** (65.6% with vs 70.3% without). Cross-validation marginally
   preferred it, the held-out session preferred without. Kept the CV-selected model as default
   rather than choosing on the test set.

### What has been learned
- **Fixed pixel-velocity gates were the original bug.** The palm itself moves at 73-110 px/s;
  a 60 px/s "descent" gate fires on hand travel. Noise-relative (MAD) gates fixed that.
- **Cross-finger suppression is the single biggest hand-crafted win** (+23 F1): the raw signal
  pulses on every finger of a hand at once.
- **Palm-frame compensation alone *hurts*** unless the press axis is estimated correctly;
  PCA over a 1 s window picks lateral key-to-key travel, not the press.
- **Hand-crafted rules overfit badly.** `taps_flexion` scored 68.5% on the session it was
  tuned on and 39.7% on the next one. Any number not measured on a held-out session is noise.
- **Which finger tapped: solved, after an early wrong diagnosis.** An argmax-over-fingers rule
  does sit at chance, and that was first read as a camera-angle limitation. It is not. A
  supervised classifier reaches **56% top-1 / 77% top-2 on 9 classes (chance 11%)** and
  **75% hand attribution (majority baseline 54%)**, validated against a shuffled-label control
  that scores exactly the majority baseline.
- **Cross-validation is meaningless below ~1,000 labels.** On 107 labels, blocked CV gave
  F1 0.415 ± 0.274 with one fold at 0.00; at 1,050 labels it is 0.664 ± 0.041.
- Timing, when a tap is detected at all, is good: median error 17-29 ms.

## Why "good enough" is lower than it sounds
TouchInsight (Meta RL + ETH, UIST'24) had a vision front-end wrong about the key **25.3% of
the time**, and still reached **2.9% uncorrected error** after beam search over a character +
word language model. Target ~75-85% detection with tight timing; the decoder recovers the rest.
Chasing 95% detection is the wrong optimisation.

## Not worth doing
- **Training on internet videos of typing.** The entire advantage here is that the keylogger
  provides perfect free labels. Public video has none, plus a different keyboard, hands and angle.
- **A large from-scratch vision network.** MediaPipe already solves pixels→21 joints. The missing
  piece is a small temporal model on top, and that layer is inherently person-specific.

### Calibration — important for the next stage
The winning model's probabilities are **monotone but systematically over-confident**
(ECE 0.055; predicted 0.85 corresponds to an observed 0.62), because `scale_pos_weight=8`
deliberately distorts the prior. The ranking is sound (AP 0.613 at a 15% base rate). Since the
eventual text decoder wants a *distribution* rather than an argmax — TouchInsight gained 2.5
points of character error from modelling uncertainty — these scores need post-hoc isotonic or
Platt recalibration on a held-out session before being fed downstream.

## Finger attribution

| Task | Accuracy | Baseline |
|---|---|---|
| Finger, 9 classes, top-1 | 56.4% | 11% uniform / 17% majority |
| Finger, 9 classes, top-2 | 77.0% | — |
| Hand (left/right) | 74.6% | 53.8% majority |
| Thumb / space | 88.6% | — |
| Both index fingers | ~70% | — |

Ring, middle and pinky are where it degrades, and a third of errors cross hands.

### Why the obvious rule fails — the useful diagnostic
Measured over the training sessions, in physical units (1 px ~ 0.35 mm at this framing):

| Quantity | Value |
|---|---|
| Tracker jitter | **0.28 mm / frame** |
| Fingertip excursion during a keypress | **9.5 mm** |
| Motion SNR | **~34x** |
| Pressing finger's excursion vs its siblings' | **0.7 mm SMALLER** |

The fingers move enormously relative to tracker noise, so **resolution was never the constraint**.
The problem is contrast: the non-pressing fingers move *more* than the presser, because they are
already travelling toward the next key. "Whichever finger moved most" is therefore close to
anti-correlated with the truth. Finger identity lives in the joint posture of both hands.

Depth helps only this naive rule (lifting it from 1.0x to 1.9x chance) and is neutral-to-harmful
for the supervised model — consistent with the tap-detection result.

### How to use it in Phase 0
- **Use hand attribution now** (75%) — it already separates the per-key landing data usefully.
- **Do not trust a single tap's finger label** at 56%. Aggregate per key: with the contract's
  minimum of 15 samples per key, a majority vote over per-tap predictions is far more reliable
  than any individual prediction.
- **What would actually improve it**, in order: a second camera from the side (the missing
  information is finger *elevation*, which this along-the-desk view foreshortens and which
  MediaPipe's `z` does not recover); more labelled data (913 training taps for a 9-class problem
  is the binding constraint); a temporal model over the tap *sequence*, since letter-transition
  statistics constrain which finger can follow which. Higher resolution is not the fix.

Caveat: labels come from a standard touch-typing key→finger map and the user is demonstrably not
a strict touch typist, so some of the 44% error is label error. Treat 56% as a lower bound.

## Data scaling saturates early — measured, superseding the earlier extrapolation

A properly-powered study (109 training runs, leave-one-session-out, 5 seeds/point, contiguous
calibration windows) replaced the earlier single-split curve. It changes the picture:

- **The curve is flat above ~300 keypresses (~1.5 minutes).** On the two well-behaved folds,
  300 keys already gives 64.6% / 69.1%; going to 1,164 buys ~5 points, inside the seed spread.
- **A 10-minute recording confirmed it from the other side.** Going 1,050 -> 2,743 labels moved
  F1 from 76.3% to 75.7% — flat — though it did take still-period false fires from 11 to **0**
  and improved timing. The earlier "+7-13 points per doubling, curve not flattened" claim is
  superseded by measurement.
- **The headline 76.3% was a favourable single draw.** Reproduced over 5 seeds, the same
  configuration gives mean 68.1%, median 71.2%, IQR [61.5, 72.4] — the published number sits
  *above* the IQR. Honest expected value at that configuration: **~68-71%**.
- **Realistic plateau: 65-70% measured.** 80% is not reachable by adding calibration typing.
  Saturating-curve fits do not identify an asymptote (pooled 95% CI spans 57-100%), so all
  extrapolations past the measured range are indicative only.

### The finding that matters most for a product
**Test-session identity explains 22.6% of F1 variance; training-set size explains only 13.7%.**
Session `015217` as the held-out set never exceeds 56% regardless of calibration volume. Two
sessions of the *same person, same rig, forty minutes apart* differ more than data volume does.
**Cross-session robustness is a prerequisite for cross-user generalisation**, and it is the larger
product risk — bigger than camera-angle diversity.

### Free money found and taken
`EVENT_GRID["refractory"]` was `(5, 7, 9, 11)` frames, so the tuner never tried below 83 ms.
A sweep showed 67 ms (4 frames) better on all three folds. Widening the grid to `(3, 4, 5, 7, 9, 11)`
took the held-out session from F1 75.7% to **77.5%**, with recall **72.4% -> 82.8%**.
Recall was throttled by a search-grid boundary, not by typing physics.

## Honest limits on the current 77.5%
- All sessions are the same person, same camera, same keyboard, same night. This says nothing
  about another typist or rig.
- ~~Recall has a hard ceiling from rollover typing.~~ **This was wrong and is retracted.** ~28% of
  keydowns do follow within 80 ms, but `eval_taps` pairs within +/-80 ms, so an extractor may fire
  *early* for a fast pair and still match both. Computed exactly on the real frame grid, maximum
  achievable recall under an 80 ms refractory is **99.8%** (max F1 99.9%). The ceiling only binds
  if +/-20 ms timing is demanded (89%). Rollover is not the constraint.
- Finger attribution inside the model is still a placeholder, not a measured capability.

## Live preview
The preview now runs the real model (`--detector model`, the default), not the placeholder
heuristic that fired on resting hands.

| | F1 | still-FP | timing err |
|---|---|---|---|
| Offline `taps_gb` | 76.3% | 11 (5%) | 20 ms |
| **Online replay of the same session** | **74.7%** | **10 (4%)** | 23 ms |
| Old inline heuristic | 43.4% | constant | 28 ms |

Costs 1.6 F1 points versus offline, entirely in precision, because a causal first-come
refractory replaces height-ordered peak suppression. **Latency 9-12 frames (149-199 ms)** before
a flash appears; tap *timestamps* are unaffected. Preview runs at **29-34 fps** end-to-end at
1280x720 with the model in the loop (45 fps with the heuristic), so no frame decimation was needed.

Measured, not guessed: the lookahead window is non-monotone and short wins — 8 frames scores
75.6%, while 90 frames scores 72.2%, because long lookaheads lose real detections to warm-up and
cool-down edges.

## Testing note
`phase0/tests/test_online_taps.py` must be run **separately** — 210 tests pass in the main suite
and its own 20 pass alone, but running both in one process aborts inside a model load
(LightGBM and MediaPipe together). Not a product bug; a test-runner constraint.

## How to run it

```bash
cd ~/Documents/misc/computer-vision-test

# 1. open WideCam on the iPhone, then watch it work
.venv/bin/python -m phase0.capture.preview --backend net --camera auto \
      --width 1280 --height 720 --rotate ccw90

# 2. record (keyboard present)
.venv/bin/python -m phase0.capture.recorder --condition kbd --backend net --camera auto \
      --width 1280 --height 720 --fps 60 --seconds 600

# 3. analyse
.venv/bin/python -m phase0.analysis.extract_landmarks <session>
.venv/bin/python -m phase0.analysis.detect <session>     # writes taps.jsonl with the trained model
.venv/bin/python -m phase0.analysis.eval_taps <session> --clip   # score vs ground truth
```

`phase0/analysis/detect.py` is the single entry point — it runs the best available model and
writes `taps.jsonl`, which is what `analyze_drift` reads. `--detector kinematic` falls back to
the old rule-based one.

Retrain whenever you add sessions:
```bash
.venv/bin/python -m phase0.analysis.taps_gb train --sessions data/sessions/*-kbd
```

## LiDAR — tested on device, verdict no

Depth was measured rather than reasoned about, and the earlier physics prior was **wrong in every
detail** — it fails for a different reason than assumed.

| | Assumed | Measured on this iPhone 15 Pro |
|---|---|---|
| Depth resolution | 256x192 | **320x180, 1.01 mm/px at the 227 mm working distance** |
| Static noise floor | ~centimetre | **0.57 mm** (0.42 mm drift-free) |
| Fingertip coverage | 2-3 px | **~16x16 px** |

Depth quality was never the problem. The hardware trade is:

| Camera | FOV | Depth | Accuracy |
|---|---|---|---|
| `builtInUltraWideCamera` (in use) | **106.2°** | **none — 0 of 61 formats** | — |
| `builtInLiDARDepthCamera` | 74.6° | 320x180 @ 30 fps | absolute / high |
| `builtInDualWideCamera` (stereo) | 106.2° | 160x90-320x180 @ 30 fps | **relative / low** |

`builtInLiDARDepthCamera.constituentDevices` is `[builtInWideAngleCamera]` — **LiDAR and the
ultra-wide are mutually exclusive.** Taking depth costs 67% of the ground area and half the frame
rate, and the consequence was measured directly: MediaPipe found hands in **0 of 898 frames** in
LiDAR mode versus **1200 of 1200** on the ultra-wide, on the same rig minutes later. The whole
detector is lost to gain a depth channel.

The stereo alternative that keeps the wide view reports `depthDataAccuracy = .relative` — not
metric at all — with a 10.8 mm noise floor against the 9.5 mm signal, i.e. SNR below 1.

Even favourably: 2.5-3.4 mm on-skin noise, 30 fps cap (~3 depth samples per keypress), noise that
spatial averaging cannot reduce (1x1 0.57 -> 5x5 0.56 mm, so it is spatially correlated), and 8-9%
invalid pixels. Three independent measurements now agree that depth does not help this problem.

## Public training data — a dataset does exist
An earlier judgement in this report that internet footage could not help was **wrong**.

[`andrewt28/keystroke-typing-videos`](https://huggingface.co/datasets/andrewt28/keystroke-typing-videos)
— 800 clips, **51,336 labelled keydowns** (~40x our data), ~90 min, 0.71 GB, AFL-3.0, ungated.
Fixed camera above a MacBook looking down at the keyboard with both hands visible: the same
geometry as ours, rotated 180°. Labels carry millisecond timestamps and were validated against the
clips' own audio onsets (88/96 within 150 ms, median offset 9.5 ms).

Trap: the container claims 30 fps but real capture drifted (27.98-30.52). Timestamps must be
converted with the per-clip `actual_fps` field or labels misalign by up to ~800 ms.

Limits: 640x480 @ ~29.5 fps against our 1280x720 @ 60, so labels quantise to ~34 ms — twice our
tolerance. Different person and keyboard, no key-up events. The plausible use is **pretraining then
fine-tuning on our own labels**, not naive pooling. That experiment is running.

Also available if multi-person data is ever wanted: **How We Type** (Aalto, CHI'16) — 30
participants at **120 fps** from atop the monitor, plus 240 fps mocap and keylogs, CC-BY-NC, 29 GB.
Video-to-keylog sync is undocumented and would need verifying first.

## The desk experiment — first valid measurement

### A retraction first
The desk session `20260910-181947-desk` has **invalid ground truth**. The recorder prompted phrases
expecting transcription; the user typed their own free-form thoughts instead, because the prompter's
output went to a background log they never saw. Every number derived from it is void: the 0/12
alignment failures, the "2x over-detection" (tap/char ratios were computed against a character count
from phrases never typed), the "typed at half speed on the desk" claim, and the desk CER of 0.83.
The keyboard-side results are unaffected — they use real keylogger ground truth.

### The valid run: `20260910-202149-desk`
20 prompted phrases, **676 characters actually transcribed**, 11,950 frames, 0 dropped, 59.7 fps,
591 taps. Natural sentences rather than pangrams (pangrams defeat the language model, and
over-sample the rare keys we have least training data for).

**Tap detection on a bare desk works**: 18/20 phrases within the healthy tap/char band, median
ratio **0.88**. The detector slightly *under*-detects; it does not double-fire.

### Decoding result

| | Character error | Word error |
|---|---|---|
| No camera (language model alone) | 0.794 | 0.964 |
| **With the camera** | **0.683** | 0.956 |

**The camera contributes ~14% relative. It does not read the typing.** Roughly a third of characters
survive; essentially zero words are correct. Actual output:

```
typed:    'it took longer than i expected'
produced: 'it took out the there it he hatt'

typed:    'the team agreed on the new timeline'
produced: 'the title to the the time the'
```

The decoder sprays high-frequency English ("the", "that", "it", "out") because the vision does not
constrain it enough. It knows roughly how many keys were struck and when, and guesses the rest.

## The ceiling: landing positions do not localise keys
Verified twice, using **real keystrokes as labels** on `20260910-131629-kbd`:

| home-row pair | centroid distance |
|---|---|
| a-s | **22 px** |
| s-d | **64 px** |
| k-l | **120 px** |

Adjacent keys, 5x apart — and **within-key spread is 140 px median**, over twice the distance
*between* keys. Per-key scatter is 2.0-2.4x the calibrated key pitch on all three sessions.

Consequences: a per-key Gaussian on this (x,y) scores **5.1% top-1** (chance 4.8%) and makes the
decoder *worse than no vision at all*. And Phase 0's variance-ratio criterion is undetectable —
injecting 40% label noise moves the median ratio only 1.00 -> 1.11, because the clean baseline
scatter is already keyboard-wide. **Phase 0 cannot be answered until this is fixed**, regardless of
how good the labelling gets.

The information exists: the **pose** model, using whole-hand configuration rather than a single
point, reaches 38% top-1 cross-session / 59% held-out. It is the contact-point estimate that is broken.

## Is this realistic? — an honest assessment
The physics is settled: TouchInsight (Meta RL + ETH, UIST'24) reached **37 WPM at 2.9% error** on a
bare surface from cameras alone. The question is only whether this setup can get there.

| | TouchInsight | Here |
|---|---|---|
| Raw per-key accuracy | ~75% | **38%** |
| Character error after decoding | 8.3% | **68%** |
| Training contact events | **5,290,000** | **~3,000** |
| Participants | 385 | 1 |
| Ground truth | Mocap + capacitive touchpad | Free, from a keylogger |

**The gap is data, by ~1,700x** — not algorithms and not hardware. Their front-end was wrong 25% of
the time and the language model rescued it; ours is wrong 62% of the time, past what a decoder can
recover.

What makes it plausible anyway: **keystrokes label themselves**. ~2,000 free labels per 10 minutes of
ordinary work; ~100,000 is roughly 8 hours of passive recording spread over weeks. That is the one
asset TouchInsight had to buy with a research lab's budget.

Conservative expectation, explicitly extrapolated:
- ~3,000 labels (now): 38% key accuracy, unreadable. *Measured.*
- ~30,000: plausibly 55-65%; fragments readable, prose not.
- ~100,000+: TouchInsight's regime; *might* be usable with correction.

Two caveats not to soften: detection saturated at ~300 keypresses, and key identification is a harder
27-way problem, so the curve may flatten sooner. And all data is one person, one camera position, one
day — session identity already affects results more than data volume does.

**Verdict: not unrealistic, but a data-collection project measured in weeks, not a weekend.** What
exists today is a working *instrument* — capture, detection, labelling, decoding, evaluation — and a
model that is starved rather than wrong.

## Overnight push — desk CER 0.794 -> 0.448

The headline metric is character error rate decoding the valid desk session
(`20260910-202149-desk`, 20 prompted phrases, 676 transcribed characters).

| stage | desk CER |
|---|---|
| no camera (language model alone) | 0.794 |
| pose `key_probs`, keyboard-tuned | 0.728 |
| + desk-tuned extractor | 0.680 |
| + CORAL feature standardisation | 0.617 |
| + EM weak supervision from the known text | 0.602 |
| + sharpened observation weight | 0.555 |
| + pixel CNN on per-fingertip crops | 0.479 |
| **+ high-density extraction (~2.1 taps/char)** | **0.448** [0.399, 0.498], WER 0.745 |
| *ceiling if keys were perfect* | *~0.23-0.32* |

Sample output at the best operating point:
```
typed:    'can you let me know if this works'   produced: 'day you let me of this worse'
typed:    'let me know if you have questions'   produced: 'like her now if you have tone'
typed:    'we should talk about this in person' produced: 'keep all about the soon person'
```

### What worked, and why
1. **Raw pixels carry key information the landmarks discard.** A CNN on per-fingertip crops at
   contact beats the landmark model (LOSO key top-1 0.371 -> 0.448; fused 0.497). Verified it is
   **not** key-legend reading: masking to the hand silhouette *gains* accuracy, finger accuracy
   improves, and background-only is redundant. Roughly half the gain is a better encoder for pose
   we already had (a CNN on the silhouette alone scores 0.410), half is genuine contact appearance.
2. **Detection density was a hard limit, and it was set by a search grid.** Sweeping to 3.6
   taps/char located an optimum at ~1.7-2.1 — roughly 3x the keyboard-tuned operating point.
   Relaxed non-maximum suppression (letting a merged double-tap emit twice) was the most
   productive single lever. Worth -0.068, CI excluding zero.
3. **Sharpening beats calibrating.** Every calibration method (temperature, vector, beta,
   isotonic, Dirichlet, conditional) was neutral-to-harmful; the winning configuration applies an
   effective temperature ~3.7x *sharper* than the NLL optimum. The decoder does not want a
   calibrated probability — it wants the observation term's dynamic range to match the
   insertion/deletion penalty scale. Worth -0.034, and a ~4-line change.
4. **Desk and keyboard have opposite cost structures.** On the desk, insertions are cheaper than
   deletions and high density wins; on the keyboard the reverse. Tuning for keyboard F1 actively
   *hurts* desk decoding — measured twice.

### What was ruled out, with measurements
| idea | result |
|---|---|
| Language-model upgrade | **Nothing.** Modern corpora cut entropy 2.9 -> 1.7 bits/char and bought -0.013 (CI touching zero). GPT-2 rescoring, word bigrams, full hyperparameter grid: nothing. A *clairvoyant* 20-way LM still leaves CER 0.546, so >=75% of the remaining error is the vision's. |
| Motor-sequence / timing layer | +0.007, CI spans zero. Same-finger digraphs are 1.74x slower but the interval resolves only 4.4% of the uncertainty. The character LM already had it. |
| Probability calibration | Neutral to -0.04 worse. |
| Contact-point correction | Geometry fixable (98.8% of the 174 px scatter was one bug) but loses to the pose distribution downstream. |
| Recall-trained detector | Worse at every matched density — the extra recall is the wrong recall. |
| Public-corpus pretraining | Worth ~40-60 seconds of the user's own typing. |
| Depth / LiDAR | Mutually exclusive with the ultra-wide lens; three independent measurements say depth does not help. |

### Corrections made tonight
- **The "cross-session collapse" is withdrawn.** The 27.4-point spread was a protocol artifact; under
  honest tuning the buggy model scores 71.1 / 72.3 / 71.8. Cross-session robustness is *not* the
  outsized product risk this report previously claimed.
- **The hand-slotting bug was worth ~2 F1**, not the +4.9 the first seed suggested, and it does
  nothing for cross-session generalisation.
- **The rollover ceiling claim stays retracted** — max achievable recall under an 80 ms refractory
  is 99.8%, not ~72%.
- The invalid desk session (`20260910-181947-desk`) remains void; the user typed freely rather than
  transcribing, so its ground truth never matched.

## Public footage for key identification — tested, does not help

Question: can online typing footage replace the user's own recording time? The one suitable
corpus (`andrewt28/keystroke-typing-videos`: 800 clips, 51,336 keydowns, top-down MacBook) was
already shown not to help tap *detection*. It was then tested for the current bottleneck, **key
identification**, where per-class data is thinnest (~110 labels per key for us vs ~1,900 public).
Code: `phase0/analysis/keypre.py`. Results: `results/keypre/`.

Setup notes that would otherwise bite: public keyboard is US QWERTY (y/z mapped by physical
position onto the user's QWERTZ); public wrists are out of frame so MediaPipe's guessed wrist made
every hand ~4x too large (hand scale now taken from knuckle width); contact crop taken 67 ms after
keydown; labelled taps are 2,038 train + 172 held-out (not ~3,000), i.e. ~142 usable labels per
minute of typing.

**Pixel CNN, LOSO top-1 (3 seeds), from scratch vs public-pretrained then fine-tuned:**

| our data | from scratch | pretrained | paired delta [95% CI] |
|---|---|---|---|
| 25% | 0.349 | 0.330 | -0.019 [-0.034, -0.004] |
| 50% | 0.392 | 0.377 | -0.015 [-0.031, +0.000] |
| 100% | 0.432 | 0.426 | -0.006 [-0.024, +0.011] |

Worth -0.6 to -1.0 minutes of the user's recording time. Zero-shot (public model only) scores
0.165 on the user's taps, below always guessing space (0.22), with hand identity at chance.

**Desk CER** (same 40 tap streams, same reduced grid, leave-one-phrase-out): from scratch 0.416
[0.357, 0.472] vs pretrained 0.442 [0.378, 0.504]; paired delta **+0.027 [-0.005, +0.061]** —
no help.

**Why it fails (measured):** on its own validation clips the public model gets 0.63 top-1, but
0.40 from the still contact frame alone and only 0.15 from motion alone (chance-level). Shuffling
fingertip patches between slots drops it to 0.19. It learned what that MacBook and camera look
like, not the press. The user's crops and public crops are perfectly separable (domain AUC 1.000).
Resolution, the y/z layout swap and tracking quality were each checked and ruled out.

**Completed accuracy table** (mean over 3 seeds; LOSO = leave-one-session-out over the three
keyboard sessions, held = `20260910-015948-kbd`; `results/keypre/keys.json`):

| model | condition | 25% | 50% | 100% |
|---|---|---|---|---|
| pose (absolute, production) | from scratch | 0.346 / 0.469 | 0.358 / 0.556 | **0.369 / 0.576** |
| pose (hand-relative) | from scratch | 0.349 / 0.380 | 0.361 / 0.417 | 0.280 / 0.452 |
| pose (hand-relative) | public-pretrained | 0.346 / 0.293 | 0.292 / 0.147 | 0.146 / 0.126 |
| pose | zero-shot (public only) | — | — | 0.186 / 0.114 |
| pixel CNN | from scratch | 0.349 / 0.434 | 0.392 / 0.516 | **0.432 / 0.560** |
| pixel CNN | public-pretrained | 0.330 / 0.407 | 0.377 / 0.469 | 0.426 / 0.545 |
| pixel CNN | zero-shot (public only) | — | — | 0.165 / 0.153 |
| **fused pose+pixel** | **both from scratch** | 0.368 / 0.523 | 0.433 / 0.574 | **0.495 / 0.638** |
| fused pose+pixel | pixel pretrained | 0.355 / 0.481 | 0.443 / 0.581 | 0.480 / 0.610 |
| fused pose+pixel | both pretrained | 0.376 / 0.308 | 0.338 / 0.421 | 0.426 / 0.545 |

(cells are LOSO top-1 / held-out top-1.) Pretraining is at or below from-scratch in every family at
every fraction, and the fine-tuned hand-relative pose model degrades catastrophically as more of our
data is added (0.346 -> 0.146 LOSO), the collapse described below. Best configuration remains
**both models trained from scratch on our data alone: 0.495 LOSO / 0.638 held-out**.

**Verdict:** online footage cannot substitute for the user's own typing for this project. The only
suitable public dataset has now been tested for detection, key identification and desk decoding;
none improves.

**Integrity note on 0.416:** that number is **not** a new headline. The reduced grid was centred on
operating points earlier desk folds had selected, so it carries desk-derived information and is
optimistically biased. The honest best desk CER remains **0.448** (`hirecall.py`). The pretrained
vs from-scratch *delta* is still valid because both arms share the same streams and grid.

**Pose model:** pretraining on public pose ties from-scratch at 25% of our labels and hurts clearly
from 50% up. The absolute-coordinate pose model (production) scores 0.369 LOSO, matching keymax's
0.371, and does not collapse. The hand-relative pose variant does collapse: with only two training
sessions per LOSO fold it maps a whole held-out session onto one key (from scratch, 829 of 1,221
taps in session 131629 predicted 'u'; fine-tuned from the public model, 1,018 taps predicted 'e' at
mean top probability 0.99). No non-finite probabilities — this is a real failure mode, not a bug.
The matched desk check was still finishing on the Mac mini when this was saved; its output lands in
`.cache/keypre/` on the mini and must be copied into `results/keypre/`.

## Next, in order of value
1. **Record ~8 more minutes of ordinary typing.** The scaling curve has not flattened; this is
   the cheapest available improvement and should reach the low-to-mid 80s. Prioritise *varied*
   hand position and lighting over simply more minutes of the same — all current data is one
   person, one rig, one night, so session diversity is now worth more than label count.
2. **Record the desk session** (keyboard removed, phrases prompted) and run `analyze_drift`.
   This is the actual Phase 0 question and everything for it is built and tested.
3. **Recalibrate the model's probabilities** (isotonic or Platt on a held-out session) before
   any text-decoding work. A strict add-on; it cannot change the detection numbers above.
4. Only then: gesture shortcuts (Goal A), or the decoder.

## Test suite
211 tests, plus 20 in `test_online_taps.py` which must run separately (LightGBM and MediaPipe
in one process abort during a model load).
