# 45° camera trial — recording protocol and one-command scoring

Everything below runs on a Mac with the phone you already use. Read
`results/angle45/SUMMARY.md` first: the simulations say 45° is a **modest** change, not a breakthrough,
so the point of this recording is to settle it on your own hands and your own rig.

## 1. Mount the phone at 45°

Target geometry (this is exactly what was simulated, so the numbers stay comparable):

* **Elevation 45°** above the desk surface, looking down at the hands from **in front of them** — i.e.
  the camera is beyond your fingertips, on the far side of the keyboard, looking back toward you.
* **Distance to the middle of your hands: 23 cm** along the optical axis. With a 45° tilt that puts the
  lens **16 cm above the desk** and **16 cm beyond your fingertips** (23 × sin45 and 23 × cos45).
* Both hands must be in frame with margin: at that distance 25 cm of hand span uses about 590 of the
  1280 px, so there is room. Aim the centre of the frame between the two hands.
* Keep the **same lens and the same 1280×720 @ 60 fps** as the top-down sessions; only the pose changes.
* A phone tripod with a ball head, or the phone leaned against a stack of books at 45°, is enough.
  Measure the two distances with a ruler — a 10° error moves the numbers more than the effect we are
  looking for.

Check the framing before recording:

```bash
cd ~/cvt
.venv/bin/python -m phase0.capture.preview --backend net --camera auto \
    --width 1280 --height 720 --rotate ccw90
```

In the preview, confirm: both hands fully visible through a whole typing burst, fingertips not clipped
at the frame edge, and the far hand not hidden behind the near one when you reach for the top row.
If the hands are cut off, move the phone back in 2 cm steps and note the final distance.

## 2. Record

Two blocks, same session or back to back:

```bash
# (a) 12 minutes of natural typing on the physical KEYBOARD — the keylogger labels every frame for free
.venv/bin/python -m phase0.capture.recorder --condition kbd --backend net --camera auto \
    --width 1280 --height 720 --fps 60 --seconds 720

# (b) 3 minutes of BARE-DESK phrases, the product condition (keyboard pushed aside, same phone pose)
.venv/bin/python -m phase0.capture.recorder --condition desk --backend net --camera auto \
    --width 1280 --height 720 --fps 60 --seconds 180
```

Type normally in (a) — real work, not test phrases. Do not move the phone between (a) and (b);
the whole point is that the two blocks share one camera pose.

Write the measured geometry into the session's `meta.json` notes field (or a note next to it):
elevation, distance to hand centre, lens height above the desk. Without it the session cannot be
compared to anything later.

## 3. Score it — one command

```bash
cd ~/cvt && PYTHONPATH=. .venv/bin/python -m phase0.analysis.angle45_trial run \
    --session <new-45deg-session-id> --baseline 20260911-164237-kbd
```

This extracts landmarks, then for the new session AND for the top-down baseline session runs the same
two measurements and prints the difference:

* **key-down detection AUC** — kinematic feature stack, time-blocked cross-validation inside each
  session, 10 s block-bootstrap CI, plus a time-shuffled control. Top-down reference on
  `20260911-164237-kbd` is **0.928** (results/pressure/SUMMARY.md).
* **CTC character error rate** on the held-out last quarter of the session, fine-tuned from the matching
  pretrained model (`.cache/hwt/pre45/hwt_el45_s0.pt` for the trial, `hwt_el90_s0.pt` for the baseline),
  and from scratch as a control.

It writes `results/angle45/trial_<session>.json`.

## 4. What counts as a result

* **Landmark rate first.** If MediaPipe finds both hands in fewer than ~90% of frames at 45°, the angle
  fails on tracking and nothing else matters. (On real 39–51° footage from PressureVisionDB the tracker
  held up — 0.84 detection versus 0.84 at 90° on the same frames — so this is expected to pass.)
* **key-down AUC**: the simulations predict ~0 change (±0.01). A drop of more than 0.03 is a real cost;
  a gain of more than 0.03 would be the first evidence that the angle helps on our own rig.
* **CER**: the honest expectation is "no better than top-down". A CER improvement larger than 0.03
  would justify re-recording the labelled desk corpus at 45°; anything smaller does not, because
  re-recording invalidates all three labelled desk sessions and every baseline built on them.
* Report the negative plainly if that is what comes out — that is the expected outcome.
