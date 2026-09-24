# Pad session — trackpad contacts as desk touch ground truth

**The problem.** Desk decoding is limited by the tap stream. On the desk about 20% of touches are
missed and about 1.3 extra taps per character are detected. We cannot measure or train against
this, because a bare desk records no touches. The keyboard keylogger gives exact labels, but on a
surface with key travel and a different look.

**The idea.** The MacBook trackpad is flat, hard glass with no travel, and it is already in the
camera's top-down shot. macOS (MultitouchSupport) reports every finger contact on it at about
125 Hz, on the same clock as the video frames. So: "type" on the trackpad, record as usual, and
get exact press/lift labels on a desk-like surface. Data format: CONTRACT amendment 8.

---

## 0. One-time checks (2 minutes)

1. **Does the trackpad logger see your fingers?** Run this and tap the trackpad a few times
   during the 10 seconds:

   ```bash
   PYTHONPATH=. .venv/bin/python -m phase0.capture.touchpad --seconds 10
   ```

   Expected:
   - one `down`/`up` line per tap, with `x`/`y` in mm that follow your finger
   - a summary with `host_per_dev_second` around 1.009 on this Mac (the pad clock runs slow;
     `pad_labels` corrects it)
   - `RESULT: PASS` at the end
   - `NO TOUCHES SEEN` means the framework started but never delivered a contact. Report it; do
     not record.
   - `CHECK summary` with `xy_out_of_range > 0` means the struct layout is wrong on this macOS
     build. Also do not record.
   - No permission is needed. It does not use Input Monitoring or Accessibility.
2. **Stop the Mac acting on your taps.** In System Settings > Trackpad:
   - Point & Click: **Tap to click OFF**, **Force Click and haptic feedback OFF**,
     **Look up & data detectors OFF**.
   - More Gestures: turn off **Notification Center**, **App Exposé** and **Mission Control**.

   Light taps then only nudge the pointer. Turn these back on after the session.
3. **Camera.** The WideCam phone goes above the MacBook as for the kbd sessions. The **whole
   trackpad glass and both hands** must be in frame. Nothing else changes.

## 1. Record (about 10 minutes)

Keep the recorder's Terminal frontmost and make it full screen, so stray pointer motion lands on
the Terminal:

```bash
.venv/bin/python -m phase0.capture.recorder --condition pad --backend net --camera auto \
      --width 1280 --height 720 --fps 60 --phrases phase0/phrases_pad.txt --phrase-seconds 12
```

`--condition pad` turns on `--touchpad`. It prints `trackpad 124.8x76.8 mm -> touches.jsonl` at
start and a `trackpad : N contacts` line at the end. It stops by itself when the 47 prompts are
done (~9.5 min). If no contact is seen within 20 s, it warns loudly.

The prompts come in blocks:

| block | prompts | ~time | what to do | what it trains |
|---|---|---|---|---|
| **CALIB** | 1 | 12 s | **Right index finger** only. Press and hold ~1 s on each glass corner in turn: far-left, far-right, near-right, near-left. | pad-to-image homography |
| **typing A** | 15 | 3 min | Type the phrase on the glass as if it were a shrunken keyboard (layout below). Correct fingers, thumb for space, natural rhythm, **no corrections**. | press vs hover and tap count in real typing sequences |
| **DRILL** | 10 | 2 min | Follow the text: single fingers, thumbs, same-finger double taps, rolls, fast bursts, very soft taps. | every finger evenly; merged double taps and soft taps (the current misses) |
| **NEG** | 6 | 1.2 min | Hover, twitch above the glass, rest still on the glass, slide, hands off. | hard negatives (the current extra taps) |
| **typing B** | 15 | 3 min | Same as typing A, new phrases. | held-out test block |

**Mini layout.** The sensor is 12.5 cm wide, so an imagined keyboard shrinks to about 60% scale:
- home row across the middle of the pad
- index fingers about 3 cm apart
- top row toward the far edge, bottom row toward you
- thumbs just inside the near edge

**Keep every tap on the glass.** Resting palms or wrists on the aluminium palm rest is fine: it is
still, and it is not a tap. A finger tap that lands **off** the glass is an unlabelled real touch,
and it teaches the detector that a visible tap is "no touch". If a key would fall off the pad,
reach onto the glass anyway. `pad_labels` reports `pad_coverage` so you can check afterwards.

Never press hard enough to click. A physical click is a real contact too, but it adds key-travel
motion the desk does not have.

## 2. Process

```bash
PAD=data/sessions/<id>-pad                       # printed by the recorder
.venv/bin/python -m phase0.analysis.extract_landmarks "$PAD"
PYTHONPATH=. .venv/bin/python -m phase0.analysis.pad_labels "$PAD"
```

`pad_labels` writes three things:
- `$PAD/taps_pad.jsonl`: taps.jsonl schema, one row per pad tap. `t` is the exact touch-down,
  `i` the nearest frame, hand/finger/x/y from the nearest fingertip landmark.
- `$PAD/pad_labels.json`: counts by kind (tap/rest/slide) and by block, calibration corners,
  `d_px_median` (contact-to-fingertip distance), `pad_coverage`, taps per minute.
- `$PAD-padtrain/`: symlinks to video/frames/landmarks/meta/phrases, plus a `keys.jsonl` with
  one down/up per tap. It looks like a keyboard session to every existing tool.

**Sanity gates before trusting the labels:**
- `kinds.tap` should be roughly the number of characters typed plus the drill taps.
- `d_px_median` should be well under half a finger spacing (< ~25 px at 720p). If it is not,
  calibration failed.
- `pad_coverage` should be > 0.8.
- `taps_by_block.NEG` should be near 0.

If corner calibration fails (hands not tracked at a corner, or a corner skipped), open frame 0 and
read the four glass corners in **landmark pixel** coordinates. Then rerun with
`--corners-px "x,y;x,y;x,y;x,y"`, in the order far-left, far-right, near-right, near-left.
`--refine` refits the homography on all taps; it is opt-in, and the report shows the before/after
median distance.

Camera frame times are stamped on arrival over Wi-Fi, so frames lag the physical moment by a
roughly constant amount. The keyboard labels have the same bias. `--lag-ms` shifts which frame a
touch is paired with. Leave it at 0 to stay comparable with the kbd labels, unless you are
measuring the lag.

## 3. How the labels are used

### (a) Desk touch detector: press vs hover, and count

This is the main use. The pad view drops into the existing detector tooling:

```bash
# 1. What does the CURRENT keyboard-trained detector do on a flat surface? (first real measurement)
.venv/bin/python -m phase0.analysis.taps_gb predict "$PAD-padtrain" --model models/taps_gb.joblib
.venv/bin/python -m phase0.analysis.eval_taps "$PAD-padtrain" --taps "$PAD-padtrain/taps_gb.jsonl"
#    -> recall = 1 - miss rate, and (taps - hits)/keydowns = extra taps per touch, on glass.

# 2. Train with pad labels, session-held-out CV alongside the keyboard sessions
.venv/bin/python -m phase0.analysis.taps_gb experiment --what ablation --cv session \
      --sessions data/sessions/20260910-021315-kbd data/sessions/20260910-131629-kbd \
                 data/sessions/20260911-164237-kbd "$PAD-padtrain"

# 3. Fit a model that includes the pad session
.venv/bin/python -m phase0.analysis.taps_gb train --out models/taps_gb_pad.joblib \
      --sessions data/sessions/20260910-021315-kbd data/sessions/20260910-131629-kbd \
                 data/sessions/20260911-164237-kbd "$PAD-padtrain"
```

The contact-detector experiments (`touch_common` / `touch_runs` / `contact_ego`) take a session
id. `<id>-pad-padtrain` resolves under `data/sessions/` like any other, and `touch_common.kt()`
reads its `keys.jsonl`.

The honest test is **typing B**. Train on everything else, then score on the typing-B phrase
windows (`phrases.jsonl`). Only then check whether desk CER moves, using the desk decoding
harness on `20260910-202149-desk`. The pad is a proxy for the desk, not the desk.

What this answers that nothing else can:
- the true miss rate and extra-tap rate on a flat surface
- whether the NEG block's hovers and twitches are where the extra taps come from
- whether the DRILL block's double taps and soft taps are where the misses come from

### (b) Optional: fingertip contact position

Each `taps_pad.jsonl` row has two positions:
- `contact_x/contact_y`: where the glass was touched, projected into the image.
- `x/y`: the fingertip landmark.

The difference per finger is the landmark-to-contact offset (MediaPipe's tip joint sits at the
nail, not the pad of the finger). That gives a measured correction target for tap position
models, and `d_px` gives its spread.

**Caution:** hand/finger is assigned by nearest landmark, so it is *not* independent finger ground
truth. Filter on a margin (`d2_px - d_px` large) before using it to judge finger attribution.

## 4. Limitations (read before over-interpreting)

- **Contact, not layout.** The sensor is **124.8 x 76.8 mm** (measured on this Mac15,6; the glass
  is a bit larger). A real keyboard is about 28 cm wide. Hands are cramped to about 60% key pitch,
  so key positions, hand spread and per-key landing geometry are **not** desk-representative. This
  session trains *when* a finger touches, not *which key* it meant. Do not add pad taps to key-ID
  training.
- **Surface appearance differs.** The surface is silver glass and aluminium, not a dark desk.
  Pose/landmark features should transfer; pixel/appearance features (the contact CNN) may learn
  "silver under fingertip". Compare pose-only and pixel models separately.
- **Capacitive, not force.** Very light grazes register as touches, and a nail tap may not.
  `density` is a capacitance proxy, not force. Hover (in-range) states are deliberately not logged
  as contacts.
- **Nearest-landmark assignment.** Mistracked or occluded fingertips mis-assign hand/finger. The
  time label stays correct regardless.
- **Pointer side effects.** Even with Tap to click off, the pointer moves and multi-finger rests
  can scroll. That is harmless to the data, but keep the Terminal frontmost.
