# Phase 0 operator protocol

Two ~10-minute recordings, one analysis command, one verdict. Read this once end to end
before you start: two of the steps (camera placement, no corrections on the desk) cannot be
fixed afterwards and force a full re-record.

The question: **do per-key fingertip landing positions survive removing the keyboard?**
Session A gives ground truth with a keyboard, session B gives the same typing on bare desk,
and `analyze_drift` compares them. Thresholds are frozen in `phase0/CONTRACT.md`.

---

## 1. Physical setup

**Phone on a stand, oblique top-down, about 60-70 degrees from horizontal.**
Not a pure overhead (nadir) shot. Directly overhead puts the fingertip's descent along the
optical axis, which is the worst possible geometry: the tap becomes a few pixels of scale
change instead of visible vertical travel, and the tap detector keys off exactly that
vertical travel. Tilt the phone so you can see the *front faces* of the fingers as well as
the desk surface.

- **Framing**: both hands fully in frame, including the little fingers at the outside of the
  home row, with 10-15 cm of bare desk visible below the hands. Nothing important within
  ~5% of any frame edge — MediaPipe drops hands that clip the border.
- **Distance**: as close as you can get while both hands stay framed. Bigger hands in pixels
  means finer landing resolution. 40-60 cm is typical.
- **Stability**: a real stand or tripod clamp. Not a stack of books, not leaning on a mug.
  Any camera movement between or during the two sessions is measured as finger drift and
  silently corrupts the result.
- **Lighting**: steady, diffuse, from the front or side. Overhead room light plus a desk lamp
  aimed at the hands is fine. No window behind the hands (backlight turns the hands into
  silhouettes and MediaPipe confidence collapses), no direct sun, no flicker. Blinds down —
  daylight changing over ten minutes changes the tracker's behaviour mid-session.
- **Desk surface**: matte and plain. Avoid glass, high-gloss black, and busy woodgrain or
  patterned mats; the tap detector needs the fingertip to be separable from the surface.
- Turn off auto-lock and Do Not Disturb on the phone. A screen-dim mid-session ends the run.

**Once the phone is placed, do not touch it again until both sessions are recorded.** Not to
check framing, not to charge it, not between sessions. The whole comparison is between two
sets of pixel coordinates in the same camera frame; a 1 cm nudge is larger than the effect
being measured. If the phone does get moved, both sessions are void — record both again.

---

## 2. macOS permissions

The recorder captures video *and* hooks the keyboard from one process, so it needs two
permissions, and both are granted to **the terminal application you launch it from** (Terminal,
iTerm2, VS Code, …) — not to Python.

System Settings > Privacy & Security >

| Permission | Needed for | Symptom if missing |
|---|---|---|
| **Camera** | frames from the phone/webcam | recorder exits with a camera error |
| **Input Monitoring** | `keys.jsonl` in the kbd session | **fails silently**: video records, `keys.jsonl` stays empty |

Add your terminal app to both lists, toggle it on, then **fully quit and reopen the terminal**
— macOS only applies the new grant to a fresh process.

> **Warning:** Input Monitoring on a terminal grants keystroke visibility to *everything* you
> ever run from that terminal, for as long as it stays enabled. The recorder itself only ever
> writes the allowlisted keys (letters, digits, and a handful of named keys; everything else
> becomes `"unknown"`), but the grant is not scoped to it. Prefer a terminal app you use for
> nothing else, and turn the permission back off when Phase 0 is finished.

Also grant the phone-as-webcam link (Continuity Camera): iPhone unlocked, on the same Apple
account, near the Mac. Verify the camera list before recording:

```bash
.venv/bin/python -m phase0.capture.recorder --list-cameras
```

---

## 3. Session A — keyboard (`kbd`)

Keyboard in its normal position, hands as usual. This session provides ground truth: every
tap is labelled by the keystroke it coincides with (±80 ms), and the key pitch used by the
whole analysis is measured from it.

```bash
cd <repo root>
.venv/bin/python -m phase0.capture.recorder --condition kbd --camera iPhone --seconds 600
```

Then type for the full ten minutes — **real typing**, not key-mashing:

- Do actual work: answer email, write notes, transcribe an article. Natural text gives natural
  key frequencies, which is what fills the per-key sample counts.
- Steady, normal speed. Do not slow down for the camera and do not perform.
- Correct typos normally; backspaces are fine here, the analysis ignores them.
- Keep both hands in frame. If you reach for the trackpad or scratch your nose, just return
  to the home row — those stretches simply produce unpaired taps.
- Never move the phone or the keyboard.

Aim for **at least 15 landings on every key you care about**, which is the contract's minimum.
Ten minutes of English prose comfortably clears that for the common letters; the rare ones
(`q`, `z`, `x`, `j`) may not qualify, and that is expected.

## 4. Session B — bare desk (`desk`)

Immediately after session A, without touching the phone:

1. **Lift the keyboard straight up and out of frame.** Do not slide it — sliding drags your
   hands' reference position with it. Note where the home row was; you are about to put your
   hands back in the same spot.
2. Put your hands down in the **same place they were resting in session A**, on the bare desk.
   Same posture, same wrist height, same elbow angle.
3. Record with the prompter on:

```bash
.venv/bin/python -m phase0.capture.recorder \
    --condition desk --camera iPhone --seconds 600 \
    --phrases phase0/phrases.txt
```

There are **no key events on a bare desk**, so this session's ground truth is the phrase text
itself (CONTRACT amendment 1). The prompter shows one phrase at a time from `phase0/phrases.txt`
and records the window in `phrases.jsonl`; the analysis aligns the taps detected inside each
window to that phrase's characters.

Type each shown phrase once, then continue to the next.

- **Do not correct mistakes.** There is no backspace to observe on a desk, so a correction
  makes the tap sequence disagree with the phrase text and the analysis drops the phrase.
  Mistyped a word? Keep going and let the phrase be dropped.
- **Space with your thumb, as normal.** Spaces are the anchors the alignment relies on:
  a space is the one character that is identifiable from the video alone (a thumb tap), which
  is what pins each word to the right place in the phrase.
- Type at a natural rhythm, and pause *between* phrases rather than in the middle of a word.
- Keep hitting where the keys *would* be. Do not creep inward, and do not look for your hands.
- Roughly ten minutes, same as session A. Volume matters: only phrases that align cleanly are
  used, so expect to need noticeably more raw phrases than you would guess.

---

## 5. Commands, in order

From the repo root, with the venv:

```bash
# 0. one-time: check cameras and permissions
.venv/bin/python -m phase0.capture.recorder --list-cameras

# 1. record both sessions (10 min each, phone untouched in between)
.venv/bin/python -m phase0.capture.recorder --condition kbd  --camera iPhone --seconds 600
.venv/bin/python -m phase0.capture.recorder --condition desk --camera iPhone --seconds 600 \
      --phrases phase0/phrases.txt

# 2. note the two session ids it printed, e.g.
KBD=data/sessions/20260910-101500-kbd
DESK=data/sessions/20260910-103000-desk

# 3. landmarks (downloads the MediaPipe model on first run; a few minutes per session)
.venv/bin/python -m phase0.analysis.extract_landmarks "$KBD"
.venv/bin/python -m phase0.analysis.extract_landmarks "$DESK"

# 4. taps
.venv/bin/python -m phase0.analysis.detect_taps "$KBD"
.venv/bin/python -m phase0.analysis.detect_taps "$DESK"

# 5. the verdict
.venv/bin/python -m phase0.analysis.analyze_drift --kbd "$KBD" --desk "$DESK"
```

Step 5 prints the verdict, and writes `drift_report.md` and `drift_scatter.png` into the desk
session directory. Exit code: `0` = PASS, `1` = FAIL, `3` = INCONCLUSIVE, `2` = unusable input.

---

## 6. Reading the result

**A good run looks like this:**

- kbd: **> 85%** of keystrokes paired to a tap, and most detected taps paired to a keystroke.
- desk: **most phrases used**, median alignment cost low, a solid majority of characters
  confidently aligned. Some phrases dropped is normal and healthy — that is the analysis
  refusing to guess.
- Key pitch measured from **5/5 home-row pairs**, and a plausible number (tens of px, in the
  same ballpark as the visible key spacing in the scatter plot).
- **15+ qualifying keys**, comfortably above the minimum of 8.
- In the scatter plot, per-key blue and orange clouds sit on top of each other with short
  arrows between the centroids.

**PASS** means median centroid shift < 0.5 key pitch *and* median variance ratio < 2.0: the
hand keeps a per-key spatial map without the keyboard, and Phase 2 proceeds.
**FAIL** means it does not, and Phase 2 does not proceed.
**INCONCLUSIVE** (fewer than 8 qualifying keys) is neither — do not report it as a result, and
do not report a median computed from three keys. Record more data and re-run.

Also read the two secondary signals:

- **Dominant finger changes.** A key hit by the middle finger on the keyboard and the index
  finger on the desk is a real behavioural finding even if the centroid barely moved.
- **Within-session drift**, reported in px/min and key-pitch/min for the desk session with the
  kbd session as control. Published invisible-keyboard work reports ~1.32 mm/min of drift
  versus ~0.25 mm/min with a visible keyboard; one key pitch is 19 mm, so ~0.07 pitch/min vs
  ~0.013 pitch/min. Monotonic migration in most keys is the signature to look for.
- **Trimmed variance ratio.** If it is far below the untrimmed one, a handful of mislabelled
  taps are inflating the spread — a labelling problem, not a real result.

## 7. If the numbers look wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `keys.jsonl` empty after the kbd session | Input Monitoring not granted, or granted before the terminal was restarted | grant it, fully quit and reopen the terminal, re-record |
| **Low kbd pairing rate (< 60% of keystrokes)** | taps not being detected: too dim, too shallow a camera angle, hands too small in frame, or detector thresholds off | re-check lighting and the 60-70° angle, move the camera closer, then re-tune `detect_taps` before trusting anything downstream |
| Many taps, few keystrokes paired | detector firing on non-typing motion (hand repositioning, gestures) | tighten the detector's rebound thresholds; keep hands still between phrases |
| **Most desk phrases dropped** | corrections while typing, skipped or half-typed phrases, or spaces not hit with the thumb | re-record the desk session typing each phrase once, straight through, no corrections, thumb spaces |
| Both rates fine, few qualifying keys | just not enough typing | record longer sessions; sample count is the binding constraint |
| Key pitch reported as FALLBACK or ASSUMED | too few home-row keys have data | usually a symptom of a low pairing rate — fix that first; every pitch-unit number is untrustworthy until it says "home-row pairs" |
| Everything shifted by a large constant | the phone moved between sessions | both sessions are void, re-record both |

A low pairing rate invalidates everything downstream. `analyze_drift` prints it as a loud
warning rather than a footnote — treat that warning as a stop sign, not a caveat.
