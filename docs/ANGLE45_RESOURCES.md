# Public resources for a ~45° typing / hand-contact camera (surveyed 2026-09-25)

Scope: datasets and papers with hands typing or touching a surface filmed at roughly 30–60° above that
surface, usable to pretrain or validate a 45° version of this project. Licence is called out for each,
because **nothing derived from a non-commercial dataset may be presented as shippable**.

## Summary table

| resource | view(s) | labels | size / availability | licence | verdict for us |
|---|---|---|---|---|---|
| **PressureVisionDB** | 4 fixed cameras; measured **~39°, ~40°, ~51°** oblique + one **~90°** near-normal | per-frame 105×185 pressure image | 140 GB "small" (3 fps) or 960 GB full, as 6 fold-zips / 43 participant-zips | **MIT** | **best fit.** Simultaneous 90° and 40° views of the *same* hands, incl. `type_sentence_5x_both`. Used here: 533 MB subset pulled by ranged reads. |
| **EgoPressure** | 7 static third-person views (incl. oblique) + egocentric | MANO hands + per-frame pressure | ~9 GB subset already on this machine | **CC BY-NC-SA** | already used for `results/angle/`; research only |
| **How We Type** (Aalto CHI'16) | 3D mocap, no camera — re-projectable to any angle | 240 Hz finger mocap + keylogger | 29 GB; 30 participants; streams already cached here | **CC BY-NC 4.0** | the only source of *real ten-finger typing* with key labels; research only |
| **Keystroke Typing Videos** (HF `andrewt28/keystroke-typing-videos`) | fixed camera **above** a laptop keyboard (top-down) | 51,336 key-downs, ms timestamps | 0.71 GB, 800 clips | **AFL-3.0** | top-down, not oblique; already assessed in REPORT.md |
| **TypeNet / TypingHands26** (WACV 2022) | camera **in front of the user, on the surface**, low oblique | frame-level characters; 4:50 h keyboard (26 users) + 5:50 h flat-surface (10 users) | announced at github.com/benadar293/typeNet — **repo contains only a 110-byte README; no data, no licence** | n/a | closest prior system, but the data is *not actually published* |
| TypeNet (keystroke biometrics, Acuña et al.) | no video | keystroke timings only | large | varies | unrelated to vision despite the name clash |

## PressureVisionDB — the one that matters

* Paper: PressureVision (ECCV 2022) and PressureVision++ (WACV 2024), Grady et al.
  Code + data: https://github.com/facebookresearch/PressureVision — **MIT licence** for code and data.
* Download: `python -m recording.downloader` (add `--full` for 960 GB). The "small" set is 6 zips of
  ~19–23 GB each (3 fps, lower JPEG quality); the full set is 43 per-participant zips of ~22 GB. **No
  published bundle fits a 12 GB budget**, so this machine used HTTP range reads into the zip's central
  directory instead: `phase0/tools/zipsub.py list|get <url> --grep … --bulk`. 533 MB fetched
  (3 participants × `type_sentence_5x_both` × 4 cameras + force + calibration) out of a 19.4 GB archive.
* Structure: `<participant>/<sequence>/camera_{0..3}/NNNNN.jpg` (1920×1080), `force/NNNNN.pkl`
  (pickled float32 105×185 pressure image), `metadata.json` (per-camera `ModelViewMatrix`, fx/fy/cx/cy,
  OpenCV distortion, participant metadata incl. skin tone, lux, light direction).
* **Camera geometry, measured here** from `metadata.json` rather than taken from the paper: taking
  camera_2's optical axis as the surface normal (confirmed by eye — it is the only view looking squarely
  down onto the pad), camera_0 sits **39°**, camera_3 **40°** and camera_1 **51°** above the surface.
  The papers describe the rig loosely as "~45°"; the measured spread is 39–51°.
* 74 sequence types per participant. The relevant ones for us: **`type_sentence_5x_both`** (two-handed
  sentence typing, 127–260 frames), `type_ipad_5x_{left,right}`, `press_fingers_*`, `index_press_*`.
* Caveat: the small set is **3 fps**, so anything velocity-based (our stop-motion cue, dwell, CTC over
  time) cannot be evaluated on it — only per-frame pose. The full set is 15 fps.

## What was actually used in this study

* `results/angle45/pvdb_*.json` — MediaPipe run on all four PressureVisionDB views of the same typing
  frames, then our own feature stack and probe protocol (see results/angle45/SUMMARY.md).
* `.cache/hwt/streams_el{90,60,45,30}/` — How We Type re-projected through virtual cameras
  (`phase0/analysis/hwt45.py`), for the typing-specific angle measurement and the 45° pretraining arm.

## Checked and rejected

* **TypingHands26** — the only public flat-surface typing video corpus, and the repository is empty.
  If it ever appears it is the single most valuable resource for this project; worth re-checking.
* **EgoPressure full release** — CC BY-NC-SA, so usable for measurement only, and we already hold the
  subset the angle study used.
* Keystroke-dynamics datasets (Aalto 136M, EmoSurv, HF keystroke-*) — timings only, no video.
* AR/VR typing work (TouchInsight, StegoType, GAZEploit) — egocentric headset views, not a desk camera,
  and no public desk-camera corpus attached.

## Practical note on licences

PressureVisionDB (MIT) and the HF keystroke videos (AFL-3.0) are the only surveyed sources whose terms
allow commercial use. How We Type and EgoPressure are non-commercial: every number in this project that
derives from them — including the 45° pretraining arm — is a **research measurement**, and a shipped
model must not be trained on them.
