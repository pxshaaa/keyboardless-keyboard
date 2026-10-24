# RTMPose hand keypoints vs MediaPipe for the desk CTC model — SUMMARY (2026-09-20, Mac mini)

## Verdict: NO measurable difference. Keep MediaPipe.
Swapping the landmark tracker for RTMPose-m (rtmlib, onnxruntime CPU) and retraining the exact ctc_v4
desk recipe changes WER by less than the noise on every decoder and training label. Every paired
confidence interval includes zero. RTMPose is also slower (25-48 fps vs 59 fps for MediaPipe on the
same session), so it costs speed and buys nothing.

Pooled over the 3 labelled desk sessions, leave-one-session-out, 31 lines / 250 words, paired bootstrap on
identical lines (RTM minus MediaPipe; negative = RTM better):

| label   | decoder | WER MediaPipe | WER RTMPose | d WER [95% CI]          | CER MP / RTM  | d CER [95% CI]          |
|---------|---------|---------------|-------------|-------------------------|---------------|-------------------------|
| zs      | qwen    | 0.272 | 0.288 | +0.016 [-0.043, +0.082] | 0.173 / 0.157 | -0.016 [-0.051, +0.020] |
| loso_ft | qwen    | 0.236 | 0.228 | -0.008 [-0.065, +0.050] | 0.141 / 0.135 | -0.006 [-0.036, +0.024] |
| zs      | char    | 0.452 | 0.412 | -0.040 [-0.094, +0.012] | 0.183 / 0.175 | -0.008 [-0.032, +0.016] |
| loso_ft | char    | 0.320 | 0.332 | +0.012 [-0.034, +0.056] | 0.130 / 0.143 | +0.013 [-0.005, +0.031] |

loso_ft + qwen is the deployed configuration; there RTM is -0.008 WER, i.e. two words of 250.

Per session (qwen decoder):

| session               | label   | WER MP | WER RTM | d [95% CI]              |
|-----------------------|---------|--------|---------|-------------------------|
| old desk 0910-202149  | zs      | 0.234  | 0.263   | +0.029 [-0.037, +0.101] |
| blind-1 0912-174542   | zs      | 0.274  | 0.177   | -0.097 [-0.182, -0.016] |
| blind-2 0913-125733   | zs      | 0.373  | 0.490   | +0.118 [+0.000, +0.276] |
| old desk              | loso_ft | 0.255  | 0.248   | -0.007 [-0.087, +0.073] |
| blind-1               | loso_ft | 0.161  | 0.161   | +0.000 [-0.143, +0.139] |
| blind-2               | loso_ft | 0.275  | 0.255   | -0.020 [-0.062, +0.000] |

The two zero-shot sessions that individually "reach significance" go in opposite directions (blind-1
better, blind-2 worse) on 5-6 lines each and cancel in the pool. That is what noise on this much data
looks like, not a tracker effect.

## Tracker quality, for the record
Still-period fingertip jitter on 20260911-164237-kbd (no key event within +-300 ms; 27,177 frames):
MediaPipe median 1.32 px, p90 7.10, rms 6.31 | RTMPose median 1.00 px, p90 4.25, rms 5.13
(same hand width ~140 px, same 1.6 hands/frame). RTMPose is 24% quieter at the median and 40% at p90 —
and that does not move WER. Jitter has again failed to predict WER in this project.
Joint mapping verified as identity (every RTM joint's nearest MediaPipe joint is itself, median
offset 12 px). Raw numbers: results/rtm/jitter_20260911-164237-kbd.json.

## Protocol
- RTMPose-m hand5 via rtmlib 0.0.16 / onnxruntime 1.30, CPU, on all 8 sessions; boxes tracked from
  each hand's previous keypoints, re-seeded from the MediaPipe boxes (TRACK_MIN_MEAN 0.3). Output written
  to data/sessions/<sid>/landmarks_rtm.parquet — MediaPipe landmarks.parquet untouched. z = 0 (2D
  tracker); the recipe has zfeat=False so z is zeroed for MediaPipe too.
- Training: phase0/analysis/ctc_rtm.py, identical to ctc_v4: stage 1 hwtmix zero-shot 3 seeds x 1500
  steps, stage 2 LOSO desk fine-tune 3 seeds x 3 folds x 300 steps (lr 5e-4, desk_w 0.5, ftmix 0.3),
  under phase0.tools.gpulock. Streams cached separately in .cache/seqctc/streams_rtm/.
- Eval: same proxy decoders (qwen-0.5B and char) on CPU, same lines. Baseline is
  .cache/ctc_v4/results/loso_v4.json (MediaPipe, same recipe, same decoders, same seeds).
- Compare: ctc_rtm compare -> results/rtm/loso_compare.json; paired bootstrap over identical per-line
  word-edit arrays. Log: .cache/rtm/train.log (ends TRAIN_ALL_DONE).

## Caveats
- n is small: 31 lines, 250 words pooled; blind-1 + blind-2 is 11 lines, so one word is ~0.02 WER on
  those. The pooled CIs are about +-0.05 WER wide; a real effect smaller than that would not show here,
  but nothing that small changes the picture (the gap to the goal is 0.2+ WER).
- Nothing about the fine-tune recipe was re-tuned for RTMPose; it inherits MediaPipe's settings. A
  recipe re-tune might move a point or two, but the zero-shot rows (no fine-tune) show the same
  no-difference.
- Not tested: RTMPose-l or the wholebody model, hi-res crops, or fusing both trackers. Given that a 40%
  jitter reduction produced no WER change, none of these is a promising next step.
