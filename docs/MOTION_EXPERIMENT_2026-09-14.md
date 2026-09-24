# Motion residual experiment — 14 September 2026

Implemented and completed using only the existing recordings. No production model replaced, no commits, no new recordings, and no remote compute.

## Outcome

A longer temporal residual model, combined at a fixed 50:50 posterior weight with the existing whole-session-held-out fine-tune, reduces **character-beam word errors from 79 to 77 out of 250**. Character error falls from 13.29% to 12.62%. The paired WER change is −0.8 percentage points, with 95% interval [-4.08, 2.79] percentage points. This does **not** establish a reliable improvement.

Optical flow adds no improvement to the combined output: the geometry-only and optical-flow versions have identical aggregate error counts (their individual transcriptions are not all identical). It is not justified to call this a breakthrough or replace the validated pipeline. The full 8B decoder has not been evaluated with these posteriors, so these numbers must not be compared to its 10% development WER.

| Three-seed ensemble, same frozen character decoder | CER | WER |
|---|---:|---:|
| Original zero-shot camera model | 18.69% | 44.8% |
| Longer geometry residual | 17.94% | 40.8% |
| Optical-flow residual | 17.86% | 42.0% |
| Existing session-held-out fine-tune | 13.29% | 31.6% |
| Geometry residual + existing fine-tune | 12.62% | 30.8% |
| Optical-flow residual + existing fine-tune | 12.62% | 30.8% |

## What changed

The previous 24-pixel crop encoder could discard fine motion. This experiment instead uses [OpenCV pyramidal Lucas–Kanade tracking](https://docs.opencv.org/4.7.0/dc/d6b/group__video__track.html) on half-resolution video around all 21 joints of each hand. A forward/backward check rejects tracks whose return error exceeds 1.5 pixels. Per joint, the features contain normalized optical displacement, its difference from landmark displacement, and tracking validity. Missing two-hand observations are zeroed. Hand ordering is by wrist image y, as in the earlier crop pipeline; crossings and occlusion remain limitations.

A 64-channel residual temporal convolution stack has dilation 1, 2, 4 and 8, giving a 61-frame receptive field (about two seconds at the model's 30 Hz output). It receives geometry, optical flow and base class probabilities. Its final layer begins at zero, preserving the existing probabilities at initialization. The loss combines CTC with a KL anchor to the base model.

Three seeds × two conditions × three held-out desk recordings = **18 adapted model fits**, plus keyboard pretraining. Each condition uses 300 pretraining steps on 81 windows from two keyboard recordings, then 600 adaptation steps with equal probability of keyboard or other-session desk examples. No tested visual model trains on that recording's labels. The existing fine-tune comparison verifies its exclusion metadata before loading its cached posteriors.

Timing controls shuffle optical flow or replace it with zero at evaluation. Raw flow often slightly beats shuffled motion, but zero-flow controls can match or improve it, and the final aggregate error counts are identical without flow. This is insufficient evidence that the new motion channel adds useful information.

## Evaluation limits

The reference data consist of 31 phrases across three desk recordings, 250 normalized words. Phrase boundaries are supplied. These are historically exposed development recordings; whole-session visual-model exclusions do not make the language model or experiment design blind. The bootstrap uses 10,000 paired phrase resamples within each recording and says nothing about variation across unseen users or camera placements. There is no near-100% result.

## Reproduce and artifacts

```sh
.venv/bin/python -m phase0.analysis.motion_residual --seed 0
.venv/bin/python -m phase0.analysis.motion_residual --seed 1
.venv/bin/python -m phase0.analysis.motion_residual --seed 2
.venv/bin/python -m phase0.analysis.motion_evaluate
.venv/bin/python -m pytest phase0/tests/test_motion_residual.py phase0/tests/test_decoder_probes.py -q
```

Code: `phase0/analysis/motion_residual.py`, `motion_evaluate.py`.

Checkpoints, flow features, source hashes, frozen run settings and per-phrase posteriors: `.cache/nextgen/motion_residual/`. Training reports: `results/nextgen/motion_residual_s0.json` through `s2.json`. Full decoder comparisons and paired uncertainty: `results/nextgen/motion_evaluation.json`; compact summary: `results/nextgen/motion_summary.json`.

Generation uses local CPU, two Torch threads and one OpenCV thread. Caches validate the video, landmarks and frame-list hashes. Training commands overwrite their own seed outputs; preserve reports before changing the recipe. Eight focused tests pass, including recovery of a known synthetic image translation and initialization/backpropagation checks. All experiment jobs have finished.
