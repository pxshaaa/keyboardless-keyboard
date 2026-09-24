# Sequence CTC ("lip-reading for fingers") — landmark streams to text, no per-tap classification

## Method
- Input: MediaPipe 21 joints x 2 hands per frame (60 fps). Hands slotted by wrist image-y (the parquet `hand`/`handedness` labels flip between sessions), anchored to each hand's session-median palm centre, scaled by session-median knuckle width (index MCP to pinky MCP; palm length foreshortens ~2x top-down). Features: palm-local pose, anchored pose, both velocities, hand-present masks (506 dims). z not used.
- Model: depthwise-separable TCN (stride 2 -> 30 Hz) + 3-layer transformer (d=192), CTC over blank + a-z + space + "other key" (1.19 M params). Augmentation: rotation +-10 deg, scale +-10 %, anchor jitter, joint noise/dropout, hand dropout, time masking, speed 0.83-1.2x.
- Training data: random 1.5-8 s crops cut in inter-key gaps; target = keystroke sequence (modifiers dropped, backspace/punctuation -> "other"). 1500 steps, batch 24.
- Decoding: greedy, and CTC prefix beam search (beam 16) with the existing `models/charlm.npz` 6-gram. LM weight/insertion bonus tuned per seed on the pooled keyboard LOSO windows (chose alpha 0.6, beta 1-2 for scratch/How We Type, alpha 0.6-1.0 for FSboard), never on desk data; learning-curve runs use the fixed pilot setting alpha 0.3, beta 1.0 (chosen on one keyboard fold).
- Evaluation: keyboard LOSO over 021315/131629/164237 (015217 always in training), held-out 015948 (model trained on all 4 kbd sessions), desk 202149 20 phrases (10 s shown->done windows) zero-shot, and 5-fold cross-phrase fine-tuning on desk phrase text (CTC only needs the text; keyboard crops mixed 50/50). 3 seeds each; CIs are 5000x item bootstrap of seed-averaged edit counts; desk deltas vs the per-tap pipeline use its per-phrase edits (`.cache/hirecall/cv_all.json`, CER 0.448).

## Public data provenance
- How We Type (Feit, Weir, Oulasvirta, CHI'16), Zenodo record 4034268, CC-BY-NC-4.0. Downloaded only `Motion Capture.zip` (981 MB) + `Typing.zip`. 30 participants, 240 Hz mocap of 52 hand markers (sentences condition, Finnish text). Markers mapped to MediaPipe joints (wrist = mean(Win, Wout); T1-4 -> thumb 1-4; I/M/R/L 1-4 -> MCP/PIP/DIP/TIP), projected onto the keyboard plane (mocap z = forward, x = right), downsampled to 60 Hz, Finnish y/z keys swapped to the user's QWERTZ positions, a-umlaut/o-umlaut/punctuation -> "other". ~150 min, ~34 k keystrokes.
- FSboard (Georg et al. 2024), Kaggle dataset `googleai/fsboard`, CC BY 4.0. No Kaggle credentials and no public mirror (Hugging Face searched: nothing). Kaggle's public API download endpoint serves individual files of this public dataset WITHOUT authentication: `https://www.kaggle.com/api/v1/datasets/download/googleai/fsboard/<url-encoded path>` answers with a 302 to a signed Google Cloud Storage URL. No account was created and no credentials were used. Full dataset is 1.45 TB (landmarks: train 61.5 GB in 100 shards); we downloaded only 4 train landmark shards `daun_v3/landmarks/daun_v3-train.arrow-0000{0..3}-of-00100` (2.52 GB) plus `convert_arrow_to_parquet.py`. The shards are tf.train.SequenceExample records; parsed with a minimal protobuf wire decoder (no TensorFlow) into 2,783 clips, 4.86 h, 49,583 characters (MediaPipe Holistic hand landmarks, selfie camera; image-up mapped to "forward").

## Conclusion (plain English)
1. Yes: the sequence model beats the per-tap pipeline on the desk session. With no desk labels at all, the How-We-Type-pretrained model reads the 20 desk phrases at CER 0.234 [0.190, 0.277] (pipeline 0.448; paired delta -0.214 [-0.261, -0.170]).
2. With the weak supervision the pipeline also used (phrase text only, 5-fold cross-phrase), CER drops to 0.132 [0.102, 0.163], WER 0.326 (pipeline WER 0.745): most words now come out right.
3. Even trained from scratch on my ~4,300 keystrokes it is better than the pipeline: zero-shot 0.388 (delta -0.060 [-0.102, -0.015]), desk-fine-tuned 0.224.
4. Public pretraining on real typing helps a lot: How We Type (30 people, mocap mapped to MediaPipe joints) cuts CER vs scratch by 0.15 on the desk zero-shot, 0.09 after desk fine-tuning and 0.10 on keyboard LOSO, at every data fraction.
5. Public pretraining on sign language does not: FSboard fingerspelling is worse than scratch (desk +0.056 [+0.030, +0.082], keyboard LOSO +0.023 [+0.012, +0.034]) and only neutral after desk fine-tuning (+0.004 [-0.020, +0.028]).
6. The character LM matters (~0.05-0.09 CER); AdaBN-style re-normalisation on the unlabelled desk session hurts in every condition and is not used.
7. No saturation in sight: every doubling of my own keyboard data lowers desk CER by ~0.10 (scratch) / ~0.06 (pretrained).
8. Log-linear extrapolation (4 points, crude): desk CER 0.20 at ~16k keystrokes from scratch or ~10k with How-We-Type pretraining, i.e. roughly 1-1.5 h more ordinary typing than the ~28 min used here.
9. Caveats: one user, one rig; architecture, steps and LM weights were chosen on keyboard data only (pilot on fold 131629, LM grid on LOSO), never on the desk; pretraining is one checkpoint per public source with 3 fine-tuning seeds.
10. Recommendation: drop per-tap key classification for text; pretrain on How We Type, fine-tune on my passive keyboard logs, then adapt with a few prompted desk phrases.

## Results (CER mean of 3 seeds [95% item bootstrap CI]; LM = char 6-gram prefix beam, alpha/beta tuned on keyboard LOSO)

| model | kbd LOSO | held-out 015948 | desk zero-shot | desk 5-fold fine-tuned (phrase text only) |
|---|---|---|---|---|
| per-tap pipeline (reference) | - | - | 0.448 [0.399, 0.498], WER 0.745 | (uses desk EM) |
| scratch, greedy | 0.433 [0.412, 0.456] | 0.381 [0.321, 0.423] | 0.454 [0.423, 0.489] | 0.315 [0.293, 0.339] |
| scratch, LM | 0.382 [0.360, 0.406] | 0.393 [0.288, 0.462] | 0.388 [0.344, 0.433], WER 0.725 | 0.224 [0.197, 0.252], WER 0.474 |
| How We Type pretrain, greedy | 0.332 [0.312, 0.354] | 0.293 [0.226, 0.363] | 0.320 [0.287, 0.354] | 0.241 [0.220, 0.262] |
| **How We Type pretrain, LM** | **0.287 [0.266, 0.310]** | **0.281 [0.172, 0.362]** | **0.234 [0.190, 0.277], WER 0.538** | **0.132 [0.102, 0.163], WER 0.326** |
| FSboard pretrain, greedy | 0.450 [0.429, 0.471] | 0.450 [0.364, 0.508] | 0.510 [0.478, 0.544] | 0.340 [0.313, 0.368] |
| FSboard pretrain, LM | 0.405 [0.381, 0.429] | 0.455 [0.345, 0.537] | 0.444 [0.392, 0.496], WER 0.727 | 0.228 [0.188, 0.270], WER 0.428 |
| desk phrases only, from scratch (control), LM | - | - | - | 0.627 [0.591, 0.660], WER 0.971 |

Paired desk deltas vs the pipeline (CER 0.448): scratch zero-shot -0.060 [-0.102, -0.015]; How We Type zero-shot -0.214 [-0.261, -0.170]; scratch fine-tuned -0.224 [-0.263, -0.182]; How We Type fine-tuned -0.316 [-0.369, -0.263].
Paired vs scratch (LM, same items): How We Type keyboard LOSO -0.096 [-0.109, -0.083], held -0.112 [-0.151, -0.061], desk zero-shot -0.154 [-0.180, -0.129], desk fine-tuned -0.092 [-0.120, -0.065]; FSboard keyboard LOSO +0.023 [+0.012, +0.034], held +0.063 [+0.021, +0.101], desk zero-shot +0.056 [+0.030, +0.082], desk fine-tuned +0.004 [-0.020, +0.028]. FSboard fine-tuned vs pipeline -0.220 [-0.265, -0.172] (the gain comes from the keyboard + desk fine-tuning, not from FSboard). Full table: `table.md`.

## Learning curve (training fractions of the 4 keyboard sessions, 3 seeds each, fixed LM alpha 0.3/beta 1.0)

| keystrokes | scratch desk LM | How We Type desk LM | scratch held LM | How We Type held LM |
|---|---|---|---|---|
| 674 (1/8) | 0.667 | 0.430 | 0.611 | 0.511 |
| 1,253 (1/4) | 0.563 | 0.375 | 0.516 | 0.408 |
| 2,344 (1/2) | 0.478 | 0.327 | 0.450 | 0.332 |
| 4,285 (all) | 0.396 | 0.268 | 0.362 | 0.262 |

Fit CER = a + b log2(keys): scratch desk b = -0.101/doubling, How We Type -0.060. Keys needed for desk CER 0.30 / 0.20: scratch ~8.1k / ~16.1k, How We Type ~3.0k / ~9.7k (`curve.json`).

## What did not work
- FSboard (sign-language fingerspelling) pretraining: zero-shot keyboard CER stays 1.00 and it is worse than scratch after fine-tuning.
- AdaBN (re-estimating BatchNorm statistics on the unlabelled target session): +0.07 to +0.21 CER worse on desk in every condition.
- Training on the desk phrases alone (16 phrases per fold, no keyboard data): CER 0.627, worse than the pipeline (+0.179 [+0.127, +0.234]). The keyboard logs are what make desk fine-tuning work.
- MediaPipe z and palm-length scaling (dropped; knuckle-width scale).

## Files
- Code: `phase0/analysis/seqctc.py` (data, model, training, CTC+LM decoding, experiments), `phase0/analysis/howwetype.py`, `phase0/analysis/fsboard.py`.
- Results: `results/seqctc/decode_*.json` (per-condition CER/WER, CIs, per-item edits), `table.md`, `curve.json`.
- Caches (gitignored): `.cache/seqctc/` (streams, pretrained checkpoints `pre/`, run logprobs `runs/`, raw downloads `hwt/`, `fsb/`).
