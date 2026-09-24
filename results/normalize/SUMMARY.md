# Camera/session-robust key identification — `phase0/analysis/normalize.py`

## Conclusion (10 lines)
1. **Raw-pixel pose does not survive a camera re-mount.** Trained on the three 2026-09-10 sessions, fused key-ID scores 0.534 on the re-mounted 2026-09-11 session (0.395 pose-only; 3 seeds). Per-key contact centroids move 1.11 key pitches, almost all of it a rigid shift.
2. **What works: `tap_g`.** One similarity transform per session, fitted with NO labels from the median hand configuration over detected taps. Cross-day fused 0.534 -> 0.698 (+0.164 [+0.141, +0.188]); 4-session LOSO 0.582 -> 0.713 (+0.130 [+0.116, +0.144]). It beats label-free CORAL by +0.050 [+0.032, +0.068] cross-day and +0.052 [+0.040, +0.063] LOSO.
3. **What does not.** Cross-day fused: CORAL with label-free moments 0.648 (an earlier 0.695 had used labelled-tap moments and was corrected); CORAL stacked on tap_g 0.658. On one seed (all clearly below tap_g): rest-pose/home-row anchoring 0.651, per-session wrist anchor 0.623, palm-relative 0.609. Per-hand transforms were worse in the label-free diagnostic. A palm-travel keyboard-scale estimate collapsed pose accuracy to 0.39.
4. **Warm-up, no labels.** A new setup needs ~200-400 detected taps (~1.5-3 min of typing) before the frame locks in: 0.706-0.713 mean over 8 windows vs 0.716 with the whole session. Below 100 taps it is unstable (worst window 0.49-0.55).
5. **Enrolment: with tap_g, a same-user re-mount needs 0 labelled taps.** Label-based calibration adds only +0.016 at 50 taps and +0.024 at 100 (fine-tuning ~0, CIs span zero). Raw pixels need enrolment: 25 taps of calibration give +0.101 (0.539 -> 0.640), but the best raw-pixel result (200 taps, calibration + fine-tune, 0.658) is still below tap_g with zero labels (0.708 on the same eval taps).
6. **Self-training without a keylogger does not help once tap_g is on.** Rows 6-7 use the day-2 session's unlabelled first half and score on its second half. LM-decoded pseudo-labels give +0.003 [-0.013, +0.019] fused. Even the true labels of the same taps give only +0.016 [-0.002, +0.035]. Confidence-only pseudo-labels hurt: -0.017 [-0.033, -0.001].
7. **Self-training does help raw pixels.** LM self-training gives +0.037 [+0.021, +0.053] fused and keyboard CER -0.02 to -0.03 on every seed, but it ends at ~0.54, far below tap_g without adaptation.
8. **Desk CER: tap_g does NOT change desk CER.** Seed-averaged, nested leave-one-phrase-out, with the text-based EM step: tap_g 0.452 vs the matched production path (abs+CORAL) 0.459, delta -0.007 [-0.036, +0.023]. Against official hirecall 0.448: +0.004 [-0.045, +0.054]. Without the EM step tap_g is worse, 0.540 vs 0.482 (+0.059 [+0.007, +0.106]). Desk self-training without text also does not help (0.460 with EM, 0.522 without). This fits the autocorrect finding: desk errors are dominated by the tap stream, not by which key a correctly detected tap is.
9. **Cross-user limit.** tap_g ignores camera zoom, rotation and shift by construction, but NOT hand size: +-15% hand scale drops it 0.68 -> 0.47 pose-only, because hand size and camera distance are confounded in one view. Training with +-15% hand-size augmentation restores fused to 0.66 at both scales (from 0.53-0.56). It costs ~0.03 at native size (0.697 -> 0.666; 2 seeds).
10. **Product recommendation and caveats.** Recommendation: use tap_g, fit it on the first ~300 detected taps, skip enrolment for the same user, use hand-size augmentation for new users, and do not count on self-training for key ID. Caveats: one user and one keyboard; the cross-day test is one re-mount; desk results cover 20 phrases over a reduced grid (20 streams, 8 decoder settings), so the matched 0.459 baseline is the fair comparator, not 0.448; keyboard CER used 10 segments with fixed decoder weights; rel/palm/rest were run on 1 seed.

## Key numbers

| question | before | after | delta [95% CI] |
|---|---|---|---|
| cross-day fused top-1 (train 09-10, test 09-11 re-mount) | abs 0.534 | tap_g 0.698 | +0.164 [+0.141, +0.188] |
| 4-session LOSO fused top-1 | abs 0.582 | tap_g 0.713 | +0.130 [+0.116, +0.144] |
| tap_g vs label-free CORAL, cross-day fused | coral 0.648 | tap_g 0.698 | +0.050 [+0.032, +0.068] |
| enrolment, tap_g, 100 labelled taps (calibration) | 0.708 | 0.732 | +0.024 [+0.011, +0.037] |
| enrolment, abs, 25 labelled taps (calibration) | 0.539 | 0.640 | +0.101 [+0.080, +0.123] |
| self-training, tap_g, LM pseudo-labels r1 | 0.705 | 0.707 | +0.002 [-0.012, +0.017] |
| self-training, abs, LM pseudo-labels r1 | 0.506 | 0.543 | +0.037 [+0.021, +0.053] |
| desk CER with text EM, tap_g vs matched abs+coral | 0.459 | 0.452 | -0.007 [-0.036, +0.023] |
| desk CER without EM, tap_g vs matched abs+coral | 0.482 | 0.540 | +0.059 [+0.007, +0.106] |
| hand size x1.15, fused, tap_g vs hand-size-augmented | 0.531 | 0.663 | 2 seeds, no CI |

Full tables: `numbers.md` (regenerate with `python -m phase0.analysis.normalize_report`); self-training log summary: `selftrain_summary.txt`; desk per-seed rows: `desk_main.json`, seed-pooled: `desk_pooled.json`.
