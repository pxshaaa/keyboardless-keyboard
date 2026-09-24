# Touch detection on a bare desk: press vs hover

## Conclusion (10 lines)
1. Only one idea made the tap stream itself better: a fingertip-crop CNN that recognises contact appearance (pad occlusion/shadow), trained on keyboard crops and used to rescore desk candidates. Combining it with the kinematic features gives the best desk numbers: count error 0.088 (-0.076 [-0.111, -0.042]), oracle CER 0.195 (-0.036 [-0.069, -0.006]), end-to-end CER 0.395 (-0.026 [-0.051, +0.000]); exploratory, chosen after seeing both parts.
2. On held-out keyboard it cuts misses 19.5% -> 17.6% at 1.0 taps/char and 13.7% -> 11.3% at 1.3 (extra taps 0.199 -> 0.183 and 0.436 -> 0.412 per char); on desk it lowers density-matched oracle CER -0.048 [-0.081, -0.018] and end-to-end CER 0.421 -> 0.397 (-0.024 [-0.051, +0.002]). Every kinematic arm leaves misses/extras unchanged (~19-20% / 0.20 at 1.0).
3. New kinematic features (stop-motion, relative finger drop, pooled over fingertips) cut desk per-phrase count error 0.164 -> 0.108 (-0.055 [-0.076, -0.034]), but ~25% of their gain is a static hand-posture column; dynamics-only still gives -0.039 [-0.061, -0.018].
4. That count gain does not carry into tap-stream quality: density-matched perfect-key oracle CER -0.012 [-0.035, +0.012] (touch), +0.004 (dynamics-only).
5. A constant "average phrase length" guess scores count error 0.053, better than every detector: counts vary too little across the 20 phrases for this metric to certify detection.
6. EgoPressure (21 people, MANO + pressure) rendered as a top-down MediaPipe view: same index motion pressed vs hovering is only AUC 0.61 from 2D, 0.78 with perfect relative depth, 0.92 from true fingertip height.
7. EgoPressure transfer is weak: zero-shot keyboard F1 33.6%; stacking its score into our detector: count error -0.022, oracle CER -0.003 [-0.022, +0.015].
8. Desk self-training from known character counts fails: forced count alignment labels only ~80% of keyboard taps correctly (= a count-matched threshold), and retraining worsens count error (+0.016) and oracle CER (+0.016).
9. End-to-end desk CER (hirecall stack, reduced grid, 3 seeds): base 0.421, kinematic features 0.402 (-0.019 [-0.050, +0.010]), dynamics-only 0.423 (+0.001), pixel verifier 0.397 (-0.024 [-0.051, +0.002]). A sequence TCN is worse than LightGBM on keyboard (F1 75.2 vs 78.5), and adding a desk count loss hurts tap-stream quality (oracle +0.054 [+0.003, +0.103]).
10. Press-vs-hover from one top-down camera looks physically limited, not model-limited: the information is in fingertip height, which a 2D overhead view barely encodes and MediaPipe's depth does not recover.

## Numbers
| detector | seeds | kbd held-out F1 | kbd miss @1.0 / @1.3 taps/char | kbd extra/char @1.0 / @1.3 | desk count err (d vs base) | desk oracle CER @1.0+1.25 (d vs base) | desk end-to-end CER (d vs base) |
|---|---|---|---|---|---|---|---|
| base (taps_gb features, 4 kbd sessions) | 3 | 78.5 | 19.5% / 13.7% | 0.199 / 0.436 | 0.164 (-) | 0.231 (-) | 0.421 (-) |
| idea 2: + kinematic touch features | 3 | 79.3 | 19.9% / 13.6% | 0.204 / 0.436 | 0.108 (-0.055 [-0.076, -0.034]) | 0.219 (-0.012 [-0.035, +0.012]) | 0.402 (-0.019 [-0.050, +0.010]) |
| idea 2: + touch features, dynamics only | 3 | 79.5 | 19.0% / 13.7% | 0.198 / 0.436 | 0.125 (-0.039 [-0.061, -0.018]) | 0.235 (+0.004 [-0.017, +0.025]) | 0.423 (+0.001 [-0.028, +0.032]) |
| idea 2b: + fingertip-crop contact CNN | 3 | 78.5 | 17.6% / 11.3% | 0.183 / 0.412 | 0.112 (-0.052 [-0.080, -0.024]) | 0.183 (-0.048 [-0.081, -0.018]) | 0.397 (-0.024 [-0.051, +0.002]) |
| idea 2+2b combined | 3 | 79.3 | - | - | 0.088 (-0.076 [-0.111, -0.042]) | 0.195 (-0.036 [-0.069, -0.006]) | 0.395 (-0.026 [-0.051, +0.000]) |
| idea 1: + EgoPressure contact score | 3 | 79.0 | 19.8% / 14.6% | 0.196 / 0.446 | 0.142 (-0.022 [-0.042, -0.003]) | 0.228 (-0.003 [-0.022, +0.015]) | - (-) |
| idea 3: desk self-training (count-forced labels) | 3 | 77.4 | - | - | 0.180 (+0.016 [-0.025, +0.059]) | 0.247 (+0.016 [-0.008, +0.040]) | - (-) |
| idea 4: TCN, no count loss | 3 | 75.2 | - | - | 0.148 (-0.016 [-0.072, +0.038]) | 0.246 (+0.015 [-0.019, +0.052]) | - (-) |
| idea 4: TCN + desk count loss | 3 | 75.0 | - | - | 0.152 (-0.011 [-0.072, +0.044]) | 0.285 (+0.054 [+0.003, +0.103]) | - (-) |
| ref: production taps_gb.joblib | 1 | - | - | - | 0.148 (-0.001 [-0.050, +0.049]) | 0.255 (+0.030 [-0.016, +0.080]) | - (-) |
| ref: hirecall recall detector | 1 | - | - | - | 0.112 (-0.038 [-0.081, +0.004]) | 0.289 (+0.064 [+0.031, +0.098]) | - (-) |

Rows marked `-` were not measured for that detector (the self-trained, TCN and reference detectors have no saved held-out keyboard stream at matched density; end-to-end CER was run for base, touch, touchdyn, pix and pix+touch). Reference rows are single fitted models, not seeds.

## Protocol and caveats
- Keyboard: LightGBM detectors train on 4 kbd sessions (015217, 021315, 131629, 164237); held-out 20260910-015948-kbd scored with eval_taps (+/-80 ms 1:1, clipped). Event params tuned on leave-one-session-out OOF probs for seed 0; seeds 1-2 reuse that arm's seed-0 params (base keyboard F1 with per-seed tuning on the MacBook: 77.9 / 79.9). Held-out session is 90 s: touch vs base F1 +0.8 [-1.1, +1.7] (10 s block bootstrap).
- Missed/extra taps are measured on the held-out keyboard at matched tap density (threshold set so taps/char = 1.0 or 1.3). The desk has no tap truth, so desk tap quality is proxied by oracle CER.
- Desk metrics use 20260910-202149-desk (20 phrases, 676 chars), all leave-one-phrase-out, seed-averaged per phrase, paired phrase bootstrap (10k) vs base.
  - (a) count error: |taps - chars|/chars at 1 tap/char, threshold from the other 19 phrases. Trivial floor: guessing the mean phrase length scores 0.053, better than every detector, so it ranks detectors but cannot certify one.
  - (b) perfect-key oracle CER: key identity handed over by aligning taps to the true text (pipeline.ceiling_row's construction), decoded with the char LM. The LOPO-selected version is too noisy across seeds (0.13-0.25 for one arm); the table uses the density-matched version at 1.0 and 1.25 taps/char (the densities every arm reaches; the self-trained detector saturates at ~1.5). Not comparable with REPORT's older 0.23-0.32 ceiling.
  - (c) end-to-end CER: hirecall's stack (pose + pixel-CNN key probs + CORAL + EM + char/word LM), reduced grid (30 streams x alpha {.5,.75,1} x obs {2,3,4} x deletion {-9,-5,-3}) centred on hirecall's fold medians, nested LOPO. The grid carries mild desk-derived information, equally for every arm; deltas are valid, absolute values are optimistic. Pixel key probs snapped to the nearest crop-bank frame (97% within 1 frame).
- Pixel contact CNN (touch_pix): trained only on keyboard crops (keydown frames vs near-miss and random typing frames), weight 0.5 fixed a priori (p^0.5 * r^0.5), never tuned on desk. Desk frames more than 2 frames from hirecall's crop bank (6%) get the median score.
- Self-training: 5 phrase folds; stitched probabilities mix 5 models whose calibrations differ (fold thresholds 0.18-0.39); the deployment-like in-fold threshold variant is worse still (count error 0.254).
- EgoPressure: 12 press/type/draw gestures x 2 hands x 21 participants (annotation + pressure only, 8.9 GB), MANO joints projected through overhead static camera 4 and mapped into our image convention per hand, resampled 30 -> 60 Hz; 2D view keeps x, y only. The "3D" view uses MANO relative depth, far cleaner than MediaPipe's real z, so it is an upper bound.
- All sessions are one person, one rig; nothing here says anything about other typists.

## Files
- Code: phase0/analysis/touch_common.py (harness + desk metrics), touch_runs.py (runner), touch_report.py (table), touch_feats.py (idea 2), touch_pix.py (idea 2b), contact_ego.py (idea 1), desk_selftrain.py (idea 3), taps_seq_count.py (idea 4), touch_missextra.py
- Results: results/contact/table.md, table.json, arm_*.json, missextra.json, pix_kbd_missextra.json, ego_press_hover.json, ego_zero_shot.json, selftrain_infold.json, count_err_trivial.json, ref_detectors_count.json, kbd_touch_vs_base_ci.json
- Caches: .cache/contact_ego/ (feature matrices, desk/kbd probabilities, EgoPressure subset in egop/, crop banks in pixbank/ and pixbank_cand/)
