# Autocorrect vs camera accuracy (autocorrect.py)

**Headline:** tap detection is the gate. A much smarter autocorrect only pays off once the camera detects taps almost perfectly. With today's desk tap stream, no decoder tested beat the current 0.448 CER.

## Conclusion
1. Tap detection, not key identity, is the bottleneck: with the desk detector (20% of keystrokes missed, 1.3 extra taps/char) no decoder reaches 50% words correct even at 95% key accuracy.
2. 80–90% words correct needs ≤5% missed and ≤5% extra taps plus ~90% key accuracy with the word-level decoder (31k vocabulary scored by local Qwen2.5-0.5B): 89% words at 90% keys [84–93%], 75% at 80%.
3. With perfect tap detection the word-level decoder reaches 92% words at 70% keys and 97% at 80%; the current beam decoder needs ~91% keys for 90% words (held-out model today: 65.7%).
4. Rescoring the beam's top 30 with the neural LM: +3–4 points words with perfect or 5%/5% taps (CI excludes zero), +1–2 with keyboard-detector taps, nothing with desk taps.
5. Why rescoring stops there: with noisy taps the right sentence is rarely in the beam's top 30 — best-of-30 by hand still gives only 47% (desk taps) / 56% (keyboard-detector taps) at 90% keys.
6. The word-level decoder breaks down with noisy taps: at 80% keys −2.6 points vs beam with keyboard-detector taps, −5.9 with desk taps (both CIs include zero). An LLM rewrite of the top 10 (Qwen2.5-0.5B-Instruct) changed nothing (±1.1 points).
7. Real desk session 202149: CER stays 0.448 [0.399, 0.498] (WER 0.745). Neural rescoring 0.441, Δ −0.007 [−0.020, +0.004]; word-level decoder 0.713, Δ +0.265 [+0.174, +0.361]; LLM rewrite 0.478, Δ +0.030 [+0.008, +0.053]; best-of-30 ceiling 0.393.
8. Checks are clean: the clean-tap gain repeats on the 20 project phrases (100% words at 80% keys, +7.3 points over beam [+3.0, +11.9]); no sign Qwen memorised MacKenzie (bits/char vs char n-gram 0.816 MacKenzie vs 0.800 project phrases).
9. Simulator caveats: resampling real vectors instead of shifting the true key's probability gives 4–7 points fewer words at the same top-1; the real desk decodes like 74–78% simulated key accuracy while its tap statistics look like 55–60%, so simulated desk noise may be harsher than reality; a second noise seed moves words by ≤1.6 points.
10. Order of work: get desk tap detection to ≤5% missed and ≤5% extra first; then 70–80% key accuracy is already enough for ~90% words with the word-level decoder.

## Words correct, test split (MacKenzie; beam/n-best n=350, word decoder n=100 clean / 60 at 5%/5%)
| Tap errors | Decoder | 70% keys | 80% keys | 90% keys |
|---|---|---|---|---|
| none | beam (current) | 0.722 | 0.830 | 0.894 |
| none | n-best + Qwen | 0.757 | 0.873 | 0.936 |
| none | word-level + Qwen | **0.921** | **0.971** | **0.991** |
| 5% missed / 5% extra | beam | 0.540 | 0.686 | 0.768 |
| 5% missed / 5% extra | n-best + Qwen | 0.562 | 0.728 | 0.809 |
| 5% missed / 5% extra | word-level + Qwen | — | 0.746 | **0.890** |
| keyboard detector (16% / 15%) | best decoder | 0.33 | 0.43 | 0.51 |
| desk detector (20% / 130%) | best decoder | 0.25 | 0.34 | 0.41 |

## Method
Simulator draws real out-of-fold fused key-probability vectors (9,957) per true key; accuracy set by shifting the true key's probability (at 65.7% top-1, top-5 is 92.5% vs 93% real). Desk tap errors measured by aligning real desk taps to known text and matched in simulation (overdispersed counts, extra taps repeat neighbouring keys). MacKenzie 150 dev / 350 test + 20 project phrases; tuned on dev, paired bootstrap on test. All models local; heavy runs on the Mac mini.

## Limitations
Word-decoder sweep uses 30–100 sentences (~9 s/sentence). The 5%/5% setting reuses keyboard-tuned word-decoder weights.

## Files
`phase0/analysis/autocorrect.py`; `results/autocorrect/`: `sweep.json`, `curve.png`, `desk_qwen2.5-0.5b.json`, `robust.json`, `probe_qwen2.5-0.5b.json`; caches in `.cache/autocorrect/`.
