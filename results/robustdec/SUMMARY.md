# Noise-robust tap-stream word decoder (robustdec.py)

1. Explicit tap-noise model: missed taps, bursts of extra taps repeating the previous/next key, missed spaces; sums over tap alignments per word lattice; words ranked with Qwen2.5-0.5B; word boundaries need no detected space tap.
2. Real desk 202149, settings tuned only on simulated desk dev: CER 0.441 [0.396, 0.488] vs beam 0.448; ΔCER −0.007 [−0.067, +0.047]; words 40.1% vs 32.1%, Δ +8.0 points [−3.1, +19.8].
3. Leave-one-phrase-out setting choice on the real desk: CER 0.419 [0.366, 0.470], ΔCER −0.030 [−0.089, +0.027]; words 42.3%, Δ +10.2 [−0.7, +21.2]. Every fold picked the same setting (tap weight 0.75) — effectively one setting chosen on desk.
4. Fixes the old word-level decoder's collapse (CER 0.713 on the same session); several phrases come out nearly right, none fully.
5. Simulated desk dev (80% keys, 20 tuning sentences, favourable): CER 0.274 / 55.6% words vs beam 0.420 / 35.2%.
6. What mattered on dev: summing over alignments (0.274 vs 0.326 max), spurious-tap weight 0.667 from the simulator (vs 0.308 at 0.5, 0.450 at 1.0), low LM weight 0.6, word bonus 2.0.
7. Detector confidence/timing don't help: real-vs-spurious AUC 0.59 / 0.57; same-finger duplicates within 120 ms real 34% vs 40% otherwise; adding either term made desk slightly worse (0.451–0.454).
8. A tuning-search flaw (dropping the in-use value) was found and fixed; desk tuning re-run.
9. `models/vocab_blocklist.txt` (+ inflections) removes 74 of 30,894 words; none in dev/test/project phrases.
10. Verdict: best tap-stream decoder so far but not a proven gain over beam, 10–20 s/phrase; seqctc (0.234 zero-shot / 0.132 fine-tuned) is far better — stop investing in the tap-stream path.

| Real desk 202149 (20 phrases) | CER [95% CI] | Words correct | ΔCER vs beam |
|---|---|---|---|
| Beam (current) | 0.448 [0.399, 0.498] | 32.1% | — |
| Old word-level + Qwen | 0.713 | — | +0.265 [+0.174, +0.361] |
| New decoder, tuned on simulation | 0.441 [0.396, 0.488] | 40.1% [29.5, 51.1] | −0.007 [−0.067, +0.047] |
| New decoder, leave-one-phrase-out | 0.419 [0.366, 0.470] | 42.3% [32.3, 52.1] | −0.030 [−0.089, +0.027] |

Simulated test sweep skipped (priority moved to seqctc). Files: `phase0/analysis/robustdec.py`, `results/robustdec/` (`report_qwen_test.json`, `real_desk.png`, `tune_sim_dev.json`, `sim_dev_and_softtap.json`), caches `.cache/robustdec/`.
