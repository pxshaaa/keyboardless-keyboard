# Swipe-style word decoding — feasibility probes (swipe_*.py)

**Verdict: no.** Word-level "swipe path" decoding does not push bare-desk accuracy toward 100% with the current camera evidence. Best measured change: −2 word errors / 250 across dev desk sets. Remaining blind-2 errors are cases where CTC evidence itself favours the wrong words.

- P0 error anatomy (frozen v3): old desk 8 errors/137 words, blind-1 10/62, blind-2 9/51. Blind-2: "sorry claude but" +11.7 nats and "bifi" +8.2 favour truth; "by" −13.6, "wonder this does" −43.7, "deceived" −3.6 disfavour truth. Segment pad 0.5→1.0 s changes log-liks ≤1.6 nats.
- P1 lexicon-constrained CTC word beam (35.8k merged lexicon): claude/codex/best enter at rank 1 when CTC supports them; sorry, bifi, by, wonder, deceived, does never enter under any config; blind-2 seg1 is a model error, not a search error. Union with frozen pool: old 0.058 (=), blind-1 0.129→0.097 oracle, blind-2 0.176 (=).
- P2 swipe-path scorer on keyboard (728 words, LOSO): top-1 0.96 with the true touch-typing finger (label-derived upper bound), 0.32 label-free, ≤0.10 with detector finger; 10%/20% tap drop+insert → 0.68/0.45 even with true finger. Shape channel never helps.
- P3 on desk: realistic fusion (CTC peaks) = CTC + prior; apparent +13–18 pt gain with aligned frames is a leak (constant tap log-lik reproduces it); fingertip position adds nothing measurable.
- P4 sentence level (frozen weights, word-beam candidates injected): old 0.058→0.066, blind-1 0.161→0.113 (CI [−0.158, 0.000], not truly blind), blind-2 0.176 (=).

| milestone | verdict |
|---|---|
| M0 error tool | done |
| M1 lexicon word beam | go small (extra candidate source only; drop pad change) |
| M2 names boost | no-go as own milestone (keep names list in M1) |
| M3 calibration data + desk fine-tune | **go, top priority** |
| M4 motor/template scorer | no-go (gate fails) |
| M5 learned word scorer | no-go (finger attribution at chance) |
| M6 cross-segment 8B | no-go for now (at pool ceiling) |
| M7 suggestion bar | go (per-word top-5 0.80–0.94 with oracle spans) |

Caveats: one user, 250 desk words; blind-1 not blind; word-beam configs inspected on all sets. Code: `phase0/analysis/swipe_*.py`; results JSON in this folder.
