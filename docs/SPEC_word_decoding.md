# Spec: word-level ("gesture-style") decoding toward ~100% bare-desk words

> **Update 2026-09-13 (feasibility probes, `results/swipe/SUMMARY.md`, report `docs/swipe_study.html`):** swipe-style path matching does not transfer to the bare desk with current camera evidence (finger attribution unknown; best case −2 word errors / 250). Revised priorities: M3 own labelled desk data first, M7 suggestion bar, M1 only as a small extra candidate source; M2 standalone, M4, M5, M6 are no-go for now.


> **Update 2026-09-14 (M3 data step, revised after the Codex review `docs/PROJECT_REVIEW_2026-09-13.md`):** (1) `ctc_v4` desk models = hwtmix fine-tuned on all 31 labelled desk phrases (old desk 20 + blind-1 6 + blind-2 5; v3 d26 recipe, 3 seeds); manifest `.cache/ctc_v4/deploy/manifest.json` (v3 layout). Leave-one-session-out (CPU proxy decoders; frozen v3 decoder needs the mini, twin sets `data/sessions/zz-v4loso-*` prepared): on the two blind sessions, fine-tuning on the other two sessions beats old-session-only v3 finals, Qwen-0.5B WER 0.257→0.212 (dWER −0.044 [−0.113, 0.000]), char decoder 0.416→0.310 (−0.106 [−0.153, −0.067]); zero-shot 0.319 / 0.504. Old desk held out (11 training phrases): Qwen 0.234 zs→0.255, char 0.409→0.328. Pooled over all 3 folds vs zero-shot: Qwen −0.036 [−0.093, +0.025], char −0.132 [−0.190, −0.075]. (2) Calibration prompts are now **a–z + space only** (the legacy alphabet drops digits/umlauts, so `get2germany` became `getgermany`; checked with `text_contract.legacy_training_text`): `phase0/phrases_calib.txt` (65) + `phrases_calib_natural.txt` (15) = 80 prompts, split by `calibration_plan.py` into `phase0/phrases_calib_block{1..4}.txt` (4 × 20, ~4.7 min each, ~25 min with phone remounts between blocks for camera diversity); all 55 names/product/confusable terms ≥2×, 253/300 top personal words present, 0 shared 4-grams with blind-1/2; protocol `phase0/CALIB_PROTOCOL.txt`. (3) `phase0/analysis/calib_ingest.py` no longer certifies its own labels (alignment accepts 48% of one-word-deleted references): windows with no typing or typing cut off by the next prompt are dropped, everything else needs human confirmation in `<session>/calibration_review.json` (`calibration_review.py`), and only confirmed text becomes a label. Training runs on the **Mac mini** (rsync + ssh + gpulock; `--compute local` only for tiny tests). Dry run on the old desk session (simulated confirmations): 13/20 confirmed usable, 7 cut off by the 10 s prompt window, windows 100% inside the stored ones, mini training + manifest + CV all ran.

Status 2026-09-13: blind test 2 = 82% words (official scorer, frozen v3). Target on a fresh sealed blind-3: **≥92% words automatic (stretch 95%), ≥98% with ≤1 tap per wrong word (top-3)**. 100% fully automatic with one camera is not realistic.

## Blind-2 diagnosis
- "sorry claude but": not in the candidate pool, but the CTC evidence FAVOURS the truth (−129.9 vs −161.4 nats) → fixable by better candidate generation.
- "by the day", "no wonder this does … deceived": truth scores worse on CTC evidence (+10 / +44 nats) → needs new evidence or strong context.
- sorry/claude/bifi/deceived/by were in no stored pool although the personal lexicon has "claude" ×405, "bifi" ×19.

## How swipe decoders work (and what transfers)
SHARK² (shape + location channels, template pruning, LM prior; ~80% top-1), Google Gboard (spatial model + lexicon FST + n-gram LM; Alsharif 2015 BLSTM-CTC on touch points + lexicon FST), FUTO Swipe TCN (92.9% top-1), How We Swipe dataset. Doesn't transfer literally (we have ten fingers making discrete, often undetected taps, not one continuous trace). Transfers: (1) decoding against a closed personal lexicon from the posteriors (CTC + lexicon trie/FST), (2) template pruning, (3) SHARK² location/shape channels on the tap polyline in a calibrated keyboard plane.

## Milestones (ranked by expected gain: M1 > M3 > M2 > M6 > M4 > M5)
| # | Work | Effort | Machine | Acceptance |
|---|---|---|---|---|
| M0 | Error-type tool + dev harness (truth-vs-output log-lik per word) | 0.5 d | MacBook CPU | per-word error types (a–f) reported |
| M1 | Lexicon-constrained CTC word beam over merged lexicon (generic + personal + names), optional spaces, personal uni/bigram prior, name boost; top-50 into gen-verify pool; start pad 0.5→1.0 s | 1–1.5 d | MacBook CPU; 8B verify on mini | pool oracle ↓ on kbd/old/blind-1 (0.358/0.131/0.161); blind-2 seg1 contains "sorry claude but"; no kbd WER regression > 0.01 |
| M2 | Proper-noun class + contextual boost; user-editable `names.txt` | 0.5 d | MacBook | names recall ≥ 90% on dev, no extra false name insertions |
| M3 | Data: 15-min calibration (top 300 personal words incl. all names ×2–3 in prompted phrases + ~40 sentences) + 10-min blind-3 (≥150 words, sha256 of truth committed before decoding); fine-tune 3 desk seeds | 1 d + 25 min user | MacBook GPU (one job) | — |
| M4 | Motor model (per-key 2-D Gaussians + P(finger\|key) from keyboard taps, MAP-adapted to desk via CTC forced alignment) + template word scorer (pair-HMM/DTW, location × finger × shape channels, explicit miss/extra probabilities, SHARK² pruning) as generator + score channel | 2.5–3 d | MacBook CPU | gate: true word top-5 recall ≥ 60% on CTC-aligned desk spans and dev WER −0.02, else stop |
| M5 | (conditional) learned dual-encoder word scorer (motion span ↔ word template, contrastive) | 3–5 d | MacBook MLX | top-5 recall ≥ M4 + 10 pts |
| M6 | Cross-segment word lattice (top-3 per slot) + 8B joint rescoring for context errors | 1 d | mini (gpulock) | dev WER −0.02, ≤ 30 s/phrase; cap λ |
| M7 | Suggestion bar: top-3 per word + names quick list | 1 d | — | words correct after ≤1 tap; taps/100 words |
| M8 | Freeze (md5 manifest), blind-3 run, report with ablations v3→+M1→+M2→+M4→+M6 | 0.5 d | mini + MacBook | Wilson CIs, per-error-type breakdown |

## Evaluation rules
Dev (tuning allowed): keyboard windows, old desk, blind-1, blind-2. Everything recorded after the freeze is test-only; no re-tuning after seeing blind-3; drop lexicon/LoRA text sharing a 4-gram with blind-3 truth and report it. One GPU job per machine at any time.

## Sources
SHARK² (Kristensson & Zhai, UIST 2004) http://pokristensson.com/pubs/KristenssonZhaiUIST2004.pdf · Alsharif et al., ICASSP 2015 https://research.google.com/pubs/archive/43461.pdf · Gboard blog https://research.google/blog/the-machine-intelligence-behind-gboard/ · How We Swipe https://luis.leiva.name/web/docs/papers/shapewriting-mobilehci2021-preprint.pdf · FUTO Swipe https://arxiv.org/html/2606.25247 · Gesture2Text https://arxiv.org/abs/2410.18099 · neural-swipe-typing https://github.com/proshian/neural-swipe-typing · Richardson et al. UIST 2020 https://dl.acm.org/doi/10.1145/3379337.3415816 · StegoType UIST 2024 https://dl.acm.org/doi/10.1145/3654777.3676343
