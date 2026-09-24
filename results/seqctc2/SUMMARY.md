# Sequence model to goal (seqctc2.py)

**Headline:** bare-desk goal met after desk fine-tuning — WER 0.139 [0.096, 0.185] (~86% words); with no desk labels WER 0.253 [0.178, 0.327] (~75% words). Session 20260910-202149-desk, 20 phrases; best model 5 seeds, others 3.

1. Qwen word decoder (weights tuned on keyboard windows ≥80% in-lexicon, scored on WER; never on desk) takes fine-tuned desk WER 0.326 → 0.151 [0.105, 0.201] on the same posteriors (dWER −0.175 [−0.242, −0.115]).
2. Best model: baseline How-We-Type checkpoint fine-tuned with 30% How We Type mixed into each batch + own-tuned Qwen decoder. Zero-shot desk CER 0.157, WER 0.253 (dWER −0.224 [−0.311, −0.140] vs baseline+own Qwen); fine-tuned CER 0.083, WER 0.139.
3. Zero-shot gain is not a tuning/seed artifact: char decoder CER 0.174 vs 0.234; baseline-tuned Qwen WER 0.333 vs 0.477; per-seed WER 0.226–0.292; seeds 3–4 replicated with no new tuning/selection.
4. After desk fine-tuning, models barely differ (WER 0.14–0.15 with own-tuned decoder, CIs span 0).
5. Continuous desk stream with automatic pause segmentation matches phrase windows: fine-tuned WER 0.143 [0.094, 0.196], zero-shot 0.254 [0.188, 0.315].
6. Stronger geometry augmentation improves keyboard LOSO (−0.025 CER) but not desk; larger model (d=256, 6 layers) best held-out keyboard, desk no better; masked-landmark self-supervised pretraining no gain; augmentation+mix worse than mix alone.
7. Label-free session frame did not help (similarity: unchanged loss; affine: worse).
8. Leakage audit clean: no desk phrase in keyboard, How We Type or LM texts (corpora contain ≤7 of 9 words of a phrase); only "let me know" shared; folds split by phrase; unused desk session 201720 has reworded prompts and is excluded; fine-tuning deterministic.
9. Caveats: one user, 20 desk phrases, best model picked among 6 conditions on those phrases; closed 30.8k-word lexicon contains all 91 desk words; decoder tuning on 75 keyboard windows is noisy (±0.03–0.06 WER); How We Type is CC-BY-NC-4.0 (research only).
10. Recommendation: How-We-Type checkpoint + keyboard logs, fine-tune with 30% How We Type mix, Qwen word decoder tuned on my keyboard windows; a few prompted desk phrases take ~75% → ~86% words. Confirm on a fresh desk session (`phase0/phrases_desk2.txt`) with models frozen.

| model | kbd LOSO char CER | zero-shot char CER | zero-shot Qwen (own) CER / WER | zero-shot dWER | fine-tuned Qwen (own) CER / WER | fine-tuned dWER | continuous fine-tuned WER |
|---|---|---|---|---|---|---|---|
| published (char 6-gram) | 0.287 | 0.234 | WER 0.538 | – | 0.132 / 0.326 | – | – |
| baseline + Qwen | 0.287 | 0.234 | 0.298 / 0.477 | ref | 0.094 / 0.151 | ref | 0.168 |
| stronger aug, 16k pretrain | 0.261 | 0.236 | 0.288 / 0.406 | −0.071 [−0.148, +0.003] | 0.101 / 0.139 | −0.012 | 0.158 |
| masked-landmark → stronger aug | 0.269 | 0.223 | 0.288 / 0.443 | −0.034 | 0.167 / 0.246 | +0.095 [+0.027, +0.171] | 0.255 |
| **How We Type mix (5 seeds)** | 0.282 | **0.174** | **0.157 / 0.253** | **−0.224 [−0.311, −0.140]** | **0.083 / 0.139** | −0.012 [−0.057, +0.028] | **0.143** |
| stronger aug + mix | 0.262 | 0.169 | 0.247 / 0.397 | −0.080 | 0.131 / 0.185 | +0.034 | 0.200 |
| larger model | 0.262 | 0.207 | 0.330 / 0.438 | −0.039 | 0.113 / 0.185 | +0.034 | 0.197 |

Files: `phase0/analysis/seqctc2.py`; `results/seqctc2/` (`summary.json`, `hwtmix_5seeds.json`, `decode_*.json`, `audit.json`, `geom.json`, `diag.json`, `tuning/`); hypotheses/logs in `.cache/seqctc2/`.
