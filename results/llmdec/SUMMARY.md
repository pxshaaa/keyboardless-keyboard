# Large-LLM + personal-context decoder (llmdec2 / decipher2)

**Recommended for blind test #2: frozen v3 personal** — `PYTHONPATH=. .venv/bin/python -m phase0.analysis.decipher2 data/sessions/<new-id> --version v3 --config recommended` (~8 min for a 2-min session incl. landmarks; `--version v2` = frozen v2 personal; `--score truth.txt` to score).

1. Best method: Qwen3-8B-4bit generate-and-verify (8B rewrites the n-best; every candidate scored by exact CTC log-lik + λ·LLM + bonuses). Constrained (LLM-guided CTC prefix) decoding lost at every 3B setting; 8B constrained grid dropped.
2. Personal context tuned on my keyboard windows only: profile+context prompt with a personal LoRA (Qwen2.5-0.5B) kbd WER 0.464→0.451; + keyboard-aware fuzzy candidates over the merged personal lexicon 0.430. Personal lexicon/word-LM terms in the 8B selection ≤0.007.
3. Cross-phrase context helped slightly; seed ensembling barely.
4. v3 fuzzy candidates lowered the pool oracle (kbd 0.379→0.358, old desk 0.146→0.131) and recovered "claude", "codex", "really" on blind-1; "div" (not in lexicon) and "center" still fail.
5. Selection is near its pool ceiling (blind-1 v3 0.210 vs oracle 0.161; old desk v2 0.131 vs 0.117) — remaining errors are letters never in the candidates, i.e. the camera/CTC model.
6. Blind-1 is NOT truly blind (truth seen; v3 designed from its errors); 62 words, ±3 words ≈ 0.05 WER. Blind-2 is the real test.
7. Leakage: keystroke personal source excluded; anything sharing a 4-gram with tuning sets or blind-1 truth dropped; LoRA training data 0 overlaps.
8. 14B skipped (mini swapping from non-agent processes). Safety: one GPU job via gpulock, MLX memory caps, nice 10.

| variant | kbd WER | old desk WER | blind-1 WER / CER / words |
|---|---|---|---|
| Qwen-0.5B word decoder (previous) | 0.536 | 0.226 | 0.274 / 0.135 / 73% |
| + personal lexicon | 0.529 | 0.234 | 0.226 / 0.120 / 77% |
| gen-verify 8B (v1) | 0.474 | 0.139 | 0.242 / 0.099 / 77% |
| frozen v2 generic (8B + cross-phrase context) | 0.464 | 0.131 | 0.258 / 0.109 / 74% |
| frozen v2 personal (+ profile + personal LoRA) | 0.451 | 0.131 | 0.210 / 0.139 / 81% |
| **frozen v3 personal (+ fuzzy candidates + vocab bonus)** | **0.430** | 0.139 | **0.210 / 0.073 / 79%** |
| constrained 3B, best | 0.529 | 0.255 | 0.419 / 0.186 / 63% |

Runtime/phrase on the mini: 0.5B ~1.5 s, 8B gen-verify ~14 s (v2) / ~19 s (v3). Files: `results/llmdec/` (frozen configs, blind-1 scores, oracles, tuning, `report_v2.json`, `status_v2.json`); code `phase0/analysis/llmdec2.py`, `llmdec_fuzzy.py`, `llmdec3_freeze.py`, `llmdec3_blind1.py`, `decipher2.py`.
