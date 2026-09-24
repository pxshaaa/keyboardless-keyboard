# Decoder v4 — word beam + suggestion bar (decipher2_v4)

Command: `PYTHONPATH=. .venv/bin/python -m phase0.analysis.decipher2_ctc3 data/sessions/<id> --version v4 [--manifest <path>] [--score <truth.txt>]` → `data/sessions/zz-ctc3-<id>/decipher2_v4.json` (per-word top-5 alternatives, start/end times, names list) + `.txt`. Dry run on a 2-min blind-2 copy: 318 s (landmarks 132 s, 8B rewrites 98 s, LLM scoring 70 s).

| set (words) | version | WER | auto words | ≤1 tap top-3 | taps/100 |
|---|---|---|---|---|---|
| old desk CV-safe (137) | v3 | 0.058 | 94.2% | 94.9% | 0.7 |
| | v4 | 0.058 | 94.2% | 95.6% | 1.5 |
| blind-1 (62) | v3 | 0.161 | 83.9% | 93.5% | 9.7 |
| | v4 | 0.129 | 87.1% | 93.5% | 6.5 |
| blind-2 (51) | v3 | 0.176 | 82.4% | 82.4% | 0 |
| | v4 | 0.176 | 82.4% | 82.4% | 0 |

- Word beam (small config, top-50 per segment) added to the v3 pool with unchanged selection weights; 27 → 25 word errors pooled.
- Suggestion bar: candidate scores / T (T=8) aligned per word slot; names quick list (12, incl. non-names "gyn", "german" to prune) adds 0.
- Blind-2 misses ("sorry claude but", "bifi", "wonder this does", "deceived") are in no candidate — only better camera evidence can fix them.
- NOT blind: configs and boosted names were chosen with all three dev sets in view. Needs a sealed blind-3.
Files: `phase0/analysis/wordbeam_v4.py`, `v4_suggest.py`, `decipher2_v4.py`, `predict_manifest.py`, `v4_dev.py`, `v4_freeze.py`; `results/llmdec/frozen_config_v4.json`, `results/v4/`.
