# Camera-to-text CTC v3 (ctcv3 / decipher2_ctc3)

**Blind-2 pipeline:** `PYTHONPATH=. .venv/bin/python -m phase0.analysis.decipher2_ctc3 data/sessions/<id> --version v3 --config recommended` (optional `--selftrain`, off by default; `--score <truth.txt>`). Deployed: hwtmix zero-shot ×3 seeds + hwtmix fine-tuned on all 26 labelled desk phrases ×3 seeds; frozen v3 decoder on the Mac mini.

| Old desk session, CV-safe, frozen v3 decoder | WER [95% CI] | CER | Words |
|---|---|---|---|
| Zero-shot, 3 seeds | 0.146 [0.066, 0.237] | 0.086 | 85% |
| **+ desk fine-tuned (deployed recipe)** | **0.058 [0.029, 0.091]** | **0.052** | **94%** |
| + self-training on held-out phrases | 0.066 | 0.050 | 93% |
| + noisy student (session 181947) | 0.073 | 0.050 | 93% |
| syn25 (synthetic personal vocab) instead of hwtmix | 0.073 | 0.046 | 93% |

| Blind-1 (not truly blind), frozen v3 decoder, no model trained on blind-1 | WER | CER | Words |
|---|---|---|---|
| decoder agent's CTC (v2) | 0.210 | 0.073 | 79% |
| **v3: zero-shot + 20-phrase finals** | **0.161 [0.088, 0.238]** | 0.084 | **84%** |
| syn25 20-phrase finals | 0.194 | 0.091 | 84% |

- Fine-tuned vs zero-shot under v3: dWER −0.088 [−0.153, −0.030] — the one gain that survives the strong decoder.
- TTA, self-training, noisy student, desk synthetic motion, syn25 helped at most with the 0.5B decoder; none beat plain fine-tuning under v3. TTA hurts continuous decoding.
- Learning curve (0.5B decoder, continuous WER) at 0/4/8/12/16 desk phrases: 0.226/0.212/0.168/0.117/0.117 — flattens at ~12 same-session phrases.
- Keyboard LOSO char CER hwtmix 0.291 vs syn25 0.271 (syn25's keyboard gain doesn't transfer to desk).
- Caveats: one user; blind-1 truth seen during development (62 words, ±3 words ≈ ±0.05 WER).

Files: `.cache/ctc_v3/README.json`, `deploy/manifest.json`, `predict.py`, `selftrain.py`, `posteriors/v3_deploy/`, `results/`; code `phase0/analysis/ctcv3.py`, `ctcv3_eval.py`, `decipher2_ctc3.py`.
