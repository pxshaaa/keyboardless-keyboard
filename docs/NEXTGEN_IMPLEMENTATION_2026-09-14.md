# Implementation and measured results — 14 September 2026

The first implementation round is complete. Calibration and text-contract fixes are implemented. Context spotting and RGB fusion were built and evaluated locally, but neither has enough evidence to replace the frozen pipeline. No remote connections or new recordings were used.

**Changes that are ready**

- `text_contract.py` provides NFC-preserving literal CER/WER, explicit supported symbols, strict encoding and head expansion. It preserves digits, German letters, case and punctuation in the literal metric. Unsupported characters raise errors instead of disappearing.
- `calibration_review.py` creates `calibration_review.json` for each calibration recording, bound to its prompt file by SHA-256. Every entry starts pending. Only explicitly confirmed labels can enter training; rejected/pending labels remain in the review record. Confirmed, supported labels with poor model agreement are retained and flagged. Missing/cut alignment windows still require attention.
- `calib_ingest.py` uses that review, retains the original text, refuses unsupported legacy labels, and starts no training when no usable labels are confirmed. It no longer claims model agreement establishes correct typing. Existing inference/decoder files are unchanged.
- `literal_checkpoint.py` produced `.cache/nextgen/literal/init_s0.pt`: 104 symbols, preserving all 29 legacy head rows. It is explicitly marked **not deployable / new symbols not trained**. This is initialization for future literal-label training, not a demonstrated multilingual upgrade. The historical keyboard recorder also lowercases and filters characters, so missing historical labels cannot be reconstructed by enlarging the head.
- `calibration_plan.py` selected 80 prompts into four 20-phrase blocks, using diminishing-return character-bigram coverage. Each block is about 4 minutes 40 seconds at 14 seconds per prompt, excluding setup and review. Three unsupported prompts were excluded from this legacy-compatible plan. Collect the blocks as separate sessions, with realistic remounts between blocks. Its advantage over the original prompt list has not yet been measured.

**Context spotting: no measured improvement**

The new `context_spotter.py` searches CTC probabilities for local terms independently of the sentence beam. It handles repeated-letter blank transitions and proposes replacements against aligned word spans. The pilot used 64 existing personal terms selected by frequency, excluding keyboard-source and other-author Notion counts; it did not construct a term list from the reference phrases. The personal corpus itself is historical development context, so this is not a blind test.

Across the three development sessions it proposed 106 additional candidates in approximately 4 seconds. The unchanged v4 outputs have 25 word errors in 250 words. CTC-only replacement with margins 0 or 2 produced 26 errors; margins 5 or 10 retained 25 errors. The candidate-pool oracle also retained 25 errors: the new candidates do not repair additional words in this pilot.

No production integration is enabled. This result applies to the frequency-selected context list, local replacement rule and tested thresholds. It does not evaluate live application context, missing-prefix insertion, or an LLM verification stage.

**Visual fusion: a small appearance signal, no established motion gain**

`visual_fusion.py` extracts continuous masked hand crops from the original recordings and embeds them with a frozen, locally cached ImageNet ResNet18. It combines those features with landmarks through a small residual temporal CTC head. A matched control receives landmarks and zeroed image features. The pilot holds out each whole session, trains on the other two, and repeats with three seeds: 18 residual-head training runs in total.

All normalization for the residual head is fitted on the training sessions. Video frame IDs are checked against the recording, and image caches are bound to source-file hashes. Existing zero-shot CTC posteriors supply the common base. The experiment uses known phrase windows and greedy decoding, not the full deployed decoder or automatic segmentation.

| Held-out session | Geometry residual CER | Geometry + RGB CER | RGB shuffled in time |
|---|---:|---:|---:|
| Old desk | 33.83% | 32.69% | 32.20% |
| First free-phrase session | 28.13% | 26.27% | 26.27% |
| Second free-phrase session | 33.46% | 32.05% | 32.56% |
| Pooled | 32.48% | 31.12% | 30.95% |

The unchanged visual baseline is **31.56% CER**. Thus RGB beats the geometry residual control by 1.36 percentage points, but beats the unchanged baseline by only 0.44 points. The descriptive paired phrase-bootstrap interval for RGB minus geometry is approximately **[−2.77, +0.03] points**. With only three sessions, this does not establish generalization to future recordings.

Shuffling the RGB features in time barely affects the result. The experiment does not demonstrate additional temporal typing evidence; static appearance/setup information or regularization may explain it. Keep the current deployed model. The next visual experiment should use a typing-specific temporal crop encoder and explicit motion controls, rather than treating a larger generic image encoder as the answer.

**Verification**

The focused tests cover literal character preservation, strict rejection, checkpoint row preservation, review hashes/statuses, hard confirmed labels passing ingestion, repeated-letter CTC transitions, and exhaustive-search checks of the spotter and oracle. Source files compile. The pre-existing frozen decoder and core model files are checked against the hashes recorded before this implementation round.

**Commands** — run from `/Users/pashaalidadi/Documents/misc/computer-vision-test`:

```sh
.venv/bin/python -m pytest phase0/tests/test_nextgen.py -q
.venv/bin/python -m phase0.analysis.context_spotter_eval
.venv/bin/python -m phase0.analysis.nextgen_report
.venv/bin/python -m phase0.analysis.calibration_plan
.venv/bin/python -m phase0.analysis.text_contract truth.txt predictions.txt --out literal_score.json
```

Reproduce one visual seed with the shared GPU lock:

```sh
.venv/bin/python -m phase0.tools.gpulock --wait 1 -- .venv/bin/python -m phase0.analysis.visual_fusion --seed 0 --out results/nextgen/visual_fusion_s0.json
```

For a future calibration recording, use one of `.cache/nextgen/calibration/part1.txt` through `part4.txt` with the existing prompter at 14 seconds per phrase. Run ingestion with `--no-train` first. Review `calibration_review.json`: set confirmed only when the text was actually typed, correct its text only when known, and reject incomplete/mistyped examples. Rerun ingestion to train a new version. A run with no confirmed usable labels exits 2 after saving diagnostics.

**Artifacts and remaining work**

- `results/nextgen/summary.json`: consolidated metrics and limitations.
- `results/nextgen/visual_fusion_s{0,1,2}.json`: per-phrase counts and seed results.
- `results/nextgen/context_spotter.json`: generation/selection ablation.
- `.cache/nextgen/`: private candidate texts, feature banks, experimental checkpoints and calibration packs.
- `results/nextgen/baseline_hashes.json`: code/frozen-config hashes captured before changes.

Fresh reviewed literal labels are still needed to train and validate new character classes. Live task-context retrieval, a typing-specific pixel encoder, uncertainty-driven reinspection and streaming distillation remain follow-up experiments, not completed features. No new accuracy percentage is promised for the next recording.
