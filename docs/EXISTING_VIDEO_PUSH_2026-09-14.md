# Existing-video decoder experiments — 14 September 2026

This round is complete. No new recording, production decoder change, training-label modification, SSH operation, or commit. **No near-100% result was achieved.**

## Measured outcome

The existing development baseline makes 25 word edits over 250 reference words (10.0% WER). Five new selection pipelines did not improve on it. Baseline fallback is included before alternatives, so ties retain the baseline. Each recording's settings are selected using the other two recordings' total word errors, not its own score.

| Method | Old (137 words) | B1 (62 words) | B2 (51 words) | Total |
|---|---:|---:|---:|---:|
| Existing baseline | 8 | 8 | 9 | 25/250 (10.0% WER) |
| Consensus / minimum-risk decoding | 8 | 8 | 9 | 25/250 (10.0% WER) |
| Raw-character 8B repairs | 8 | 8 | 9 | 25/250 (10.0% WER) |
| 8B word-and-gap repairs | 8 | 8 | 9 | 25/250 (10.0% WER) |
| Two-sided vocabulary search | 8 | 8 | 9 | 25/250 (10.0% WER) |
| Vocabulary candidates + 8B scoring | 10 | 8 | 9 | 27/250 (10.8% WER) |

These are **development measurements**, not an untouched test. The underlying model/decoder and this investigation have historical access to reference text. Splitting hyperparameter selection by recording does not undo that exposure. The legacy scorer ignores case, punctuation, digits and some spelling distinctions. Also, 1−WER is not necessarily the fraction of reference words matched: WER includes insertions. Do not use this table to claim literal typing accuracy or expected accuracy on a new recording.

## Six approaches actually tested

1. **Minimum-risk and word consensus decoding.** Softmax temperatures 0.5–16 over the top 100 eligible candidates; select the lowest expected word-edit-loss candidate or align and vote over word slots. Every recording selects the original MAP decoder. Inspired by [consensus/lattice decoding research](https://www.microsoft.com/en-us/research/publication/an-improved-consensus-like-method-for-minimum-bayes-risk-decoding-and-lattice-combination/); this is a small local prototype, not a reproduction of the paper.
2. **Raw-character 8B repair.** Two generic prompts ask Qwen3-8B to recover material discarded by polished predictions or use other predictions within the recording. Both character groups are supplied, never reference text. Parsed string and transcription-object responses are supported. No selected gain.
3. **Repeated-evidence fusion.** Detect similar long predicted phrases within one recording, align their camera probability sequences using dynamic time warping with Hellinger costs, and average aligned evidence. One repeated pair was found in B1. The small beam changes from `really want this to sort out and if to eas i shall feat` to `i really want this to work out and if it was i all feat`. That is still below the full existing decoder; no deployment. Similar predictions alone also do not prove two utterances are identical.
4. **8B word-and-gap search.** Propose three single-word completions at every word replacement and insertion gap: all 518 slots completed. Incomplete JSON output is retried in six-slot batches using the same prompt. Verify candidates with exact CTC likelihood and LM scores. Evaluate single repairs and combinations. Combination scores use additive single-edit gains, not an exact joint likelihood.
5. **Full-vocabulary two-sided search.** Rank the personal bigram vocabulary using both neighboring words, generate 20 proposals per slot plus deletion, and verify with CTC. The diagnostic best single-edit candidates reduce the per-segment reference error sum from 25 to 17. **17 is an oracle diagnostic, not achieved recognition:** the actual selection cannot choose those candidates reliably. It also does not measure the ceiling of arbitrary multi-edit search.
6. **8B rescoring of vocabulary candidates.** Retain the baseline plus the top 24 camera-ranked and top 24 bigram-ranked candidates without looking at references. Score with local Qwen3-8B and select camera/LM weights on the other recordings. This produces 27 errors, worse than the baseline's 25.

## What this establishes

Reranking the original eligible pool has very little headroom: its per-segment diagnostic oracle is 24 errors versus 25 actual errors. Expanding the vocabulary finds some better text, but the camera/LM combination still favors incorrect alternatives. Generic grammatical correction can delete supported words or insert plausible words such as `can i i`; grammatical fluency is not evidence of what was typed.

The experiments do not establish an absolute limit of the videos or disprove larger visual models. They establish that these six tested decoder strategies do not deliver the requested improvement. The existing tiny-pixel controls from the previous foundation study also failed to demonstrate added motion information. A larger, temporally aligned visual model remains an untested direction, not a measured breakthrough. Training until known transcripts are reproduced would measure memorization rather than recognition.

## Artifacts and reproduction

Results: `results/nextgen/existing_video_push.json`, `consensus/report.json`, `span_repair.json`, `repeat_consensus.json`, `cloze_probe.json`, `lexical_cloze.json`, `lexical_rescore.json`. Cached raw LLM outputs, candidates, scores and prompts are under `.cache/nextgen/` in the correspondingly named directories.

Implementation: `phase0/analysis/consensus_probe.py`, `span_repair.py`, `repeat_consensus.py`, `cloze_probe.py`, `lexical_cloze.py`, `lexical_rescore.py`.

From the repository root:

```sh
.venv/bin/python -m phase0.analysis.cloze_probe eval
.venv/bin/python -m phase0.analysis.lexical_cloze
.venv/bin/python -m phase0.analysis.lexical_rescore eval
.venv/bin/python -m pytest phase0/tests/test_decoder_probes.py phase0/tests/test_nextgen.py phase0/tests/test_nextgen_workflows.py -q
```

Generation uses the cached `mlx-community/Qwen3-8B-4bit` via `.cache/personal_llm/venv/bin/python`, offline, one job under `phase0.tools.gpulock`, with a 7 GB MLX cap and 0.5 GB cache. Generation commands are `cloze_probe generate` and `lexical_rescore generate`. Candidate/scoring files are reusable caches; evaluation overwrites its own reports.

**Validation:** 26 focused tests passed, including scalar/vectorized bigram agreement, minimum-risk selection, word-slot consensus, identity posterior fusion, response parsing and repair composition. All six experiment modules compile. No experimental path was promoted.
