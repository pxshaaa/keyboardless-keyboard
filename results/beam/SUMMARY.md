# Is the 8B generation stage redundant? — oracle study of the CTC word beam

`results/beam/` · 2026-09-20/21 · code `phase0/analysis/beam_oracle.py`, `beam_sweep_par.py`, `beam_e2e.py`
Operational handoff (what is running, exact next commands): **`results/beam/HANDOFF.md`**.

---

## Verdict

**The word beam alone cannot replace the 8B. But the 8B *generation* stage adds essentially nothing to the
candidate pool once char n-best + fuzzy + a wide word beam are all in it.**

* A lexicon-constrained CTC word beam, tuned and widened to beam 2048 / 500-best, has an *oracle* ceiling of
  **0.898 / 0.871 / 0.765 / 0.860** words (old / blind-1 / blind-2 / live). That is below the v4 union's
  oracle on all four sets, and below what v4's selector **already picks** on blind-1, blind-2 and live. A
  perfect rescorer over the beam's N-best would score *worse than v4 does today*. Answer to the literal
  hypothesis: **no**.
* Replace "8B rewrites" with "word beam top-50" inside the existing pool and the ceiling is unchanged on three
  of four sets: **char n-best + fuzzy + beam = 0.934 / 0.903 / 0.843 / 0.895** vs the real v4 union's
  **0.942 / 0.903 / 0.843 / 0.895**. The 139 s stage buys **0.8 pt of oracle ceiling on old desk and 0 pt on
  blind-1, blind-2 and live.**
* So the 139 s is removable — not by deleting generation in favour of the beam, but by deleting the *8B* and
  keeping the cheap generators (char n-best ~1.3 s, fuzzy ~9 s, beam 0.05–4 s per phrase, all CPU) that
  already cover what it found.
* The beam earns its place either way: it is the only source for personal-lexicon phrases the 8B never
  produces (blind-1 "claude code or codex which one is the best", beam rank 4, in no 8B list). The union beats
  both sources individually, which is why v4 pools them.

---

## Oracle table

Per-segment truth; oracle = best candidate in the list chosen *with* knowledge of the truth.
Cells are **words · CER · exact-in-list · median rank of truth · mean #candidates**.
Word beam = `wordbeam_v4.beam` at the tuned config below.

| candidate source | old desk (21 seg, 137 w) | blind-1 (6, 62) | blind-2 (5, 51) | live (6, 57) |
|---|---|---|---|---|
| **(a)** word beam top-50 | 0.876 · .089 · .48 · 10 · 50 | 0.839 · .078 · .17 · 1 · 50 | 0.745 · .166 · .00 · – · 50 | 0.860 · .132 · .33 · 27 · 50 |
| **(a)** word beam top-100 | 0.891 · .084 · .48 · 10 · 100 | 0.855 · .078 · .17 · 1 · 100 | 0.745 · .162 · .00 · – · 100 | 0.860 · .132 · .33 · 27 · 100 |
| **(a)** word beam top-500 | 0.898 · .074 · .48 · 10 · 496 | 0.871 · .056 · .17 · 1 · 500 | 0.765 · .143 · .00 · – · 500 | 0.860 · .120 · .33 · 27 · 500 |
| **(b)** 8B rewrites, frozen prompt p7 | 0.898 · .053 · .48 · 1 · 6 | 0.806 · .104 · .17 · 2 · 4 | 0.804 · .154 · .20 · 1 · 6 | 0.842 · .124 · .17 · 1 · 6 |
| **(b)** 8B rewrites, all prompts | 0.920 · .046 · .62 · 1 · 11 | 0.823 · .086 · .17 · 1 · 10 | 0.824 · .131 · .40 · 1 · 9 | 0.842 · .120 · .17 · 1 · 11 |
| char n-best + greedy + 0.5B only | 0.854 · .067 · .29 · 18 · 36 | 0.839 · .071 · .17 · 18 · 36 | 0.784 · .104 · .20 · 4 · 36 | 0.789 · .140 · .17 · 2 · 36 |
| char n-best + beam-500 (no fuzzy, no LLM) | 0.905 · .049 · .48 · 18 · 527 | 0.903 · .045 · .17 · 18 · 533 | 0.804 · .104 · .20 · 4 · 535 | 0.860 · .096 · .33 · 12 · 516 |
| **char n-best + fuzzy + beam-50 (NO LLM gen)** | **0.934** · .041 · .62 · 36 · 117 | **0.903** · .048 · .17 · 18 · 121 | **0.843** · .093 · .40 · 26 · 121 | **0.895** · .092 · .33 · 14 · 104 |
| char n-best + fuzzy + beam-500 (no LLM gen) | 0.934 · .040 · .62 · 36 · 560 | 0.903 · .045 · .17 · 18 · 570 | 0.843 · .093 · .40 · 26 · 570 | 0.895 · .084 · .33 · 14 · 551 |
| v3 pool minus the 8B (keeps 0.5B) | 0.891 · .055 · .43 · 18 · 71 | 0.887 · .059 · .17 · 18 · 74 | 0.824 · .093 · .40 · 26 · 74 | 0.860 · .096 · .33 · 14 · 56 |
| v3 pool (char + greedy + 0.5B + 8B + fuzzy) | 0.934 · .041 · .62 · 31 · 74 | 0.887 · .059 · .17 · 18 · 78 | 0.843 · .093 · .40 · 20 · 79 | 0.877 · .096 · .33 · 14 · 61 |
| **(c) v4 union — what v4 actually uses** | **0.942** · .037 · .67 · 32 · 121 | **0.903** · .048 · .17 · 18 · 125 | **0.843** · .093 · .40 · 20 · 126 | **0.895** · .092 · .33 · 14 · 109 |
| — *what the CURRENT selector picks* | *0.898* · .084 · .43 · 1 · 1 | *0.887* · .071 · .17 · 1 · 1 | *0.784* · .139 · .20 · 1 · 1 | *0.877* · .088 · .33 · 1 · 1 |
| — word beam top-1 (no rescoring) | 0.701 · .221 | 0.677 · .156 | 0.647 · .220 | 0.737 · .200 |

Reading it:
* **Selection, not generation, is the binding constraint on three of four sets.** The union contains enough
  for 0.942 / 0.903 / 0.843 / 0.895; the selector delivers 0.898 / 0.887 / 0.784 / 0.877. The largest single
  gap is blind-2 (0.784 picked vs 0.843 available, 6 pt).
* The truth is *literally* in the pool for only 0.67 / 0.17 / 0.40 / 0.33 of segments, at median rank ~20–36.
  Most of the oracle gain is partial-credit words, not whole correct sentences.
* The 8B's rewrites are a short, high-precision list (4–11 candidates, truth at median rank 1 when present).
  The beam is a long, low-precision list (50–500, truth at median rank 10–27). They fail on different
  segments, which is the whole argument for the union.

### What the 8B does that the beam cannot
The 8B reconstructs words whose CTC evidence is **absent**, using sentence-level semantics. The beam is
anchored to the posterior: each extra character costs CTC log-prob and a word bigram cannot outvote it.

| truth (old desk) | beam top-1 | 8B rewrite |
|---|---|---|
| the meeting **moved** to tuesday afternoon | the meeting move to the after on | the meeting moved to tuesday afternoon ✅ |
| thanks for **getting** back to me | the tint back to | thank you for getting back to |
| it took longer than i **expected** | it look longer than i e | it took longer than i thought |
| i have **added** a few notes for you | i have a a re it for you | i have added a free note for you |

And the reverse — where the beam is the only source:

| truth (blind-1) | beam | 8B |
|---|---|---|
| claude code or codex which one is the best | rank 4 ✅ | absent from every prompt |

Note that the *fuzzy* generator recovers most of what the 8B finds at a fraction of the cost, which is why
"char n-best + fuzzy + beam" ties the union: fuzzy edits words toward the personal lexicon, the beam supplies
lexicon-legal reparses, and between them the 8B's unique contributions collapse to one old-desk segment.

### Beam hyperparameters (tuned on OLD DESK only)
Beam **width**, not N, is the binding constraint: at the frozen v4 setting (beam 128) the beam emits only
~80 distinct finals, so N above ~100 does nothing.

| beam | old@100 | old@500 | b1@500 | b2@500 | live@500 | s/segment (1 CPU core) |
|---|---|---|---|---|---|---|
| **128 (v4 frozen)** | 0.803 | 0.803 | 0.855 | 0.745 | 0.825 | 0.05–0.08 |
| 256 | 0.839 | 0.839 | 0.871 | 0.765 | 0.842 | 0.10–0.19 |
| 512 | 0.854 | 0.854 | 0.887 | 0.784 | 0.842 | 0.24–0.43 |
| 1024 | 0.861 | 0.869 | 0.903 | 0.784 | 0.842 | 0.63–1.18 |
| 2048 | 0.869 | 0.876 | 0.903 | 0.784 | 0.842 | 1.4–2.9 |

Chosen by staged coordinate search on old desk, `oracle_words @N=100` (`results/beam/chosen.json`):
`alpha 1.5, beta 2.0, gamma −8.0, boost 0.0, beam 2048, prune −12.0`, lexicon **generic**.
Two surprises worth a follow-up: the *generic* lexicon beat the merged generic+personal one, and the personal
name boost was best at **0.0**. Both are the opposite of the frozen v4 choice and were selected on 21 dev
segments, so treat them as weak signals.

**Cost warning.** `docs/JEV_ASSESSMENT_2026-09-20.md` assumes a "~10 ms" beam search. This one is pure Python
over a ~300k-word trie with a bigram LM: 0.05–0.08 s/segment at beam 128 and **4–7 s/segment** at beam 2048 on
the mini. The useful operating point (beam 512, N=50) is ~0.25–0.45 s/segment. A compiled implementation is a
prerequisite for the latency claim, and it has not been written.

---

## Reference points (live set, measured here)

| decoder | words | WER | wall clock / phrase |
|---|---|---|---|
| v4, 8B generate + 8B/0.5B score | **87.7 %** (88.5 % on the 5 high-confidence phrases) | 0.123 | **27 s** (216 s / 8 phrases) |
| live in-session 0.5B word beam (`seqctc2.word_beam` + Qwen-0.5B) | 73.7 % (75.0 % high-conf) | 0.281 | 3.3 s (MacBook CPU) |
| word beam top-1, tuned, no rescoring | 73.7 % | 0.200 | 6.9 s at beam 2048 |

v4 stage split (`data/sessions/zz-live-20260915-010345/decipher2_v4.json` → `timing_s`, 8 phrases):
candidates 1.3 s · **8B generation 139.3 s** · fuzzy 9.0 s · word beam 1.2 s · **LM scoring 62.7 s** ·
select+suggest 2.5 s.

The brief's "~67 % @ ~1 s" for the live 0.5B beam is 73.7 % @ 3.3 s against this truth set and on this machine.

---

## End-to-end (beam N-best + small local rescorer) — NOT RUN

`phase0/analysis/beam_e2e.py` is written and staged (`nbest` → `lmscore` under `gpulock` → `eval`), reusing
`llmdec_mlx.lm_scores`, which is already batched teacher-forced prefill, not generation. It was not executed:
the MacBook session ended first. Exact commands are in `HANDOFF.md` §4 step 3. Available MLX models on the
mini are only `personal_llm/base` (0.5B, ± LoRA), `Qwen2.5-3B-Instruct-4bit` and `Qwen3-8B-4bit` —
`HF_HUB_OFFLINE=1`, so there is no 1.5B.

What the oracle already bounds: a rescorer over **beam-only** N-best cannot exceed 0.898 / 0.871 / 0.765 /
0.860; over the **no-LLM pool** it cannot exceed 0.934 / 0.903 / 0.843 / 0.895 — i.e. the same ceiling v4 has.
The open question is purely how much of that ceiling a 0.5B/3B rescorer realises, and at what wall clock.

---

## Protocol

**Desk sets** — the ctc_v4 **leave-one-session-out** posteriors used by `.cache/ctc_v4/results/loso_v4.json`
(`s_zz-v4loso-<sid>`), so no phrase is scored with a model trained on it. Posterior group `ens` (zs + desk),
matching frozen v4 `run["groups"]`. Per-segment truth via `swipe_common.assign_truth` (word-align concatenated
truth against the frozen v3 recommended output, then refine each boundary ±3 words by CTC log-lik) — the same
routine `decipher2_v4.score` uses.

**Live set** — `data/live/20260915-010345`, llmdec set `s_zz-live-20260915-010345`, i.e. the same posteriors
the v4 live run used.

**Pool reconstruction** — `beam_oracle.pool_texts` rebuilds the v4 pool exactly as `llmdec.build_pools` does:
char n-best + greedy + 0.5B per group, all 8B and personal-0.5B rewrites for every prompt, fuzzy candidates;
the v4 union adds the word beam top-50.

**Tuning discipline** — the beam hyperparameter search selects on **old desk only**, `oracle_words @N=100`
(`chosen.json` records this). blind-1, blind-2 and live are carried through every table but never selected on.
Nothing was tuned on blind-2 or the live phrases.

### Live ground truth (reconstructed; `data/live/20260915-010345/truth.jsonl`, copy in `live_truth.jsonl`)

`truth.jsonl` did not exist. Rebuilt from `results/live1/whisper_medium/audio.json` plus the coordinator notes
in `results/live1/PROGRESS_NOTES.md`; the spoken phrase precedes each typed phrase.

| k | truth | confidence | source |
|---|---|---|---|
| 0 | — | **unusable** | no utterance; typed raw `ooo`. Excluded. |
| 1 | — | **unusable** | no utterance; typed raw `rrtha nour`. Excluded. |
| 2 | i will try this out now to talk as well | high | whisper 170.6–177.4 s + coordinator |
| 3 | i will write to you what i want to write to you | high | whisper 376.3–385.2 s + coordinator |
| 4 | if i talk slowly like i am doing right now i hope that this will still work | high | whisper 398.8–411.5 s. Coordinator note has an extra "it" ("doing **it** right now"); the typed raw shows no evidence of it, so whisper's wording is used. |
| 5 | okay all campaigns are paused right now | medium | coordinator only — whisper hallucinated this window |
| 6 | did you fix it now | **low** | coordinator note, itself parenthesised/uncertain; no whisper support |
| 7 | actually this is not working out | high | whisper 665.0–671.1 s + coordinator |

All live numbers above use k=2..7 (6 phrases, 57 words). The high-confidence subset k=2,3,4,5,7 (52 words)
moves v4 from 87.7 % to 88.5 % and the 0.5B beam from 73.7 % to 75.0 %; no conclusion depends on which is used.

⚠️ **Whisper medium hallucinated heavily** on this audio, repeating "Okay, I will try this out now to talk as
well." across ~15 unrelated windows. Only the windows above are real. Do not re-derive truth from that
transcript without checking each window against `events.jsonl`.
⚠️ In the live set `zs` and `desk` are the **same array** (no zero-shot ensemble ran live).

## Caveats

1. **Not blind.** All three desk sets are v4 dev sets (`results/llmdec/frozen_config_v4.json` → `disclosure`);
   the name-boost seed `{claude, bifi, codex}` came from blind-1/2 errors. The beam config here was tuned on
   old desk. A sealed blind-3 is still required for an unbiased number.
2. **Small n.** 37 segments, 307 truth words in total; blind-1, blind-2 and live are 5–6 segments each. A
   single segment moves any of those columns by 1–2 pt. No confidence intervals are reported because at this
   n they would swamp every difference discussed.
3. **Live truth is partly reconstructed** (k=5 coordinator-only, k=6 weak) — see above.
4. The v4loso sets have no `__v4` fuzzy tag and their 8B/0.5B LM scores cover only the v3 pool, so the real v4
   selector cannot be run over the beam-extended pool on the desk sets without new MLX scoring. `SELECTOR
   current` is the frozen **v3** recommended output for the desk sets and the real **v4** output for live.
5. Oracle numbers are per-segment (each segment's best candidate chosen independently), not the official
   concatenated `score_variant` metric. They are a ceiling, not a decoder result.

## Files

`sweep.json` (full hyperparameter sweep) · `chosen.json` · `oracle.json` (tuned config) ·
`oracle_beam128_default.json` (frozen v4 beam config, i.e. what v4 ships) · `live_truth.jsonl` ·
`sweep_par.log`, `oracle.log` · `HANDOFF.md`.
Code: `phase0/analysis/beam_oracle.py`, `beam_sweep_par.py`, `beam_e2e.py`.
