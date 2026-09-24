# Jev (TypeSafe AI) — verification and fit assessment for the bare-desk typing decoder

Date: 2026-09-20
Scope: verify the MarkTechPost write-up against TypeSafe's own primary sources, then assess fit
for the CTC → candidate → decoder pipeline.

Every claim below is tagged:
- **[P]** verified from a TypeSafe primary source (typesafe.ai, docs.typesafe.ai, their legal pages)
- **[S]** secondary source only (press, third-party blogs, third-party benchmarks)
- **[?]** could not determine

---

## 0. Corrections to the MarkTechPost article

| MarkTechPost claim | Reality | Tag |
|---|---|---|
| "available behind an early-access waitlist" | Half right. TypeSafe's **own** API (`api.typesafe.ai`) is early access with a waitlist **[P]**. But Jev is served **today with no TypeSafe waitlist** through Vercel AI Gateway (announced 2026-09-16) and Cloudflare Workers AI **[S/P-third-party]**. So there *is* a public way to call it today. | mixed |
| "not open source" | Correct. No weights, no paper, no self-host, no on-device variant, no announced timeline **[P: absent from all docs; S: multiple]**. | ✅ |
| "returns typed, calibrated decisions rather than free text" | Correct, and stronger than stated: TypeSafe's own jaggedness page lists **"text generation — not designed for generative tasks"** as a documented weakness **[P]**. It is structurally incapable of stage (a). | ✅ |
| "fast, cheap" | Correct on price. Latency is 70–500 ms *model-side*, not the 0.114 s marketing number **[P]**. | ✅ |
| (omitted) context window | 64k tokens per request total; 32k for state + longest question **[P]** | added |
| (omitted) calibration evidence | TypeSafe publishes **no ECE or reliability numbers at all**. The only ECE figures in existence are third-party, and they are unflattering out-of-distribution. | added |

---

## 1. Input / output, option sets, ranking, calibration

**Input** **[P]** — `POST https://api.typesafe.ai/v1/systemone`, Bearer auth. Body:
`state` (string, JSON object, or array of strings), `model` (`"jev-latest"`), `questions` (a map of
question objects). **Text only** — no image/audio; non-text must be preprocessed.

**Output** **[P]** — three primitives, all caller-constrained:

| Primitive | Returns | Limits |
|---|---|---|
| **Choice** | `choice` (argmax option), `probabilities` (full distribution over your options, sums to 1.0), `confidence` (0–1) | **max 255 options** |
| **Score** | probability-weighted value over an ordered rubric + `confidence` | **2–10 ordered levels** |
| **Noul** | a single float 0–1 = P(yes). No separate confidence — the value *is* the answer and the certainty | — |

Response also carries `usage.{input_tokens, output_tokens}`.

**Can it generate free text?** **No** **[P].** There is no generative primitive. Output is always
drawn from a set you supply. TypeSafe's `model-jaggedness/jev-1.13` page explicitly lists text
generation as out of scope.

**Can it rank/score a list of candidate strings?** **Yes, two ways** **[P]**:
1. One **Choice** question with the candidates as options (≤255) → one call, full posterior over
   candidates. This is the efficient shape for us.
2. TypeSafe's own re-ranking cookbook (`cookbooks/rerank_typesafe.md`) uses **one Noul per
   candidate** — 30 candidates/query, 1,200 pairs for $0.0645, run 12-way concurrent. Reported
   top-1 5%→18%, top-10 38%→62% on a legal-citation retrieval set **[P]**. Note that is a
   *relevance* re-rank, not a language-model rescore.

**Calibrated probabilities?** Yes, that is the product claim — trained with "RLCD"
(Reinforcement Learning for Calibrated Decisions) **[P]**.

**Is there published evidence the calibration is good?**
- **From TypeSafe: no.** The blog and docs assert "higher confidence means higher accuracy" and
  give a three-tier thresholding pattern, but publish **zero** ECE, Brier, or reliability-diagram
  numbers **[P — verified absent]**.
- **From third parties: mixed, and bad where it matters to us.** The most rigorous is
  `github.com/scienthoon/jev-ood-calibration` **[S]**:

  | Set | Accuracy | ECE / temperature |
  |---|---|---|
  | OpenBookQA | 94.2% | ECE 0.024 |
  | CommonsenseQA | 88.1% | ECE 0.032 |
  | HellaSwag | 86.1% | ECE 0.029 |
  | **Synthetic OOD tickets (900)** | — | **ECE 0.107 vs a 0.024 noise floor — 4.4×** |
  | ↳ queue (Choice) | 89.0% | refit T = **3.29** (overconfident) |
  | ↳ anger (Noul) | 91.7% | refit T = 0.66 (underconfident) |
  | ↳ priority (Score) | 44.7% | refit T = **3.40**, mean 0.74 prob on an unknowable label |

  Conclusion of that study: calibration holds on public benchmarks (likely in-distribution / in
  training data) and **degrades to roughly fine-tuned-classifier-out-of-distribution levels on a
  novel task**, with the sign of the error differing per primitive. Their explicit advice: don't
  rely on the `confidence` field; recalibrate locally per question type.
- A second third-party benchmark (`github.com/themsquared/jev-benchmark`, agent tool-call risk,
  60 cases) found 91.7% accuracy and that **no wrong answer ever came back at confidence 1.000** —
  i.e. confidence was usable for escalation routing on *that* task **[S]**.

**Implication for us:** our task (picking among garbled CTC hypotheses) is about as
out-of-distribution as it gets. Expect Choice overconfidence in the T≈3 range and plan to fit our
own temperature on held-out recordings before thresholding a suggestion bar.

---

## 2. Latency

- **Published, model-side:** 70–500 ms end-to-end **[P]** (TypeSafe blog). Homepage demo figure
  0.114 s for one workflow **[P]** — marketing best case, not a p50.
- **No published p50/p99 from TypeSafe** **[P — verified absent]**.
- **Independent, measured** **[S]** (`themsquared/jev-benchmark`, residential Portland OR,
  2026-09-17): **p50 421.6 ms, p95 542.0 ms** (`jev-latest`); p50 378.5 / p95 484.3 (`jev-preview`).
- **Streaming:** not documented anywhere; there is nothing to stream (the answer is a fixed-size
  object) **[P — absent]**.
- **Batch:** no async batch endpoint. Instead, **many questions in one request over one state**,
  evaluated in parallel — TypeSafe's own measurement: 13 questions batched = **0.27 s** vs 2.71 s
  sequential, and 12.2× cheaper because the state is sent once **[P]**.
- **Rate limits:** 250,000 tokens/sec, 1,200 requests/min **[P]**.

**Realistic round-trip from Germany:** **[?/estimate]** TypeSafe does not publish an EU endpoint or
any PoP list, and the privacy policy says the service is **hosted in the United States** **[P]**.
Frankfurt→US-East adds roughly 90–110 ms RTT, US-West 150–170 ms, on a warm TLS connection. So
budget **p50 ≈ 0.5–0.6 s, p95 ≈ 0.7–0.8 s** per call from Germany. Cold TLS handshake adds another
~200–300 ms; keep a persistent connection pool. This is not verified — it must be measured from
our own box before anyone designs against it.

---

## 3. Price and free tier

- **$0.042 per 1M input tokens** ($42/billion). **Output tokens free** ("too cheap to meter")
  **[P]**.
- **Free tier: none from TypeSafe** **[P — verified absent from pricing/docs]**.
- Vercel AI Gateway ran Jev **free until 2026-09-25** **[S]** — a launch promotion, already
  expiring; do not plan on it.

---

## 4. Access and — critically — data handling

**Access** **[P]**
- Direct: waitlist at typesafe.ai, keys from `console.typesafe.ai`. Early access, batched invites.
- Today, without a TypeSafe waitlist: **Vercel AI Gateway** (2026-09-16) and **Cloudflare
  Workers AI**; OpenRouter also listed **[S]**.
- Official open-source **SDKs** (Python `typesafe-sdk`, JS `@typesafe-ai/sdk`) — the SDKs are open,
  the model is not **[P]**.

**Data handling — this is the section that matters for us.**

| Question | Answer | Tag |
|---|---|---|
| Do they train on submitted data? | **No.** Privacy policy, verbatim: *"We will not train or fine tune any artificial intelligence or machine learning models on your prompts or other Input."* Docs `models.md` repeats it. | **[P]** |
| Zero data retention? | **Enterprise only, on request**, via `privacy@typesafe.ai`. Not self-serve, not default, not in the standard terms. | **[P]** |
| Default retention window? | **Unstated.** Privacy policy says only *"as long as reasonably necessary to provide you with the Services."* DPA says *"as long as necessary taking into account the purpose of the Processing."* **No number anywhere.** No published abuse-monitoring window. | **[P — verified absent]** |
| EU / GDPR hosting? | **No EU region.** Privacy policy: the service is *"hosted in the United States"*; personal data is transferred to and processed in the U.S. | **[P]** |
| Legal transfer mechanism? | DPA uses **EU SCCs Module 2** (controller→processor), **Irish law, Dublin courts**, plus the **UK Addendum v B1.0 (March 2022)**. | **[P]** |
| Sub-processors | Registry at `trust.typesafe.ai/subprocessors`, 15-day advance notice before adding one. **The page did not render for me — contents unverified.** | **[P** for the mechanism / **[?]** for the list |
| Security measures | DPA commits only to *"reasonable and appropriate technical and organizational security measures"*, details deferred to `trust.typesafe.ai`. Generic. | **[P]** |
| Deletion on request | Not explicitly addressed in the DPA beyond forwarding data-subject requests to the customer. | **[P — verified absent]** |

### The privacy verdict, stated plainly

This system transcribes **everything the user types**. The candidate sentences we would send to
Jev *are* the user's raw keystrokes — passwords, messages, medical notes, whatever. There is **no
de-identification path**: the payload is, by construction, the personal data.

What their terms actually permit, and what they do not:
- ✅ They contractually will **not train** on it. That is a real, primary-source commitment.
- ❌ They **do retain** it, for an **unspecified** period, **in the United States**.
- ❌ **ZDR is not available to us** at our stage — it is gated to enterprise contracts.
- ❌ There is **no EU data residency option at all**, so the entire flow is a third-country
  transfer requiring SCCs (which they offer) plus our own transfer impact assessment.
- ⚠️ Going via Vercel or Cloudflare to skip the waitlist **adds a second processor** to the chain
  and a second set of logs. It makes the privacy posture worse, not better.

For a German-deployed keyboard, routing raw typed text to a US API with an unbounded retention
window and no ZDR is, in my view, the single largest blocker — larger than latency or accuracy.
It is very likely to include GDPR Art. 9 special-category data (health, beliefs) as an unavoidable
side effect of being a keyboard. **Recommendation: do not route raw typed text off-device.** If a
hosted call is ever made, it should be for a non-reconstructive signal, and Jev's whole value here
is precisely that it sees the sentences — so that carve-out does not exist.

---

## 5. Context window and input size

- **64k tokens per request total; 32k for `state` + the longest question** **[P]**.
- Choice: **≤255 options** **[P]**. Score: **2–10 levels** **[P]**. Nouls: unbounded in practice,
  batched in parallel **[P]**.

**Can it take a few hundred candidate strings plus per-candidate numeric scores in one call?**
- *Fitting them:* **yes, easily.** 100 candidates × ~12 tokens ≈ 1.2k tokens, well inside 32k. Even
  255 candidates with descriptions fits.
- *Using the numbers:* **no — this is a documented failure mode.** The jaggedness page states Jev
  **cannot reliably do arithmetic or counting** and treats numbers as text **[P]**. Handing it CTC
  log-likelihoods and asking it to weigh them against linguistic plausibility is exactly the thing
  it is documented to be bad at. The correct architecture is: let Jev return the **posterior over
  candidates**, and do `argmax_i [ log P_jev(i) + λ·log P_CTC(i) + μ·log P_profile(i) ]` **in our own
  code**. That is also what the docs advise generally ("move calculations to code").

---

## 6. Model size and how the speed is achieved

- TypeSafe describes "a new model architecture, parallel sampler for maximum efficiency" and
  non-autoregressive decoding — it emits a distribution over your option set in one shot rather
  than sampling tokens **[P]**.
- **No parameter count, no architecture details, no paper** **[P — verified absent]**.
- Training: **RLCD** (Reinforcement Learning for Calibrated Decisions), contrasted with RLHF/RLVR
  **[P]**. No RLCD paper published **[?]**.
- **Self-hostable / on-device variant: none, now or announced** **[P — absent; S — multiple
  sources confirm no weights, no VPC, no on-prem, no timeline]**.
- Third-party commentary speculates it is a BERT-scale encoder-classifier decomposition **[S,
  speculation — treat as unverified]**.

---

## 7. Independent evaluations

Two exist and both are reproducible:

1. **`github.com/scienthoon/jev-ood-calibration`** — 900 rule-generated support tickets (OOD) plus
   OpenBookQA / CommonsenseQA / HellaSwag, via Vercel AI Gateway, ~$0.06 to reproduce. Numbers in
   §1. Headline: **calibration is good in-distribution and ~4.4× the noise floor out of
   distribution, with per-primitive sign flips.** **[S]**
2. **`github.com/themsquared/jev-benchmark`** — agent tool-call risk, 60 hand-labelled cases
   (14 ambiguous, 12 adversarial). 91.7% accuracy; p50 421.6 ms / p95 542.0 ms; ~$0.0000173/call;
   no wrong answer at confidence 1.000. Caveat from the author: no frontier-LLM comparison was run,
   so the 40–200× speed/cost multipliers remain **unverified by anyone outside TypeSafe**. **[S]**

Nothing peer-reviewed. No Artificial-Analysis-style independent leaderboard entry found. **[?]**

---

# Fit assessment for our decoder

## The pipeline arithmetic first

Measured today (Mac mini M4, real recordings):

| Stage | Per 8 phrases | Per phrase | Share |
|---|---|---|---|
| (a) generate candidates from noisy letter stream (Qwen 8B) | 139 s | **17.4 s** | 69% |
| (b) score/rank candidates vs motion evidence + profile (Qwen 8B) | 63 s | **7.9 s** | 31% |
| total | 202 s | ~25–27 s | |
| Target | | **~2 s** | |

**This single table settles the question.** Jev cannot touch stage (a). Even if Jev made stage (b)
*instantaneous*, we would go from ~25 s to ~17.4 s per phrase — still **9× too slow**. Jev is not a
solution to our latency problem, because our latency problem is 69% generation and Jev cannot
generate.

## Stage (b) — SELECTION: could Jev do it?

**Yes, technically, and it's a clean fit for the API shape.**

- Shape: one request. `state` = the CTC letter stream + a compact personal-profile blurb.
  One **Choice** question, the 30–100 candidates as options (cap 255 — we fit). Get back the full
  posterior, then combine with CTC log-likelihood **in our code** (§5).
- **Latency:** ~0.5–0.6 s p50 from Germany (estimate, §2), ~0.75 s p95. Fits a 2 s budget with
  room, but consumes ~30% of it and adds a hard network dependency to every keystroke burst.
- **Cost:** state + 100 candidates + question ≈ 2.5k input tokens → **$0.000105 per phrase**.
  At ~1,000 phrases/user/day (≈2 h of typing at 40 wpm) that is **~$0.10/user/day ≈ $3/user/month**.
  Output free. Negligible. Rate limit 1,200 req/min supports ~1,200 concurrent typists at one call
  per phrase — fine at our scale.
- **Would the calibration let us threshold "not confident → show a suggestion bar"?**
  **Not out of the box.** Our task is maximally OOD, and the one rigorous independent study
  measured Choice overconfidence needing T≈3.3 on a novel task (§1). We would have to fit our own
  temperature on held-out recordings per user or per condition. That is a few hours of work and it
  is doable — but it means Jev's headline selling point (calibration you can trust immediately)
  does not transfer to us, and we end up maintaining a calibration layer anyway. At which point a
  local model's logits, which we can calibrate identically, lose much of their disadvantage.
- **Would it be *accurate*?** **Unknown, and I would bet against it.** Jev is documented to be
  literal, to degrade with noisy/irrelevant context, and to be bad at multi-step indirection.
  "Which of these 80 garbled strings is the most plausible English sentence given this letter
  posterior" is a language-modelling question, not a classification question. A model trained for
  routing/triage has no particular reason to be good at it, and nobody has benchmarked it on
  anything resembling this. This needs a 200-phrase pilot before any design commitment.

## Stage (a) — GENERATION: can Jev do it?

**No. Categorically.** Not a tuning problem, an architecture problem. There is no generative
primitive; output is always an element of a caller-supplied set; TypeSafe's own docs list text
generation as out of scope **[P]**. Producing "if it takes daily luke..." from
`"if itak dely lluke i ak dpoi gtivht nois"` requires free-text generation. Jev will never do it.

## Other decision points

| Use | Fit | Why |
|---|---|---|
| **"Has the user finished a phrase"** (segment boundary) | ❌ **No** | Needs to fire every ~100 ms against motion/timing features. A 500 ms network round trip per decision is a non-starter, and it is a signal-processing question (pause duration, motion energy), not a semantic one. Use a local threshold or HMM. |
| **"Is this decoded word wrong / worth flagging"** | ✅ **Genuinely good fit** | Batch ~10 Nouls (one per word) in one request over one state — TypeSafe's own batching measurement says that costs barely more than one question (0.27 s for 13). Runs *after* the phrase is committed, so it is off the critical path. This is the one place the API shape matches the job exactly. Still gated by the privacy problem. |

## Blunt verdict on Jev

- **Could serve:** stage (b) selection (with our own recalibration), and post-hoc per-word error
  flagging.
- **Cannot serve:** stage (a) generation (architecturally impossible), segment boundary detection
  (latency + wrong modality).
- **Expected latency:** ~0.5–0.6 s p50 per phrase from Germany (estimate; unmeasured for us).
- **Expected cost:** ~$0.0001/phrase, ~$3/user/month. Cost is a non-issue.
- **Blockers, in order:**
  1. **Privacy.** Raw keystrokes to a US processor, unbounded retention, no ZDR at our tier, no EU
     region. This alone should stop the idea for a shipped product.
  2. **It doesn't fix the actual bottleneck.** 69% of our time is generation. Best case Jev takes
     us from 25 s to 17 s per phrase.
  3. **Unproven on our task type**, with documented OOD overconfidence.
  4. **Hard network dependency** on the typing hot path — offline typing stops working.

---

# The alternatives, and what actually gets us to 87% @ 2 s

## The key insight the current pipeline is missing

**Stage (a) does not need an LLM at all.** "Generate candidate sentences from a noisy character
posterior" is the textbook job of a **CTC beam search with a lexicon + n-gram language model**
(KenLM / WFST — `pyctcdecode`, `torchaudio`'s CTC decoder, or a hand-rolled prefix beam search).
It runs in **single-digit milliseconds** on CPU, it consumes the CTC posteriors *exactly* rather
than through a text serialization, and it naturally emits an N-best list with scores — which is
precisely the input stage (b) wants. The personal language profile drops straight in as the n-gram
LM, which is what a personal profile actually is.

Using an 8B LLM to do this is paying 17.4 s for something a beam search does in ~10 ms, and
throwing away the posterior in the process.

**Likewise, stage (b) does not need generation.** Scoring N candidates is a *teacher-forced
forward pass* — one batched prefill, no token-by-token decoding. That is a completely different
cost class from what a generative 0.5B/8B does. The measured "0.5B ≈ 1 s/phrase, 67%" number is a
*generation* number and does not describe a 0.5B used as a scorer.

## The comparison

| Option | Stage (a) | Stage (b) | Est. latency/phrase | Cost | Privacy | Gets 87% @ 2 s? |
|---|---|---|---|---|---|---|
| **Beam search + local small-LM rescore** ⭐ | CTC beam search + personal n-gram, ~10 ms | Qwen 1.5B/3B teacher-forced batch scoring of top-20, est. 0.2–0.8 s on M4 (MLX, 4-bit) | **~0.3–1 s** | €0 | fully local | **Most likely yes** |
| Optimise the current 8B | replace with beam search (same as above) | 8B as *scorer* not generator: top-20 × ~40 tok ≈ 800 tok prefill, est. 1–3 s on M4 | ~1.5–3.5 s | €0 | fully local | Borderline; strongest accuracy |
| Local 0.5B as-is (generative) | 0.5B generation | 0.5B | ~1 s | €0 | fully local | **No — 67%** |
| **Jev** | ❌ impossible | Choice over candidates, ~0.5–0.6 s | **~17–18 s** (stage (a) still local 8B) | $0.0001 | ❌ US, retained | **No** |
| **Haiku 4.5 API** (`claude-haiku-4-5`, $1/$5 per MTok, 200k ctx) | ✅ can generate | ✅ can rank | ~0.6–1.2 s/call from DE, 1–2 calls | ~$0.0025/phrase → **~$75/user/month** at 1k phrases/day | ❌ US-hosted third party | Latency yes; **cost and privacy no** |

Notes on the numbers:
- All local-latency figures are **estimates from model size and M4 MLX throughput, not measured**.
  They must be benchmarked. The ranking between them is robust; the absolute values are not.
- Haiku 4.5 is ~24× Jev's input price ($1/M vs $0.042/M) and also bills output. It is the only
  hosted option that could do *both* stages, but at ~$75/user/month it is not viable for a
  consumer keyboard, and it has the identical raw-keystroke privacy problem.
- "Quantisation, speculative decoding, shorter prompts, batching, caching" on the 8B: speculative
  decoding only helps *generation*, which we are deleting. Quantisation is presumably already on.
  The real 10–50× on stage (b) comes from **switching from generative scoring to teacher-forced
  batch scoring**, and the real 1000× on stage (a) comes from **deleting the LLM**. Those two
  changes subsume the rest of the optimisation list.

## Recommendation

1. **Build the CTC beam search + personal n-gram LM first.** It is a day or two of work, it is
   free, it is local, and it plausibly removes 69% of our latency outright. Measure its N-best
   oracle accuracy — if the correct sentence is in the top-20 at >95%, stage (b) becomes an easy
   rescoring problem.
2. **Re-implement stage (b) as batched teacher-forced scoring**, and sweep model size
   0.5B → 1.5B → 3B → 8B against the ~2 s budget. Interpolate `log P_LM + λ·log P_CTC +
   μ·log P_profile`, tune λ/μ on held-out recordings.
3. **Only if (2) plateaus below 87%** should a hosted rescorer be reconsidered — and then the
   privacy question has to be answered first, not last. Jev would need an EU region and contractual
   ZDR before it could ship in a keyboard for German users; neither exists today.
4. Jev's one defensible niche for us is **post-hoc per-word error flagging** (batched Nouls,
   off the critical path) — and even that sends the user's text to the US.

---

## Sources

Primary (TypeSafe):
- https://typesafe.ai/ — homepage claims (193.6× faster, $42/B input)
- https://typesafe.ai/blog/introducing-system-one-models-and-jev — 70–500 ms, RLCD, pricing, waitlist
- https://docs.typesafe.ai/introduction — primitives overview
- https://docs.typesafe.ai/llms.txt — full docs index
- https://docs.typesafe.ai/models.md — 64k/32k context, rate limits, no-training, enterprise ZDR
- https://docs.typesafe.ai/api.md — endpoint, schemas, 255-option and 2–10-level caps
- https://docs.typesafe.ai/primitives/choice.md — Choice semantics and limits
- https://docs.typesafe.ai/primitives/noul.md — Noul semantics
- https://docs.typesafe.ai/confidence.md — thresholding guidance (no numbers)
- https://docs.typesafe.ai/model-jaggedness/jev-1.13.md — documented weaknesses incl. no generation, no arithmetic
- https://docs.typesafe.ai/cookbooks/rerank_typesafe.md — re-ranking pattern and measured results
- https://docs.typesafe.ai/cookbooks/parallel_questions.md — 13 questions in 0.27 s batched
- https://docs.typesafe.ai/introduction/quickstart.md — curl example
- https://docs.typesafe.ai/legal.md — legal index
- https://typesafe.ai/legal/privacy — US hosting, no-training commitment
- https://typesafe.ai/legal/data-processing — SCCs Module 2, UK Addendum, retention language
- https://trust.typesafe.ai/subprocessors — referenced; **did not render, contents unverified**

Third party:
- https://github.com/scienthoon/jev-ood-calibration — independent OOD calibration study
- https://github.com/themsquared/jev-benchmark — independent latency/accuracy benchmark
- https://developers.cloudflare.com/ai/models/typesafe/jev/ — 32k context, availability
- https://vercel.com/changelog/typesafe-ai-jev-now-available-on-ai-gateway — 2026-09-16, no waitlist
- https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/ — the article under review

Anthropic comparison: Claude Haiku 4.5 = `claude-haiku-4-5`, $1.00 / $5.00 per MTok, 200K context.
