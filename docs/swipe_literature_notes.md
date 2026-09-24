# Gesture & surface-typing decoders — literature notes (2026-09-13)

Key takeaways for bare-desk camera typing (full comparison table and sources below).

## Ranked transferable techniques
1. **Sentence-level beam fusing CTC frame posteriors + char n-gram + word LM with optional spaces, plus post-correction; LLM rescores the final lattice/N-best** — StegoType touchpad 5.6→0.6% with sentence beam; TouchInsight greedy ~7% → beam ~3% UER; Gboard post-correction −12 to −20% rel. WER. Richardson 2020: frame-wise CTC beats discrete contact decoding because missed/spurious contacts break contact decoders. No new data needed; largest expected gain.
2. **Intent-labelled per-user data at scale** — StegoType: intent labels 7% vs physical-key labels 18–25% CER; per-user 6.1% vs cross-user 24.6%; Richardson ~60 min/user. Needs 30–60 min of the user's natural typing labelled with final intended text.
3. **Personal/contextual biasing with a literal (character-level) escape path** — class-based/dynamic LMs for names (ASR shallow fusion −31 to −46% name errors), prefix-level boosts; an LM without escape path hurts OOV text (StegoType random letters 16.9→28.5%).
4. **Per-user, drift-invariant motor model adapting online** (TOAST relative inter-tap vectors 86.2→92.1%, per-user >96% top-1, 10-word sliding refit; per-finger Gaussians 5.66→4.76% UER) — second score channel, gated.
5. **Confidence-gated top-3 suggestion bar + post-correction** — top-3 lifts +3 to +14 pts; only route in the literature to ≥98% effective words.
Lower transfer: SHARK² template matching (42% on surface traces), synthetic trajectories (mixed), small fine-tuned LLM decoders (weak on short words).

## Ceilings
Best flat-surface hand-motion→text: 2.2–2.4% UER (mocap, experts, ~1 h/user), ~3% UER (headset + touch model + word-LM beam), 7% UER without LM. ≈88–95% words (estimate). ≥99% word figures all involve touch/depth sensors, in-vocabulary prompts and/or candidate selection. No surveyed system reaches 100% automatically. Realistic for us: ~92–95% automatic, ≥98% with top-3 tap.

## Comparison table
| System | Input | Surface | Decoder | Users/data | Accuracy | WPM |
|---|---|---|---|---|---|---|
| Richardson UIST'20 | Mocap hand pose 60 Hz | Touchpad | TCN + CTC, char Transformer LM beam | 20 experts, ~60 min each | 2.38% UER | 73 |
| StegoType UIST'24 | Headset tracker latents | Flat surface | Emformer + CTC, no LM live | 606 typists | live 7% UER | 42 |
| TouchInsight UIST'24 | Headset → touch time/finger/Gaussian | Any surface | beam + char LM + word trigram | 385 participants | beam 2.65–3.50% UER | 37 |
| TOAST IMWUT'18 | Touch sensor | Tabletop | word Bayesian, relative vectors, online refit | 16 | per-user >96% top-1, top-3 >99% | 43–45 |
| ATK UIST'15 | Leap Motion | Mid-air | word-level, same-length words | 8 | 99.6% with 5-candidate choice | 29 |
| TapType CHI'22 | Wrist IMU | Any surface | Bayes net + n-gram | 10 | CER 0.6% online (selection) | 19 |
| TypeAnywhere CHI'22 | Tap Strap | Table | neural LM over finger sequences | 2.5 h | CER 1.5% | 71 |
| VelociTap CHI'15 | Touchscreen | Phone/watch | sentence decoder w/ insert/delete, 12-gram char + 4-gram word LM | 48 | nearest-key 20.2% → 4.7% CER | 41–50 |
| Gboard FST '17 | Taps/swipes | Phone | lexicon+LM FST, optional space, literal path, post-correction | — | gesture WER 11.9→9.2 | — |
| Alsharif ICASSP'15 | Swipe | Phone | BLSTM + CTC + trie lexicon | 45K real words | 89.2% vs 67% baseline | — |
| FUTO Swipe '26 | Swipe | Phone | TCN + CTC, trie beam | 1.04M swipes | 93.5% top-1, 97.9% top-3 | — |
| Gesture2Text '24 | Swipe | Surface | pretrained neural | 100 | 83.0% (SHARK² 42.1%) | — |

## Sources
SHARK² http://pokristensson.com/pubs/KristenssonZhaiUIST2004.pdf · Alsharif https://research.google.com/pubs/archive/43461.pdf · Ouyang FST https://arxiv.org/abs/1704.03987 · Gboard personalisation https://arxiv.org/abs/2209.11311 · QuickPath GAN https://arxiv.org/abs/2004.07800 · FUTO https://arxiv.org/html/2606.25247 · How We Swipe https://luis.leiva.name/web/docs/papers/shapewriting-mobilehci2021-preprint.pdf · Gesture2Text https://arxiv.org/abs/2410.18099 · Richardson https://research.facebook.com/file/456120199154190/Decoding-Surface-Touch-Typing-from-Hand-Tracking.pdf · StegoType https://www.keithv.com/pub/stegotype/ · TouchInsight https://arxiv.org/abs/2410.05940 · TapType https://arxiv.org/abs/2410.06001 · TypeAnywhere https://dl.acm.org/doi/10.1145/3491102.3517686 · ATK https://pi.cs.tsinghua.edu.cn/lab/papers/p539-yi.pdf · TOAST https://pi.cs.tsinghua.edu.cn/lab/papers/TOAST.pdf · KeySense https://arxiv.org/abs/2602.12432 · VelociTap https://www.keithv.com/pub/velocitap/velocitap.pdf · Vertanen CHI'18 https://dl.acm.org/doi/10.1145/3173574.3174200 · Contextual density ratio https://arxiv.org/abs/2206.14623 · Quinn & Zhai https://dl.acm.org/doi/10.1145/2858036.2858305
