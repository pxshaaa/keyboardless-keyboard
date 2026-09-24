# Open-source survey for bare-desk typing from one overhead camera

**Date:** 2026-09-20
**Method:** web research only, primary sources (GitHub, arXiv, Hugging Face model cards) fetched and verified where possible.
**Evidence key used throughout:**
- **[MEASURED]** — a number I read from a primary source (paper, model card, repo README). URL given.
- **[REPORTED]** — a number from a secondary source (search summary, blog, third-party benchmark) that I could not verify against the primary document.
- **[ESTIMATE]** — my own arithmetic or judgement. Not measured by anyone.
- **[UNVERIFIED]** — I looked and could not confirm it. Do not quote.

---

## 0. System context this survey is written against

- Bare desk, **no keyboard**. One overhead phone camera, 1280x720, 60 fps, hands ~136 px knuckle width.
- Pipeline: MediaPipe hand landmarker (21 2D keypoints/hand) -> small Transformer/conv CTC over landmark sequences -> character posteriors -> word/LLM decoder.
- Best measured: ~82-94% words on labelled desk sessions, ~73-82% on blind tests.
- **Measured bottleneck is the visual signal, not the decoder.** From top-down, press vs hover gives **AUC 0.61** from 2D landmarks; **AUC 0.92** if true fingertip height were known.
- Depth/LiDAR tested and rejected (kills the wide lens).
- Assets: ~1 hour of keyboard video with exact keystroke labels; ~35 labelled desk phrases; Mac mini M4, 16 GB, MPS/MLX. **Cannot record new data right now.**

---

## 1. What is "JEV"?

**Plain answer: two different things exist under that name, and only one of them is plausibly what was meant.**

### (a) There IS a real, recently released project literally called "Jev" — and it is irrelevant here

TypeSafe AI announced a model called **Jev** on approximately 15 September 2026, five days before this survey. It is the first release in what they call a "System One" model class: it returns **typed decisions with calibrated probabilities instead of text**.

- **It is text-only.** No vision, no video, no multimodal capability described. **[MEASURED]**
- **It is not open source.** Verbatim from the coverage: *"TypeSafe has not published weights, a parameter count, or a self-hosting option."* It is a hosted API in early access behind a waitlist, at `POST https://api.typesafe.ai/v1/systemone`. **[MEASURED]**
- Source: https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/
- Their stated official resources: `typesafe.ai/blog/introducing-system-one-models-and-jev`, `docs.typesafe.ai`, `github.com/typesafe-ai/skills`. **I did not independently fetch these.** **[UNVERIFIED]**

This Jev has **nothing to do with video, hands, pose, or typing recognition**. If the user read about "JEV" in the last week, this is almost certainly the headline they saw — and it does not help this project at all.

### (b) V-JEPA 2 is the likely intended referent

Given the phrasing "recently released open source project" in the context of video understanding, **V-JEPA 2** (Meta FAIR's Video Joint-Embedding Predictive Architecture) is the probable intent — the acronym is close and it is genuinely open source and genuinely recent.

- Repo: https://github.com/facebookresearch/vjepa2
- Paper: https://arxiv.org/abs/2506.09985
- Blog: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/

### (c) What I could not confirm

I searched for other open-source projects named JEV or Jev in 2025-2026 and found **no vision, video, hand-tracking, or typing-related project under that name**. I did not find a "JEV" that is a video model. I am not going to invent one. If the user has a specific link, that would settle it — but on the public record as of 2026-09-20, "JEV" resolves to TypeSafe's closed text API, and "the recently released open-source video model people are excited about" resolves to V-JEPA 2 / 2.1.

**Bottom line for the user:** JEV as such will not help. V-JEPA 2 might, as a frozen feature extractor — but it is ranked 7th below, not 1st, and section 3 explains why.

---

## 2. THE HEADLINE FINDING: PressureVision — the press signal is in the PIXELS

This is the most important result in the survey and the reason item #1 in the ranking is what it is.

### 2.1 What it is and why it matters to us

**PressureVision** (ECCV 2022) and **PressureVision++** (WACV 2024), Grady et al., estimate **the pressure a hand applies to a surface from ordinary RGB images**. The premise of the work is precisely our bottleneck, and the ECCV abstract states the mechanism outright:

> *"The central insight is that the application of pressure by a hand results in informative appearance changes. Hands share biomechanical properties that result in similar observable phenomena, such as **soft-tissue deformation, blood distribution, hand pose, and cast shadows**. ... We also show that the output of our model **depends on the appearance of the hand and cast shadows near contact regions**."*

— https://arxiv.org/abs/2203.10385 **[MEASURED — abstract read directly]**

**Why this is the key insight for us:** our AUC of 0.61 is the AUC of *21 landmark coordinates*. Nail blanching, fingerpad compression, and the contact shadow converging under the fingertip are all **destroyed by the landmark abstraction before our model ever sees them**. AUC 0.61 may be the ceiling of the *features*, not the ceiling of the *pixels*. That is a testable claim and it is cheap to test.

Independent corroboration of the same physical cue, from the opposite direction: a June 2026 paper built a bare-surface keyboard whose press detector was **hand-crafted fingernail hue-variance analysis** and got **~36% touch detection accuracy** — the cue is real but hand-crafting it barely works. https://arxiv.org/html/2606.26508 **[MEASURED]**

### 2.2 The two releases, side by side

| | PressureVision (v1, ECCV 2022) | PressureVision++ (WACV 2024) |
|---|---|---|
| Paper | https://arxiv.org/abs/2203.10385 | https://arxiv.org/abs/2301.02310 (HTML: https://arxiv.org/html/2301.02310v3) |
| Repo | https://github.com/facebookresearch/PressureVision | https://github.com/pgrady3/pressurevision2 |
| **Licence** | **MIT — code AND dataset** **[MEASURED, from repo]** | **MIT** **[MEASURED, from repo]** |
| Weights released | **Yes** — `python -m recording.downloader` | **Yes** — Dropbox link in README, place at `data/model/paper_29.pth` |
| Dataset released | **Yes. PressureVisionDB: 36 participants, 4 cameras, 16 hours. 140 GB (3 fps sample) / 960 GB (15 fps). MIT licence.** | **No.** README: *"our team is still working on dataset hosting... training and evaluation is not possible at this time."* |
| Architecture | CNN encoder-decoder (PressureVisionNet) | **SE-ResNeXt-50 encoder + FPN decoder**, trained end-to-end |
| Input | RGB, hand crop | **RGB, 448x448 crop**; hand bbox from **MediaPipe** |
| Camera setup | *"a webcam about 60 cm above a white table, pointing at a 45 degree angle downwards"* | Logitech Brio ~55 cm above table, ~45 deg down; 1080p 30 fps downsampled to 15 fps |
| Data scale | 36 participants | **51 participants, 20 environments, 2.9M frames** (0.5M fully labelled with a Sensel Morph sensor, 2.4M weakly labelled) |
| Demo | `python -m prediction.webcam_demo --config paper` | `python -m prediction.demo_webcam --config paper` |

### 2.3 What it predicts (exactly)

A **per-pixel pressure image** over the hand crop, plus a contact/no-contact label. PressureVision++ adds a binary "contact label" classification head (`Lw`) alongside the pressure regression head (`Lp`) and an adversarial domain-adaptation loss (`Ld`) for the weakly-labelled data. For a keyboard you find **local maxima in the pressure blob** to get the touch location. **[MEASURED — https://arxiv.org/html/2301.02310v3]**

So: it gives you, per frame, per fingertip, **a press probability and a contact location** — which is exactly the missing variable in our pipeline.

### 2.4 Reported accuracy — including the single strongest number

**Contact detection (PressureVision++):** **[MEASURED]**
- **Contact accuracy 89.3%** on the fully-labelled test set; **80.5%** on the weakly-labelled test set.
- **It outperformed human annotators (78.4%)** on the fully-labelled test set.
- Contact IoU 41.9%; volumetric IoU 27.5%.

**THE SINGLE STRONGEST MEASURED NUMBER FOR US — they built our exact demo:**
A keyboard layout projected onto a tabletop, 10 participants, net words per minute:

- **PressureVision++ keyboard: 25.8 net WPM**
- **Pose-based "Direct Touch" baseline: 14.4 net WPM**
- 9 of 10 participants preferred the pressure system.

**That is a ~1.8x throughput advantage of the appearance cue over the pose cue, measured on a tabletop typing task.** It is the most direct published evidence that our landmark abstraction is throwing away the signal we need. **[MEASURED — https://arxiv.org/html/2301.02310v3]**

**Speed:** ~50 FPS on an RTX 3090, including hand detection plus pressure estimation for both hands. **[MEASURED]**

### 2.5 Will it run on the M4?

**Yes, comfortably — this is the cheapest experiment in the whole survey.**

- The model is an **SE-ResNeXt-50 + FPN** built on `segmentation_models_pytorch`. That is roughly 25-30M parameters — a small CNN, not a ViT. **[MEASURED — architecture from paper; parameter count is my ESTIMATE]**
- Pure PyTorch convolutions map cleanly onto MPS. No custom CUDA kernels in the pipeline.
- The README documents CUDA setup only and gives **no CPU or Apple Silicon notes**. **[UNVERIFIED — nobody has published an M4 benchmark.]**
- **[ESTIMATE]** At 50 FPS on a 3090, expect single-digit-to-low-teens FPS on M4 MPS. Irrelevant for us: we are running **offline over 1 hour of existing footage**, not real time. One overnight pass.
- Caution: the repo pins Python 3.10 / PyTorch 1.12.1 / CUDA 11.3. Expect to update the pins for a modern MPS build.

### 2.6 Exactly how it attaches to our pipeline

Three integration points, in increasing order of commitment:

**(A) Diagnostic probe — do this first, ~1 day.**
Run the released PressureVision++ checkpoint over our 35 labelled desk phrases. We already use MediaPipe, so we already have the hand bbox it needs for the 448x448 crop. For each frame, take the predicted pressure under each fingertip landmark. **Compute press-vs-hover AUC from that scalar and compare it against our measured 0.61.** This is a pure measurement with no training and no integration. It answers "is the signal in the pixels?" definitively.

**(B) Extra input channel to the existing CTC model — ~2-3 days.**
Append the 10 per-fingertip pressure scalars (or a small pooled feature vector from the FPN decoder) to the landmark feature vector we already feed the CTC encoder. No architecture change, no new loss, and the CTC model retrains in minutes on the M4.

**(C) Fine-tune the pressure head on our own footage — ~1 week.**
Our ~1 hour of keyboard video **has exact keystroke labels**, which is per-frame contact ground truth for free. Fine-tune the SE-ResNeXt-50 on our top-down view using those labels. This is the step that fixes the viewpoint mismatch. A ~25M-param CNN fine-tune on 1 hour of video is well within a 16 GB M4.

### 2.7 Risks — stated bluntly

1. **Viewpoint mismatch.** Their camera is at **~45 degrees**, ours is **~90 degrees top-down**. This is out of distribution and **nobody has quantified it**. PressureVision++ also showed poor cross-dataset generalisation to PressureVisionDB's *"very harsh, artificial lighting conditions"* — so the model is not viewpoint/lighting-invariant. **[MEASURED limitation]**
2. **Occlusion in ten-finger typing — the big one.** PressureVision++'s own failure analysis: *"pressure is not estimated for occluded fingertips."* During their typing study *"fingertips are often occluded"* in five-finger typing, so **they restricted the study to index-finger-only input**. The 25.8 WPM headline is an index-finger number. **Ten-finger touch typing is exactly the case they avoided.** **[MEASURED]** — this is the single biggest reason the experiment might fail for us. Note however that a top-down view is arguably *better* for nail visibility than their 45-degree view, which cuts the other way.
3. **No viewpoint-generalisation study exists** in either paper. **[UNVERIFIED]**
4. No gloved-hand evaluation; bare hands only. **[MEASURED — stated limitation]**

### 2.8 Active successor line (2025-2026)

- **EgoPressure** (CVPR 2025) — https://arxiv.org/abs/2409.02224 · https://github.com/eth-siplab/EgoPressure · https://huggingface.co/datasets/eth-siplab/EgoPressure
  21 participants, 5.0 hours, **1 egocentric camera + 7 STATIC third-person Azure Kinects**, Sensel Morph pressure GT, MANO hand meshes, ~1.96 TB, 30 Hz. Baseline `PressureFormer` predicts pressure as a UV map on the hand mesh.
  **Licence: CC BY-NC-SA 4.0 — NON-COMMERCIAL.** Plus MANO restrictions. **[MEASURED]** Research-usable, product-blocked.
- **HOPE: Hand-Object Pressure Estimation from Monocular Videos** — https://arxiv.org/html/2608.06192 (per-vertex pressure on the hand mesh) **[REPORTED — not fetched directly]**
- **EgoPressDiff** (2026) — https://arxiv.org/pdf/2606.06872 (conditional video diffusion for UV pressure maps) **[REPORTED]**
- **EgoTactile** (2026) — https://arxiv.org/pdf/2606.09243 **[REPORTED]**

**PressureVisionDB (MIT) is the only commercially usable pressure/contact dataset in this line.**

---

## 3. V-JEPA 2 / 2.1 — verified facts and a blunt verdict

### 3.1 Licence

- Repo README: *"The majority of V-JEPA 2 is MIT-licensed"*; `randaugment.py`, `randerase.py`, `worker_init_fn.py` are Apache 2.0. **[MEASURED — https://github.com/facebookresearch/vjepa2]**
- HF model card `facebook/vjepa2-vitl-fpc64-256`: **MIT**. https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 **[MEASURED]**
- HF model card `facebook/vjepa2-vitg-fpc64-384`: **Apache 2.0**. https://huggingface.co/facebook/vjepa2-vitg-fpc64-384 **[MEASURED]**
- **V-JEPA 2.1 weights licence: COULD NOT CONFIRM.** The arXiv HTML for 2.1 carries CC BY-NC-ND 4.0, but that is the *paper's* licence, not the weights'. [HF issue #137](https://github.com/facebookresearch/vjepa2/issues/137) shows Hugging Face asking Meta to host the 2.1 checkpoints, still open with no reply — so 2.1 weights ship from the repo, not the hub, and I found no explicit weights licence. **[UNVERIFIED — verify before any commercial use of 2.1.]**

### 3.2 Model sizes and inputs **[MEASURED — repo README + HF config]**

**V-JEPA 2:** ViT-L/16 300M @256 · ViT-H/16 600M @256 · ViT-g/16 1B @256 · ViT-g/16 1B @384
**V-JEPA 2.1:** ViT-B/16 80M @384 · ViT-L/16 300M @384 · ViT-g/16 1B @384 · ViT-G/16 2B @384

`VJEPA2Config` defaults: `patch_size=16, crop_size=256, frames_per_clip=64, tubelet_size=2, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16`.
https://huggingface.co/docs/transformers/model_doc/vjepa2

**Token count: 64 frames / tubelet 2 = 32 temporal x (256/16)^2 = 256 spatial = 8,192 tokens per clip.** That is why it is far heavier than an image ViT. **[MEASURED arithmetic from the config]**

Two facts that make it *cheaper than it looks*, both verified:
- **It uses RoPE, not learned absolute position embeddings** (`VJEPA2RopeAttention`, with `apply_rotary_embeddings` / `rotate_queries_or_keys` decomposed over depth/height/width), and patch counts are computed dynamically (`# ensure we are using dynamic patch size`). **So arbitrary crop sizes and frame counts work at inference.** https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/vjepa2/modeling_vjepa2.py **[MEASURED]**
- The config docs state `frames_per_clip` *"does not impact inference"*, and **Meta's own SSv2 evaluation uses 16-frame clips (16x2x3), not 64.** **[MEASURED]**
  **[ESTIMATE]** A 128x128 hand crop at 16 frames is 8 temporal x 64 spatial = **512 tokens** — 16x cheaper than the default. This is what makes item 7 feasible at all.

### 3.3 macOS problem

Repo README, verbatim: *"V-JEPA 2 relies on `decord`, which does not support macOS (and, unfortunately, is also no longer under development)."* Workarounds suggested: eva-decord, decord2. Or pre-decode to frames with ffmpeg, which is what we should do. **[MEASURED]**

### 3.4 How you attach a task head

**A 4-layer attentive probe on frozen features:** *"composed of four transformer blocks, the last of which replaces standard self-attention with a cross-attention layer using a learnable query token,"* then a linear classifier. **[MEASURED — https://arxiv.org/html/2506.09985v1]**

Reference command: `python -m evals.main --fname configs/eval/vitl16/ssv2.yaml --devices cuda:0 cuda:1` — i.e. **the official recipe assumes two CUDA GPUs**. There is **no documented CPU or MPS path**, and **the repo states no hardware requirements anywhere** for inference or probing. **[MEASURED / UNVERIFIED respectively]**

For our purposes the attentive probe is beside the point: we would extract frozen features, cache them, and feed them to our existing CTC head.

### 3.5 Results on fine-grained hand / manipulation video — the genuinely encouraging part

**[MEASURED — repo README benchmark table and paper]**
- **Something-Something v2: 77.3% top-1, frozen** (previous best 69.7). SSv2 is overwhelmingly close-range hand-object manipulation — the most relevant benchmark that exists.
- **EPIC-KITCHENS-100 action anticipation: 39.7 recall@5** (previous best 27.6; a 44% relative improvement). Egocentric hands.
- **Diving48: 90.2%.**
- Video QA with LLM alignment at 8B: PerceptionTest 84.0, TempCompass 76.9.
- Pretraining: **over 1 million hours of internet video.**
- V-JEPA 2-AC (action-conditioned world model) post-trained on **under 62 hours of unlabelled Droid robot video**, zero-shot on Franka arms. **Note: that 62 h is UNLABELLED post-training, not a low-label-supervision result. Do not cite it as evidence of label efficiency.**

**V-JEPA 2.1 (March 2026)** — https://arxiv.org/html/2603.14482v1 — adds a Dense Predictive Loss (all tokens contribute, not just masked ones) and Deep Self-Supervision. It substantially fixes V-JEPA 2's weak *spatial* features: **[MEASURED]**
- NYUv2 depth RMSE: **0.307 (2.1 ViT-G)** vs **0.642 (V-JEPA 2 ViT-g)**; beats DINOv3 ViT-7B (0.309).
- ADE20K semantic segmentation: **47.9 mIoU** vs 22.2. VOC12 85.0, Cityscapes 73.5.
- YouTube-VOS 72.7 J&F, DAVIS-17 69.0.
- Ego4D short-term object-interaction anticipation: **7.71 mAP**, +35% over prior SOTA.
- Paper describes V-JEPA 2's features as *"noisy and show only fragmented local spatial structure"* vs 2.1's *"spatially structured, semantically coherent, and temporally consistent."*
- **If we use this family at all, use 2.1, not 2** — subject to the licence caveat in 3.1.

Note for scale calibration: the NYUv2 RMSE of 0.307 is **metres, room-scale**. It is nowhere near the millimetre precision our press signal needs. It is evidence of *better spatial structure*, not of millimetre depth.

### 3.6 VERDICT: would a frozen V-JEPA encoder over hand crops replace landmark features?

**Plausible as an ADDITIONAL stream. Not a replacement. And not our highest-value experiment.** Reasons, in order of severity:

1. **No evidence it detects millimetre-scale contact.** SSv2, EK100 and Diving48 are coarse action *categories*. **There is no published result anywhere showing V-JEPA features separate press from hover, or anything at that temporal/spatial granularity.** Treat it as an unvalidated hypothesis. **[UNVERIFIED]**
2. **Data.** V-JEPA 2's headline probes are trained on SSv2 (~169k clips) and K400. We have ~1 hour of labelled keyboard video and 35 desk phrases — two to three orders of magnitude less. **I specifically looked for the 5%/10%/50% low-shot label-efficiency curves and they are NOT in the V-JEPA 2 paper** — that experiment belongs to V-JEPA 1. **Do not assume V-JEPA 2 probes are label-efficient. It is unverified.**
3. **Fine-tuning the encoder on a 16 GB M4 is not realistic** for ViT-L or larger. Frozen feature extraction at reduced frames and crop size is realistic. **[ESTIMATE — no M4 benchmark for V-JEPA 2 exists publicly; I searched and found none.]**
4. `decord` does not build on macOS (3.3).

**Also worth knowing — SALT.** "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (arXiv:2509.24317, Xianhang Li et al.) replaces V-JEPA's EMA teacher with a *frozen* teacher in a two-stage scheme, and reports students that **"outperform recently proposed V-JEPA 2 encoders under frozen backbone evaluation across diverse benchmarks"** at matched pretraining FLOPs — with the striking finding that *"student quality is remarkably robust to teacher quality: high-performing students emerge even with small, sub-optimal teachers."* If we ever pretrain on our own unlabelled desk footage, this is the cheap recipe. **Code/weights release: COULD NOT CONFIRM.** https://arxiv.org/abs/2509.24317

---

## 4. Hand pose / hand reconstruction survey — the honest answer is "this does not fix it"

### 4.1 The table

| Model | Licence | Best reported error | Camera-space / metric? | Apple Silicon |
|---|---|---|---|---|
| **HaMeR** (CVPR'24), ViT-H **671M** | code **MIT**; **weights require MANO -> registration-gated, RESEARCH-ONLY**; paper CC BY-NC-SA 4.0 | FreiHAND **PA-MPJPE 6.0 mm**, PA-MPVPE 5.7, F@5 0.785; HO3D-v2 **PA-MPJPE 7.7 mm** | **No.** Weak-perspective, regressed translation. Absolute MPJPE **not reported**. | No ONNX/CoreML/MPS export found **[UNVERIFIED]**. ~27 FPS on RTX 4060 Ti **[REPORTED — measured by the Fast-HaMeR authors, not by HaMeR]** |
| **WiLoR** (CVPR'25) | **CC-BY-NC-ND for code AND models** + Ultralytics + MANO. **No-derivatives = hard product blocker.** | FreiHAND **PA-MPJPE 5.5 mm**, PA-MPVPE 5.1; HO3D **PA-MPJPE 7.5 mm** | **No.** Authors' own stated limitation (verbatim): *"WiLoR estimates 3D hand poses in camera space, which may lead to inaccurate assumptions about the overall 3D scene"* — they suggest bolting on a 3D metric foundation model. | **The "130-175 FPS" figure is the DETECTOR only.** The ViT reconstruction head's FPS is not reported anywhere in the paper. **[MEASURED caveat]** |
| **HandOccNet** (CVPR'22) | **No LICENSE file at `main` or `master` — treat as all-rights-reserved.** + MANO. | DexYCB PA-MPJPE ~5.80 mm; HO3D-v3 ~10.7 mm **[REPORTED — from secondary summaries, not read off the paper's tables]** | No. | — |
| **RTMPose-hand / MMPose** | **Apache 2.0** — the only genuinely commercial-friendly option in this table | COCO-WholeBody-Hand RTMPose-m: PCK@0.2 **0.815**, AUC 0.837, **EPE 4.51 px**. FreiHand2D ResNet50 EPE 3.27 px. RHD2D EPE 2.18 px. OneHand10K HRNetv2+Dark EPE 23.96 px | **2D ONLY. No z, no depth, no world landmarks.** | **Best Apple story.** 90+ FPS on i7-11700 CPU, 430+ on GTX 1660 Ti, RTMPose-s 70+ FPS on Snapdragon 865. ONNX Runtime + CoreML EP via `rtmlib`. |
| **MediaPipe** (current) | Apache 2.0 | **MNAE 10.09%** (Full), 12.02% (Lite), across 14 regions; human inter-annotator 6.0% | z is **root-relative AND synthetic-trained AND never evaluated** — see 4.2 | 12-17 ms on Pixel 6; trivially real-time on M4 |
| **HandDGP** (ECCV'24, Niantic) | **COULD NOT CONFIRM** | **camera-space mean vertex error: FreiHAND 46.3 mm, HO3D-v2 50.3 mm** (vs RootNet 62.5, test-time PnP/DLT 50.0-50.1, MobRecon 121.7) | **YES — genuinely camera-space** (differentiable DLT + image rectification) | — |
| **ScaleHP** (Jun 2026, metric SOTA) | paper CC BY 4.0; code/weights **COULD NOT CONFIRM** | **CS-MPJPE: FreiHAND 35.8 mm, HO3Dv3 50.7 mm, DexYCB 136.3 mm**; PA-MPJPE 4.6 / 5.9 | **YES — metric space**, via a "scale token" on anatomical bone proportions | Speed **not reported** |
| **Fast-HaMeR** (Mar 2026) | **CC BY 4.0** | PA-MPJPE 8.1 mm (vs HaMeR 7.7) | Inherits HaMeR's limitations | ConvNeXt-L **240M** (vs ViT-H 671M); **40 FPS vs HaMeR's 27 FPS on RTX 4060 Ti**. Most plausible M4 candidate of the mesh models, still a stretch. |

URLs: HaMeR https://arxiv.org/abs/2312.05251 · https://github.com/geopavlakos/hamer · MANO https://mano.is.tue.mpg.de/ | WiLoR https://arxiv.org/abs/2409.12259 · https://github.com/rolpotamias/WiLoR · WiLoR-mini https://github.com/warmshao/WiLoR-mini | HandOccNet https://github.com/namepllet/HandOccNet | MMPose https://github.com/open-mmlab/mmpose · zoo https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html · rtmlib https://github.com/Tau-J/rtmlib | HandDGP https://arxiv.org/abs/2407.15844 · https://github.com/niantic-labs/HandDGP | ScaleHP https://arxiv.org/html/2606.25619v1 · https://laiang8086.github.io/scalehp | Fast-HaMeR https://arxiv.org/html/2603.16444v1 · https://github.com/hunainahmedj/Fast-HaMeR

### 4.2 MediaPipe's own model card kills the z channel

Model card (Oct 2021): https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf
Docs: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

**Verbatim from p.4** **[MEASURED]**:
> *"The model provides 3D coordinates, but as the **z screen coordinates as well as metric world coordinates are obtained from synthetic data**, so for a fair comparison with human annotations, **only 2D screen coordinates are employed**."*

Translation: **Google has never published any evaluation of the z output.** It is a GHUM mesh fitted to 2D projections of synthetic renders — a learned anatomical prior, not a measurement. The "world" landmarks are also **wrist-origin root-relative**, so they carry no information about distance to the desk even in principle.

Also from the card: **per-joint MNAE *"is the smallest at the base of each finger, and gets larger toward the fingertip"*** — the joints we care about are the worst ones. Out-of-scope explicitly includes hands holding objects and occlusions; described as *"meant for experimental usage."*

**Is there a newer MediaPipe hand landmarker?** **No evidence of one.** The shipped asset path is still `.../hand_landmarker/float16/1/hand_landmarker.task` (version **1**), and the model card is dated **Oct 2021**. The docs page is maintained (last updated June 2026) but the model is not new. **I could not confirm any 2024-2026 retrain.** **[UNVERIFIED]**

**Sanity check against our rig:** 10.09% MNAE x ~136 px palm ~= **~14 px of 2D fingertip error**. That is a clean explanation for AUC 0.61.

### 4.3 The three findings that settle the question

1. **Procrustes alignment deletes exactly what we need.** HaMeR, WiLoR and HandOccNet all report **PA-MPJPE / PA-MPVPE** — Procrustes alignment removes rotation, translation **and scale**. Fingertip-height-above-desk *is* the global translation. All three also use a **weak-perspective** camera, which is scale-depth-ambiguous by construction.
2. **HandDGP quantifies the gap in one sentence** **[MEASURED]**: root-relative error is low-single-digit mm, *"camera space error is **6 to 7 times larger**"* — attributed to 2D-to-3D depth ambiguity.
3. **ScaleHP's own depth-axis ablation** **[MEASURED]**: the scale token improves the **z-axis** root-relative error from **3.5 mm -> 3.0 mm**. **A keypress travel is ~2-4 mm. SNR ~= 1 at the state of the art**, before any top-down penalty. And its *absolute* error is 35.8 mm, ~10x the signal.

### 4.4 View-match caveat — applies to EVERY number in 4.1

**Every headline error above comes from FreiHAND, HO3D, DexYCB or InterHand2.6M.** These are green-screen or studio multi-camera **third-person / frontal** rigs with the hand filling the frame:
- FreiHAND: green-screen multi-view rig, hand centred and large — https://arxiv.org/pdf/1909.04349
- DexYCB: 8 RealSense cameras at fixed third-person viewpoints around a table — https://arxiv.org/html/2104.04631v1.pdf

**None** is an overhead view of hands resting flat on a desk with a 136 px knuckle width. **Nobody has measured our domain.** Expect degradation that is entirely unquantified. HaMeR/WiLoR's in-the-wild training helps with *robustness*, not with the specific top-down near-degenerate-depth geometry.

### 4.5 Geometry — why this is a camera problem, not a model problem

**[ESTIMATE, from first principles]** From directly overhead, a 3 mm vertical fingertip move is almost purely **along the optical axis**. With a hand ~136 px across at ~0.5-0.7 m, that produces roughly a **0.5% apparent-scale change** — well under one pixel at the fingertip. There is barely any depth signal in the pixels to recover, regardless of model. The *appearance* signal (section 2) is a different matter and survives this argument, which is why PressureVision ranks first and HaMeR does not.

### 4.6 Models that explicitly do NOT apply to us

- **HaWoR** (CVPR'25 Highlight) — https://arxiv.org/abs/2501.02973 · https://github.com/ThunderVVV/HaWoR · **CC-BY-NC-ND** + MANO. World-space hand motion, but derives scale from **camera motion** via DROID-SLAM + Metric3D. **A static overhead phone gives it nothing to SLAM on.**
- **Dyn-HaMR** (CVPR'25 Highlight) — https://github.com/ZhengdiYu/Dyn-HaMR. 4D interacting hands from a **dynamic** camera; optimisation-based, not real-time; static camera is its degenerate case.
- **EgoForce** (May 2026) — https://arxiv.org/pdf/2605.12498 — uses the **forearm** as a metric anthropometric cue to resolve depth-scale ambiguity. Conceptually interesting since our top-down frame may include forearms, but **the PDF was too large to fetch; numbers and code release UNVERIFIED.**
- **GeoHand** — https://arxiv.org/abs/2605.17354 — monocular hand geometry with metric scale. **UNVERIFIED**, listed for completeness only.
- **Meta HOT3D / UmeTrack** — https://arxiv.org/abs/2411.19167 · https://facebookresearch.github.io/hot3d/ · https://github.com/facebookresearch/hand_tracking_toolkit · https://github.com/facebookresearch/UmeTrack_data. **Important caveat: Meta has released the datasets and format, NOT a runnable production hand tracker.** All of it is egocentric multi-view. I found **no Meta release of a monocular metric hand tracker.**
- **Synthetic-to-real gap study** (CVPR'25) — https://arxiv.org/html/2503.19307v1 · https://github.com/delaprada/HandSynthesis. Identifies **forearm presence**, image frequency statistics, pose distribution and occlusion as the factors that close the synth-to-real gap, and shows synthetic-only training can match real data when handled. Relevant because our view is out-of-distribution for every public dataset — this is the recipe if we ever build a synthetic set.

---

## 5. Typing / keystroke / hand-motion-to-text — what actually works and what it cost

### 5.1 Systems that work, and their data bills

| System | Input | Result | Data required |
|---|---|---|---|
| **Meta UIST 2020**, "Decoding Surface Touch Typing from Hand-Tracking" — https://dl.acm.org/doi/10.1145/3379337.3415816 | **Marker-based OptiTrack** 3D hand pose + temporal conv net motion model + language model + beam search | **73 WPM, 2% error** **[REPORTED — ACM paywalled, from search summary]** | studio mocap capture |
| **TouchInsight** (UIST 2024, Meta RL + ETH) — https://arxiv.org/html/2410.05940 | Quest 3, **4 mono wide-FOV VGA cameras @30 Hz**; input is a **92-d per-frame vector** of 3D keypoints at pad/edge/tip of each finger + hand confidence (**landmarks only, no RGB**); **8-layer Transformer-XL with conv layers before attention**, 4-frame left context; **CTC** + cross-entropy + beta-NLL; outputs a **bivariate Gaussian** over touch location fused with an LM prior | touch event **F1 0.99**, finger ID F1 0.96, **6.3 mm** mean spatial error, <3 ms temporal offset, **37.0 WPM @ 2.9% uncorrected error** | **385 participants, 5.29 M unique contact events** (376 train / 9 eval, ~17k touches each) |
| **StegoType** (UIST 2024, Meta) — https://dl.acm.org/doi/10.1145/3654777.3676343 | egocentric cameras, hand-tracking only, end-to-end, closed-loop data collection | **42.4 WPM @ 7% UER** (n=18); physical keyboard baseline **74.5 WPM @ 0.8% UER** | *"a sufficiently diverse training set of hand motions paired with typed text"* — **exact volume COULD NOT CONFIRM (ACM 403)** |
| **Typing on Any Surface** — https://arxiv.org/abs/2309.00174 | **MediaPipe 2D landmarks** + adaptive C-RNN, 27 classes, egocentric RGB | **91.05% accuracy at 40 WPM, ~32 FPS** | dataset size **not stated**; no code or data release found |
| **USENIX Security 2023** video keystroke inference — https://www.usenix.org/conference/usenixsecurity23/presentation/yang-zhuolin | Single commodity phone RGB camera, frontal. Noisy hand tracking -> keystroke detection/clustering -> **language-based HMM assigns letters** -> those become **pseudo-labels** -> trains two 3D-CNNs **on the target's own video, from scratch, no pretraining** | ~90% top-1; CER 0.3-1.1% at 0.8 m; >98% semantic similarity. **Claims it also works on INVISIBLE keyboards.** **[REPORTED — PDF would not extract; numbers from abstract/snippets only. Invisible-keyboard-specific numbers UNVERIFIED.]** | a single target video (exact minutes **UNVERIFIED**) |

**TouchInsight is our system.** Same input class (hand pose from cameras), same architecture class (conv + Transformer + CTC + LM). The differences are (a) Quest 3's multi-camera stereo gives genuine 3D, and (b) **5.29 million labelled contact events versus our ~1 hour**. Our 82-94% / 73-82% is not far off what this literature predicts for our data budget. It shipped as the Quest 3 "Surface Keyboard" in Horizon OS v85 and is **still labelled experimental** — https://www.meta.com/help/quest/1589938228821220/

**Two design ideas to steal from TouchInsight regardless of sensor:**
1. **Stop predicting hard key IDs.** Predict a **2-D Gaussian over contact location plus a press probability**, and fuse with the LM. This is the published answer to "the sensor cannot disambiguate press from hover."
2. Conv-before-attention causal Transformer-XL with a tiny left context — small enough for the M4, and causal, so it can run online.

### 5.2 The negative results (useful, because they bound the naive approach)

- **"Empirical Evaluation of Multi-Modal Touch Detection in Over-the-Shoulder Video Surveillance"** (June 2026) — https://arxiv.org/abs/2606.29504. MediaPipe landmarks + HSV skin + frame differencing + Canny edges. **Motion-only F1 18.5%, edge-only 18.2%, combined 16.7%.** MediaPipe and skin detection *"fail to run autonomously."* On real video: a **median of 57 spurious touch points per frame** (peak 205). Conclusion: *"does not achieve reliable keystroke reconstruction outside the calibrated staged setting."* **Weak evidence** — single-author preprint, 120-frame staged dataset, no peer review, no code — but it independently corroborates that naive RGB press detection fails.
- **"Budget-Aware Keyboardless Interaction"** (June 2026) — https://arxiv.org/html/2606.26508. Closest hardware story to ours: one iPhone 13 Pro Max at 45-60 deg, printed A4 keyboard. YOLOv8n-seg keyboard region **92% AP**, key detection **70%**, and press detection via **fingernail hue-variance**: **~36%** (right index 89% TPR, ring/little 57-67%). No end-to-end typing accuracy. Requires natural light; fails under artificial light. **No code released.** Paper CC BY-NC-SA 4.0. **This is the best citable evidence that press-vs-hover is THE bottleneck.**
- **Open-source bare-desk typing systems:** I searched GitHub specifically. Everything findable is hobby-grade MediaPipe virtual keyboards keyed on fingertip-to-landmark distance (`rishraks/Virtual_Keyboard`, `jatin-cse/virtual-keyboard`, etc.). **There is no serious open-source competitor to what we have built.**

### 5.3 Does adding an RGB crop stream to a landmark stream help? YES — controlled ablation

**HandReader (2025)** — https://arxiv.org/abs/2505.10267 · https://github.com/ai-forever/handreader
Three variants trained head-to-head on identical data: RGB-only (Temporal Shift-Adaptive Module), keypoint-only (Temporal Pose Encoder), and **RGB+KP joint encoder**. **Letter accuracy** **[MEASURED]**:

| Dataset | RGB | Keypoints | **RGB+KP** |
|---|---|---|---|
| ChicagoFSWild | 72.0 | 69.3 | **72.9** |
| ChicagoFSWild+ | 73.8 | 72.4 | **75.6** |
| Znaki (37,252 videos) | 92.39 | 92.65 | **94.94** |

**Fusion wins all three (+0.9 to +2.3 points), and RGB alone beats keypoints alone on both Chicago sets.**
**Licence: CC BY-NC-SA 4.0 (code, weights and the Znaki dataset) — NON-COMMERCIAL.** Use as a design reference, not shipped code.

**This is a lower bound for our case.** In fingerspelling, landmarks already capture the discriminative signal (large-amplitude in-plane motion). Our discriminative signal is sub-pixel and out-of-plane and is *structurally absent* from landmarks. Expect more than 2 points.

Corroborating (suggestive, **not** a controlled comparison — different metrics and decoders): on **FSboard**, Google's own **landmark-only** baseline (MediaPipe Holistic -> ByT5-Small) gets **11.1% CER / 52.9% top-1** (https://arxiv.org/html/2407.15806v1), while **MiCT-RANet-18**, an **RGB** MiCT/ResNet + recurrent visual attention model, reports **92.7% letter accuracy on FSboard** and 77.2% on ChicagoFSWild+ (https://github.com/fmahoudeau/MiCT-RANet-ASL-FingerSpelling — weights downloadable but **NO LICENSE file**, so not safely usable).

**Counter-evidence to weigh honestly — Uni-Sign** (ICLR 2025) — https://arxiv.org/html/2501.15187v1 · https://github.com/ZechengLi19/Uni-Sign · https://huggingface.co/ZechengLi19/Uni-Sign. CSL-Daily CSLR WER **28.2/27.4 pose-only -> 26.7/26.0 pose+RGB** (~1.4 WER); SLT BLEU4 25.27 -> 26.25. Architecture: stage-1 pose-only via spatial GCNs, stage-2 RGB via EfficientNet-B0 + Prior-Guided Fusion with deformable attention, 69 keypoints, into mT5-Base. Pretraining: **CSL-News 1,985 hours** (751,320 clips) + YouTube-ASL 984 hours. **Weights licence cc-by-nc-4.0 — NON-COMMERCIAL.**
**Do not over-read "RGB only adds 1.4 points."** Same argument as above: sign language is large-amplitude in-plane motion. The PressureVision evidence is the relevant prior for us, not the SLR ablation.

### 5.4 Architecture: what beats vanilla conv/transformer CTC on landmark sequences

**Kaggle Google ASL Fingerspelling Recognition 2023, 1st place (Henkel & Hanley)** — https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution · writeup https://www.kaggle.com/competitions/asl-fingerspelling/writeups/darragh-dieter-1st-place-solution-improved-squeeze
- **Encoder: heavily modified Squeezeformer** (original: https://arxiv.org/abs/2206.00888) with the speech front-end swapped for a MediaPipe-landmark front-end. **Decoder: a 2-layer transformer — encoder-decoder, NOT CTC.** Plus an auxiliary **confidence head** to flag corrupted examples.
- Input: **130 landmarks** — 21/hand, 6 pose per arm, 76 face (lips, nose, eyes).
- **The augmentations were reported as decisive: CutMix, FingerDropout, TimeStretch, DecoderInput masking.** On 35 labelled phrases, augmentation is where the wins are.
- PyTorch -> manually ported to TF -> TFLite.
- **Licence: Apache-2.0 on the code — COMMERCIALLY USABLE.**
- **Final leaderboard score: COULD NOT CONFIRM.** **Whether trained weights are distributed: COULD NOT CONFIRM.** Do not quote a score.
- 2nd place writeup benchmarks speech architectures on landmarks: https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434588

**The convergent pattern across the Kaggle winner, emg2qwerty and TouchInsight: local temporal convolutions feeding attention.** Plain transformer-CTC over raw landmarks is the weakest of the three. If we stay with CTC, emg2qwerty says the **backspace-aware n-gram beam search** is doing serious work.

**SignBERT+** (TPAMI 2023) — https://arxiv.org/abs/2305.04868 (earlier: https://arxiv.org/abs/2110.05382) — masked modelling over detector-derived hand pose with **multi-level masking (joint / frame / clip)** explicitly designed to mimic detector failure modes. Directly applicable to MediaPipe dropouts in our desk footage.

**Ignore these:**
- **SignLLM** (https://arxiv.org/abs/2405.10718) is sign *production* (text->pose) and the authors state it **will not be open-sourced**.
- **PHONSSM / "State Space Models are Effective Sign Language Learners"** (https://arxiv.org/abs/2604.08761) claims 72.1% on WLASL2000 from skeleton only. Three-author ICLR 2026 **workshop** paper at a fairness-and-agents workshop, no code found. **Low confidence — do not build on it.**
- **CorrNet+** (https://arxiv.org/pdf/2404.11111) and KD-MSLRT's claimed **16.9% WER on PHOENIX14T** (https://ojs.aaai.org/index.php/AAAI/article/download/35037/37192) — **not verified against the papers' own tables. Indicative only.**

### 5.5 Fingerspelling SOTA vs data volume — the brutal scaling pattern

| Benchmark | Best public number | Approach | Data volume |
|---|---|---|---|
| ChicagoFSWild (7k seqs) | **72.9%** letter acc (HandReader RGB+KP) | fusion | 7k sequences |
| ChicagoFSWild+ (55k seqs) | **77.2%** (MiCT-RANet-18) / 75.6% (HandReader) | RGB attention / fusion | 55k sequences |
| FSboard (266 h, 3.2M chars) | **92.7%** letter acc (MiCT-RANet-18 finetuned) vs **11.1% CER** landmark baseline | RGB vs MediaPipe->ByT5 | 266 h, 147 signers |
| Znaki (37k videos) | **94.94%** (HandReader RGB+KP) | fusion | 37k videos |

ChicagoFSWild: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm

**At 7k sequences you get ~72%; at 266 hours you get ~93%.** Our ~35 labelled desk phrases are **three orders of magnitude below ChicagoFSWild**, and ChicagoFSWild is the weakest row.

### 5.6 Data efficiency — what ~1 hour actually buys, and the number that explains our blind-test gap

**emg2qwerty** (NeurIPS 2024 D&B) — https://arxiv.org/html/2410.20081v3 · https://github.com/facebookresearch/emg2qwerty
108 users, 1,135 sessions, **346 hours**, keylogger ground truth while touch typing. Baseline is *exactly our pipeline shape*: spectral features -> rotation-invariant layer -> **4 time-depth-separable conv blocks (1 s receptive field) -> CTC -> 6-gram Kneser-Ney char LM with backspace-aware beam search**. **[MEASURED]**

- **Generic (cross-user) CER: 51.78% +/- 4.61**
- Personalised **from random init**: **9.55%**
- Personalised by **fine-tuning the generic model**: **6.95% +/- 3.61**
- They state models become usable below ~10% CER.
- **Licence: CC-BY-NC-4.0 for code AND data. Pretrained checkpoints included. NOT commercially usable.**

**This is the single most important number in the survey for diagnosing our labelled-vs-blind gap.** A 52% -> 7% CER swing from *personalisation alone*, on a fixed architecture, says the gap is **domain shift, not model capacity** — and that **fine-tuning a generic model beats per-user training from scratch**.

Corroborating: **Meta, Nature 2025**, "generic neuromotor interface" — https://www.nature.com/articles/s41586-025-09255-w · https://github.com/facebookresearch/generic-neuromotor-interface. 100 participants per task, 51.4-98.8 hours per task; handwriting **20.9 WPM closed-loop**, >90% open-loop held-out-participant classification; **"a small amount of personalization based on limited individual data can improve handwriting recognition accuracy by up to 16%."** Code, data and checkpoints released, **CC-BY-NC-4.0, non-commercial.**

**Self-supervised pretraining on unlabelled hand video helps low-data landmark tasks — this is well established:**
- **OpenHands** (ACL 2022) — https://aclanthology.org/2022.acl-long.150/ · https://arxiv.org/abs/2110.05877. Explicit finding: SSL pretraining on unlabelled pose beats from-scratch, and **the gains are largest when labelled samples per class are fewest**. Also shows **cross-lingual transfer** (pretrain on Indian SL, improve American/Chinese/Argentinian SL) — i.e. the pretraining domain need not closely match the task. Checkpoints for 6 languages released; **licence not confirmed.**
- **SHuBERT** (ACL 2025 oral) — https://arxiv.org/abs/2411.16765 · https://shubert.pals.ttic.edu/ · https://github.com/ShesterG/SHuBERT. Masked cluster prediction over **four streams (two hands, face, body pose)** from **~1,000 hours of unlabelled ASL video**. +10.0 BLEU on OpenASL, +20.6% on SEM-LEX. Crucially, it fine-tunes with **rank-1 LoRA on every linear layer — 0.2% of parameters** — explicitly motivated by *"plentiful unlabeled video for pre-training but very limited parallel data."* **That is our situation exactly, and LoRA-on-frozen-encoder is the right recipe for 35 phrases on a 16 GB M4.** **Repo licence NOT CONFIRMED** (paper is CC BY 4.0).
- **"Self-supervised Learning Matters: A Simple Ensemble Solution for Micro-Gesture Recognition"** (2026) — https://arxiv.org/abs/2606.09261. Masked video modelling on 120k unlabelled clips then fine-tuned; 69.2% top-1 on iMiGUE.
- **CODASPY 2022**, "Leveraging Disentangled Representations to Improve Vision-Based Keystroke Inference Attacks **Under Low Data Constraints**" — https://github.com/jlim13/keystroke-inference-attack-deep-learning (+ synthetic generator https://github.com/jlim13/keystroke-inference-attack-synthetic-dataset-generator-). Literally about our regime: disentangled representations + synthetic data + domain adaptation. **NO LICENSE file** -> not usable commercially. **Accuracy numbers COULD NOT BE EXTRACTED.**

**What I could NOT find: any published data-scaling curve for a landmark->CTC->text system in the 0.5-5 hour labelled regime.** FSboard's "scaling" section is frame-rate and landmark-subset ablations (30->15 Hz costs 0.7 pts CER; dropping face costs 0.9 pts; dropping pose is free) — **not** a labelled-data-volume curve. emg2qwerty explicitly provides none. **So I cannot tell you that 1 hour is enough, and 5.5 suggests it is not for a from-scratch model.**

### 5.7 Released datasets of typing video with keystroke ground truth

**Blunt answer: there are none that match our setup. Not one.**
- **emg2qwerty** — exact keystroke ground truth, **but sEMG, no video**. CC-BY-NC-4.0.
- **TouchInsight's 5.29M-event corpus** — **not released**.
- **KD-MultiModal** — 243.2k frames RGB+D, 20 subjects typing (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10610624/). It is an RGB-D *identification* dataset. **Public download and licence COULD NOT BE CONFIRMED**; not top-down bare-desk.
- **Clarkson University Keystroke Dataset** — https://citer.clarkson.edu/research-resources/biometric-dataset-collections-2/clarkson-university-keystroke-dataset/ — includes video of hand movement, but is a biometrics set behind a data-sharing agreement. **Frame-accurate keystroke alignment and licence NOT VERIFIED.**

**Implication: our ~1 hour of keyboard video with exact keystroke labels is, in licensing terms, an asset nobody else has.**

### 5.8 Other systems, for completeness

- **ATK: Ten-Finger Freehand Typing in Air** (UIST 2015) — https://dl.acm.org/doi/10.1145/2807442.2807504 · https://pi.cs.tsinghua.edu.cn/lab/papers/p539-yi.pdf — 23.0 -> 29.2 WPM, 0.3% uncorrected word error, but with a **Leap Motion 3D hand tracker** and a Bayesian touch-point decoder. Old, no code.
- **TypeAnywhere** (CHI 2022) — https://dl.acm.org/doi/10.1145/3491102.3517686 · https://faculty.washington.edu/wobbrock/pubs/chi-22.03.pdf — **not camera-based** (Tap Strap finger accelerometers). Worth reading only for the decoder: a **BERT-derived neural decoder mapping finger-tap sequences (which finger, not which key) to text**. The right mental model if we can only recover *which finger tapped and roughly where*.
- **Zoom on the Keystrokes** (NDSS 2021) — https://arxiv.org/pdf/2010.12078 — infers from shoulder/arm motion in video calls. No code.
- **RadKey** (IEEE S&P 2026) — https://arxiv.org/abs/2606.10148 — RF backscatter, not video. Relevant for one trick only: **using an LLM's outputs as pseudo-ground-truth to adapt the classifier online**, reducing CER with no victim-specific training data. Directly stealable for our blind sessions.
- **GAZEploit** (CCS 2024) — https://arxiv.org/pdf/2409.08122 — keystrokes from avatar gaze in VR. Irrelevant except as a reminder that people route *around* the visual press problem.
- **Apple Vision Pro** — no hand-tracking surface-typing *research* from Apple found. Only an evaluation study: Arnold, Epperson & Chaparro, https://journals.sagepub.com/doi/10.1177/10711813251367736 (direct tap typing <20 WPM; beats gaze+pinch). Nothing releasable.
- **"I-Keyboard" and "VISAR"** — **I found NO credible primary sources under those names and will not invent them.** The "invisible keyboard at 11 WPM / 3.3% error" figure appears in an XR text-entry survey (https://arxiv.org/pdf/2503.11357) but **I could not trace it to a primary paper.**

---

## 6. Video backbones — what is actually fine-tunable on a 16 GB M4

| Model | Licence | Frozen inference on M4? | **Fine-tune on M4 16 GB?** |
|---|---|---|---|
| **VideoMAEv2 ViT-S / ViT-B** — https://github.com/OpenGVLab/VideoMAEv2 · https://huggingface.co/OpenGVLab/VideoMAE2 | **MIT**, weights freely downloadable (migrated to HF, no request form) **[MEASURED from repo]** | Yes | **YES — the realistic full-fine-tune option.** The best control experiment for "can a video model see the press at all?" |
| **V-JEPA 2.1 ViT-B/16 (80M)** | **UNVERIFIED — check before commercial use** | Yes | **Marginal-to-yes** at reduced frames/resolution. Best 2.x candidate. **[ESTIMATE]** |
| **V-JEPA 2 ViT-L (300M)** | **MIT** | **Yes**, at 16-frame clips and small crops (~512 tokens) | **No** for a full fine-tune. LoRA on attention only: maybe, **untested [ESTIMATE]** |
| **V-JEPA 2 ViT-g / 2.1 ViT-G (1-2B)** | Apache 2.0 / unverified | Inference only, slowly | **No** |
| **InternVideo2.5** — https://github.com/OpenGVLab/InternVideo | MIT repo **[REPORTED]**; released Jan 2025 | — | **No — it is an ~8B video-LLM. Wrong tool entirely.** Also note its InternVideo2 base weights were historically behind a request form. |
| **DINOv3** — https://ai.meta.com/blog/dinov3-self-supervised-vision-model/ | **Meta commercial licence, gated** (requires DOB + approval). **NOT Apache 2.0, unlike DINOv2** — https://ai.meta.com/resources/models-and-libraries/dinov3-license/ **[MEASURED]** | Small variants yes | Small variants yes — **but it is IMAGE-ONLY. No temporal modelling, which is where our signal lives.** |
| **SAM 2** — https://github.com/facebookresearch/segment-anything-2 | **Apache 2.0** (code, weights and the SA-V dataset) **[REPORTED — widely stated, not read off a LICENSE file]** | Yes | Inference/tracking only. **Useful for stable hand-crop tracking across frames, not as a feature backbone.** |
| **PressureVision++ SE-ResNeXt-50 + FPN (~25M)** | **MIT** | Yes | **YES, easily.** |

**Caveat on all M4 claims in this table: no published Apple Silicon benchmark exists for any of these video models. I searched and found none. Every "yes/no" here is my [ESTIMATE] from parameter counts and token arithmetic.**

---

## 7. RANKED SHORTLIST — what to try this week with existing footage on the M4

Ranked by **(measured evidence it addresses OUR bottleneck) x (feasibility on OUR hardware and data)**.

### 1 (joint). Run PressureVision++ over existing footage as a press detector
- **Benefit:** replaces a feature with AUC 0.61 with a model whose published contact accuracy is **89.3%** and which beat human annotators (78.4%).
- **Evidence [MEASURED]:** contact acc 89.3% / 80.5%; tabletop typing **25.8 vs 14.4 net WPM** against a pose-based baseline. https://arxiv.org/html/2301.02310v3
- **Compute:** hours. Weights are a Dropbox download; ~25M-param CNN in `segmentation_models_pytorch`; runs on MPS. Inference only over 1 hour of video. **Zero training.**
- **Licence: MIT.** Clean for product.
- **Main risk:** their camera is 45 deg, ours is 90 deg top-down — out of distribution, unquantified. **And their own typing study fell back to index-finger-only because of fingertip occlusion in five-finger typing.** This may simply not transfer. Which is exactly why it is a one-day test, not a bet.
- **Decisive measurement:** press-vs-hover AUC on our 35 labelled desk phrases vs the current 0.61.

### 1 (joint). Add an RGB fingertip-crop stream to the existing CTC model
- **Benefit:** our pipeline discards every appearance cue — nail blanching, pad deformation, contact shadow — at the landmark bottleneck.
- **Evidence [MEASURED]:** (a) HandReader's controlled ablation, RGB+KP beats KP-only on all three datasets by +0.9 to +2.3 letter-accuracy points, and RGB-only beats KP-only on two of three (https://arxiv.org/abs/2505.10267); (b) PressureVision's cue analysis — *"depends on the appearance of the hand and cast shadows near contact regions"* (https://arxiv.org/abs/2203.10385); (c) the hand-crafted nail-hue detector reaching 36% with no learning (https://arxiv.org/html/2606.26508).
- **Compute:** near zero. Small CNN over 5 fingertip crops per hand per frame, concatenated to existing landmark features. Trains in minutes on the M4.
- **Licence:** none needed if we train our own crop encoder. **Do not copy HandReader code — CC BY-NC-SA.**
- **Main risk:** 1 hour of labels may be too little to learn the cue from scratch; overfits to our lighting/skin/desk. **Mitigation: initialise from PressureVisionDB — MIT, 36 participants, 4 cameras, 16 hours, 140 GB — which is precisely the pretraining set for this.**
- Complementary to #1, not an alternative.

### 3. Per-session self-labelling / personalisation loop
- **Benefit:** attacks the labelled-vs-blind gap directly rather than the per-frame signal.
- **Evidence [MEASURED]:** emg2qwerty, same pipeline shape: generic CER **51.78% -> 6.95%** by fine-tuning on one user; from-scratch personalisation only reaches 9.55%. https://arxiv.org/html/2410.20081v3. Corroborated by Meta's Nature 2025 result (up to **+16%** from small-scale personalisation). **Mechanisms to copy:** USENIX'23 bootstraps pseudo-labels from the target's own unlabelled video via clustering + a language-model HMM, **with no pretraining at all**; RadKey uses LLM output as pseudo-labels for online adaptation.
- **Compute:** trivial. No new models.
- **Licence:** none — we are copying a method, not code.
- **Main risk:** pseudo-label feedback loops reinforce their own errors. Needs a confidence gate. Also, USENIX'23's numbers are **[REPORTED]** only — the PDF would not extract and no code release could be confirmed. Take the mechanism, not the numbers.

### 4. Predict a Gaussian over contact location + a press probability, instead of hard key IDs
- **Benefit:** this is the published answer to "the sensor cannot disambiguate press from hover" — move the ambiguity into a distribution and let the LM resolve it.
- **Evidence [MEASURED]:** TouchInsight outputs a bivariate Gaussian fused with an LM prior and reaches touch F1 0.99, 6.3 mm, 37.0 WPM @ 2.9% UER. https://arxiv.org/html/2410.05940
- **Compute:** small — a head and loss change (beta-NLL) on the existing model.
- **Licence:** paper is CC BY-NC-ND, but we are reimplementing a described method, not using their code (none released).
- **Main risk:** TouchInsight had 5.29M contact events to calibrate that uncertainty. With 35 phrases the covariance estimate may be garbage. Mitigate by starting with a fixed isotropic covariance and only learning it if data allows.

### 5. Squeezeformer encoder + the ASL-fingerspelling augmentation recipe
- **Benefit:** the Kaggle ASL Fingerspelling winner solves the same problem shape (MediaPipe landmark sequences -> text) and credits **CutMix, FingerDropout, TimeStretch, DecoderInput masking** as decisive. In a 1-hour regime, augmentation is worth more than architecture. The convergent pattern across three independent systems is conv-before-attention.
- **Evidence:** competition-winning, but on a different task — **no transfer measurement exists [UNVERIFIED]**. Leaderboard score could not be confirmed.
- **Compute:** a day or two; trains on the M4.
- **Licence: Apache-2.0.** Clean.
- **Main risk:** the bottleneck is visual, not the decoder — cap expectations at a few points. Cheap enough to do anyway.

### 6. Masked-landmark SSL pretraining on unlabelled desk footage + LoRA fine-tune on the 35 phrases
- **Benefit:** operates on features we already have, fits the M4, and has direct low-data evidence.
- **Evidence [MEASURED]:** SHuBERT fine-tunes with **rank-1 LoRA, 0.2% of parameters**, explicitly for "plentiful unlabeled video, very limited parallel data" (https://arxiv.org/abs/2411.16765); OpenHands shows SSL gains are **largest when labels are scarcest**, and transfer works across sign languages (https://aclanthology.org/2022.acl-long.150/); SignBERT+'s joint/frame/clip masking is designed for detector dropouts, which is exactly MediaPipe's failure mode on a desk (https://arxiv.org/abs/2305.04868).
- **Compute:** moderate — SSL pretraining on our own unlabelled footage, then LoRA. Feasible on M4 at landmark-sequence scale (not video-pixel scale).
- **Licence:** methods only. For *supervised* pretraining data we can legally ship, **FSboard is CC BY 4.0, 266 hours, 147 signers, 30 fps, MediaPipe landmarks already extracted** — https://www.kaggle.com/datasets/googleai/fsboard. **It is the only large, commercially usable hand-motion corpus in this entire survey.**
- **Main risk:** we have little unlabelled desk footage either. The SSL corpus may be too small to matter.

### 7. V-JEPA 2.1 (or V-JEPA 2 ViT-L) as a FROZEN feature extractor over hand crops, as an extra stream
- **Benefit:** best-in-class frozen motion features on hand-manipulation-heavy benchmarks (SSv2 77.3%, EK100 39.7 R@5), and 2.1 fixes the spatial weakness (NYUv2 RMSE 0.642 -> 0.307).
- **Compute:** frozen extraction at 16 frames x 128-192 px crops ~= 500-1500 tokens/clip — **feasible on the M4**. One-time pass over 1 hour, cache features, then train only the head. **[ESTIMATE] budget roughly a day of extraction.** Pre-decode to frames with ffmpeg to dodge `decord`.
- **Licence:** V-JEPA 2 **MIT / Apache 2.0** — clean. **V-JEPA 2.1 weights licence UNVERIFIED.**
- **Main risks:** (a) **no published evidence V-JEPA features separate press from hover**; (b) ~1 hour of labels vs probes normally trained on 100k+ clips, and **V-JEPA 2's low-shot curves do not exist** (that was V-JEPA 1); (c) `decord` does not build on macOS; (d) **fine-tuning the encoder is off the table on 16 GB.**
- **Ranked 7th deliberately: it is the most fashionable option and the least evidenced one for THIS bottleneck.**

### 8. VideoMAEv2 ViT-S/B fine-tuned on hand crops for binary press/hover
- The only video backbone here we can genuinely **fine-tune end-to-end** on a 16 GB M4, and it is **MIT** with freely downloadable weights. Worth one run as the control experiment against #7's frozen-probe result.

### 9. RTMPose-hand instead of MediaPipe, for cleaner 2D only
- **Benefit:** COCO-WholeBody-Hand **EPE 4.51 px** vs MediaPipe's ~10.09% MNAE (~14 px at our 136 px hand scale, and worst at the fingertips). Better temporal stability into the CTC model.
- **Compute:** low. **Apache 2.0**, ONNX -> CoreML via `rtmlib`, 90+ FPS on CPU.
- **Main risk: strictly 2D. Gives no height. Reduces noise; does not create signal.**

---

## 8. Explicitly NOT recommended

- **HaMeR / WiLoR / HandOccNet / Fast-HaMeR.** We would trade Apache 2.0 for research-only (MANO) or **CC-BY-NC-ND** licences, 60 fps for an **[ESTIMATE]** of well under 10 FPS on M4 MPS (no Apple benchmark exists for any of them), and get a depth axis the field's own papers describe as **6-7x worse than root-relative error** — measured on datasets that look nothing like a top-down desk. **Expect a null result.**
- **HandDGP / ScaleHP** — only if we want the metric-depth experiment for completeness. Go in expecting **35-50 mm absolute error against a 2-4 mm signal.**
- **HaWoR / Dyn-HaMR** — structurally inapplicable (they need camera motion).
- **InternVideo2.5** — an 8B video-LLM; wrong tool.
- **DINOv3** — image-only, and its licence is gated and non-standard.

---

## 9. Things that would need data or hardware we do not have — say these out loud

1. **MOVE THE CAMERA to 30-45 degrees.** This is almost certainly the single largest win available and it costs one camera repositioning. Every piece of evidence points the same way: from directly overhead a keypress is a sub-pixel, optical-axis event **[ESTIMATE, section 4.5]**; PressureVision (45 deg), PressureVision++ (45 deg), the Budget-Aware paper (45-60 deg), TouchInsight and StegoType (egocentric) **all** use oblique or egocentric views. An oblique view converts fingertip height into an **in-plane** quantity that MediaPipe's existing 2D accuracy can already resolve. **But it needs new recording.**
2. **More data.** TouchInsight needed **385 participants and 5.29M contact events** to reach 37 WPM @ 2.9% from camera hand-tracking. StegoType's own framing is that a *"sufficiently diverse training set"* is what makes it work. **No backbone swap substitutes for this.** The fingerspelling scaling table (7k seqs -> ~72%; 266 h -> ~93%) is the same message.
3. **Stereo or a second camera.** Every system that works (Quest 3's four cameras, OptiTrack, Leap Motion) has genuine 3D. Depth/LiDAR was tested and rejected here for good reason, but a **second cheap phone** is a different proposition from a depth sensor and does not cost the wide lens.
4. **EgoPressure** (5 h, 21 participants, 7 static third-person views + pressure GT + MANO meshes) is the best public dataset for pretraining a contact head — **but CC BY-NC-SA 4.0, non-commercial.** Fine for research, blocked for product. **PressureVisionDB (MIT) is the commercially usable one.**

---

## 10. Licence summary — the commercial-use picture

**Uncomfortable summary: nearly every high-quality released model and dataset in this space is CC-BY-NC or unlicensed.**

**Commercially usable (MIT / Apache 2.0 / CC BY 4.0):**
- **PressureVision + PressureVisionDB** — MIT, code + weights + 16 h dataset
- **PressureVision++** — MIT, code + weights (no dataset)
- **V-JEPA 2** — MIT (ViT-L@256) / Apache 2.0 (ViT-g@384)
- **VideoMAEv2** — MIT, weights on HF
- **SAM 2** — Apache 2.0 **[REPORTED]**
- **RTMPose / MMPose** — Apache 2.0
- **MediaPipe** — Apache 2.0
- **Kaggle ASL Fingerspelling 1st place (code)** — Apache 2.0
- **FSboard dataset** — CC BY 4.0, 266 h — *the only large commercially usable hand-motion corpus found*
- **Fast-HaMeR** — CC BY 4.0
- HaMeR *code* — MIT, but **its weights need MANO and are research-only**

**NOT commercially usable:**
- WiLoR (CC-BY-NC-**ND**, code and models) · HaWoR (CC-BY-NC-ND) · HandReader + Znaki (CC BY-NC-SA) · Uni-Sign (cc-by-nc) · emg2qwerty (CC-BY-NC) · Meta generic-neuromotor-interface (CC-BY-NC) · EgoPressure (CC BY-NC-SA) · SignLLM (NC, and will not be open-sourced) · all MANO-dependent weights

**No licence file at all — treat as all-rights-reserved:** HandOccNet · MiCT-RANet · CODASPY'22 keystroke repos

**Licence could not be confirmed:** V-JEPA 2.1 weights · HandDGP · ScaleHP · SHuBERT repo · OpenHands · Uni-Sign GitHub (only the HF weights licence is confirmed)

---

## 11. Consolidated "could not verify" list — read before quoting anything

- The official TypeSafe Jev pages (`typesafe.ai`, `docs.typesafe.ai`) — not fetched directly.
- **V-JEPA 2.1 weights licence.**
- Any V-JEPA 2 low-shot / label-efficiency curve — **it does not exist in the V-JEPA 2 paper.** (The 5%/10%/50% result belongs to V-JEPA 1.)
- Any Apple Silicon / M4 / MPS benchmark for **any** model in this survey. **None exists publicly.** All M4 feasibility claims are my estimates.
- SALT (arXiv:2509.24317) code/weights release.
- USENIX'23 keystroke-attack exact numbers, training-data requirements, invisible-keyboard-specific results, and whether code exists at all (project site failed TLS; no GitHub found).
- Kaggle ASL 1st place final leaderboard score and whether trained weights are published.
- StegoType's training-data volume (ACM returned 403 on both the PDF and the fullHtml).
- Meta UIST 2020's exact numbers — **[REPORTED]** from a search summary, ACM paywalled.
- "Typing on Any Surface" dataset size; no code found.
- Uni-Sign's Phoenix14T / CSL-Daily WER; Uni-Sign GitHub licence.
- SHuBERT repo licence and its specific low-data ablations.
- HandOccNet's numbers read off the paper's own tables (secondary summaries only) and its licence.
- HandDGP and ScaleHP code/weights availability; ScaleHP speed.
- EgoForce numbers and code (PDF too large to fetch); GeoHand entirely.
- KD-MultiModal and Clarkson keystroke-video datasets: availability, licence, frame alignment.
- **"I-Keyboard" and "VISAR" — no credible primary sources found under those names. Not invented here.** The "invisible keyboard, 11 WPM, 3.3% error" figure appears in a survey (https://arxiv.org/pdf/2503.11357) but could not be traced to a primary paper.
- 2026 arXiv IDs surfaced by search but **not verified and not relied upon**: VRSafe (2604.21001), SurfaceXR (2603.19529), TouchSight (2609.20414), HOPE (2608.06192), EgoPressDiff (2606.06872), EgoTactile (2606.09243).
- CorrNet+ / KD-MSLRT WER figures (indicative, not read off the papers' tables).
- PHONSSM (arXiv:2604.08761) — workshop paper, no code, low confidence.
