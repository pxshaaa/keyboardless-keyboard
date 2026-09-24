# Finger attribution & keystroke count — research gate (finger_*.py)

**Gate: NO-GO.** The swipe-path idea stays shelved.

- Finger truth on keyboard (grid-consistent key/fingertip solution, 3,185 keystrokes): I touch-type — 94.1% of keystrokes use the touch-typing finger; least consistent keys l (0.51), u (0.76), k (0.78), b (0.80).
- Label-free finger classifier (LightGBM on kinematics, keyboard LOSO): top-1 0.44–0.48, top-2 0.68–0.70, hand 0.66–0.69. With true tap positions/counts on keyboard it adds +15 pts word top-1 (0.409→0.560).
- Desk (truth-aligned anchors, optimistic): finger agreement 0.54–0.57 vs chance 0.14–0.17 — the classifier transfers.
- Keystroke count per word span: kinematics add nothing over CTC; CTC + duration 0.684 exact / 0.964 ±1.
- Fusion into the swipe scorer on desk (249 words, CTC + prior baseline top-1 0.715): best finger arm +2.4 [−0.4, +5.6]; shuffled/uniform/constant controls match or beat it; the only ≥ +5 pt arms come from a duration term measured on truth-aligned spans (leak). Why: label-free CTC peaks miscount and fingertip positions at those peaks carry no key information after desk mapping.

Caveats: one user, 249 desk words; blind-1 not blind; finger truth for l/k/u/i/b uncertain. Results `results/finger/`, caches `.cache/finger/`.
