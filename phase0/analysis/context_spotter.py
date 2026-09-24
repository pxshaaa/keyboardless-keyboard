"""Local CTC context search independent of sentence beams; outputs candidates, not certified corrections."""
from __future__ import annotations

import numpy as np

from phase0.analysis.text_contract import legacy_training_text


def spot(M: np.ndarray, term: str, max_cost=3.0, topk=3) -> list[dict]:
    term = legacy_training_text(term)
    if topk < 1 or max_cost < 0:
        raise ValueError("topk must be positive and max_cost nonnegative")
    if M.ndim != 2 or M.shape[1] != 28 or not len(M):
        raise ValueError("Expected nonempty [frames,28] log probabilities")
    if np.isnan(M).any() or np.isposinf(M).any() or not np.isfinite(M.max(1)).all():
        raise ValueError("Invalid frame probabilities")
    syms = "_abcdefghijklmnopqrstuvwxyz "
    ext = np.zeros(2 * len(term) + 1, dtype=int)
    ext[1::2] = [syms.index(c) for c in term]
    n = len(ext)
    skip = np.zeros(n, bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    prev, starts = np.full(n, -np.inf), np.zeros(n, int)
    prev[0] = 0
    hits = []
    for t, row in enumerate(M - M.max(1, keepdims=True)):
        options = np.stack((prev, np.r_[-np.inf, prev[:-1]],
                            np.where(skip, np.r_[-np.inf, -np.inf, prev[:-2]], -np.inf)))
        choice = options.argmax(0)
        source = np.maximum(0, np.arange(n) - choice)
        starts = starts[source]
        prev = options[choice, np.arange(n)] + row[ext]
        prev[0], starts[0] = 0, t + 1
        cost = -prev[-2] / len(term)
        if cost <= max_cost:
            hits.append({"term": term, "start": int(starts[-2]), "end": t + 1, "cost": float(cost)})
    selected = []
    for h in sorted(hits, key=lambda h: (h["cost"], h["end"] - h["start"])):
        if any(h["start"] < x["end"] and x["start"] < h["end"] for x in selected):
            continue
        selected.append(h)
        if len(selected) == topk:
            break
    return selected


def candidates(M: np.ndarray, baseline: str, terms: list[str], max_cost=3.0) -> list[dict]:
    from phase0.analysis.swipe_common import letter_frames
    baseline = legacy_training_text(baseline)
    frames = letter_frames(M, baseline)
    spans, offset = [], 0
    for word in baseline.split():
        ff = [f for f in frames[offset:offset + len(word)] if f[0] >= 0]
        spans.append((min(f[0] for f in ff), max(f[1] for f in ff) + 1) if ff else (-1, -1))
        offset += len(word) + 1
    words = baseline.split()
    out = {baseline: {"text": baseline, "source": "baseline"}}
    for term in dict.fromkeys(legacy_training_text(t) for t in terms):
        for hit in spot(M, term, max_cost):
            overlap = [i for i, (a, b) in enumerate(spans)
                       if a >= 0 and max(0, min(b, hit["end"]) - max(a, hit["start"])) / max(1, b - a) >= 0.5]
            if not overlap or len(overlap) > 3:
                continue
            lo, hi = min(overlap), max(overlap) + 1
            text = " ".join(words[:lo] + term.split() + words[hi:])
            if text not in out:
                out[text] = {"text": text, "source": "context_spotter", "hit": hit, "word_range": [lo, hi]}
    return list(out.values())
