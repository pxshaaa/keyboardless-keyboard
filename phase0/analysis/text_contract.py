"""Explicit alphabets and literal metrics for new experiments; legacy checkpoints stay unchanged."""
from __future__ import annotations

import argparse
import json
import string
import unicodedata
from pathlib import Path

LEGACY_TOKENS = ("<blank>", *string.ascii_lowercase, " ", "<other>")
LITERAL_TOKENS = (*LEGACY_TOKENS, *string.ascii_uppercase, *string.digits,
                  *"äöüÄÖÜß", *string.punctuation)


def literal(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n"))


def encode(text: str, tokens=LITERAL_TOKENS) -> list[int]:
    lookup = {c: i for i, c in enumerate(tokens) if len(c) == 1}
    text = literal(text)
    missing = sorted(set(text) - lookup.keys())
    if missing:
        raise ValueError(f"Unsupported characters: {missing!r}")
    return [lookup[c] for c in text]


def legacy_training_text(text: str) -> str:
    text = " ".join(literal(text).lower().split())
    encode(text, LEGACY_TOKENS)
    if not text:
        raise ValueError("Empty training label")
    return text


def edits(ref, hyp) -> int:
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        nxt = [i]
        for j, b in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def metrics(ref: str, hyp: str) -> dict:
    ref, hyp = literal(ref), literal(hyp)
    ce, we = edits(ref, hyp), edits(ref.split(), hyp.split())
    return {"contract": "literal-nfc-v1", "char_edits": ce, "reference_chars": len(ref),
            "word_edits": we, "reference_words": len(ref.split()),
            "cer": ce / len(ref) if ref else None,
            "wer": we / len(ref.split()) if ref.split() else None, "exact": ref == hyp}


def expand_head(model, tokens=LITERAL_TOKENS):
    import torch
    from torch import nn
    old = model.head
    if tuple(tokens[:29]) != LEGACY_TOKENS or len(set(tokens)) != len(tokens):
        raise ValueError("Expanded alphabet must preserve legacy indices and contain unique tokens")
    if old.out_features != len(LEGACY_TOKENS):
        raise ValueError("Expected an unchanged 29-symbol legacy head")
    head = nn.Linear(old.in_features, len(tokens), device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        head.weight[:old.out_features].copy_(old.weight)
        head.bias[:old.out_features].copy_(old.bias)
        head.weight[old.out_features:].zero_()
        head.bias[old.out_features:].fill_(-12)
    model.head = head
    model.kw = dict(model.kw, nsym=len(tokens))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("hypothesis", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = metrics(a.reference.read_text().rstrip("\r\n"), a.hypothesis.read_text().rstrip("\r\n"))
    output = json.dumps(result, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
