"""FSboard (ASL fingerspelling, CC-BY-4.0) hand landmarks -> seqctc clips, via a minimal protobuf decoder.
Run: python -m phase0.analysis.fsboard build [--shards N]"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from phase0.analysis.seqctc import CACHE, OTHER, SPACE_S, SYMS, normalise

FSB = CACHE / "fsb"
CLIPS = FSB / "clips.npz"
WANT = {f"{c}_{h}_hand" for c in "xyz" for h in ("left", "right")}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        if c < 0x80:
            return v, i
        s += 7


def _fields(b: bytes):
    """Yield (field_number, wire_type, value) where value is bytes for length-delimited fields."""
    i, n = 0, len(b)
    while i < n:
        tag, i = _varint(b, i)
        f, wt = tag >> 3, tag & 7
        if wt == 2:
            ln, i = _varint(b, i)
            yield f, wt, b[i:i + ln]
            i += ln
        elif wt == 0:
            v, i = _varint(b, i)
            yield f, wt, v
        elif wt == 5:
            yield f, wt, b[i:i + 4]
            i += 4
        elif wt == 1:
            yield f, wt, b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"wire type {wt}")


def _feature(b: bytes):
    for f, _, v in _fields(b):
        if f == 1:
            return [x for g, _, x in _fields(v) if g == 1]
        if f == 2:
            out = []
            for g, wt, x in _fields(v):
                out.append(np.frombuffer(x, "<f4") if wt == 2 else np.frombuffer(x, "<f4"))
            return np.concatenate(out) if out else np.zeros(0, np.float32)
        if f == 3:
            vals = []
            for g, wt, x in _fields(v):
                if wt == 2:
                    j = 0
                    while j < len(x):
                        y, j = _varint(x, j)
                        vals.append(y)
                else:
                    vals.append(x)
            return vals
    return None


def _map(b: bytes):
    key = val = None
    for f, _, v in _fields(b):
        if f == 1:
            key = v.decode()
        elif f == 2:
            val = v
    return key, val


def parse_example(blob: bytes) -> dict:
    ctx, lists = {}, {}
    for f, _, v in _fields(blob):
        if f == 1:
            for _, _, e in _fields(v):
                k, val = _map(e)
                ctx[k] = _feature(val)
        elif f == 2:
            for _, _, e in _fields(v):
                k, val = _map(e)
                if k in WANT:
                    lists[k] = [_feature(x) for g, _, x in _fields(val) if g == 1]
    return {"ctx": ctx, "lists": lists}


def clip_arrays(ex: dict, fps_out: float = 60.0):
    """-> (A[T,2,21,3] normalised, M[T,2], symbols) or None; image-up becomes 'forward'."""
    ctx, L = ex["ctx"], ex["lists"]
    prompt = ctx.get("prompt")
    if not prompt or "x_left_hand" not in L:
        return None
    text = prompt[0].decode().lower()
    w = float(ctx.get("image/width", [1])[0])
    h = float(ctx.get("image/height", [1])[0])
    fps = float(ctx["image/frame_rate"][0]) if "image/frame_rate" in ctx else 30.0
    T = len(L["x_left_hand"])
    P = np.full((T, 2, 21, 3), np.nan, np.float32)
    for k, hand in enumerate(("left", "right")):
        for t in range(T):
            x, y, z = (L[f"{c}_{hand}_hand"][t] for c in "xyz")
            if x is None or len(x) != 21 or np.isnan(x).all():
                continue
            P[t, k, :, 0] = -y * h
            P[t, k, :, 1] = x * w
            P[t, k, :, 2] = z * w
    if np.isnan(P[..., 0]).all(axis=(1, 2)).mean() > 0.8:
        return None
    n = max(8, int(round(T * fps_out / fps)))
    src = np.clip(np.round(np.linspace(0, T - 1, n)).astype(int), 0, T - 1)
    A, M = normalise(P[src])
    syms = np.array([SYMS.index(c) if c in SYMS[1:27] else SPACE_S if c == " " else OTHER for c in text],
                    np.int64)
    return A.astype(np.float16), M, syms, text


def cmd_build(a) -> int:
    files = sorted(glob.glob(str(FSB / "daun_v3-train.arrow-*[0-9]")))[:a.shards]
    As, Ms, Ss, texts = [], [], [], []
    for fn in files:
        tb = feather.read_table(fn, columns=["serialized"])
        nclip = 0
        for chunk in tb.column("serialized").chunks:
            for blob in chunk.to_pylist():
                r = clip_arrays(parse_example(blob))
                if r is None or len(r[2]) == 0 or len(r[0]) < 2 * len(r[2]):
                    continue
                As.append(r[0])
                Ms.append(r[1])
                Ss.append(r[2])
                texts.append(r[3])
                nclip += 1
        print(f"{Path(fn).name}: {nclip} clips", flush=True)
    off = np.cumsum([0] + [len(x) for x in As])
    soff = np.cumsum([0] + [len(x) for x in Ss])
    np.savez(CLIPS, A=np.concatenate(As), M=np.concatenate(Ms), off=off, S=np.concatenate(Ss), soff=soff,
             text=np.array(texts))
    print(f"{len(As)} clips, {off[-1] / 3600 / 60:.2f} h, {soff[-1]} chars -> {CLIPS}")
    return 0


class ClipSource:
    """Whole FSboard clips as seqctc training samples (speed/geometry augmentation happens in train)."""

    def __init__(self, path: Path = CLIPS, idx=None):
        z = np.load(path)
        self.A, self.M, self.off, self.S, self.soff = z["A"], z["M"], z["off"], z["S"], z["soff"]
        self.idx = np.arange(len(self.off) - 1) if idx is None else np.asarray(idx)

    def draw(self, rng):
        k = self.idx[rng.integers(len(self.idx))]
        a, b = self.off[k], self.off[k + 1]
        st = _ClipStream(self.A[a:b].astype(np.float32), self.M[a:b])
        return st, 0.0, (b - a) / 60.0, self.S[self.soff[k]:self.soff[k + 1]]


class _ClipStream:
    def __init__(self, A, M):
        self.A, self.M = A, M
        self.t = np.arange(len(A)) / 60.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--shards", type=int, default=4)
    b.set_defaults(fn=cmd_build)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
