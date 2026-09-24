"""How We Type (Feit et al., CHI'16, CC-BY-NC-4.0) mocap -> seqctc Streams in MediaPipe joint order.
Run: python -m phase0.analysis.howwetype extract"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv

from phase0.analysis.seqctc import CACHE, OTHER, SPACE_S, SYMS, Stream, normalise

HWT = CACHE / "hwt"
ZIP = HWT / "mocap.zip"
OUT = HWT / "streams"
FINGERS = ("T", "I", "M", "R", "L")
MP_MARKERS = [None] + [f"{f}{k}" for f in FINGERS for k in (1, 2, 3, 4)]  # joint 0 = mean(Win, Wout)
MODS = {"Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Multi_key", "ISO_Level3_Shift"}
SWAP = {"y": "z", "z": "y"}  # Finnish QWERTY -> the user's QWERTZ by physical key position


def key_sym(k: str) -> int | None:
    if k in MODS:
        return None
    if len(k) == 1 and "a" <= k.lower() <= "z":
        return SYMS.index(SWAP.get(k.lower(), k.lower()))
    return SPACE_S if k == "space" else OTHER


def _fill(v: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs of at most max_gap samples."""
    ok = ~np.isnan(v)
    if ok.all() or ok.sum() < 2:
        return v
    idx = np.arange(len(v))
    filled = np.interp(idx, idx[ok], v[ok])
    run_start = np.maximum.accumulate(np.where(ok, idx, -1))
    run_end = np.minimum.accumulate(np.where(ok, idx, len(v))[::-1])[::-1]
    short = (~ok) & (run_start >= 0) & (run_end < len(v)) & (run_end - run_start - 1 <= max_gap)
    return np.where(ok | short, filled, np.nan)


def parse_csv(raw: bytes, step: int = 4) -> Stream:
    tb = pcsv.read_csv(io.BytesIO(raw), read_options=pcsv.ReadOptions(skip_rows=2),
                       parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
                       convert_options=pcsv.ConvertOptions(null_values=["NaN", ""], strings_can_be_null=True))
    cols = set(tb.column_names)
    t = tb.column("time").to_numpy().astype(np.float64)

    def xyz(name):
        arr = [tb.column(f"Hands_{name}_{c}").cast(pa.float64()).to_numpy(zero_copy_only=False)
               if f"Hands_{name}_{c}" in cols else np.full(len(t), np.nan) for c in "xyz"]
        return np.stack([_fill(a, 60) for a in arr], -1)

    P = np.full((len(t), 2, 21, 3), np.nan)
    for k, side in enumerate("LR"):
        P[:, k, 0] = (xyz(f"{side}_Win") + xyz(f"{side}_Wout")) / 2
        for j, m in enumerate(MP_MARKERS[1:], 1):
            P[:, k, j] = xyz(f"{side}_{m}")
    Q = np.empty_like(P)
    Q[..., 0] = P[..., 2]                          # forward
    Q[..., 1] = P[..., 0]                          # user's right
    Q[..., 2] = -(P[..., 1] - P[:, :, :1, 1])      # MediaPipe-like z: wrist-relative, negative = higher
    keys = tb.column("key_symbol").to_pylist()
    kd = [(t[i], s) for i, k in enumerate(keys) if k is not None and (s := key_sym(k)) is not None]
    sel = np.arange(0, len(t), step)
    A, M = normalise(Q[sel].astype(np.float32))
    return Stream("", t[sel], A, M, np.array([x for x, _ in kd]), np.array([s for _, s in kd], np.int64))


def cmd_extract(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".csv"):
                continue
            uid = Path(info.filename).name.split("_")[0]
            f = OUT / f"{uid}.npz"
            if f.exists():
                continue
            st = parse_csv(z.read(info))
            st.save(f)
            print(f"{uid}: {len(st.t)} frames ({len(st.t) / 3600:.1f} min @60), {len(st.kt)} keys, "
                  f"hands present {st.M.mean():.2f}, letters {(st.ks < SPACE_S).sum()}", flush=True)
    return 0


def streams(uids=None) -> list[Stream]:
    return [Stream.load(p.stem, p) for p in sorted(OUT.glob("*.npz")) if uids is None or p.stem in uids]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract").set_defaults(fn=cmd_extract)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
