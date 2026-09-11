"""Push the extractor past combine.py's 1.13 taps/char and find where density stops paying.
Run: python -m phase0.analysis.hirecall {streams | model | bank | pix | table | cv | curve}"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis import adapt as ad
from phase0.analysis import combine as CB
from phase0.analysis import desk_tune as dt
from phase0.analysis import pipeline as pl
from phase0.analysis import tap_pos as tp
from phase0.analysis.decode import edit_distance, wer
from phase0.analysis.tap_pos import H as FRAME_H
from phase0.analysis.tap_pos import W as FRAME_W

HR_CACHE = Path(".cache/hirecall")

DESK = pl.DESK
KBD_TRAIN = pl.KBD_TRAIN
MODE = "none"       # unmasked pixels: combine.py's best expert
RECALL_MODEL = HR_CACHE / "taps_gb_recall.joblib"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
OBS = (1.0, 1.5, 2.0, 2.5, 3.0)
DELETIONS = (-9.0, -5.0, -3.0, -2.0, -1.0)

THR = (0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75)
GRID = {"smooth": (1, 3), "gate_thr": (0.0, 0.25, 0.45), "refractory": (1, 3, 4, 6, 8),
        "n_consec": (1, 2), "thr": THR, "nms": ("peak", "shoulder")}
BAND = (0.55, 3.6)
N_STREAMS = 90
BINS = (0.55, 0.85, 1.0, 1.15, 1.35, 1.6, 1.9, 2.3, 2.8, 3.6)


# ------------------------------------------------------------------ frame probabilities
_P: dict = {}


def probs(src: str, session: str = DESK) -> dict:
    """-> desk per-frame tap probability and gate from `src` ('base' or 'recall')."""
    key = (src, session)
    if key not in _P:
        s = pl.session_path(session)
        d = pl.frame_probs(s) if src == "base" else dt.probs(s, RECALL_MODEL)
        _P[key] = {"p": d["p"], "gate": d["gate"], "t": d["t"]}
    return _P[key]


# ------------------------------------------------------------------ relaxed peak picking
def shoulders(p: np.ndarray) -> np.ndarray:
    """Flank inflections: where two taps merge into one peak the second never dips to a
    local minimum, so strict non-maximum suppression can only ever emit one of them."""
    d = np.diff(p)
    a, b, c = d[:-2], d[1:-1], d[2:]
    rise = (b > 0) & (a > 0) & (b < a) & (b <= c)
    fall = (b < 0) & (c < 0) & (b > a) & (b >= c)
    return np.where(rise | fall)[0] + 2


def _nms(cand: np.ndarray, h: np.ndarray, rf: int, n: int) -> np.ndarray:
    """Greedy highest-first suppression, so shoulder candidates obey the same refractory."""
    if rf <= 1 or not len(cand):
        return np.sort(cand)
    blocked = np.zeros(n, bool)
    keep = []
    for i in cand[np.argsort(-h, kind="stable")]:
        if blocked[i]:
            continue
        keep.append(int(i))
        blocked[max(0, i - rf + 1):i + rf] = True
    return np.array(sorted(keep), int)


def pick(pg: np.ndarray, thr: float, rf: int, nc: int, nms: str) -> np.ndarray:
    if nms == "peak":
        idx, pr = find_peaks(pg, height=thr, distance=max(1, rf))
    else:
        c = np.union1d(find_peaks(pg, height=thr)[0], shoulders(pg))
        c = c[pg[c] >= thr]
        idx = _nms(c, pg[c], rf, len(pg))
    if nc > 1 and len(idx):
        idx = idx[dt.run_length(pg >= thr * 0.6)[idx] >= nc]
    return np.asarray(idx, int)


def candidate_streams(src: str = "base", grid: dict = GRID, band=BAND,
                      session: str = DESK) -> list[tuple[dict, np.ndarray]]:
    """Every grid point, deduped by the event set it produces and kept inside a taps/char
    band that reaches far above the 1.13 ceiling combine.py's grid could not pass."""
    d = probs(src, session)
    nch = nchars(session)
    seen, out = set(), []
    for sm in grid["smooth"]:
        ps = dt.smooth(d["p"], sm)
        for gt in grid["gate_thr"]:
            pg = ps if gt <= 0 else np.where(d["gate"] >= gt, ps, 0.0)
            for rf in grid["refractory"]:
                for nc in grid["n_consec"]:
                    for nms in grid["nms"]:
                        for thr in grid["thr"]:
                            ev = pick(pg, thr, rf, nc, nms)
                            if not band[0] <= len(ev) / nch <= band[1]:
                                continue
                            k = ev.tobytes()
                            if k in seen:
                                continue
                            seen.add(k)
                            out.append(({"src": src, "smooth": sm, "gate_thr": gt,
                                         "refractory": rf, "n_consec": nc, "nms": nms,
                                         "thr": float(thr)}, ev))
    return out


_NCH: dict = {}


def nchars(session: str = DESK) -> int:
    if session not in _NCH:
        from phase0.analysis.decode import desk_segments
        s = pl.session_path(session)
        d = pl.frame_probs(s)
        segs = desk_segments(s, pl.taps_from(d, pl.event_idx(d, dict(d["cfg"],
                                                                    refractory_ms=0.0))))
        _NCH[session] = sum(len(t) for t, _ in segs)
    return _NCH[session]


# ------------------------------------------------------------------ higher-recall detector
def train_recall_model(label_half: float = 0.050, pos_weight: float = 24.0,
                       out: Path = RECALL_MODEL) -> Path:
    """A model moved along the precision/recall curve rather than slid down one threshold:
    a wider positive window and a heavier positive class, both fitted on keyboard only."""
    from phase0.analysis import taps_gb as gb

    data = gb.Data([pl.session_path(s) for s in KBD_TRAIN])
    X = data.X(gb.GROUPS)
    y = gb.labels(data.t, data.kt_all, label_half)
    yg = gb.labels(data.t, data.kt_all, gb.TYPING_HALF_WIDTH_S)
    print(f"recall model: frames={len(y)} pos={int(y.sum())} ({y.mean():.4f}) "
          f"half={label_half*1000:.0f}ms spw={pos_weight}", flush=True)
    t0 = time.time()
    m = gb.make_model("lgbm", scale_pos_weight=pos_weight).fit(X, y)
    g = gb.make_model("lgbm", scale_pos_weight=8.0).fit(X, yg)
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"kind": "lgbm", "model": m, "gate": g, "groups": gb.GROUPS,
                 "cfg": {"thr": 0.5, "smooth": 1, "refractory": 4, "n_consec": 1,
                         "gate_thr": 0.25},
                 "label_half": label_half, "pos_weight": pos_weight,
                 "sessions": list(KBD_TRAIN)}, out)
    print(f"saved {out} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------ stream book-keeping
def streams_path(tag: str) -> Path:
    return HR_CACHE / f"streams_{tag}.json"


def save_streams(tag: str, streams: list[tuple[dict, np.ndarray]]) -> None:
    HR_CACHE.mkdir(parents=True, exist_ok=True)
    streams_path(tag).write_text(json.dumps(
        [{"sid": i, "cfg": c, "ev": ev.tolist()} for i, (c, ev) in enumerate(streams)]))


_ST: dict = {}


def load_streams(tag: str) -> list[dict]:
    if tag not in _ST:
        _ST[tag] = json.loads(streams_path(tag).read_text())
    return _ST[tag]


def ev_of(tag: str, sid: int) -> np.ndarray:
    return np.array(load_streams(tag)[sid]["ev"], int)


def subsample(streams, n: int):
    """Thin by tap count only, never by CER, so the pool cannot smuggle held-out CER in."""
    if len(streams) <= n:
        return streams
    order = sorted(range(len(streams)), key=lambda i: (len(streams[i][1]), i))
    keep = sorted({order[i] for i in np.linspace(0, len(order) - 1, n).round().astype(int)})
    return [streams[i] for i in keep]


# ------------------------------------------------------------------ the joint table
def _one(args):
    CB._shim()
    use_cache()
    tag, sid, mode, alphas, obs_grid, dels = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = ev_of(tag, sid)
    rows = []
    for a in alphas:
        _, segs, p, _, _ = CB.stack_proba(s, ev, kf, CB.Cfg(alpha=a, em=False), mode)
        for w in obs_grid:
            for dl in dels:
                c = CB.Cfg(alpha=a, obs=w, deletion=dl, em=False)
                r = CB.decode_rows(p, segs, kf, c)
                rows.append({"sid": sid, "alpha": a, "obs": w, "deletion": dl, "mode": mode,
                             "n_taps": int(len(ev)), "per": [(x[2], x[3]) for x in r]})
    return rows


def build_table(tag: str, sids: list[int], mode: str = MODE, procs: int = 9,
                obs=OBS) -> list[dict]:
    import multiprocessing as mp

    jobs = [(tag, i, mode, ALPHAS, obs, DELETIONS) for i in sids]
    out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for n, rows in enumerate(pool.imap_unordered(_one, jobs, chunksize=1)):
            out += rows
            if (n + 1) % 5 == 0:
                print(f"  {n+1}/{len(jobs)} streams, best {min(CB.pooled(r) for r in out):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return out


def table_path(tag: str, mode: str, obs=OBS) -> Path:
    extra = "" if tuple(obs) == OBS else "_obs" + "-".join(f"{w:g}" for w in obs)
    return HR_CACHE / f"table_{tag}_{mode}{extra}.json"


def load_table(tag: str, mode: str) -> list[dict]:
    """Shards written under different obs grids merge, so a fold's choice of observation
    weight ranges over every value ever swept for this stream set."""
    out = []
    for f in sorted(HR_CACHE.glob(f"table_{tag}_{mode}.json")) + \
            sorted(HR_CACHE.glob(f"table_{tag}_{mode}_obs*.json")):
        out += json.loads(f.read_text())
    return out


# ------------------------------------------------------------------ nested CV
def dense_par(segs) -> "ad.AlignParams":
    """adapt.py fixes the spurious-tap rate at 8%; above ~1.1 taps/char that is arithmetically
    impossible, so give EM the insertion rate the tap and character counts actually imply."""
    n = sum(len(i) for _, i in segs)
    ch = sum(sum(c in ad.A_INDEX for c in t) for t, i in segs if len(i))
    return ad.AlignParams.from_counts(n, ch, float(np.clip(1.0 - ch / max(n, 1), 0.08, 0.5)))


def _cv_fold(args):
    CB._shim()
    use_cache()
    j, tag, pick_, mode, em = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = ev_of(tag, pick_["sid"])
    c = CB.Cfg(alpha=pick_["alpha"], obs=pick_["obs"], deletion=pick_["deletion"], em=em != "off")
    _, segs, p, sess, k = CB.stack_proba(s, ev, kf, c, mode)
    if em != "off":
        tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
        st = replace(pl.FULL, deletion=pick_["deletion"])
        if em == "dens":
            p = ad.adapt(pl._AdaptTarget(sess, k), tr, p, st.iters, st.lam, st.l2, "em",
                         par=dense_par(tr), verbose=False)[1]
        else:
            p = pl.weakly_supervise(sess, k, p, tr, st)
    return j, CB.decode_rows(p, segs, kf, c, keep={j})[0]


def nested_cv(tag: str, rows: list[dict], mode: str = MODE, em: str = "pipe", procs: int = 9,
              verbose: bool = False) -> list[tuple]:
    """combine.py's protocol verbatim: stream, alpha, obs weight and deletion cost are all
    chosen on the other 19 phrases, and EM only ever sees those 19 phrases' text."""
    import multiprocessing as mp

    n = len(rows[0]["per"])
    picks = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    jobs = [(j, tag, picks[j], mode, em) for j in range(n)]
    out: dict = {}
    with mp.get_context("spawn").Pool(procs) as pool:
        for j, r in pool.imap_unordered(_cv_fold, jobs):
            out[j] = r
            if verbose:
                p = picks[j]
                s = load_streams(tag)[p["sid"]]["cfg"]
                print(f"  phrase{j:>2} src={s['src']} nms={s['nms']} thr={s['thr']:.3f} "
                      f"rf={s['refractory']} gate={s['gate_thr']:.2f} taps={p['n_taps']} "
                      f"({p['n_taps']/nchars():.2f}/char) a={p['alpha']} obs={p['obs']} "
                      f"del={p['deletion']:.0f} CER={r[2]/max(1,r[3]):.3f}", flush=True)
    return [out[j] for j in range(n)]


def report(name: str, rows) -> dict:
    c, lo, hi = pl.boot_ci(rows)
    w = sum(edit_distance(r[0].split(), r[1].split()) for r in rows) / \
        max(1, sum(len(r[0].split()) for r in rows))
    print(f"{name:<44} CER={c:.3f} [{lo:.3f}, {hi:.3f}]  WER={w:.3f}")
    return {"name": name, "cer": c, "lo": lo, "hi": hi, "wer": w,
            "per": [(r[2], r[3]) for r in rows]}


def picked(tag: str, rows: list[dict]) -> str:
    n = len(rows[0]["per"])
    p = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    return (f"alpha {np.median([x['alpha'] for x in p]):.2f}, obs "
            f"{np.median([x['obs'] for x in p]):.2f}, del "
            f"{np.median([x['deletion'] for x in p]):.0f}, taps/char "
            f"{np.median([x['n_taps'] for x in p])/nchars():.2f} (fold medians)")


def examples(rows, n: int) -> None:
    for ref, hyp, e, r in rows[:n]:
        print(f"  CER={e/max(1,r):.3f} WER={wer(ref, hyp):.3f}")
        print(f"    typed:    {ref!r}")
        print(f"    produced: {hyp!r}")


# ------------------------------------------------------------------ density curve
def density(tag: str, r: dict) -> float:
    return r["n_taps"] / nchars()


def bin_rows(tag: str, rows: list[dict], lo: float, hi: float) -> list[dict]:
    return [r for r in rows if lo <= density(tag, r) < hi]


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def provenance(tags: list[str]) -> None:
    print("pinned inputs:")
    print(f"  {DESK}/landmarks.parquet md5 "
          f"{md5(tp.sess_path(DESK) / 'landmarks.parquet')[:12]}  (taps.jsonl is never read)")
    print(f"  models/taps_gb.joblib  md5 {md5('models/taps_gb.joblib')[:12]}  (base detector)")
    if RECALL_MODEL.exists() and any(t.startswith("rec") or t == "all" for t in tags):
        import joblib
        b = joblib.load(RECALL_MODEL)
        print(f"  {RECALL_MODEL} md5 {md5(RECALL_MODEL)[:12]}  (recall detector: "
              f"label +/-{b['label_half']*1000:.0f}ms, scale_pos_weight={b['pos_weight']}, "
              f"fitted on {', '.join(b['sessions'])})")
    print(f"  models/tap_pos.pkl     md5 {md5('models/tap_pos.pkl')[:12]}  (pose key model)")
    print(f"  models/charlm.npz      md5 {md5('models/charlm.npz')[:12]}")
    for sid in KBD_TRAIN:
        print(f"  {sid}/taps_contact.jsonl md5 "
              f"{md5(tp.sess_path(sid) / 'taps_contact.jsonl')[:12]} (pixel CNN labels)")
    for t in tags:
        p = streams_path(t)
        if p.exists():
            print(f"  {p} md5 {md5(p)[:12]} ({len(load_streams(t))} pinned tap streams)")


# ------------------------------------------------------------------ CLI
def _pool_tags(a) -> list[str]:
    return [t for t in a.tags.split(",") if t]


def _merged(a) -> tuple[str, list[dict]]:
    """Tables built against different stream files are re-keyed onto one merged file, so a
    fold may choose any stream from any detector."""
    tags = _pool_tags(a)
    if len(tags) == 1:
        return tags[0], load_table(tags[0], a.mode)
    merged, rows, off = [], [], 0
    for t in tags:
        st = load_streams(t)
        merged += [dict(x, sid=x["sid"] + off) for x in st]
        for r in load_table(t, a.mode):
            rows.append(dict(r, sid=r["sid"] + off))
        off += len(st)
    tag = "+".join(tags)
    HR_CACHE.mkdir(parents=True, exist_ok=True)
    streams_path(tag).write_text(json.dumps(merged))
    _ST.pop(tag, None)
    return tag, rows


def cmd_streams(a) -> int:
    allc = []
    for src in _pool_tags(a):
        c = candidate_streams(src)
        print(f"{src}: {len(c)} distinct streams, taps/char "
              f"{min(len(e) for _, e in c)/nchars():.2f}-{max(len(e) for _, e in c)/nchars():.2f}")
        allc.append((src, c))
    for src, c in allc:
        keep = subsample(c, a.streams)
        save_streams(src, keep)
        d = np.array([len(e) / nchars() for _, e in keep])
        print(f"{src}: kept {len(keep)} -> {streams_path(src)}  density "
              f"{d.min():.2f}/{np.median(d):.2f}/{d.max():.2f}")
    fr = sorted({int(i) for src, _ in allc for r in load_streams(src) for i in r["ev"]})
    (HR_CACHE / "union_frames.json").write_text(json.dumps(fr))
    print(f"union of {len(fr)} distinct tap frames -> {HR_CACHE/'union_frames.json'}")
    return 0


def cmd_model(a) -> int:
    train_recall_model(a.label_half, a.pos_weight)
    return 0


def use_cache() -> None:
    """combine.py's frame bank and pixel probabilities are keyed by its module-level CACHE;
    point it at ours so the denser streams get their own bank instead of overwriting."""
    CB.CACHE = HR_CACHE


def _stub_masks() -> None:
    """build_frame_banks caches one full 720x1280 mask per pose row; over this many frames
    that is ~14 GB, and mode 'none' never reads the mask bank, so feed it a shared blank."""
    blank = np.zeros((int(FRAME_H), int(FRAME_W)), np.uint8)
    CB._frame_mask = lambda cv2, P, span, shape: blank


def cmd_bank(a) -> int:
    if a.mode != "none":
        raise SystemExit("this bank is built unmasked-only; use --mode none")
    use_cache()
    _stub_masks()
    fr = np.array(json.loads((HR_CACHE / "union_frames.json").read_text()), int)
    frames = pl.frame_probs(pl.session_path(DESK))["frames"][fr]
    print(f"{len(frames)} distinct tap frames")
    CB.build_frame_banks(DESK, frames, force=a.force)
    return 0


def cmd_pix(a) -> int:
    use_cache()
    p = CB.pixel_probs(DESK, a.mode, force=a.force)
    print(f"{a.mode}: {p.shape} mean max prob {p.max(1).mean():.3f}")
    return 0


def cmd_table(a) -> int:
    obs = tuple(float(x) for x in a.obs.split(",")) if a.obs else OBS
    for tag in _pool_tags(a):
        p = table_path(tag, a.mode, obs)
        if p.exists() and not a.force:
            print(f"{p} exists")
            continue
        sids = [r["sid"] for r in load_streams(tag)
                if len(r["ev"]) / nchars() >= a.min_density]
        print(f"[{tag}] {len(sids)} streams x {len(ALPHAS)} alphas x {len(obs)} obs x "
              f"{len(DELETIONS)} deletions", flush=True)
        rows = build_table(tag, sids, a.mode, a.procs, obs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows))
        print(f"-> {p}")
    return 0


def cmd_curve(a) -> int:
    tag, rows = _merged(a)
    print(f"\n== CER against detection density, {tag} ({a.mode} pixels) ==")
    print(f"{'taps/char':>13}{'streams':>9}{'best':>8}{'median':>8}{'nestedCV':>10}"
          f"{'95% CI':>18}{'picked':>34}")
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        v = bin_rows(tag, rows, lo, hi)
        if len(v) < len(ALPHAS) * len(OBS) * len(DELETIONS):
            continue
        r = nested_cv(tag, v, a.mode, em=a.em, procs=a.procs)
        c, clo, chi = pl.boot_ci(r)
        n = len({x["sid"] for x in v})
        d = np.median([x["n_taps"] for x in v]) / nchars()
        print(f"{f'{lo:.2f}-{hi:.2f}':>13}{n:>9}{min(CB.pooled(x) for x in v):>8.3f}"
              f"{np.median([CB.pooled(x) for x in v]):>8.3f}{c:>10.3f}"
              f"{f'[{clo:.3f}, {chi:.3f}]':>18}{picked(tag, v):>34}", flush=True)
        out.append({"lo": lo, "hi": hi, "n_streams": n, "median_density": float(d),
                    "best_pooled": min(CB.pooled(x) for x in v), "cer": c, "ci": [clo, chi],
                    "per": [(x[2], x[3]) for x in r]})
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


def cmd_cv(a) -> int:
    tag, rows = _merged(a)
    if a.max_density:
        rows = [r for r in rows if density(tag, r) <= a.max_density]
    provenance(_pool_tags(a))
    print(f"\n-- nested leave-one-phrase-out over {len({r['sid'] for r in rows})} tap streams "
          f"({tag}, {a.mode} pixels) --")
    r = nested_cv(tag, rows, a.mode, em=a.em, procs=a.procs, verbose=True)
    print()
    report(f"A + {a.mode} pixels, high-recall streams (EM={a.em})", r)
    print(f"{'':<44}   picked: {picked(tag, rows)}")
    if a.baseline:
        base = json.loads(Path(a.baseline).read_text())
        b = [x for x in base if x["name"] == a.baseline_name][0]
        bp = [(e, n) for e, n in b["per"]]
        d, lo, hi = pl.boot_delta([(None, None, e, n) for e, n in bp], r)
        print(f"{'':<44}   delta vs {b['name']} ({b['cer']:.3f}): "
              f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"\n-- {min(a.examples, len(r))} decoded phrases --")
    examples(r, a.examples)
    if a.json:
        Path(a.json).write_text(json.dumps(report(f"hirecall {tag}", r), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.hirecall", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("streams", cmd_streams), ("model", cmd_model), ("bank", cmd_bank),
                     ("pix", cmd_pix), ("table", cmd_table), ("cv", cmd_cv),
                     ("curve", cmd_curve)):
        p = sub.add_parser(name)
        p.add_argument("--tags", default="base")
        p.add_argument("--mode", default=MODE)
        p.add_argument("--em", default="pipe", choices=("off", "pipe", "dens"))
        p.add_argument("--max-density", type=float, default=0.0)
        p.add_argument("--min-density", type=float, default=0.0)
        p.add_argument("--obs", default=None)
        p.add_argument("--streams", type=int, default=N_STREAMS)
        p.add_argument("--procs", type=int, default=9)
        p.add_argument("--examples", type=int, default=8)
        p.add_argument("--label-half", type=float, default=0.050)
        p.add_argument("--pos-weight", type=float, default=24.0)
        p.add_argument("--baseline", default=".cache/combine/ablate_final.json")
        p.add_argument("--baseline-name", default="A + unmasked pixels")
        p.add_argument("--json", default=None)
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
