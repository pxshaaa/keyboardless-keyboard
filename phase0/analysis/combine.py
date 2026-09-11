"""Does masked.py's hand-masked pixel expert still add on top of pipeline.py's desk-tuned stack?
Run: python -m phase0.analysis.combine {bank | pix | table | cv | ablate}"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis import pipeline as pl
from phase0.analysis import tap_pos as tp
from phase0.analysis.appearance import TIP_CROP, TIP_OFFSETS, TIP_SPANS, fit_cnn, cnn_proba
from phase0.analysis.decode import NA, Weights, beam_decode, edit_distance, wer
from phase0.analysis.finger_id import FINGERTIP_JOINTS
from phase0.analysis.masked import _frame_mask, build_dataset, tip_crops
from phase0.analysis.tap_pos import H as FRAME_H
from phase0.analysis.tap_pos import W as FRAME_W

DESK = pl.DESK
KBD_TRAIN = pl.KBD_TRAIN
CACHE = Path(".cache/combine")
EPS = pl.EPS
SEEDS = (0, 1, 2)
OFFSET = 4          # masked.py's operating point: contact is 3-5 frames after the peak
REP = "diff"
EPOCHS = 40

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
OBS = (1.0, 1.5, 2.0, 2.5)
DELETIONS = (-9.0, -5.0, -3.0, -2.0)
N_STREAMS = 100     # even subsample of the 718 distinct tap streams, by tap count only


# ------------------------------------------------------------------ tap streams
def all_streams(session: Path) -> list[tuple[dict, np.ndarray]]:
    return pl.candidate_cfgs(session)


def subsample(cands: list[tuple[dict, np.ndarray]], n: int) -> list[tuple[dict, np.ndarray]]:
    """Thin the candidate set for the joint sweep. Ordered by tap count, never by CER, so the
    restriction cannot smuggle held-out phrase performance into the candidate list."""
    if len(cands) <= n:
        return cands
    order = sorted(range(len(cands)), key=lambda i: (len(cands[i][1]), i))
    keep = sorted(order[i] for i in np.linspace(0, len(order) - 1, n).round().astype(int))
    return [cands[i] for i in dict.fromkeys(keep)]


def union_frames(session: Path) -> np.ndarray:
    d = pl.frame_probs(session)
    u = set()
    for _, ev in all_streams(session):
        u.update(d["frames"][ev].tolist())
    return np.array(sorted(u), int)


# ------------------------------------------------------------------ frame-indexed crop banks
def bank_paths(sid: str) -> tuple[Path, Path, Path]:
    return (CACHE / f"{sid}_ftips.npy", CACHE / f"{sid}_fmask.npy", CACHE / f"{sid}_frames.json")


def _tip_geometry(s, frames: np.ndarray):
    """Patch centres frozen at the tap frame, exactly as appearance.build_tip_bank does."""
    k = np.clip(np.searchsorted(s.frames, frames), 0, len(s.P) - 1)
    P = s.P[k][:, :, list(FINGERTIP_JOINTS), :2]
    fb = np.broadcast_to(s.anchor[:, None, :], P.shape[1:])
    return np.where(np.isfinite(P), P, fb[None]), TIP_SPANS * s.span


def build_frame_banks(sid: str, frames: np.ndarray, force: bool = False) -> None:
    """Tip crops and hand masks keyed by VIDEO FRAME, not by a tap file, so every candidate
    tap stream can be scored against the same pixels."""
    import cv2

    tp_p, mk_p, fr_p = bank_paths(sid)
    if tp_p.exists() and mk_p.exists() and fr_p.exists() and not force:
        if json.loads(fr_p.read_text())["frames"] == frames.tolist():
            return
    CACHE.mkdir(parents=True, exist_ok=True)
    s = tp.load_sess(sid)
    P, side = _tip_geometry(s, frames)
    n, shape = len(frames), (int(FRAME_H), int(FRAME_W))
    tips = np.lib.format.open_memmap(tp_p, mode="w+", dtype=np.uint8,
                                     shape=(n, len(TIP_OFFSETS), 2, 5, TIP_CROP, TIP_CROP))
    masks = np.lib.format.open_memmap(mk_p, mode="w+", dtype=np.uint8, shape=tips.shape)

    def warp(img, i, j, out):
        for h in (0, 1):
            sc = TIP_CROP / side[h]
            for f in range(5):
                cx, cy = P[i, h, f]
                M = np.array([[sc, 0, -sc * (cx - side[h] / 2)],
                              [0, sc, -sc * (cy - side[h] / 2)]], np.float32)
                out[i, j, h, f] = cv2.warpAffine(img, M, (TIP_CROP, TIP_CROP),
                                                 flags=cv2.INTER_AREA if out is tips
                                                 else cv2.INTER_NEAREST,
                                                 borderMode=cv2.BORDER_REPLICATE if out is tips
                                                 else cv2.BORDER_CONSTANT, borderValue=0)

    want: dict[int, list[tuple[int, int]]] = {}
    for j, off in enumerate(TIP_OFFSETS):
        for i, f in enumerate(np.clip(frames + off, 0, len(s.frames) - 1)):
            want.setdefault(int(f), []).append((i, j))
    t0 = time.time()
    cap = cv2.VideoCapture(str(tp.sess_path(sid) / "video.mp4"))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        jobs = want.get(idx)
        if jobs:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i, j in jobs:
                warp(g, i, j, tips)
        idx += 1
    cap.release()
    cache: dict[int, np.ndarray] = {}
    for f, jobs in want.items():
        r = int(np.clip(np.searchsorted(s.frames, f), 0, len(s.P) - 1))
        if r not in cache:
            cache[r] = _frame_mask(cv2, s.P[r, :, :, :2], s.span, shape)
        for i, j in jobs:
            warp(cache[r], i, j, masks)
    sample = np.asarray(tips[:: max(1, n // 200)], dtype=np.uint8)
    lo, hi = np.percentile(sample, [2.0, 99.8])
    hi = max(hi, lo + 1.0)
    for i in range(0, n, 128):
        blk = np.asarray(tips[i:i + 128], dtype=np.float32)
        tips[i:i + 128] = np.clip((blk - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    tips.flush()
    masks.flush()
    fr_p.write_text(json.dumps({"frames": frames.tolist(), "lo": float(lo), "hi": float(hi)}))
    print(f"{sid}: frame banks {tips.shape} hand-frac="
          f"{float(np.asarray(masks[::7]).mean()):.3f} ({time.time()-t0:.0f}s)", flush=True)


def frame_crops(sid: str, mode: str) -> np.ndarray:
    """[n_frames,10,2,32,32] in masked.py's 'diff' representation."""
    tp_p, mk_p, _ = bank_paths(sid)
    tips = np.load(tp_p, mmap_mode="r")
    masks = np.load(mk_p, mmap_mode="r") if mode != "none" else None
    jj = {o: j for j, o in enumerate(TIP_OFFSETS)}

    def at(o):
        o = min(TIP_OFFSETS, key=lambda x: abs(x - o))
        a = np.asarray(tips[:, jj[o]], dtype=np.float32) / 255.0
        if masks is not None:
            m = np.asarray(masks[:, jj[o]], dtype=np.float32)
            a = m if mode == "silh" else a * (m if mode == "hand" else 1.0 - m)
        return a.reshape(len(a), 10, 1, TIP_CROP, TIP_CROP)

    base = at(OFFSET)
    return np.concatenate([base, base - at(min(TIP_OFFSETS))], axis=2)


# ------------------------------------------------------------------ pixel expert
def pixel_probs(sid: str, mode: str, seeds=SEEDS, epochs: int = EPOCHS,
                force: bool = False) -> np.ndarray:
    """Key distribution per video frame from a CNN fitted on the keyboard sessions only."""
    out_p = CACHE / f"pix_{sid}_{mode}_{len(seeds)}x{epochs}.npy"
    if out_p.exists() and not force:
        return np.load(out_p)
    tr = [build_dataset(s) for s in KBD_TRAIN]
    Xtr = np.vstack([tip_crops(d, OFFSET, mode, REP) for d in tr])
    ytr = np.concatenate([d["y"] for d in tr])
    Xte = frame_crops(sid, mode)
    ps = []
    for seed in seeds:
        t0 = time.time()
        clf = fit_cnn(Xtr, ytr, NA, seed=seed, epochs=epochs, arch="small")
        ps.append(cnn_proba(clf, Xte, NA))
        print(f"  {mode} seed{seed}: {len(ytr)} training taps ({time.time()-t0:.0f}s)", flush=True)
    p = np.mean(ps, axis=0)
    p /= p.sum(1, keepdims=True)
    np.save(out_p, p)
    return p


def fuse(pix: np.ndarray, pose: np.ndarray, a: float) -> np.ndarray:
    if a <= 0:
        return pose
    q = np.exp(a * np.log(np.maximum(pix, EPS)) + (1 - a) * np.log(np.maximum(pose, EPS)))
    return q / q.sum(1, keepdims=True)


def fuse3(pix, silh, pose, a_pix, a_silh):
    q = np.exp(a_pix * np.log(np.maximum(pix, EPS)) + a_silh * np.log(np.maximum(silh, EPS))
               + (1 - a_pix - a_silh) * np.log(np.maximum(pose, EPS)))
    return q / q.sum(1, keepdims=True)


# ------------------------------------------------------------------ one stack
@dataclass(frozen=True)
class Cfg:
    alpha: float = 0.0
    obs: float = 1.0
    deletion: float = -3.0
    a_silh: float = 0.0
    contact: float = 0.0
    em: bool = True
    coral: bool = True

    def weights(self) -> Weights:
        return Weights(obs=self.obs, deletion=self.deletion, insertion=-7.0, max_deletions=3)


_PIX: dict = {}


def pix_for(session: Path, ev: np.ndarray, mode: str) -> np.ndarray:
    """Pixel probabilities for a tap stream: index the frame bank by each tap's video frame."""
    sid = Path(session).name
    if mode not in _PIX:
        _, _, fr_p = bank_paths(sid)
        frames = np.array(json.loads(fr_p.read_text())["frames"], int)
        _PIX[mode] = (frames, pixel_probs(sid, mode))
    frames, p = _PIX[mode]
    want = pl.frame_probs(session)["frames"][ev]
    j = np.searchsorted(frames, want)
    if not (j < len(frames)).all() or not (frames[np.minimum(j, len(frames) - 1)] == want).all():
        raise SystemExit("tap frame missing from the crop bank; rerun `combine bank`")
    return p[j]


def stack_proba(session: Path, ev: np.ndarray, kf, c: Cfg, mode: str = "hand"):
    """-> (taps, segments, fused observation probs, pose feature rows, session)."""
    d = pl.frame_probs(session)
    taps = pl.taps_from(d, ev)
    sess = tp.load_sess(str(session))
    k = sess.rows(taps)
    pose = pl.spatial_proba(sess, k, kf, replace(pl.FULL, coral=c.coral, contact_w=c.contact))
    p = pose
    if c.a_silh > 0:
        p = fuse3(pix_for(session, ev, mode), pix_for(session, ev, "silh"), pose,
                  c.alpha, c.a_silh)
    elif c.alpha > 0:
        p = fuse(pix_for(session, ev, mode), pose, c.alpha)
    return taps, pl.segments(session, taps), p, sess, k


def decode_rows(proba: np.ndarray, segs, kf, c: Cfg, beam: int = 30, keep=None):
    clm, wlm = kf.lm
    out = []
    for j, (text, rows) in enumerate(segs):
        if keep is not None and j not in keep:
            continue
        hyp = "" if len(rows) == 0 else beam_decode(
            np.log(np.maximum(proba[rows], EPS)), clm, wlm, c.weights(), beam)
        out.append((text, hyp, edit_distance(text, hyp), len(text)))
    return out


# ------------------------------------------------------------------ the joint table
def _shim() -> None:
    """A spawned worker unpickles models/*.pkl, which name tap_pos classes under __main__."""
    for n in ("Sess", "KeyClf", "XYReg", "GaussXY", "Selector"):
        setattr(sys.modules["__main__"], n, getattr(tp, n))


def _one(args):
    _shim()
    cfg, ev, mode, alphas, obs_grid, dels, a_silh, contact = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    rows = []
    for a in alphas:
        _, segs, p, _, _ = stack_proba(
            s, ev, kf, Cfg(alpha=a, a_silh=a_silh, contact=contact, em=False), mode)
        for w in obs_grid:
            for dl in dels:
                c = Cfg(alpha=a, obs=w, deletion=dl, a_silh=a_silh, contact=contact, em=False)
                r = decode_rows(p, segs, kf, c)
                rows.append({"cfg": cfg, "alpha": a, "obs": w, "deletion": dl,
                             "a_silh": a_silh, "contact": contact, "mode": mode,
                             "n_taps": int(len(ev)), "per": [(x[2], x[3]) for x in r]})
    return rows


def build_table(session: Path, mode: str, streams, alphas=ALPHAS, obs_grid=OBS,
                dels=DELETIONS, a_silh: float = 0.0, contact: float = 0.0,
                procs: int = 8) -> list[dict]:
    """Per-phrase edit distances for every (tap stream, alpha, obs weight, deletion cost) with
    EM off: fold-independent, so any fold's selection is a subset sum of this table."""
    import multiprocessing as mp

    jobs = [(cfg, ev, mode, alphas, obs_grid, dels, a_silh, contact) for cfg, ev in streams]
    out = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for n, rows in enumerate(pool.imap_unordered(_one, jobs, chunksize=1)):
            out += rows
            if (n + 1) % 10 == 0:
                print(f"  {n+1}/{len(jobs)} streams, best {min(pooled(r) for r in out):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return out


def pooled(r: dict) -> float:
    return sum(x[0] for x in r["per"]) / max(1, sum(x[1] for x in r["per"]))


def select(rows: list[dict], train: set[int]) -> dict:
    best, bc = None, np.inf
    for r in rows:
        e = sum(r["per"][j][0] for j in train)
        n = sum(r["per"][j][1] for j in train)
        c = e / max(1, n)
        if c < bc:
            best, bc = r, c
    return best


# ------------------------------------------------------------------ nested CV
def _cv_fold(args):
    _shim()
    j, pick, mode, em = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    d = pl.frame_probs(s)
    ev = pl.event_idx(d, pick["cfg"])
    c = Cfg(alpha=pick["alpha"], obs=pick["obs"], deletion=pick["deletion"],
            a_silh=pick["a_silh"], contact=pick.get("contact", 0.0), em=em)
    _, segs, p, sess, k = stack_proba(s, ev, kf, c, mode)
    if em:
        tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
        p = pl.weakly_supervise(sess, k, p, tr, replace(pl.FULL, deletion=pick["deletion"]))
    return j, decode_rows(p, segs, kf, c, keep={j})[0], len(ev)


def nested_cv(rows: list[dict], mode: str, em: bool = True, procs: int = 8,
              verbose: bool = True) -> list[tuple]:
    """Leave-one-phrase-out with extractor, fusion weight, obs weight and deletion cost all
    chosen inside the fold; EM only ever sees the other 19 phrases' text."""
    import multiprocessing as mp

    n = len(rows[0]["per"])
    picks = [select(rows, set(range(n)) - {j}) for j in range(n)]
    jobs = [(j, picks[j], mode, em) for j in range(n)]
    out: dict = {}
    with mp.get_context("spawn").Pool(procs) as pool:
        for j, r, ntaps in pool.imap_unordered(_cv_fold, jobs):
            out[j] = r
            if verbose:
                p = picks[j]
                print(f"  phrase{j:>2} thr={p['cfg']['thr']:.2f} rf={p['cfg']['refractory']} "
                      f"nc={p['cfg']['n_consec']} taps={ntaps} a={p['alpha']} obs={p['obs']} "
                      f"del={p['deletion']:.0f} CER={r[2]/max(1,r[3]):.3f}", flush=True)
    return [out[j] for j in range(n)]


def report(name: str, rows) -> dict:
    c, lo, hi = pl.boot_ci(rows)
    w = sum(edit_distance(r[0].split(), r[1].split()) for r in rows) / \
        max(1, sum(len(r[0].split()) for r in rows))
    print(f"{name:<38} CER={c:.3f} [{lo:.3f}, {hi:.3f}]  WER={w:.3f}")
    return {"name": name, "cer": c, "lo": lo, "hi": hi, "wer": w,
            "per": [(r[2], r[3]) for r in rows]}


def examples(rows, n: int) -> None:
    for ref, hyp, e, r in rows[:n]:
        print(f"  CER={e/max(1,r):.3f} WER={wer(ref, hyp):.3f}")
        print(f"    typed:    {ref!r}")
        print(f"    produced: {hyp!r}")


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def provenance() -> None:
    print("pinned inputs:")
    print(f"  desk tap stream: generated from {DESK}/landmarks.parquet "
          f"(md5 {md5(tp.sess_path(DESK) / 'landmarks.parquet')[:12]}) via the taps_gb model "
          f"models/taps_gb.joblib (md5 {md5(Path('models/taps_gb.joblib'))[:12]}) - taps.jsonl "
          "is never read")
    for sid in KBD_TRAIN:
        print(f"  {sid}/taps_contact.jsonl md5 "
              f"{md5(tp.sess_path(sid) / 'taps_contact.jsonl')[:12]} (pixel CNN labels)")


# ------------------------------------------------------------------ CLI
def cmd_bank(a) -> int:
    s = pl.session_path(a.session)
    fr = union_frames(s)
    print(f"{len(fr)} distinct tap frames over {len(all_streams(s))} candidate streams")
    build_frame_banks(Path(a.session).name, fr, force=a.force)
    return 0


def cmd_pix(a) -> int:
    for mode in a.modes.split(","):
        p = pixel_probs(Path(a.session).name, mode, force=a.force)
        print(f"{mode}: {p.shape} mean max prob {p.max(1).mean():.3f}")
    return 0


def _table_path(a, tag: str, obs) -> Path:
    extra = "" if tuple(obs) == OBS else "_obs" + "-".join(f"{w:g}" for w in obs)
    return Path(a.table_dir) / f"table_{tag}{extra}.json"


def _get_table(a, mode: str, a_silh: float = 0.0, contact: float = 0.0) -> list[dict]:
    """Shards written under different obs grids are merged, so the fold's choice of obs
    weight always ranges over every value ever swept for this mode."""
    tag = mode + (f"_s{a_silh}" if a_silh else "") + (f"_c{contact}" if contact else "")
    obs = tuple(float(x) for x in a.obs.split(",")) if a.obs else OBS
    p = _table_path(a, tag, obs)
    if not p.exists() or a.force:
        s = pl.session_path(a.session)
        streams = subsample(all_streams(s), a.streams)
        print(f"[{tag}] {len(streams)} streams x {len(ALPHAS)} alphas x {len(obs)} obs x "
              f"{len(DELETIONS)} deletions", flush=True)
        rows = build_table(s, mode, streams, obs_grid=obs, a_silh=a_silh, contact=contact,
                           procs=a.procs)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows))
    out = []
    for f in sorted(Path(a.table_dir).glob(f"table_{tag}.json")) + \
            sorted(Path(a.table_dir).glob(f"table_{tag}_obs*.json")):
        out += json.loads(f.read_text())
    return out


def cmd_table(a) -> int:
    for mode in a.modes.split(","):
        _get_table(a, mode, a_silh=0.25 if a.three else 0.0, contact=a.contact)
    return 0


def cmd_cv(a) -> int:
    provenance()
    rows = _get_table(a, a.mode)
    print("\n-- nested leave-one-phrase-out, combined stack --")
    r = nested_cv(rows, a.mode, em=True, procs=a.procs)
    print()
    report("A + hand-masked pixels (full)", r)
    print(f"\n-- {min(a.examples, len(r))} decoded phrases --")
    examples(r, a.examples)
    return 0


def picked(rows: list[dict]) -> str:
    n = len(rows[0]["per"])
    p = [select(rows, set(range(n)) - {j}) for j in range(n)]
    return (f"alpha {np.median([x['alpha'] for x in p]):.2f}, obs "
            f"{np.median([x['obs'] for x in p]):.2f}, del "
            f"{np.median([x['deletion'] for x in p]):.0f}, taps/char "
            f"{np.median([x['n_taps'] for x in p])/676:.2f} (fold medians)")


def cmd_ablate(a) -> int:
    provenance()
    base = _get_table(a, a.mode)
    out, runs = [], {}

    def go(name: str, rows, mode: str, em: bool = True, vs=None):
        r = nested_cv(rows, mode, em=em, procs=a.procs, verbose=False)
        runs[name] = r
        out.append(report(name, r))
        print(f"{'':<38}   picked: {picked(rows)}")
        if vs is not None:
            d, lo, hi = pl.boot_delta(runs[vs], r)
            print(f"{'':<38}   delta vs {vs}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        return r

    print("\n== leave-one-phrase-out ladder, everything tuned inside the fold ==")
    A = "A only (pose+CORAL+EM)"
    go(A, [r for r in base if r["alpha"] == 0.0], a.mode)
    go("A + hand-masked pixels", base, a.mode, vs=A)
    go("A + unmasked pixels", _get_table(a, "none"), "none", vs=A)
    go("A + silhouette expert", _get_table(a, "silh"), "silh", vs=A)
    go("hand-masked pixels only", [r for r in base if r["alpha"] == 1.0], a.mode, vs=A)
    go("A + hand-masked pixels, no EM", base, a.mode, em=False, vs="A + hand-masked pixels")
    if a.contact > 0:
        go(f"A + pixels + contact expert @{a.contact}",
           _get_table(a, a.mode, contact=a.contact), a.mode, vs="A + hand-masked pixels")
    if a.three:
        go(f"A + {a.three_mode} pixels + silhouette",
           _get_table(a, a.three_mode, a_silh=0.25), a.three_mode,
           vs=f"A + {'unmasked' if a.three_mode == 'none' else a.three_mode} pixels")

    best = min(runs, key=lambda n: pl.boot_ci(runs[n])[0])
    print(f"\n-- {a.examples} decoded phrases, best stack ({best}) --")
    examples(runs[best], a.examples)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.combine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("bank", cmd_bank), ("pix", cmd_pix), ("table", cmd_table),
                     ("cv", cmd_cv), ("ablate", cmd_ablate)):
        p = sub.add_parser(name)
        p.add_argument("--session", default=DESK)
        p.add_argument("--mode", default="hand")
        p.add_argument("--modes", default="hand,none,silh")
        p.add_argument("--streams", type=int, default=N_STREAMS)
        p.add_argument("--procs", type=int, default=8)
        p.add_argument("--table-dir", default=".cache/combine")
        p.add_argument("--examples", type=int, default=6)
        p.add_argument("--three", action="store_true")
        p.add_argument("--three-mode", default="none")
        p.add_argument("--contact", type=float, default=0.0)
        p.add_argument("--obs", default=None)
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
