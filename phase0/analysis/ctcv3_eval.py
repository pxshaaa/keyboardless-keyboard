"""ctc_v3 evaluation on CPU: log-posterior ensembles, greedy/char/Qwen decoders, desk CV, blind-1, kbd LOSO, pseudo-labels.
Run: PYTHONPATH=. python -m phase0.analysis.ctcv3_eval {run|pseudo|table} (decoder weights: seqctc2 keyboard tuning)."""

from __future__ import annotations

import functools
import os

import torch

torch.backends.mps.is_available = functools.lru_cache()(lambda: False)  # decoding stays off the shared GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from phase0.analysis import ctcv3 as C
from phase0.analysis import seqctc as S
from phase0.analysis import seqctc2 as S2
from phase0.analysis.seqctc import DESK, LOSO

RES = C.ROOT / "results"
HYP = C.ROOT / "hyps"
QWEN_CFG = json.loads(Path("results/seqctc2/tuning/tune_qwen_lex_hwtmix.json").read_text())["best"]
SID = {"desk": DESK, "blind1": C.BLIND1, "free": C.FREE}


def decoders(names) -> dict:
    from phase0.analysis.decipher import char_params
    al, be = char_params()
    D = {"greedy": {"kind": "greedy"}, "char": {"kind": "char", "alpha": al, "beta": be, "beam": 16},
         "qwen": {"kind": "qwen", "cfg": QWEN_CFG}}
    return {n: D[n] for n in names}


# ============================================================================ decoding with a persistent cache
def _init():
    torch.set_num_threads(2)


def _job(args):
    dec, lp = args
    if dec["kind"] == "qwen" and "qwen" not in S2._SC:
        from phase0.analysis import autocorrect as AC
        S2._SC["qwen"] = AC.NLM("Qwen/Qwen2.5-0.5B", device="cpu")
    return S2.run_decoder(dec, lp)


def dkey(dec, lp) -> str:
    h = hashlib.md5(json.dumps(dec, sort_keys=True).encode())
    h.update(np.ascontiguousarray(lp, np.float32).tobytes())
    return h.hexdigest()


_CACHE: dict = {}
_POOL: dict = {}


def cache() -> dict:
    if not _CACHE:
        _CACHE["_"] = None
        for f in HYP.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                    _CACHE[r["k"]] = r["h"]
                except json.JSONDecodeError:
                    pass
    return _CACHE


def decode_all(jobs, procs: int) -> list[str]:
    c = cache()
    keys = [dkey(d, l) for d, l in jobs]
    todo = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in c))
    seen, uniq = set(), []
    for i in todo:
        if keys[i] not in seen:
            seen.add(keys[i])
            uniq.append(i)
    if uniq:
        HYP.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(HYP / f"{os.getpid()}.jsonl", "a") as fh:
            if procs <= 1:
                it = (_job(jobs[i]) for i in uniq)
            else:
                if "pool" not in _POOL:
                    _POOL["pool"] = mp.get_context("spawn").Pool(procs, initializer=_init)
                it = _POOL["pool"].imap(_job, [jobs[i] for i in uniq], chunksize=1)
            for i, h in zip(uniq, it):
                c[keys[i]] = h
                fh.write(json.dumps({"k": keys[i], "h": h}) + "\n")
                fh.flush()
        print(f"  decoded {len(uniq)} items in {time.time() - t0:.0f}s", flush=True)
    return [c[k] for k in keys]


# ============================================================================ posteriors of ensemble members
def parse_members(spec: str):
    out = []
    for g in spec.split("+"):
        tag, seeds, kind = g.split(":")
        out += [(tag, int(s), kind) for s in seeds.split(",")]
    return out


def member_path(m, name: str, tta: bool) -> Path:
    tag, seed, kind = m
    sfx = "_tta" if tta and not name.startswith("loso") else ""
    d = C.rdir(tag, seed)
    if kind == "zs":
        f = {"desk_win": "zs_desk_win", "desk_cont": "zs_desk_cont", "blind1_cont": "zs_blind1_cont",
             "free_cont": "zs_free_cont", "held": "held"}.get(name, name)
        ext = ".npz" if name.endswith("win") or name in ("held",) or name.startswith("loso") else ".npy"
        return d / f"{f}{sfx}{ext}"
    ext = ".npz" if name.endswith("win") else ".npy"
    return d / kind / f"{name}{sfx}{ext}"


def compose_ft(L: np.ndarray) -> np.ndarray:
    """[5, G, NS] fold stack -> [G, NS], each frame from the fold that held out the phrase shown at that time."""
    desk = C.stream(DESK)
    wins = S.desk_windows(desk)
    tg = desk.t[::2][:L.shape[1]]
    ph = np.clip(np.searchsorted([w[0] for w in wins], tg, side="right") - 1, 0, len(wins) - 1)
    return L[ph % L.shape[0], np.arange(L.shape[1])]


def load_member(m, name: str, tta: bool):
    f = member_path(m, name, tta)
    if f.suffix == ".npz":
        return S.load_lps(f)
    L = np.load(f).astype(np.float32)
    return compose_ft(L) if L.ndim == 3 else L


def ensemble(members, name: str, tta: bool):
    got = [load_member(m, name, tta) for m in members]
    if isinstance(got[0], tuple):
        refs = got[0][1]
        return [C.logmean([g[0][j] for g in got]) for j in range(len(refs))], refs
    G = min(len(g) for g in got)
    return C.logmean([g[:G] for g in got])


# ============================================================================ metrics
def summ(ed, n, wed=None, wn=None, wok=None) -> dict:
    ed, n = np.asarray(ed, float), np.asarray(n, float)
    c, lo, hi = S.boot_ratio(ed, n)
    r = {"cer": c, "cer_ci": [lo, hi], "ed": ed.tolist(), "n": n.tolist()}
    if wed is not None:
        wed, wn = np.asarray(wed, float), np.asarray(wn, float)
        w, wlo, whi = S.boot_ratio(wed, wn)
        r.update({"wer": w, "wer_ci": [wlo, whi], "wed": wed.tolist(), "wn": wn.tolist()})
        if wok is not None:
            r["words_correct"] = float(np.sum(wok) / wn.sum())
    return r


def plan(members, sets, decs, tta: bool):
    """-> jobs [(dec, lp)], finishers [(key, n_jobs, fn(hyps) -> per-item edit arrays)]."""
    from phase0.analysis.decipher import norm, score_variant
    from phase0.analysis.decode import edit_distance
    jobs, fins = [], []
    for sname in sets:
        if sname in ("desk_win", "held"):
            lps, refs = ensemble(members, sname, tta)
            for dn, dec in decs.items():
                if sname == "held" and dn == "qwen":
                    continue
                jobs += [(dec, lp) for lp in lps]

                def fin(hy, refs=refs):
                    return dict(ed=[edit_distance(r, h) for r, h in zip(refs, hy)], n=[len(r) for r in refs],
                                wed=[edit_distance(r.split(), h.split()) for r, h in zip(refs, hy)],
                                wn=[len(r.split()) for r in refs])
                fins.append((f"{sname}/{dn}", len(lps), fin))
        elif sname == "loso":
            lps, refs = [], []
            for fold in LOSO:
                l_, r_ = ensemble(members, f"loso_{fold[9:15]}", False)
                lps += l_
                refs += r_
            for dn, dec in decs.items():
                if dn == "qwen":
                    continue
                jobs += [(dec, lp) for lp in lps]
                fins.append((f"loso/{dn}", len(lps),
                             lambda hy, refs=refs: dict(ed=[edit_distance(r, h) for r, h in zip(refs, hy)],
                                                        n=[len(r) for r in refs])))
        elif sname in ("desk_cont", "blind1_cont"):
            L = ensemble(members, sname, tta)
            st = C.stream(SID[sname[:-5]])
            times = st.t[::2][:len(L)]
            segs = S2.segments(L, times, 2.0)
            for dn, dec in decs.items():
                jobs += [(dec, L[s0:s1]) for s0, s1 in segs]
                if sname == "desk_cont":
                    phrases = [w[2] for w in S.desk_windows(C.stream(DESK))]

                    def fin(hy, phrases=phrases):
                        ref_all = " ".join(phrases)
                        hyp_all = " ".join(h for h in hy if h)
                        own = sum([[j] * (len(p) + (1 if j < len(phrases) - 1 else 0)) for j, p in enumerate(phrases)], [])
                        wown = sum([[j] * len(p.split()) for j, p in enumerate(phrases)], [])
                        return dict(ed=S2.aligned_edits(list(ref_all), list(hyp_all), own, len(phrases)),
                                    n=[len(p) + (1 if j < len(phrases) - 1 else 0) for j, p in enumerate(phrases)],
                                    wed=S2.aligned_edits(ref_all.split(), hyp_all.split(), wown, len(phrases)),
                                    wn=[len(p.split()) for p in phrases])
                else:
                    lines = [norm(l) for l in (C.SESS / C.BLIND1 / "truth.txt").read_text().splitlines() if norm(l)]

                    def fin(hy, lines=lines):
                        sv = score_variant(lines, hy)
                        return dict(ed=sv["ced"], n=sv["cn"], wed=sv["wed"], wn=sv["wn"], wok=sv["wok"],
                                    hyp=sv["hyp_by_line"])
                fins.append((f"{sname}/{dn}", len(segs), fin))
    return jobs, fins


def evaluate(members, sets, decs, tta, procs):
    jobs, fins = plan(members, sets, decs, tta)
    hy = decode_all(jobs, procs)
    out, k = {}, 0
    for key, nj, fn in fins:
        out[key] = fn(hy[k:k + nj])
        out[key]["hyps"] = hy[k:k + nj]
        k += nj
    return out


def cmd_run(a) -> int:
    members = parse_members(a.members)
    sets, decs = a.sets.split(","), decoders(a.decoders.split(","))
    groups = [members] + ([[m] for m in members] if a.each and len(members) > 1 else [])
    raw = [evaluate(g, sets, decs, a.tta, a.procs) for g in groups]
    res = {"members": a.members, "tta": a.tta, "rows": {}}
    for key in raw[0]:
        r = raw[0][key]
        res["rows"][key] = {**summ(r["ed"], r["n"], r.get("wed"), r.get("wn"), r.get("wok")), "hyps": r["hyps"]}
        if "hyp" in r:
            res["rows"][key]["hyp_by_line"] = r["hyp"]
        if len(raw) > 1:
            ed = np.mean([x[key]["ed"] for x in raw[1:]], 0)
            wed = np.mean([x[key]["wed"] for x in raw[1:]], 0) if "wed" in r else None
            wok = np.mean([x[key]["wok"] for x in raw[1:]], 0) if "wok" in r else None
            res["rows"][key]["seedmean"] = summ(ed, r["n"], wed, r.get("wn"), wok)
            res["rows"][key]["per_member"] = [{"cer": float(np.sum(x[key]["ed"]) / np.sum(r["n"])),
                                               **({"wer": float(np.sum(x[key]["wed"]) / np.sum(r["wn"]))}
                                                  if "wed" in r else {})} for x in raw[1:]]
    if a.ref and (RES / f"{a.ref}.json").exists():
        ref = json.loads((RES / f"{a.ref}.json").read_text())["rows"]
        for key, r in res["rows"].items():
            if key in ref and len(ref[key]["ed"]) == len(r["ed"]):
                r["d_cer_vs_ref"] = S.boot_delta(np.array(ref[key]["ed"]), np.array(r["ed"]), np.array(r["n"]))
                if "wed" in r and "wed" in ref[key]:
                    r["d_wer_vs_ref"] = S.boot_delta(np.array(ref[key]["wed"]), np.array(r["wed"]), np.array(r["wn"]))
        res["ref"] = a.ref
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{a.name}.json").write_text(json.dumps(res, indent=1, default=float))
    print_rows(a.name, res)
    return 0


def print_rows(name, res):
    for key, r in res["rows"].items():
        msg = f"{name:<28} {key:<20} CER {r['cer']:.3f} [{r['cer_ci'][0]:.3f},{r['cer_ci'][1]:.3f}]"
        if "wer" in r:
            msg += f" WER {r['wer']:.3f} [{r['wer_ci'][0]:.3f},{r['wer_ci'][1]:.3f}]"
        if "seedmean" in r:
            sm = r["seedmean"]
            msg += f" | seedmean CER {sm['cer']:.3f}" + (f" WER {sm['wer']:.3f}" if "wer" in sm else "")
        if "d_wer_vs_ref" in r:
            msg += f" | dWER {r['d_wer_vs_ref'][0]:+.3f} [{r['d_wer_vs_ref'][1]:+.3f},{r['d_wer_vs_ref'][2]:+.3f}]"
        elif "d_cer_vs_ref" in r:
            msg += f" | dCER {r['d_cer_vs_ref'][0]:+.3f} [{r['d_cer_vs_ref'][1]:+.3f},{r['d_cer_vs_ref'][2]:+.3f}]"
        print(msg, flush=True)


def cmd_table(a) -> int:
    for f in sorted(RES.glob("*.json")):
        if not a.filter or any(x in f.stem for x in a.filter.split(",")):
            print_rows(f.stem, json.loads(f.read_text()))
    return 0


# ============================================================================ pseudo-labels (noisy student)
def split_long(L, times, segs, maxlen: float):
    """Split segments longer than maxlen seconds at the middle of their longest no-emission run."""
    pnb = 1 - np.exp(np.logaddexp(L[:, S.BLANK], L[:, S.OTHER]))
    out = []

    def rec(s0, s1):
        if times[min(s1, len(times)) - 1] - times[s0] <= maxlen or s1 - s0 < 40:
            out.append((s0, s1))
            return
        low = np.r_[False, pnb[s0 + 10:s1 - 10] < 0.5, False]
        d = np.diff(low.astype(int))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(starts):
            k = int(np.argmax(ends - starts))
            c = s0 + 10 + (starts[k] + ends[k]) // 2
        else:
            c = (s0 + s1) // 2
        rec(s0, c)
        rec(c, s1)

    for s0, s1 in segs:
        rec(s0, s1)
    return out


def cmd_pseudo(a) -> int:
    """Confidence filter: Qwen word hypothesis and open-vocab char hypothesis agree (char edit rate <= maxdis)."""
    from phase0.analysis.decode import edit_distance
    if a.lpfile:
        L, sid = np.load(a.lpfile)[a.group].astype(np.float32), a.sid
    else:
        L, sid = ensemble(parse_members(a.members), a.set, a.tta), SID[a.set[:-5]]
    st = C.stream(sid)
    times = st.t[::2][:len(L)]
    segs = split_long(L, times, S2.segments(L, times, a.gap), a.maxlen)
    decs = decoders(["qwen", "char"])
    hy = decode_all([(decs["qwen"], L[s0:s1]) for s0, s1 in segs] + [(decs["char"], L[s0:s1]) for s0, s1 in segs],
                    a.procs)
    q, ch = hy[:len(segs)], hy[len(segs):]
    truth = S.desk_windows(st) if sid in (DESK, C.FREE) else []
    rows = []
    for (s0, s1), hq, hc in zip(segs, q, ch):
        t0, t1 = float(times[s0]), float(times[min(s1, len(times)) - 1])
        dis = edit_distance(hq, hc) / max(1, len(hq))
        r = {"t0": t0, "t1": t1, "text": hq, "char": hc, "dis": dis, "words": len(hq.split())}
        if truth:  # diagnostic only; never used for training or selection
            ov = [max(0.0, min(t1, w[1]) - max(t0, w[0])) for w in truth]
            j = int(np.argmax(ov))
            r["truth_diag"] = truth[j][2]
            r["cer_vs_truth_diag"] = edit_distance(truth[j][2], hq) / max(1, len(truth[j][2]))
        rows.append(r)
    keep = [r for r in rows if r["dis"] <= a.maxdis and r["words"] >= a.minwords and r["t1"] - r["t0"] <= 16]
    out = {"sid": sid, "members": a.members, "set": a.set, "tta": a.tta, "maxdis": a.maxdis, "keep": keep, "all": rows}
    if truth:
        for nm, rr in (("all", rows), ("keep", keep)):
            if rr:
                out[f"diag_cer_{nm}"] = float(np.mean([r["cer_vs_truth_diag"] for r in rr]))
    (C.ROOT / "pseudo").mkdir(parents=True, exist_ok=True)
    (C.ROOT / "pseudo" / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    print(f"pseudo {a.name}: {len(segs)} segments, kept {len(keep)}; diag CER all "
          f"{out.get('diag_cer_all', float('nan')):.3f} keep {out.get('diag_cer_keep', float('nan')):.3f}")
    for r in rows:
        print(f"  {'KEEP' if r in keep else 'drop'} dis {r['dis']:.2f} [{r['t0'] - st.t[0]:6.1f}-{r['t1'] - st.t[0]:6.1f}] "
              f"{r['text']!r} | char {r['char']!r}" + (f" | truth {r['truth_diag']!r}" if truth else ""))
    return 0


# ============================================================================ decipher2 / llmdec bridge
def cmd_mkdecipher(a) -> int:
    """decipher.json + .cache/decipher/out/<sid>_lp.npz (groups zs, desk) in decipher.py's format, from v3 posteriors."""
    import shutil
    if a.lpfile:
        z = np.load(a.lpfile)
        lps = {g: z[src].astype(np.float32) for g, src in (kv.split("=") for kv in a.map.split(","))}
        info = {"lpfile": a.lpfile, "map": a.map}
    else:
        lps = {"zs": ensemble(parse_members(a.zs), a.set, False), "desk": ensemble(parse_members(a.desk), a.set, False)}
        info = {"zs": a.zs, "desk": a.desk, "set": a.set,
                "cv": "ft_* members composed out-of-fold: each frame from the fold that held out the phrase shown then"}
    st = C.stream(a.src)
    G = min(len(v) for v in lps.values())
    lps = {g: v[:G] for g, v in lps.items()}
    times = st.t[::2][:G]
    comb = np.mean(list(lps.values()), 0)   # decipher.cmd_remote: common segmentation on mean evidence of all groups
    segs = S2.segments(comb, times, 2.0, pad=0.5, thr=0.5)
    if a.maxlen > 0:
        segs = split_long(comb, times, segs, a.maxlen)
    decs = decoders(["qwen", "char", "greedy"])
    jobs = [(dec, lps[g][s0:s1]) for g in lps for dec in decs.values() for s0, s1 in segs]
    hy = decode_all(jobs, a.procs)
    rows = [{"i": k + 1, "t0": float(times[s0] - st.t[0]), "t1": float(times[min(s1, G) - 1] - st.t[0]),
             "frames30": [int(s0), int(s1)]} for k, (s0, s1) in enumerate(segs)]
    k = 0
    for g in lps:
        for dn in decs:
            for r in rows:
                r.setdefault(g, {})[dn] = hy[k]
                k += 1
    sdir = C.SESS / a.out_sid
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "decipher.json").write_text(json.dumps(
        {"session": a.out_sid, "source_session": a.src, "ctc": "v3", "video_s": float(st.t[-1] - st.t[0]), "gap": 2.0,
         "pad": 0.5, "thr": 0.5, "maxlen": a.maxlen, "posteriors": info, "segments": rows}, indent=1))
    if a.truth_desk:
        (sdir / "truth.txt").write_text("\n".join(w[2] for w in S.desk_windows(C.stream(DESK))) + "\n")
    elif a.truth:
        shutil.copy(a.truth, sdir / "truth.txt")
    out = Path(".cache/decipher/out")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.out_sid}_lp.npz", times=times, **{g: v.astype(np.float16) for g, v in lps.items()})
    print(f"mkdecipher {a.out_sid}: {len(rows)} segments, groups {list(lps)} -> {sdir / 'decipher.json'}, "
          f"{out / (a.out_sid + '_lp.npz')}", flush=True)
    return 0


# ============================================================================ publishing
def cmd_export(a) -> int:
    """Ensemble posteriors: old session out-of-fold (per fold + composed), blind-1, zero-shot; float16 log-probs."""
    import shutil
    out = C.ROOT / "posteriors" / a.name
    out.mkdir(parents=True, exist_ok=True)
    desk = C.stream(DESK)
    wins = S.desk_windows(desk)
    meta = {"symbols": S.SYMS, "blank": 0, "other": 28, "frame_rate_hz": 30, "log_probs": "float16, ensemble log-mean",
            "segmentation_used_in_eval": {"gap": 2.0, "pad": 0.5, "thr": 0.5}}
    if a.cv:
        cv = parse_members(a.cv)
        F = C.logmean([np.load(member_path(m, "desk_cont", False)).astype(np.float32) for m in cv])
        lps, refs = ensemble(cv, "desk_win", False)
        off = np.cumsum([0] + [len(x) for x in lps])
        np.savez_compressed(out / "old_session_cv.npz", folds_cont=F.astype(np.float16),
                            cont_out_of_fold=compose_ft(F).astype(np.float16), times=desk.t[::2][:F.shape[1]],
                            t_start=desk.t[0], win_lp=np.concatenate(lps).astype(np.float16), win_off=off,
                            win_t0=[w[0] for w in wins], win_t1=[w[1] for w in wins], win_text=refs,
                            win_fold=[j % 5 for j in range(len(wins))])
        meta["old_session_cv"] = {"members": a.cv, "session": DESK, "folds": "phrase j held out in fold j % 5",
                                  "cont_out_of_fold": "each frame from the fold that held out the phrase shown then"}
    for key, spec in (("blind1", a.blind1), ("zeroshot_blind1", a.zs)):
        if spec:
            mem = parse_members(spec)
            L = ensemble(mem, "blind1_cont", False)
            st = C.stream(C.BLIND1)
            np.savez_compressed(out / f"{key}.npz", lp=L.astype(np.float16), times=st.t[::2][:len(L)], t_start=st.t[0])
            meta[key] = {"members": spec, "session": C.BLIND1}
    if a.zs:
        mem = parse_members(a.zs)
        L = ensemble(mem, "desk_cont", False)
        np.savez_compressed(out / "zeroshot_old_session.npz", lp=L.astype(np.float16), times=desk.t[::2][:len(L)],
                            t_start=desk.t[0])
        meta["zeroshot_old_session"] = {"members": a.zs}
    for f in a.results.split(","):
        if f and (RES / f"{f}.json").exists():
            shutil.copy(RES / f"{f}.json", out / f"result_{f}.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"exported {out}: {sorted(p.name for p in out.iterdir())}")
    return 0


def cmd_deploy(a) -> int:
    """deploy/manifest.json for .cache/ctc_v3/predict.py: groups 'name=tag:seeds:kind+...;name2=...'."""
    import shutil
    dd = C.ROOT / "deploy"
    (dd / "models").mkdir(parents=True, exist_ok=True)
    groups = {}
    for g in a.groups.split(";"):
        name, spec = g.split("=")
        mem = []
        for tag, seed, kind in parse_members(spec):
            src = C.rdir(tag, seed) / ("all.pt" if kind == "zs" else f"{kind}/model.pt")
            dst = dd / "models" / f"{tag}_s{seed}_{kind}.pt"
            shutil.copy(src, dst)
            mem.append({"model": f"deploy/models/{dst.name}", "recipe": C.recipe(tag), "tag": tag, "seed": seed, "kind": kind,
                        "trained_on": json.loads((src.parent / "meta.json").read_text()) if kind != "zs" else "keyboard+HWT"})
        groups[name] = {"members": mem, "tta": False}
    man = {"primary": a.primary, "segments": {"gap": 2.0, "pad": 0.5, "thr": 0.5, "maxlen": 15.0}, "groups": groups}
    (dd / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"wrote {dd / 'manifest.json'}: { {k: len(v['members']) for k, v in groups.items()} }")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("mkdecipher")
    s.add_argument("--out-sid", required=True)
    s.add_argument("--src", required=True, help="real session id whose stream the posteriors belong to")
    s.add_argument("--lpfile", default="")
    s.add_argument("--map", default="zs=zeroshot,desk=desk")
    s.add_argument("--zs", default="")
    s.add_argument("--desk", default="")
    s.add_argument("--set", default="desk_cont")
    s.add_argument("--maxlen", type=float, default=0.0)
    s.add_argument("--truth", default="")
    s.add_argument("--truth-desk", action="store_true")
    s.add_argument("--procs", type=int, default=2)
    s.set_defaults(fn=cmd_mkdecipher)
    s = sub.add_parser("export")
    s.add_argument("--name", required=True)
    s.add_argument("--cv", default="")
    s.add_argument("--blind1", default="")
    s.add_argument("--zs", default="")
    s.add_argument("--results", default="")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("deploy")
    s.add_argument("--groups", required=True)
    s.add_argument("--primary", required=True)
    s.set_defaults(fn=cmd_deploy)
    s = sub.add_parser("run")
    s.add_argument("--name", required=True)
    s.add_argument("--members", required=True)
    s.add_argument("--sets", default="desk_win,desk_cont")
    s.add_argument("--decoders", default="greedy,char,qwen")
    s.add_argument("--tta", action="store_true")
    s.add_argument("--each", action="store_true")
    s.add_argument("--procs", type=int, default=3)
    s.add_argument("--ref", default="")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("pseudo")
    s.add_argument("--name", required=True)
    s.add_argument("--members", required=True)
    s.add_argument("--set", default="free_cont")
    s.add_argument("--tta", action="store_true")
    s.add_argument("--gap", type=float, default=2.0)
    s.add_argument("--maxdis", type=float, default=0.25)
    s.add_argument("--minwords", type=int, default=3)
    s.add_argument("--maxlen", type=float, default=10.0)
    s.add_argument("--lpfile", default="")
    s.add_argument("--group", default="")
    s.add_argument("--sid", default="")
    s.add_argument("--procs", type=int, default=3)
    s.set_defaults(fn=cmd_pseudo)
    s = sub.add_parser("table")
    s.add_argument("--filter", default="")
    s.set_defaults(fn=cmd_table)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
