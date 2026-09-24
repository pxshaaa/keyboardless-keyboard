"""P2 swipe-path word scorer on KEYBOARD sessions (keylogger truth). LOSO over 5 kbd sessions, tap_g-style frame,
per-key Gaussian + P(finger|key) from the other sessions; each clean typed word (keylog boundaries, no backspace)
is ranked against all merged-lexicon words of length n-2..n+2 by location (DP with ins/del), shape (SHARK2), prior.
Variants: det = detector finger + its fingertip at the detected tap frame (taps_contact; finger model in-sample for
015217/021315, out-of-sample lo1316 file for 131629); oracle = touch-typing finger from the key label (upper bound,
not obtainable on desk). Simulated tap drop/insert at 10/20%.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p2 {extract|eval}"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_motor as SM

SESS = ["20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd", "20260911-164237-kbd", "20260910-015948-kbd"]
TAPF = {"20260910-131629-kbd": "taps_contact_lo1316.jsonl"}
FEAT = C.CACHE / "p2_feats.npz"


def words_of(downs):
    words, cur, bad = [], [], False
    for k, e in enumerate(downs + [{"key": "space", "t": 1e18}]):
        key = e["key"]
        if len(key) == 1 and "a" <= key <= "z":
            cur.append(k)
            continue
        if key == "backspace":
            bad = True
            continue
        if key == "shift":
            continue
        if cur and not bad:
            words.append(cur)
        cur, bad = [], False
    return words


def extract():
    from phase0.analysis import tap_pos as tp
    import __main__
    __main__.Sess = tp.Sess
    out = {}
    for sid in SESS:
        t0 = time.time()
        S = tp.load_sess(sid)
        sd = Path("data/sessions") / sid
        downs = [e for e in map(json.loads, (sd / "keys.jsonl").read_text().splitlines()) if e["event"] == "down"]
        taps = [json.loads(l) for l in (sd / TAPF.get(sid, "taps_contact.jsonl")).read_text().splitlines()]
        tt = np.array([t["t"] for t in taps])
        trow = np.clip(np.searchsorted(S.frames, [t["i"] for t in taps]), 0, len(S.frames) - 1)
        tmpl = SM.template(S.P, trow)
        letters = [k for k, e in enumerate(downs) if len(e["key"]) == 1 and "a" <= e["key"] <= "z"]
        rec = {"key": [], "det_xy": [], "det_f": [], "or_xy": [], "or_f": [], "t": [], "tips": []}
        kpos = {}
        for k in letters:
            e = downs[k]
            j = int(np.argmin(np.abs(tt - e["t"])))
            ok = abs(tt[j] - e["t"]) <= 0.08
            r = int(trow[j]) if ok else -1
            xy = S.P[r, taps[j]["hand"], taps[j]["finger"], :2] if ok else np.array([np.nan, np.nan])
            rk = int(np.clip(np.searchsorted(S.t, e["t"]), 0, len(S.t) - 1))
            g = SM.gt_fid(e["key"])
            rr = r if ok else rk
            oxy = S.P[rr, g[0], g[1], :2] if g else np.array([np.nan, np.nan])
            kpos[k] = len(rec["key"])
            rec["key"].append(SM.LI[e["key"]])
            rec["det_xy"].append(xy)
            rec["det_f"].append(SM.fid(taps[j]["hand"], taps[j]["finger"]) if ok else -1)
            rec["or_xy"].append(oxy)
            rec["or_f"].append(SM.fid(*g) if g else -1)
            rec["t"].append(e["t"])
            rec["tips"].append(np.stack([S.P[rr, h, j, :2] for h in (0, 1) for j in (4, 8, 12, 16, 20)]))
        ws = words_of(downs)
        widx = [[kpos[k] for k in w] for w in ws]
        wtext = ["".join(downs[k]["key"] for k in w) for w in ws]
        for kk, v in rec.items():
            out[f"{sid}|{kk}"] = np.array(v)
        out[f"{sid}|tmpl"] = tmpl
        out[f"{sid}|wtext"] = np.array(wtext)
        out[f"{sid}|wstart"] = np.array([w[0] for w in widx] if widx else [], int)
        out[f"{sid}|wlen"] = np.array([len(w) for w in widx], int)
        print(sid, "letters", len(letters), "paired", int(np.sum(np.isfinite(np.array(rec['det_xy'])[:, 0]))), "words", len(ws),
              f"{time.time() - t0:.0f}s", flush=True)
        del S
        tp._MEM.clear()
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEAT, **out)


def load_feats():
    z = np.load(FEAT)
    return {k: z[k] for k in z.files}


_G: dict = {}


def fold(held):
    F = _G["F"]
    lex = _G["lex"]
    ref = SM.canonical([F[f"{s}|tmpl"] for s in SESS if s != held])
    Tr = {s: SM.to_frame(F[f"{s}|tmpl"], ref) for s in SESS}
    res = {}
    rng = np.random.default_rng(0)
    for var, xyk, fk in (("det", "det_xy", "det_f"), ("oracle", "or_xy", "or_f")):
        Xs, Ks, Fs = [], [], []
        for s in SESS:
            if s == held:
                continue
            X = SM.apply_sim(F[f"{s}|{xyk}"], Tr[s])
            m = F[f"{s}|{fk}"] >= 0
            Xs.append(X[m]); Ks.append(F[f"{s}|key"][m]); Fs.append(F[f"{s}|{fk}"][m])
        km = SM.KeyModel(np.concatenate(Xs), np.concatenate(Ks), np.concatenate(Fs))
        tpl = SM.Templates(lex, km)
        Xh = SM.apply_sim(F[f"{held}|{xyk}"], Tr[held])
        fh = F[f"{held}|{fk}"]
        kh = F[f"{held}|key"]
        okall = np.isfinite(Xh).all(1) & (fh >= 0)
        # per-tap key-ID accuracy (context)
        ll_all = km.tap_ll(Xh[okall], fh[okall])
        ll_pos = km.tap_ll(Xh[okall], None, 0.0)
        res[f"{var}/tap_key_top1"] = {"pos+finger": float((ll_all.argmax(1) == kh[okall]).mean()),
                                      "pos_only": float((ll_pos.argmax(1) == kh[okall]).mean()), "n": int(okall.sum())}
        rows = []
        for w, st, ln in zip(F[f"{held}|wtext"], F[f"{held}|wstart"], F[f"{held}|wlen"]):
            w = str(w)
            if w not in lex.index:
                rows.append({"w": w, "oov": True})
                continue
            ix = np.arange(st, st + ln)
            if not okall[ix].all():
                rows.append({"w": w, "untracked": True})
                continue
            r = {"w": w, "n": int(ln)}
            for p in (0.0, 0.1, 0.2):
                for seed in ((0,) if p == 0 else (0, 1, 2)):
                    rs = np.random.default_rng([seed, abs(hash(w)) % 10000, int(st)])
                    X, f = [], []
                    pool_ix = np.where(okall)[0]
                    for k in ix:
                        if p and rs.random() < p:
                            pass
                        else:
                            X.append(Xh[k]); f.append(fh[k])
                        if p and rs.random() < p:
                            q = pool_ix[rs.integers(len(pool_ix))]
                            X.append(Xh[q]); f.append(fh[q])
                    if not X:
                        r[f"p{p}_s{seed}"] = None
                        continue
                    X, f = np.array(X), np.array(f)
                    n = len(X)
                    lens = range(max(1, n - 2), min(16, n + 2) + 1)
                    pi = max(p, 0.02)
                    cell = {}
                    for wf in ((0.0, 1.0) if p == 0 else (1.0,)):
                        sc = SM.score_words(km.tap_ll(X, f, wf), X, lex, tpl, lens, p_ins=pi, p_del=pi, log_bg=km.log_bg)
                        for nm, (wl, wsh, wp) in {"loc": (1, 0, 0), "shape": (0, 1, 0), "loc+shape": (1, 2, 0),
                                                  "loc+prior": (1, 0, 0.5), "loc+shape+prior": (1, 2, 0.5),
                                                  "loc+shape5+prior1": (1, 5, 1.0), "loc+prior1": (1, 0, 1.0),
                                                  "shape+prior": (0, 1, 0.5)}.items():
                            if wf == 0.0 and nm not in ("loc", "loc+prior", "loc+shape+prior"):
                                continue
                            rk, top = SM.rank_of(lex.index[w], lex, sc, wl, wsh, wp)
                            cell[f"{nm}{'' if wf else '|nofinger'}"] = rk
                    r[f"p{p}_s{seed}"] = cell
            rows.append(r)
        res[var] = rows
        if var == "oracle":
            km_or = km
    from phase0.analysis.swipe_p3 import FK, ll_table
    T = F[f"{held}|tips"]
    Th = SM.apply_sim(T.reshape(-1, 2), Tr[held]).reshape(T.shape)
    kh = F[f"{held}|key"]
    llf = ll_table(Th, km_or)
    okh = np.isfinite(Th[np.arange(len(Th)), FK[kh]]).all(1)
    res["hyp/tap_key_top1"] = {"pos+finger": float((llf[okh].argmax(1) == kh[okh]).mean()), "pos_only": None, "n": int(okh.sum())}
    rows = []
    for w, st, ln in zip(F[f"{held}|wtext"], F[f"{held}|wstart"], F[f"{held}|wlen"]):
        w = str(w)
        if w not in lex.index:
            rows.append({"w": w, "oov": True})
            continue
        ix = np.arange(st, st + ln)
        if not okh[ix].all():
            rows.append({"w": w, "untracked": True})
            continue
        r = {"w": w, "n": int(ln)}
        pool_ix = np.where(okh)[0]
        for p in (0.0, 0.1, 0.2):
            for seed in ((0,) if p == 0 else (0, 1, 2)):
                rs = np.random.default_rng([seed, abs(hash(w)) % 10000, int(st)])
                seq = []
                for k in ix:
                    if not (p and rs.random() < p):
                        seq.append(k)
                    if p and rs.random() < p:
                        seq.append(pool_ix[rs.integers(len(pool_ix))])
                if not seq:
                    r[f"p{p}_s{seed}"] = None
                    continue
                n = len(seq)
                lens = range(max(1, n - 2), min(16, n + 2) + 1)
                pi = max(p, 0.02)
                sc = SM.score_words(llf[seq], None, lex, None, lens, p_ins=pi, p_del=pi, log_bg=km_or.log_bg)
                cell = {}
                for nm, wp in (("loc", 0.0), ("loc+prior", 0.5), ("loc+prior1", 1.0)):
                    cell[nm] = SM.rank_of(lex.index[w], lex, sc, 1.0, 0.0, wp)[0]
                r[f"p{p}_s{seed}"] = cell
        rows.append(r)
    res["hyp"] = rows
    return held, res


def _init():
    _G["F"] = load_feats()
    from phase0.analysis import seqctc2 as S2
    B = S2.blocklist()
    lg = {w: v for w, v in C.merged_lex().items() if not S2.is_blocked(w, B)}
    _G["lex"] = SM.Lexicon(lg)


def summarise(allres):
    buckets = {"1-2": (1, 2), "3-4": (3, 4), "5-6": (5, 6), "7+": (7, 99)}
    summ = {}
    for var in ("det", "oracle", "hyp"):
        for held, res in allres.items():
            rows = res[var]
            ok = [r for r in rows if "n" in r]
            summ.setdefault(var, {})[held] = {"words": len(rows), "oov": sum(1 for r in rows if r.get("oov")),
                                              "untracked": sum(1 for r in rows if r.get("untracked")), "scored": len(ok),
                                              "tap_key_top1": res[f"{var}/tap_key_top1"]}
        pooled = [r for res in allres.values() for r in res[var] if "n" in r]
        agg = {}
        for p, seeds in ((0.0, (0,)), (0.1, (0, 1, 2)), (0.2, (0, 1, 2))):
            keys = set()
            for r in pooled:
                for s in seeds:
                    if r.get(f"p{p}_s{s}"):
                        keys |= set(r[f"p{p}_s{s}"])
            for ch in sorted(keys):
                for bn, (lo, hi) in list(buckets.items()) + [("all", (1, 99)), ("len>=3", (3, 99))]:
                    rk = [r[f"p{p}_s{s}"].get(ch) if r.get(f"p{p}_s{s}") else None for r in pooled if lo <= r["n"] <= hi for s in seeds]
                    if not rk:
                        continue
                    agg.setdefault(f"p{p}", {}).setdefault(ch, {})[bn] = {
                        "top1": float(np.mean([x == 1 for x in rk])), "top5": float(np.mean([x is not None and x <= 5 for x in rk])),
                        "n": len(rk) // len(seeds)}
        summ[var]["pooled"] = agg
    return summ


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "extract":
        extract()
        return
    helds = SESS[1:] + SESS[:1]
    with Pool(2, initializer=_init) as pool:
        allres = dict(pool.map(fold, helds))
    summ = summarise(allres)
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "p2_keyboard.json").write_text(json.dumps({"summary": summ, "per_word": allres}, default=float))
    for var in ("det", "oracle", "hyp"):
        print(var, {h: v for h, v in summ[var].items() if h != "pooled"})
        for p, d in summ[var]["pooled"].items():
            for ch, b in d.items():
                print(f"  {var} {p} {ch:<24} " + "  ".join(f"{bn}:{v['top1']:.2f}/{v['top5']:.2f}(n={v['n']})" for bn, v in b.items()))


if __name__ == "__main__":
    main()
