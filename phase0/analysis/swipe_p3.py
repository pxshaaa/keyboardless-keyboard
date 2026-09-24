"""P3 swipe-path word scorer on DESK (no tap truth). Per truth word span (CTC forced alignment of the truth = oracle
word boundaries), rank merged-lexicon words (len m-2..m+2) by
  template: hypothesis-conditioned fingertip location (letter k is read off the tip of k's touch-typing finger at the
            observation frame; label-free at test time), pair-HMM DP with ins/del, keyboard->desk via a tap_g-style
            similarity (desk template from CTC emission frames, label-free) + leave-one-phrase-out MAP adaptation;
  ctc:      exact CTC log-lik of the word over the span (+-3 frames), mean over model groups;
  fusion:   ctc + mu * template + nu * prior.
Observations: 'aligned' = one per truth letter at its Viterbi peak frame (optimistic: count+timing from truth);
'peaks' = local maxima of letter posterior mass in the span (label-free); 'taps_gb' = detector taps in the span (old desk).
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p3 [--sets old,b1,b2] [--procs 2]"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_motor as SM
from phase0.analysis.finger_id import FINGERTIP_JOINTS

KBD = ["20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd", "20260911-164237-kbd", "20260910-015948-kbd"]
FK = np.array([SM.fid(*SM.gt_fid(c)) for c in SM.LET])   # letter -> touch-typing finger id
PADF = 3


def tips_at(S, rows, T):
    out = np.full((len(rows), 10, 2), np.nan)
    for h in (0, 1):
        for fi, j in enumerate(FINGERTIP_JOINTS):
            out[:, h * 5 + fi] = S.P[rows, h, j, :2]
    return SM.apply_sim(out.reshape(-1, 2), T).reshape(out.shape)


def ll_table(tips, km):
    """[n,10,2] -> [n,26] log N(tip of letter k's finger ; mu_k, Sigma_k)."""
    X = tips[:, FK, :]
    d = X - km.mu[None]
    q = np.einsum("nki,kij,nkj->nk", d, km.icov, d)
    ll = -0.5 * q - 0.5 * km.logdet[None] - np.log(2 * np.pi)
    return np.where(np.isfinite(ll), ll, -50.0)


def prepare(name):
    from phase0.analysis import tap_pos as tp
    import __main__
    __main__.Sess = tp.Sess   # desk Sess caches were pickled from tap_pos run as __main__
    setname, sdir, src = C.SETS[name]
    segs, lines = C.load(name)
    dj = json.loads(Path(f"data/sessions/{sdir}/decipher.json").read_text())
    times = np.load(f".cache/decipher/out/{sdir}_lp.npz")["times"]
    S = tp.load_sess(src)
    F = dict(np.load(C.CACHE / "p2_feats.npz"))
    ref = SM.canonical([F[f"{k}|tmpl"] for k in KBD])
    Xs, Ks = [], []
    for k in KBD:
        Tk = SM.to_frame(F[f"{k}|tmpl"], ref)
        m = F[f"{k}|or_f"] >= 0
        Xs.append(SM.apply_sim(F[f"{k}|or_xy"][m], Tk))
        Ks.append(F[f"{k}|key"][m])
    Ks = np.concatenate(Ks)
    km_kbd = SM.KeyModel(np.concatenate(Xs), Ks, FK[Ks])
    emit_rows = []
    for s, row in zip(segs, dj["segments"]):
        s0, s1 = row["frames30"]
        assert s1 - s0 == len(s["lps"][0]), (name, s["id"], s1 - s0, len(s["lps"][0]))
        s["s0"] = s0
        M = C.ens28(s["lps"])
        s["M"] = M
        fr = np.where(1 - np.exp(M[:, 0]) > 0.5)[0]
        emit_rows += list(np.searchsorted(S.t, times[s0 + fr]))
    emit_rows = np.clip(np.array(emit_rows), 0, len(S.t) - 1)
    Td = SM.to_frame(SM.template(S.P, emit_rows), ref)
    gb = None
    gbf = Path(f"data/sessions/{src}/taps_gb.jsonl")
    if gbf.exists():
        gb = np.array([json.loads(l)["t"] for l in gbf.read_text().splitlines()])
    words = []
    for si, s in enumerate(segs):
        if not s["truth"]:
            continue
        lf = C.letter_frames(s["M"], s["truth"])
        k = 0
        for w in s["truth"].split():
            ch = lf[k:k + len(w)]
            k += len(w) + 1
            peaks = [c[2] for c in ch]
            if any(p < 0 for p in peaks):
                continue
            rows = np.clip(np.searchsorted(S.t, times[s["s0"] + np.array(peaks)]), 0, len(S.t) - 1)
            f0, f1 = min(c[0] for c in ch), max(c[1] for c in ch)
            a, b = max(0, f0 - PADF), min(len(s["M"]), f1 + PADF + 1)
            let = 1 - np.exp(np.logaddexp(s["M"][a:b, 0], s["M"][a:b, 27]))
            pk = [a + i for i in range(len(let)) if let[i] > 0.3 and (i == 0 or let[i] >= let[i - 1]) and (i == len(let) - 1 or let[i] > let[i + 1])]
            prow = np.clip(np.searchsorted(S.t, times[s["s0"] + np.array(pk, int)]), 0, len(S.t) - 1) if pk else np.zeros(0, int)
            rec = {"set": name, "seg": si, "id": s["id"], "w": w, "keys": np.array([SM.LI[c] for c in w]),
                   "tips_aligned": tips_at(S, rows, Td), "tips_peaks": tips_at(S, prow, Td) if len(prow) else np.zeros((0, 10, 2)),
                   "span": (a, b)}
            if gb is not None:
                t0, t1 = times[s["s0"] + a], times[s["s0"] + b - 1]
                gt = gb[(gb >= t0) & (gb <= t1)]
                rec["tips_gb"] = tips_at(S, np.clip(np.searchsorted(S.t, gt), 0, len(S.t) - 1), Td) if len(gt) else np.zeros((0, 10, 2))
            words.append(rec)
    del S
    return segs, words, km_kbd


_G: dict = {}


def _init():
    from phase0.analysis import seqctc2 as S2
    B = S2.blocklist()
    _G["lex"] = SM.Lexicon({w: v for w, v in C.merged_lex().items() if not S2.is_blocked(w, B)})


def ctc_job(args):
    key, lps, lens = args
    lex = _G["lex"]
    cands = [lex.words[i] for m in lens for i in lex.by_len.get(m, [])]
    return key, C.ctc_ens(lps, cands).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="old,b1,b2")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    _init()
    lex = _G["lex"]
    out_rows = []
    for name in a.sets.split(","):
        t0 = time.time()
        segs, words, km_kbd = prepare(name)
        print(name, "words", len(words), f"prep {time.time() - t0:.0f}s", flush=True)
        # CTC word ranking over the span (cached)
        cf = C.CACHE / f"p3_ctc_{name}.npz"
        cache = dict(np.load(cf)) if cf.exists() else {}
        jobs = []
        for wi, r in enumerate(words):
            m = len(r["w"])
            lens = list(range(max(1, m - 2), m + 3))
            r["lens"] = lens
            key = f"{r['id']}|{wi}"
            if key not in cache:
                s = segs[r["seg"]]
                jobs.append((key, [l[r["span"][0]:r["span"][1]] for l in s["lps"]], lens))
        if jobs:
            with Pool(a.procs, initializer=_init) as pool:
                for key, v in pool.imap_unordered(ctc_job, jobs):
                    cache[key] = v
            np.savez_compressed(cf, **cache)
        print(name, f"ctc done {time.time() - t0:.0f}s", flush=True)
        # LOPO key models
        for wi, r in enumerate(words):
            lex_ix = np.concatenate([lex.by_len.get(m, np.zeros(0, int)) for m in r["lens"]])
            if r["w"] not in lex.index:
                out_rows.append({"set": name, "id": r["id"], "w": r["w"], "oov": True})
                continue
            tgt = int(np.where(lex_ix == lex.index[r["w"]])[0][0])
            ctc = cache[f"{r['id']}|{wi}"]
            prior = lex.prior[lex_ix]
            others = [x for x in words if x["seg"] != r["seg"]]
            Xo = np.concatenate([x["tips_aligned"][np.arange(len(x["keys"])), FK[x["keys"]]] for x in others])
            Ko = np.concatenate([x["keys"] for x in others])
            models = {"kbd": km_kbd}
            for tau in (5.0, 20.0):
                models[f"map{int(tau)}"] = SM.KeyModel(Xo, Ko, FK[Ko], mean_prior=km_kbd.mu, tau_mean=tau)
            models["deskonly"] = SM.KeyModel(Xo, Ko, FK[Ko])
            row = {"set": name, "id": r["id"], "w": r["w"], "m": len(r["w"]), "n_cands": int(len(lex_ix)),
                   "n_peaks": int(len(r["tips_peaks"])), "n_gb": int(len(r["tips_gb"])) if "tips_gb" in r else None}
            ranks = {}

            def rk(v):
                return int((v > v[tgt]).sum()) + 1
            ranks["ctc"] = rk(ctc)
            ranks["prior"] = rk(prior)
            ranks["ctc+prior0.5"] = rk(ctc + 0.5 * prior)
            for obs in ("aligned", "peaks", "gb"):
                tips = r.get(f"tips_{obs}")
                if tips is None or not len(tips):
                    continue
                for mn, km in models.items():
                    if obs != "aligned" and mn not in ("kbd", "map20"):
                        continue
                    sc = SM.score_words(ll_table(tips, km), None, lex, None, r["lens"], p_ins=0.1, p_del=0.1, log_bg=km.log_bg)
                    loc = np.concatenate([sc[m][0] for m in r["lens"] if m in sc])
                    ranks[f"tpl[{obs},{mn}]"] = rk(loc)
                    ranks[f"tpl+prior[{obs},{mn}]"] = rk(loc + 0.5 * prior)
                    for mu in (0.1, 0.25, 0.5, 1.0):
                        ranks[f"ctc+{mu}tpl[{obs},{mn}]"] = rk(ctc + mu * loc)
                        ranks[f"ctc+{mu}tpl+0.5prior[{obs},{mn}]"] = rk(ctc + mu * loc + 0.5 * prior)
            row["ranks"] = ranks
            out_rows.append(row)
        print(name, f"scored {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for name in a.sets.split(","):
        R = [x for x in out_rows if x["set"] == name and "ranks" in x]
        keys = sorted({k for x in R for k in x["ranks"]})
        summ[name] = {"words": sum(1 for x in out_rows if x["set"] == name), "scored": len(R),
                      "oov": sum(1 for x in out_rows if x["set"] == name and x.get("oov"))}
        for k in keys:
            v = [x["ranks"][k] for x in R if k in x["ranks"]]
            summ[name][k] = {"top1": float(np.mean([x == 1 for x in v])), "top5": float(np.mean([x <= 5 for x in v])),
                             "top50": float(np.mean([x <= 50 for x in v])), "n": len(v)}
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "p3_desk.json").write_text(json.dumps({"summary": summ, "per_word": out_rows}, indent=1, default=float))
    for name, d in summ.items():
        print(name, {k: v for k, v in d.items() if not isinstance(v, dict)})
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {k:<40} top1 {v['top1']:.3f} top5 {v['top5']:.3f} top50 {v['top50']:.3f} n={v['n']}")


if __name__ == "__main__":
    main()
