"""P0 error anatomy: every word error of the frozen v3 decoder on old desk (CV-safe), blind-1, blind-2, classified
(a) boundary/dropped at segment edge, (b) not in pool & CTC favours truth, (c) not in pool & CTC disfavours truth,
(d) in pool not selected, (e) out of lexicon, (f) name. Exact CTC log-lik truth vs output per segment and per error span
(hybrid = output with the span replaced by truth), Viterbi frames/char of the truth span.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p0"""
from __future__ import annotations

import json
import os

import numpy as np

os.environ["LLMDEC_FUZZY_TAG"] = ""
from phase0.analysis import swipe_common as C  # noqa: E402


def spans(ops):
    out, cur = [], []
    for op in ops:
        if op[0] == "ok":
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(op)
    if cur:
        out.append(cur)
    return out


def run(name):
    from phase0.analysis import llmdec as LD
    segs, lines = C.load(name)
    setname = C.SETS[name][0]
    _, pools, _ = LD.build_pools(setname, ("8b", "p05b"))
    pool = {p["id"]: p["P"]["ens" if "ens" in p["P"] else "zs"]["texts"] for p in pools}
    lex = C.merged_lex()
    rows, segrows = [], []
    for s in segs:
        tw, hw = s["truth"].split(), s["hyp"].split()
        ops = C.word_align(tw, hw)
        P = [C.norm(x) for x in pool[s["id"]]]
        llt, llh = C.ctc_ens(s["lps"], [s["truth"], s["hyp"]])
        pool_ll = C.ctc_ens(s["lps"], P) if P else np.array([-np.inf])
        from phase0.analysis.decode import edit_distance
        orc = min(edit_distance(tw, x.split()) for x in P) if P else len(tw)
        M = C.ens28(s["lps"])
        lf = C.letter_frames(M, s["truth"]) if s["truth"] else []
        segrows.append({"set": name, "id": s["id"], "truth": s["truth"], "hyp": s["hyp"], "ll_truth": float(llt),
                        "ll_hyp": float(llh), "truth_in_pool": s["truth"] in P, "pool_size": len(P), "pool_oracle_wed": int(orc),
                        "wed": int(edit_distance(tw, hw)), "n": len(tw), "T": len(M), "max_pool_ll": float(pool_ll.max())})
        # char offsets of truth words
        offs, k = [], 0
        for w in tw:
            offs.append((k, k + len(w)))
            k += len(w) + 1
        for sp in spans(ops):
            ti = [i for op, i, j in sp if op in ("sub", "del")]
            hj = [j for op, i, j in sp if op in ("sub", "ins")]
            i0 = min(ti) if ti else sp[0][1]
            i1 = max(ti) + 1 if ti else i0
            j0 = min(hj) if hj else None
            # hybrid: hyp with the aligned hyp words replaced by the truth words
            if hj:
                hyb = hw[:min(hj)] + tw[i0:i1] + hw[max(hj) + 1:]
            else:   # pure deletion: insert truth words at the hyp position aligned before
                prev_j = max([j for op, i, j in ops if j is not None and op in ("ok", "sub") and i < i0] or [-1])
                hyb = hw[:prev_j + 1] + tw[i0:i1] + hw[prev_j + 1:]
            hyb = " ".join(hyb)
            llhy = C.ctc_ens(s["lps"], [hyb])[0]
            twords = tw[i0:i1]
            in_pool = any(all(w in x.split() for w in twords) and
                          sum(1 for op in C.word_align(tw, x.split()) if op[0] == "ok" and i0 <= op[1] < i1) == len(twords)
                          for x in P) if twords else False
            edge = bool(twords) and all(op == "del" for op, _, _ in sp) and (i0 == 0 or i1 == len(tw))
            fr = [lf[c] for w in range(i0, i1) for c in range(*offs[w]) if lf and lf[c][0] >= 0]
            fpc = (len(set(range(fr[0][0], fr[-1][1] + 1))) / max(sum(len(w) for w in twords), 1)) if fr else None
            oov = [w for w in twords if w not in lex]
            names = [w for w in twords if w in C.NAMES]
            d = float(llhy - llh)
            if edge:
                typ = "a"
            elif oov:
                typ = "e"
            elif names:
                typ = "f"
            elif in_pool:
                typ = "d"
            else:
                typ = "b" if d > 0 else "c"
            n_err = max(len(ti), len(hj)) if ti or hj else 0
            rows.append({"set": name, "id": s["id"], "truth_span": " ".join(twords), "hyp_span": " ".join(hw[j] for j in hj),
                         "n_word_errors": n_err, "n_truth_words": len(twords), "type": typ, "flags": {
                             "edge_deletion": edge, "oov": oov, "name": names, "in_pool": in_pool, "delta_ll_truth_minus_hyp": d},
                         "frames_per_char_truth": fpc, "hybrid": hyb})
    return rows, segrows


def main():
    allr, alls = [], []
    for name in ("old", "b1", "b2"):
        r, s = run(name)
        allr += r
        alls += s
        print(name, "segments", len(s), "error spans", len(r), flush=True)
        for x in r:
            print(f"  [{x['type']}] {x['id']:>5} truth='{x['truth_span']}' hyp='{x['hyp_span']}' dLL={x['flags']['delta_ll_truth_minus_hyp']:+.1f} "
                  f"pool={x['flags']['in_pool']} fpc={x['frames_per_char_truth']}", flush=True)
    summ = {}
    for name in ("old", "b1", "b2"):
        R = [x for x in allr if x["set"] == name]
        S = [x for x in alls if x["set"] == name]
        summ[name] = {"n_words": sum(x["n"] for x in S), "wed_segwise": sum(x["wed"] for x in S),
                      "pool_oracle_wed": sum(x["pool_oracle_wed"] for x in S),
                      "errors_by_type": {t: sum(x["n_word_errors"] for x in R if x["type"] == t) for t in "abcdef"},
                      "spans_by_type": {t: sum(1 for x in R if x["type"] == t) for t in "abcdef"},
                      "segments_ll_truth_gt_hyp": sum(1 for x in S if x["ll_truth"] > x["ll_hyp"] and x["truth"] != x["hyp"]),
                      "segments_wrong": sum(1 for x in S if x["truth"] != x["hyp"])}
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "p0_error_anatomy.json").write_text(json.dumps({"summary": summ, "spans": allr, "segments": alls}, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
