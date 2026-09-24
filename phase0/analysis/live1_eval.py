"""Live demo review (results/live1): build = twin session zz-live-<ts> + proxy decodes; score = truth vs live/v4/proxy
+ per-wrong-word CTC evidence (llmdec.ctc_ll). See results/live1/SUMMARY.md."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

LIVE = Path("data/live/20260915-010345")
SID = "zz-live-20260915-010345"
OUT = Path("results/live1")
PAD_FRAMES = 60   # 2 s of pure blank between phrases (30 Hz)


def load_events():
    return [json.loads(l) for l in (LIVE / "events.jsonl").read_text().splitlines() if l.strip()]


def cmd_build(a) -> int:
    from phase0.analysis import ctcv3_eval as E
    from phase0.analysis import seqctc2 as S2
    from phase0.analysis.decipher import norm, SESS
    ev = load_events()
    lps = [np.load(LIVE / f"lp_{e['k']:03d}.npy").astype(np.float32) for e in ev]
    blank = np.full((PAD_FRAMES, 29), -20.0, np.float32)
    blank[:, 0] = 0.0
    parts, rows, pos = [], [], 0
    for e, lp in zip(ev, lps):
        s0, s1 = pos + PAD_FRAMES, pos + PAD_FRAMES + len(lp)
        parts += [blank, lp]
        pos = s1
        # t0/t1: place each segment on the live clock (its end = t_last_tap + 0.5 s pad), only used for word times
        t1 = e["t_last_tap"] + 0.5
        rows.append({"i": e["k"] + 1, "k": e["k"], "t0": t1 - (len(lp) - 1) / 30, "t1": t1, "frames30": [s0, s1]})
    parts.append(blank)
    full = np.concatenate(parts)
    times = np.arange(len(full)) / 30.0
    decs = E.decoders(["qwen", "char", "greedy"])
    from phase0.analysis import autocorrect as AC
    local = Path(".cache/decipher/qwen2.5-0.5b")   # mini: offline copy of the same Qwen2.5-0.5B (llmdec cands uses it)
    S2._SC["qwen"] = AC.NLM(str(local) if local.exists() else "Qwen/Qwen2.5-0.5B", device="cpu")
    proxy = []
    for e, lp, r in zip(ev, lps, rows):
        d = {"k": e["k"], "T": int(len(lp)), "seconds": round(len(lp) / 30, 2)}
        for name, dec in decs.items():
            t0 = time.time()
            hyp = S2.run_decoder(dec, lp)
            d[name] = norm(hyp)
            d[f"{name}_s"] = round(time.time() - t0, 2)
        for g in ("zs", "desk"):
            r[g] = {n: d[n] for n in decs}
        proxy.append(d)
        print(d, flush=True)
    sdir = SESS / SID
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "decipher.json").write_text(json.dumps({
        "session": SID, "source_session": str(LIVE), "ctc": "v3", "video_s": float(times[-1]), "gap": 1.5, "pad": 0.5, "thr": 0.5,
        "maxlen": 0.0,
        "posteriors": {"lpfile": str(LIVE / "lp_*.npy"), "map": "zs=desk_live,desk=desk_live",
                       "note": "live ctc_v4 desk ensemble (.cache/ctc_v4/deploy/manifest.json group desk, live_demo.py rolling "
                               "window) logged per phrase; the same array is used for BOTH groups because no zero-shot ensemble "
                               "ran live. 'ctc': 'v3' is only the marker decipher2_v4.prepare needs to accept a prebuilt twin."},
        "segments": rows}, indent=1))
    out = Path(".cache/decipher/out")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{SID}_lp.npz", times=times, zs=full.astype(np.float16), desk=full.astype(np.float16))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "proxy_decodes.json").write_text(json.dumps({"decoders": decs, "note": "CPU proxy decoders as in loso_v4.json "
                                                        "(ctcv3_eval.decoders); qwen = Qwen-0.5B word beam = the live decoder",
                                                        "phrases": proxy}, indent=1))
    print(f"built {sdir} ({len(rows)} segments, {len(full)} frames) and {out / (SID + '_lp.npz')}")
    return 0


# ---------------------------------------------------------------------------------------------------- scoring
def wer_stats(ref: str, hyp: str) -> dict:
    from phase0.analysis.decipher import word_align
    r, h = ref.split(), hyp.split()
    ops = word_align(r, h)
    return {"n": len(r), "ok": sum(o[0] == "ok" for o in ops), "sub": sum(o[0] == "sub" for o in ops),
            "del": sum(o[0] == "del" for o in ops), "ins": sum(o[0] == "ins" for o in ops), "ops": ops}


def evidence(lp: np.ndarray, truth: str, hyp: str, ops) -> list[dict]:
    """Per wrong word: CTC log-lik of the whole truth phrase vs the hyp phrase, and of a local swap (hyp with only this
    slot corrected -> isolates the word's own evidence when several words are wrong)."""
    from phase0.analysis.llmdec import ctc_ll
    r, h = truth.split(), hyp.split()
    ll_truth, ll_hyp = ctc_ll(lp, [truth, hyp])
    rows = []
    for op, i, j in ops:
        if op == "ok":
            continue
        fixed = list(h)
        if op == "sub":
            fixed[j] = r[i]
        elif op == "del":   # truth word missing in hyp: insert it after the previous aligned hyp word
            prev = max([jj for oo, ii, jj in ops if oo in ("ok", "sub") and ii < i] + [-1])
            fixed.insert(prev + 1, r[i])
        else:   # ins: hyp word with no truth counterpart -> remove it
            fixed.pop(j)
        ll_fixed = float(ctc_ll(lp, [" ".join(fixed)])[0])
        rows.append({"op": op, "truth_word": r[i] if op != "ins" else None, "hyp_word": h[j] if op != "del" else None,
                     "ll_hyp": float(ll_hyp), "ll_hyp_with_slot_fixed": ll_fixed, "d_slot": ll_fixed - float(ll_hyp),
                     "ll_truth_phrase": float(ll_truth), "d_phrase": float(ll_truth - ll_hyp),
                     "evidence_favours": "truth" if ll_fixed > float(ll_hyp) else "hyp"})
    return rows


def cmd_score(a) -> int:
    from phase0.analysis.decipher import norm
    ev = {e["k"]: e for e in load_events()}
    truth = {t["k"]: t for t in (json.loads(l) for l in (LIVE / "truth.jsonl").read_text().splitlines() if l.strip())}
    proxy = {p["k"]: p for p in json.loads((OUT / "proxy_decodes.json").read_text())["phrases"]}
    v4p = Path("data/sessions") / SID / "decipher2_v4.json"
    v4 = json.loads(v4p.read_text()) if v4p.exists() else None
    v4seg = {s["i"] - 1: s for s in v4["segments"]} if v4 else {}
    systems = {"live_qwen05": lambda k: norm(ev[k]["decoded"]), "v4": lambda k: norm(v4seg[k]["text"]) if v4 else None,
               "proxy_qwen05": lambda k: proxy[k]["qwen"], "proxy_char": lambda k: proxy[k]["char"],
               "greedy": lambda k: proxy[k]["greedy"]}
    per, tot = [], {s: {"n": 0, "ok": 0, "sub": 0, "del": 0, "ins": 0} for s in systems}
    ev_rows = []
    for k in sorted(truth):
        t = truth[k]["text"]
        lp = np.load(LIVE / f"lp_{k:03d}.npy").astype(np.float32)
        row = {"k": k, "truth": t, "source": truth[k]["source"], "raw_greedy_live": ev[k]["raw"]}
        for s, f in systems.items():
            h = f(k)
            if h is None:
                continue
            st = wer_stats(t, h)
            row[s] = {"text": h, **{q: st[q] for q in ("n", "ok", "sub", "del", "ins")}}
            for q in tot[s]:
                tot[s][q] += st[q]
            if s in ("v4", "live_qwen05"):
                for e in evidence(lp, t, h, st["ops"]):
                    ev_rows.append({"k": k, "system": s, **e})
        per.append(row)
    summary = {}
    for s, c in tot.items():
        if c["n"]:
            summary[s] = {**c, "wer": (c["sub"] + c["del"] + c["ins"]) / c["n"], "word_acc": c["ok"] / c["n"]}
    for s in ("v4", "live_qwen05"):
        rows = [r for r in ev_rows if r["system"] == s]
        summary.setdefault(s, {})["wrong_words"] = len(rows)
        summary[s]["evidence_favours_truth"] = sum(r["evidence_favours"] == "truth" for r in rows)
        summary[s]["evidence_favours_hyp"] = sum(r["evidence_favours"] == "hyp" for r in rows)
    res = {"truth_phrases": len(per), "summary": summary, "per_phrase": per, "evidence": ev_rows,
           "v4_timing_s": v4["timing_s"] if v4 else None, "proxy_timing_s": {k: {d: proxy[k][f"{d}_s"] for d in ("qwen", "char", "greedy")}
                                                                             for k in proxy}}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    for r in per:
        print(f"k={r['k']} truth: {r['truth']}")
        for s in systems:
            if s in r:
                print(f"   {s:<13} {r[s]['text']}   ok {r[s]['ok']}/{r[s]['n']}")
    for r in ev_rows:
        print(f"k={r['k']} {r['system']:<12} {r['op']:<3} truth={r['truth_word']!s:<10} hyp={r['hyp_word']!s:<10} "
              f"d_slot={r['d_slot']:+7.1f} d_phrase={r['d_phrase']:+7.1f} -> {r['evidence_favours']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
