"""Blind desk test: continuous stream -> text with hwtmix (zero-shot) and hwtmix+desk, Qwen word + char decoders.
MacBook: python -m phase0.analysis.decipher data/sessions/<id> [--score truth.txt]; mini: train-desk | remote."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SESS = Path("data/sessions")
DC = Path(".cache/decipher")
MODELS = DC / "models"
QWEN = DC / "qwen2.5-0.5b"
TUNE_QWEN = Path("results/seqctc2/tuning/tune_qwen_lex_hwtmix.json")
DECODE_HWTMIX = Path("results/seqctc2/decode_hwtmix.json")
MINI, MINI_ROOT = "macmini", "cvt"
ARCH = {"frame": "none", "aug": "v1", "d": 192, "layers": 3}
LABEL = {"zs": "hwtmix (zero-shot)", "desk": "hwtmix+desk"}


# ============================================================================ Mac mini: training
def zs_path(seed: int) -> Path:
    return Path(".cache/seqctc2/runs/hwtmix") / f"seed{seed}" / "all.pt"


def cmd_train_desk(a) -> int:
    import torch
    from phase0.analysis import seqctc as S
    from phase0.analysis import seqctc2 as S2
    out = MODELS / f"hwtmix_desk_s{a.seed}.pt"
    if out.exists():
        return 0
    desk = S2.get_stream(S.DESK, "none")
    wins = S.desk_windows(desk)
    cfg = S2.mkcfg(ARCH, a.seed * 1000, a.steps, a.lr)
    m = S.build(cfg)
    m.load_state_dict(torch.load(zs_path(a.seed), map_location="cpu"))
    src = [(S.PhraseSource(desk, wins), a.desk_w)]
    kb = S2.kbd_srcs("none", S.KBD, a.mix, S2.hwt_streams("none"))
    tot = sum(w for _, w in kb)
    src += [(s, w / tot * (1 - a.desk_w)) for s, w in kb]
    t0 = time.time()
    S.train(m, src, cfg, log=f"[deskall s{a.seed}]")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out)
    (MODELS / f"hwtmix_desk_s{a.seed}.json").write_text(json.dumps(
        {"init": str(zs_path(a.seed)), "phrases": len(wins), "steps": a.steps, "lr": a.lr, "desk_w": a.desk_w,
         "mix": a.mix, "seed": a.seed * 1000, "secs": time.time() - t0}))
    print(f"[deskall s{a.seed}] saved {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ============================================================================ Mac mini: inference + decoding
def model_files(seeds) -> dict:
    return {"zs": [p for s in seeds if (p := zs_path(s)).exists()],
            "desk": [p for s in seeds if (p := MODELS / f"hwtmix_desk_s{s}.pt").exists()]}


def char_params() -> tuple[float, float]:
    try:
        m = json.loads(DECODE_HWTMIX.read_text())["desk/char"]["meta"]["seed0"]
        return float(m["alpha"]), float(m["beta"])
    except Exception:  # noqa: BLE001
        return 0.6, 2.0


def cmd_remote(a) -> int:
    import torch
    from phase0.analysis import autocorrect as AC
    from phase0.analysis import seqctc as S
    from phase0.analysis import seqctc2 as S2
    tt = {}
    t_all = time.time()
    sid = a.session
    cache = S.CACHE / "streams" / f"{sid}.npz"
    pq_f = SESS / sid / "landmarks.parquet"
    if cache.exists() and pq_f.exists() and pq_f.stat().st_mtime > cache.stat().st_mtime:
        cache.unlink()
    t0 = time.time()
    st = S.load_session(sid)
    tt["load_stream"] = time.time() - t0
    video_s = float(st.t[-1] - st.t[0])
    files = model_files([int(x) for x in a.seeds.split(",")])
    lps = {}
    for grp, fs in files.items():
        if not fs:
            print(f"[decipher] no {grp} models found, skipped", flush=True)
            continue
        t0 = time.time()
        L = []
        for f in fs:
            m = S2.load_model(ARCH, f)
            L.append(S2.infer_cont(m, st).astype(np.float64))
        lp = np.mean(L, 0)
        lp -= np.logaddexp.reduce(lp, axis=1, keepdims=True)
        lps[grp] = lp.astype(np.float32)
        tt[f"infer_{grp}"] = time.time() - t0
        print(f"[decipher] {grp}: {len(fs)} model(s) {[str(f) for f in fs]} {tt[f'infer_{grp}']:.1f}s", flush=True)
    G = min(len(v) for v in lps.values())
    times = st.t[::2][:G]
    comb = np.mean([v[:G] for v in lps.values()], 0)   # common segmentation: mean non-blank evidence of all models
    segs = S2.segments(comb, times, a.gap, pad=a.pad, thr=a.thr)
    print(f"[decipher] {len(segs)} segments (gap {a.gap}s, pad {a.pad}s, thr {a.thr}), video {video_s:.1f}s", flush=True)
    t0 = time.time()
    tune = json.loads(TUNE_QWEN.read_text())["best"]
    decs = {"qwen": {"kind": "qwen", "cfg": tune}, "greedy": {"kind": "greedy"}}
    al, be = char_params()
    decs["char"] = {"kind": "char", "alpha": al, "beta": be, "beam": 16}
    S2._SC["qwen"] = AC.NLM(str(QWEN))
    tt["load_qwen"] = time.time() - t0
    rows = [{"i": k + 1, "t0": float(times[s0] - st.t[0]), "t1": float(times[min(s1, G) - 1] - st.t[0]),
             "frames30": [int(s0), int(s1)]} for k, (s0, s1) in enumerate(segs)]
    for grp, lp in lps.items():
        for dn, dec in decs.items():
            t0 = time.time()
            for r, (s0, s1) in zip(rows, segs):
                r.setdefault(grp, {})[dn] = S2.run_decoder(dec, lp[s0:s1])
            tt[f"decode_{grp}_{dn}"] = time.time() - t0
            print(f"[decipher] decoded {grp}/{dn} {tt[f'decode_{grp}_{dn}']:.1f}s", flush=True)
    tt["total_remote"] = time.time() - t_all
    out = {"session": sid, "video_s": video_s, "gap": a.gap, "pad": a.pad, "thr": a.thr,
           "models": {g: [str(f) for f in fs] for g, fs in files.items()}, "decoders": decs, "segments": rows,
           "timing_s": tt, "remote_s_per_video_min": tt["total_remote"] / max(video_s / 60, 1e-9),
           "torch": torch.__version__}
    DC.joinpath("out").mkdir(parents=True, exist_ok=True)
    (DC / "out" / f"{sid}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(DC / "out" / f"{sid}_lp.npz", times=times, **{k: v.astype(np.float16) for k, v in lps.items()})
    print(f"[decipher] wrote {DC / 'out' / (sid + '.json')} ({tt['total_remote']:.0f}s)", flush=True)
    return 0


# ============================================================================ MacBook: orchestration
def sh(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def run_pipeline(sdir: Path, a) -> dict:
    sid = sdir.name
    tt = {}
    while subprocess.run(["pgrep", "-f", f"extract_landmarks.*{sid}"], capture_output=True).returncode == 0:
        print(f"waiting for a running landmark extraction of {sid} ...", flush=True)
        time.sleep(15)
    if not (sdir / "landmarks.parquet").exists():
        t0 = time.time()
        sh([sys.executable, "-m", "phase0.analysis.extract_landmarks", str(sdir)])
        tt["extract_landmarks_macbook"] = time.time() - t0
    t0 = time.time()
    files = [str(sdir / f) for f in ("frames.jsonl", "landmarks.parquet", "meta.json")
             if (sdir / f).exists()]
    sh(["ssh", MINI, f"mkdir -p {MINI_ROOT}/data/sessions/{sid}"])
    sh(["rsync", "-a", *files, f"{MINI}:{MINI_ROOT}/data/sessions/{sid}/"])
    sh(["rsync", "-a", "phase0/analysis/decipher.py", "phase0/analysis/seqctc2.py", f"{MINI}:{MINI_ROOT}/phase0/analysis/"])
    tt["sync"] = time.time() - t0
    t0 = time.time()
    sh(["ssh", MINI, f"cd {MINI_ROOT} && .cache/decipher/s2.sh phase0.analysis.decipher remote {sid} "
                     f"--gap {a.gap} --pad {a.pad} --thr {a.thr} --seeds {a.seeds}"])
    tt["remote_wall"] = time.time() - t0
    out_f = sdir / "decipher.json"
    sh(["rsync", "-a", f"{MINI}:{MINI_ROOT}/.cache/decipher/out/{sid}.json", str(out_f)])
    res = json.loads(out_f.read_text())
    res["timing_macbook_s"] = tt
    res["wall_s_per_video_min"] = sum(tt.values()) / max(res["video_s"] / 60, 1e-9)
    out_f.write_text(json.dumps(res, indent=1))
    return res


def print_segments(res: dict) -> None:
    groups = [g for g in ("zs", "desk") if res["segments"] and g in res["segments"][0]]
    print(f"\nSession {res['session']}: {len(res['segments'])} segments, video {res['video_s']:.0f}s, "
          f"models { {g: len(v) for g, v in res['models'].items()} }, gap {res['gap']}s")
    for r in res["segments"]:
        print(f"{r['i']:>2}. [{r['t0']:7.1f}s - {r['t1']:7.1f}s]")
        for g in groups:
            print(f"      {LABEL[g]:<20} {r[g]['qwen']}")
        print(f"      {'open-vocab char':<20} " + "  |  ".join(f"{g}: {r[g]['char']}" for g in groups))
    t = res.get("timing_macbook_s", {})
    print(f"\nruntime: remote {res['timing_s']['total_remote']:.0f}s, macbook {t}, "
          f"{res.get('wall_s_per_video_min', res['remote_s_per_video_min']):.0f}s wall per video minute")


# ============================================================================ scoring against revealed text
def norm(s: str) -> str:
    return " ".join("".join(c if "a" <= c <= "z" else (" " if c.isspace() or c == "-" else "") for c in s.lower()).split())


def word_align(ref: list[str], hyp: list[str]):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def score_variant(lines: list[str], seg_texts: list[str]) -> dict:
    """Concatenated segments vs concatenated lines; an inserted word is charged to the preceding aligned word's line
    if in the same segment, else to the next line (so merged/split segments are handled)."""
    from phase0.analysis.seqctc2 import aligned_edits
    rw = [w for l in lines for w in l.split()]
    rown = [k for k, l in enumerate(lines) for _ in l.split()]
    hw, hseg = [], []
    for s, t in enumerate(seg_texts):
        for w in norm(t).split():
            hw.append(w)
            hseg.append(s)
    L = len(lines)
    wed, wok, hyp_by_line = np.zeros(L), np.zeros(L), [[] for _ in range(L)]
    seg_lines = [set() for _ in seg_texts]
    ops = word_align(rw, hw)
    last = None   # (owner, segment) of the previous aligned ref word
    for k, (op, i, j) in enumerate(ops):
        if op in ("ok", "sub"):
            o = rown[i]
            wok[o] += op == "ok"
            wed[o] += op == "sub"
            hyp_by_line[o].append(hw[j] if op == "ok" else hw[j].upper())
            seg_lines[hseg[j]].add(o)
            last = (o, hseg[j])
        elif op == "del":
            wed[rown[i]] += 1
            hyp_by_line[rown[i]].append("_")
            last = (rown[i], last[1] if last else None)
        else:
            o = last[0] if last is not None and last[1] == hseg[j] else rown[min(i, len(rw) - 1)] if rw else 0
            wed[o] += 1
            hyp_by_line[o].append("+" + hw[j].upper())
            seg_lines[hseg[j]].add(o)
    ref_c = " ".join(lines)
    cown = [k for k, l in enumerate(lines) for _ in range(len(l) + (1 if k < L - 1 else 0))]
    hyp_c = " ".join(x for x in (norm(t) for t in seg_texts) if x)
    ced = aligned_edits(list(ref_c), list(hyp_c), cown, L) if ref_c else np.zeros(L)
    return {"wed": wed, "wok": wok, "wn": np.array([len(l.split()) for l in lines], float), "ced": ced,
            "cn": np.array([len(l) + (1 if k < L - 1 else 0) for k, l in enumerate(lines)], float),
            "hyp_by_line": [" ".join(h) for h in hyp_by_line], "seg_lines": [sorted(s) for s in seg_lines]}


def cmd_score(res: dict, truth: Path, sdir: Path) -> dict:
    from phase0.analysis.seqctc import boot_delta, boot_ratio
    lines = [norm(l) for l in truth.read_text().splitlines() if norm(l)]
    L = len(lines)
    segs = res["segments"]
    groups = [g for g in ("zs", "desk") if segs and g in segs[0]]
    out = {"n_lines": L, "n_segments": len(segs), "lines": lines, "variants": {}}
    print(f"\nScoring {len(segs)} segments against {L} revealed lines ({sum(len(l.split()) for l in lines)} words). "
          f"Bootstrap 95% CIs resample the {L} phrases" + (" -- with this few phrases they are VERY wide and only "
                                                           "indicative." if L < 10 else "."))
    lex = None
    try:
        from phase0.analysis.seqctc2 import lexicon
        lex = lexicon().logp
        oov = sorted({w for l in lines for w in l.split() if w not in lex})
        out["oov_words"] = oov
        print(f"words outside the Qwen decoder's closed lexicon (it cannot output them): {oov or 'none'}")
    except Exception:  # noqa: BLE001
        pass
    scored = {}
    for g in groups:
        for dn in ("qwen", "char", "greedy"):
            sv = score_variant(lines, [s[g][dn] for s in segs])
            scored[(g, dn)] = sv
            w, wlo, whi = boot_ratio(sv["wed"], sv["wn"])
            c, clo, chi = boot_ratio(sv["ced"], sv["cn"])
            k, klo, khi = boot_ratio(sv["wok"], sv["wn"])
            row = {"wer": [w, wlo, whi], "cer": [c, clo, chi], "words_correct": [k, klo, khi],
                   "per_line": [{"line": i + 1, "ref": lines[i], "hyp": sv["hyp_by_line"][i],
                                 "wer": float(sv["wed"][i] / max(sv["wn"][i], 1)),
                                 "cer": float(sv["ced"][i] / max(sv["cn"][i], 1)),
                                 "words_correct": float(sv["wok"][i] / max(sv["wn"][i], 1))} for i in range(L)],
                   "segment_to_lines": sv["seg_lines"],
                   **{kk: sv[kk].tolist() for kk in ("wed", "wok", "wn", "ced", "cn")}}
            out["variants"][f"{g}/{dn}"] = row
    print(f"\n{'model/decoder':<34} {'WER [95% CI]':<22} {'CER [95% CI]':<22} words correct [95% CI]")
    for g in groups:
        for dn in ("qwen", "char", "greedy"):
            r = out["variants"][f"{g}/{dn}"]
            f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
            print(f"{LABEL[g] + ' / ' + dn:<34} {f3(r['wer']):<22} {f3(r['cer']):<22} "
                  f"{100 * r['words_correct'][0]:.0f}% [{100 * r['words_correct'][1]:.0f}, {100 * r['words_correct'][2]:.0f}]")
    if ("zs", "qwen") in scored and ("desk", "qwen") in scored:
        a, b = scored[("zs", "qwen")], scored[("desk", "qwen")]
        d = boot_delta(a["wed"], b["wed"], a["wn"])
        out["d_wer_desk_minus_zs_qwen"] = d
        print(f"paired dWER (+desk minus zero-shot, Qwen): {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    ref_v = out["variants"][f"{groups[0]}/qwen"]
    merged = [s for s, ls in enumerate(ref_v["segment_to_lines"]) if len(ls) > 1]
    split = [i for i in range(L) if sum(i in ls for ls in ref_v["segment_to_lines"]) > 1]
    out["segmentation"] = {"merged_segments": [m + 1 for m in merged], "split_lines": [i + 1 for i in split]}
    print(f"segmentation ({LABEL[groups[0]]}/qwen alignment): {len(segs)} segments for {L} lines; "
          f"merged segments {[m + 1 for m in merged] or 'none'}; lines split over segments {[i + 1 for i in split] or 'none'}")
    print("\nper line (UPPER = substituted, +WORD = inserted, _ = deleted):")
    for i in range(L):
        print(f"{i + 1:>2}. truth: {lines[i]}")
        for g in groups:
            for dn in ("qwen", "char"):
                r = out["variants"][f"{g}/{dn}"]["per_line"][i]
                print(f"    {LABEL[g] + '/' + dn:<26} WER {r['wer']:.2f} CER {r['cer']:.2f}  {r['hyp']}")
    (sdir / "decipher_score.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {sdir / 'decipher_score.json'}")
    return out


# ============================================================================ CLI
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("train-desk", "remote"):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        s = sub.add_parser("train-desk")
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--steps", type=int, default=300)
        s.add_argument("--lr", type=float, default=5e-4)
        s.add_argument("--desk-w", type=float, default=0.5)
        s.add_argument("--mix", type=float, default=0.3)
        s.set_defaults(fn=cmd_train_desk)
        s = sub.add_parser("remote")
        s.add_argument("session")
        s.add_argument("--gap", type=float, default=2.0)
        s.add_argument("--pad", type=float, default=0.5)
        s.add_argument("--thr", type=float, default=0.5)
        s.add_argument("--seeds", default="0,1")
        s.set_defaults(fn=cmd_remote)
        a = p.parse_args(argv)
        return a.fn(a)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", type=Path)
    p.add_argument("--score", type=Path, default=None, help="revealed phrases, one per line, in typing order")
    p.add_argument("--rerun", action="store_true", help="recompute even if decipher.json exists")
    p.add_argument("--gap", type=float, default=2.0, help="pause (s) without emissions that splits segments")
    p.add_argument("--pad", type=float, default=0.5)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--seeds", default="0,1")
    a = p.parse_args(argv)
    sdir = a.session_dir
    f = sdir / "decipher.json"
    if f.exists() and not a.rerun and (a.score is not None):
        res = json.loads(f.read_text())
    elif f.exists() and not a.rerun:
        res = json.loads(f.read_text())
        print(f"(cached {f}; --rerun to recompute)")
    else:
        res = run_pipeline(sdir, a)
    print_segments(res)
    if a.score is not None:
        cmd_score(res, a.score, sdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
