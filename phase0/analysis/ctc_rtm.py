"""ctc_v4 desk recipe on RTMPose streams (CVT_LANDMARKS=landmarks_rtm.parquet -> .cache/seqctc/streams_rtm/): hwtmix
keyboard stage redone from hwt_s0, LOSO desk fine-tunes, proxy-decoder eval, paired comparison with ctc_v4 loso_v4.json."""

from __future__ import annotations

import os

os.environ.setdefault("CVT_LANDMARKS", "landmarks_rtm.parquet")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from phase0.analysis import ctcv3 as C
from phase0.analysis import ctcv4 as V4
from phase0.analysis import seqctc as S
from phase0.analysis import seqctc2 as S2
from phase0.analysis.seqctc import KBD

ROOT = Path(".cache/ctc_rtm")
TAG = "hwtmix"
DESK_SIDS = (V4.OLD, V4.BLIND1, V4.BLIND2)
QWEN_LOCAL = Path(".cache/decipher/qwen2.5-0.5b")


def zs_path(root: Path, seed: int) -> Path:
    return root / "runs" / TAG / f"seed{seed}" / "all.pt"


def cmd_zs(a) -> int:
    root = Path(a.root)
    r = C.recipe(TAG)
    for seed in map(int, a.seeds.split(",")):
        f = zs_path(root, seed)
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            print(f"[rtm zs s{seed}] exists", flush=True)
            continue
        cfg = C.mkcfg(r, seed, r["steps"], r["lr"])
        m = S.build(cfg)
        m.load_state_dict(torch.load(S2.init_path(r["init"], seed), map_location="cpu"))
        t0 = time.time()
        S.train(m, C.kbd_sources(r, KBD), cfg, log=f"[rtm zs s{seed}]", ckpt=f.with_suffix(".ckpt"))
        torch.save(m.state_dict(), f)
        (f.parent / "all.json").write_text(json.dumps({**r, "seed": seed, "landmarks": S.LANDMARKS, "streams": S.STREAMS,
                                                       "secs": time.time() - t0}))
        print(f"[rtm zs s{seed}] done {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_loso(a) -> int:
    root = Path(a.root)
    r = C.recipe(TAG)
    dfile = Path(a.data)
    data = V4.load_data(dfile)
    for sid in a.sids.split(","):
        sessions = {k: v for k, v in data.items() if k != sid}
        for seed in map(int, a.seeds.split(",")):
            d = V4.run_dir(root, seed, f"loso_{sid}")
            d.mkdir(parents=True, exist_ok=True)
            f = d / "model.pt"
            if not f.exists():
                cfg = C.mkcfg(r, seed * 1000, a.steps, a.lr)
                m = S.build(cfg)
                m.load_state_dict(torch.load(zs_path(root, seed), map_location="cpu"))
                t0 = time.time()
                S.train(m, V4.desk_sources(r, sessions, a.desk_w, a.ftmix), cfg, log=f"[rtm loso {sid[:15]} s{seed}]",
                        ckpt=d / "model.ckpt")
                torch.save(m.state_dict(), f)
                meta = {"cmd": "loso", "tag": TAG, "name": f"loso_{sid}", "seed": seed, "steps": a.steps, "lr": a.lr,
                        "desk_w": a.desk_w, "ftmix": a.ftmix, "ntrain": sum(len(v["windows"]) for v in sessions.values()),
                        "sessions": {k: len(v["windows"]) for k, v in sessions.items()}, "excluded": [sid],
                        "data": str(dfile), "data_md5": V4.data_hash(dfile), "init": str(zs_path(root, seed)),
                        "landmarks": S.LANDMARKS, "device": str(S.device()), "secs": time.time() - t0}
                (d / "meta.json").write_text(json.dumps(meta, indent=1))
            m = C.load(r, f)
            if not (d / f"{sid}_cont.npy").exists():
                np.save(d / f"{sid}_cont.npy", C.infer_cont(m, C.stream(sid)).astype(np.float16))
            print(f"[rtm loso {sid} s{seed}] done", flush=True)
    return 0


def cmd_eval(a) -> int:
    root = Path(a.root)
    if a.procs <= 1 and QWEN_LOCAL.exists() and "qwen" in a.decoders:
        from phase0.analysis import autocorrect as AC
        S2._SC["qwen"] = AC.NLM(str(QWEN_LOCAL), device="cpu")   # in-process decode: no hub download on the mini
    data = V4.load_data(Path(a.data))
    seeds = list(map(int, a.seeds.split(",")))
    sids = a.sids.split(",")
    rows = []
    for sid in sids:
        lines = V4.truth_lines(data, sid)
        rows.append({"label": "zs", "sid": sid, "lines": lines, "models": [zs_path(root, s) for s in seeds]})
        rows.append({"label": "loso_ft", "sid": sid, "lines": lines,
                     "models": [V4.run_dir(root, s, f"loso_{sid}") / "model.pt" for s in seeds]})
    decs = a.decoders.split(",")
    res = V4.eval_rows(root, rows, decs, a.procs)
    pool = {dec: V4.pooled(res, ["zs", "loso_ft"], sids, dec, ref="zs") for dec in decs}
    out = root / "results" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sids": sids, "seeds": seeds, "landmarks": S.LANDMARKS, "rows": res, "pooled": pool},
                              indent=1, default=float))
    V4.print_res(res, pool, a.name)
    print(f"-> {out}")
    return 0


def cmd_compare(a) -> int:
    """Paired bootstrap (same lines) of RTM vs original rows: per session and pooled, per decoder and label."""
    ref, new = json.loads(Path(a.ref).read_text()), json.loads(Path(a.new).read_text())
    sids = new["sids"]
    out = {"ref": a.ref, "new": a.new, "per_session": {}, "pooled": {}}
    for dec in a.decoders.split(","):
        for lab in ("zs", "loso_ft"):
            cat = {"a": [], "b": [], "n": [], "ca": [], "cb": [], "cn": []}
            for sid in sids:
                k = f"{lab}|{sid}|{dec}"
                if k not in ref["rows"] or k not in new["rows"]:
                    continue
                ra, rb = ref["rows"][k], new["rows"][k]
                wa, wb, wn = np.array(ra["wed"]), np.array(rb["wed"]), np.array(rb["wn"])
                assert len(wa) == len(wb) and np.allclose(ra["wn"], rb["wn"]), k
                d = S.boot_delta(wa, wb, wn)
                out["per_session"][k] = {"wer_orig": ra["wer"], "wer_rtm": rb["wer"], "wer_orig_ci": ra["wer_ci"],
                                         "wer_rtm_ci": rb["wer_ci"], "d_wer_rtm_minus_orig": d,
                                         "cer_orig": ra["cer"], "cer_rtm": rb["cer"],
                                         "d_cer_rtm_minus_orig": S.boot_delta(np.array(ra["ed"]), np.array(rb["ed"]), np.array(rb["n"]))}
                cat["a"].append(wa); cat["b"].append(wb); cat["n"].append(wn)
                cat["ca"].append(np.array(ra["ed"])); cat["cb"].append(np.array(rb["ed"])); cat["cn"].append(np.array(rb["n"]))
            if cat["a"]:
                A, B, N = (np.concatenate(cat[k]) for k in ("a", "b", "n"))
                CA, CB, CN = (np.concatenate(cat[k]) for k in ("ca", "cb", "cn"))
                out["pooled"][f"{lab}|{dec}"] = {
                    "wer_orig": float(A.sum() / N.sum()), "wer_rtm": float(B.sum() / N.sum()),
                    "wer_orig_ci": list(S.boot_ratio(A, N))[1:], "wer_rtm_ci": list(S.boot_ratio(B, N))[1:],
                    "d_wer_rtm_minus_orig": S.boot_delta(A, B, N),
                    "cer_orig": float(CA.sum() / CN.sum()), "cer_rtm": float(CB.sum() / CN.sum()),
                    "d_cer_rtm_minus_orig": S.boot_delta(CA, CB, CN), "lines": int(len(N)), "words": int(N.sum())}
    for k, r in out["per_session"].items():
        d = r["d_wer_rtm_minus_orig"]
        print(f"  {k:<50} WER orig {r['wer_orig']:.3f} rtm {r['wer_rtm']:.3f} | d {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
    for k, r in out["pooled"].items():
        d = r["d_wer_rtm_minus_orig"]
        print(f"  POOLED {k:<20} WER orig {r['wer_orig']:.3f} rtm {r['wer_rtm']:.3f} | d {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
              f"| CER orig {r['cer_orig']:.3f} rtm {r['cer_rtm']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("zs")
    s.add_argument("--root", default=str(ROOT))
    s.add_argument("--seeds", default="0,1,2")
    s.set_defaults(fn=cmd_zs)
    s = sub.add_parser("loso")
    s.add_argument("--root", default=str(ROOT))
    s.add_argument("--data", default=".cache/ctc_v4/desk_data.json")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--sids", default=",".join(DESK_SIDS))
    s.add_argument("--steps", type=int, default=V4.RECIPE["steps"])
    s.add_argument("--lr", type=float, default=V4.RECIPE["lr"])
    s.add_argument("--desk-w", type=float, default=V4.RECIPE["desk_w"])
    s.add_argument("--ftmix", type=float, default=V4.RECIPE["ftmix"])
    s.set_defaults(fn=cmd_loso)
    s = sub.add_parser("eval")
    s.add_argument("--root", default=str(ROOT))
    s.add_argument("--data", default=".cache/ctc_v4/desk_data.json")
    s.add_argument("--name", default="loso_rtm")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--sids", default=",".join(DESK_SIDS))
    s.add_argument("--decoders", default="qwen,char")
    s.add_argument("--procs", type=int, default=1)
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("compare")
    s.add_argument("--ref", default=".cache/ctc_v4/results/loso_v4.json")
    s.add_argument("--new", default=str(ROOT / "results" / "loso_rtm.json"))
    s.add_argument("--decoders", default="qwen,char")
    s.add_argument("--out", default="results/rtm/loso_compare.json")
    s.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
