"""Review calibration labels, flag alignment quality, then train a new version."""

from __future__ import annotations

import os

_USER_CPU = os.environ.get("CTCV3_CPU")
os.environ["CTCV3_CPU"] = "1"   # inference/decoding in this process stays on CPU; training subprocesses use the GPU lock

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from phase0.analysis import ctcv3 as C  # noqa: E402
from phase0.analysis import ctcv4 as V  # noqa: E402
from phase0.analysis import seqctc as S
from phase0.analysis.calibration_review import load_review, decision
from phase0.analysis.text_contract import legacy_training_text  # noqa: E402

LEAD, TAIL, EDGE = 0.3, 1.0, 0.6
# Model agreement flag only (never an acceptance rule): --calibrate on the 31 labelled windows keeps all true
# labels at -1.5 but also 48% of one-word-deleted and 29% of one-word-swapped references.
MIN_MARGIN, MAX_CER = -1.5, 0.8
UNUSABLE_MARGIN = -3.5   # below this the window holds something else entirely (half-typed/aborted/skipped)
MINI, MINI_ROOT = "macmini", "cvt"


# ============================================================================ windows + alignment
def ensure_stream(sdir: Path):
    sid = sdir.name
    if sdir.resolve() != (S.SESS / sid).resolve():
        raise SystemExit(f"{sdir}: session must live under {S.SESS}/")
    if not (sdir / "landmarks.parquet").exists():
        print(f"[ingest] {sid}: extracting landmarks", flush=True)
        subprocess.run([sys.executable, "-m", "phase0.analysis.extract_landmarks", str(sdir)], check=True)
    cache = S.CACHE / "streams" / f"{sid}.npz"
    if cache.exists() and (sdir / "landmarks.parquet").stat().st_mtime > cache.stat().st_mtime:
        cache.unlink()
    return C.stream(sid)


def prompt_rows(sdir: Path):
    f = sdir / "phrases.jsonl"
    if not f.exists():
        raise SystemExit(f"{sdir}: no phrases.jsonl (record with --phrases)")
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    shown = {r["idx"]: r for r in rows if r["event"] == "shown"}
    done = {r["idx"]: r for r in rows if r["event"] == "done"}
    return [(i, shown[i]["t"], done[i]["t"], shown[i]["phrase"]) for i in sorted(shown) if i in done]


def score_window(L: np.ndarray, times: np.ndarray, lo: float, hi: float, text: str) -> dict:
    f0, f1 = int(np.searchsorted(times, lo)), int(np.searchsorted(times, hi, side="right"))
    span = V.emission_span(L[f0:f1]) if f1 - f0 > 2 else None
    if span is None or not S.text_syms(text).size:
        return {"status": "no_emissions", "t0": float(lo), "t1": float(hi), "margin_per_char": -np.inf,
                "greedy_cer": 1.0, "greedy": ""}
    e0, e1 = times[f0 + span[0]], times[f0 + span[1]]
    t0, t1 = max(lo, e0 - EDGE), min(hi, e1 + EDGE)
    g0, g1 = int(np.searchsorted(times, t0)), int(np.searchsorted(times, t1, side="right"))
    r = V.align_stats(L[g0:g1], text)
    r.update({"status": "ok", "t0": float(t0), "t1": float(t1), "cut_at_end": bool(e1 > hi - 0.3)})
    return r


def unusable(r: dict) -> str:
    """Hint for prompts nobody has reviewed yet: clearly unusable windows need not be reviewed at all.
    It never overrides a human confirmation - decision() alone decides what becomes a training label."""
    if r["status"] != "ok":
        return "no_typing_in_window" if r["status"] == "no_emissions" else r["status"]
    if r.get("cut_at_end"):
        return "typing_cut_off_by_next_prompt"
    if r["margin_per_char"] < UNUSABLE_MARGIN:
        return "evidence_unrelated_to_prompt"
    return ""


# ============================================================================ GPU training via the lock
def rsync(src, dst):
    subprocess.run(["rsync", "-a", "--relative", *([src] if isinstance(src, str) else src), dst], check=True)


def sync_to_mini(out: Path, sids, host: str, root: str) -> str:
    """Everything ctcv4 train needs on the mini: code, v3 init checkpoints, HWT + session stream caches, data."""
    from phase0.analysis import howwetype as H
    rroot = f".cache/{Path(out).name}"
    subprocess.run(["ssh", host, f"mkdir -p {root}/{rroot} {root}/.cache/seqctc/streams {root}/phase0/analysis"], check=True)
    files = ["./phase0/analysis/ctcv3.py", "./phase0/analysis/ctcv4.py", "./.cache/ctc_v3/runs/hwtmix/recipe.json",
             *[f"./.cache/ctc_v3/runs/hwtmix/seed{s}/all.pt" for s in (0, 1, 2)], f"./{H.OUT}/"]
    files += [f"./.cache/seqctc/streams/{sid}.npz" for sid in sids if (S.CACHE / "streams" / f"{sid}.npz").exists()]
    print(f"[ingest] syncing {len(files)} paths to {host}:{root} (HWT streams ~272 MB on first run)", flush=True)
    rsync(files, f"{host}:{root}/")
    subprocess.run(["rsync", "-a", str(out / "desk_data.json"), f"{host}:{root}/{rroot}/desk_data.json"], check=True)
    return rroot


def train(out: Path, name: str, seeds: str, steps: int, gpu_wait: float, exclude: str = "", infer: str = "",
          compute: str = "mini", sids=(), host: str = MINI, root: str = MINI_ROOT):
    """One GPU job at a time via phase0.tools.gpulock. Heavy runs belong on the mini, not the MacBook."""
    args = ["train", "--name", name, "--seeds", seeds, "--steps", str(steps)]
    if exclude:
        args += ["--exclude", exclude]
    if infer:
        args += ["--infer", infer]
    t0 = time.time()
    if compute == "mini":
        rroot = sync_to_mini(out, sids, host, root)
        wait = "" if gpu_wait == float("inf") else f" --wait {gpu_wait}"
        remote = (f"cd {root} && .venv/bin/python -m phase0.tools.gpulock{wait} -- nice -n 10 .venv/bin/python "
                  f"-W ignore -m phase0.analysis.ctcv4 train --root {rroot} --data {rroot}/desk_data.json "
                  + " ".join(args[1:]))
        print(f"[ingest] training {name} on {host} (seeds {seeds}, {steps} steps) via gpulock", flush=True)
        subprocess.run(["ssh", host, remote], check=True)
        (out / "runs").mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{host}:{root}/{rroot}/runs/", str(out / "runs") + "/"], check=True)
    else:
        env = {k: v for k, v in os.environ.items() if k != "CTCV3_CPU"}
        if _USER_CPU:
            env["CTCV3_CPU"] = _USER_CPU
        cmd = [sys.executable, "-m", "phase0.tools.gpulock", "--wait", str(gpu_wait), "--", sys.executable,
               "-W", "ignore", "-m", "phase0.analysis.ctcv4", "train", "--root", str(out),
               "--data", str(out / "desk_data.json"), *args[1:]]
        print(f"[ingest] training {name} LOCALLY (seeds {seeds}, {steps} steps) via gpulock", flush=True)
        subprocess.run(cmd, check=True, env=env)
    print(f"[ingest] {name} done in {time.time() - t0:.0f}s", flush=True)


def parse_idx(spec: str) -> set:
    out = set()
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def set_status(sdir: Path, prompts, confirm: set, reject: set) -> int:
    """Record a HUMAN decision in the review file. Never called with model output."""
    load_review(sdir, prompts)
    path = sdir / "calibration_review.json"
    data = json.loads(path.read_text())
    n = 0
    for e in data["entries"]:
        st = "confirmed" if e["idx"] in confirm else "rejected" if e["idx"] in reject else None
        if st and e.get("status") != st:
            e["status"] = st
            n += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return n


def annotate_review(sdir: Path, rows) -> None:
    """Informational fields for the person reviewing (model text + flags); statuses/labels untouched."""
    path = sdir / "calibration_review.json"
    data = json.loads(path.read_text())
    by = {r["idx"]: r for r in rows}
    for e in data["entries"]:
        r = by.get(e["idx"])
        if r:
            e["model_reading"] = r.get("greedy", "")
            e["model_margin_per_char"] = round(float(r.get("margin_per_char", float("-inf"))), 2) \
                if r.get("margin_per_char") not in (None, float("-inf")) else None
            e["window_problem"] = r.get("unusable") or None
    data["how_to_review"] = ("Set status to 'confirmed' for prompts you typed correctly and 'rejected' otherwise. "
                             "If you typed something slightly different, correct 'text' to what you actually typed "
                             "and confirm it. model_reading/model_margin_per_char are hints only, never proof.")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def preload_hyps(out: Path, extra_roots):
    """Decoder hypothesis caches of earlier roots (same key = same decoder + posterior bytes)."""
    from phase0.analysis import ctcv3_eval as E
    E.HYP = out / "hyps"
    c = E.cache()
    for root in extra_roots:
        for f in (Path(root) / "hyps").glob("*.jsonl"):
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                    c[r["k"]] = r["h"]
                except json.JSONDecodeError:
                    pass


# ============================================================================ commands
def run_cv(a, out: Path, base: Path, sessions: dict, n_calib: int) -> dict:   # noqa: C901
    hold = [s for s in a.cv_holdout.split(",") if s in sessions]
    if not hold:
        print("[ingest] no CV hold-out sessions present; skipping CV")
        return {}
    for sid in hold:
        train(out, f"loso_{sid}", a.cv_seeds, a.steps, a.gpu_wait, exclude=sid, infer=sid,
              compute=getattr(a, "compute", "mini"), sids=list(sessions), host=getattr(a, "mini_host", MINI),
              root=getattr(a, "mini_root", MINI_ROOT))
    seeds = list(map(int, a.cv_seeds.split(",")))
    preload_hyps(out, [base])
    rows, labels = [], ["zs", "base_loso", "calib_loso"]
    for sid in hold:
        lines = V.truth_lines(sessions, sid)
        rows.append({"label": "zs", "sid": sid, "lines": lines,
                     "models": [V.V3 / "runs" / V.TAG / f"seed{s}" / "all.pt" for s in seeds]})
        bm = [V.run_dir(base, s, f"loso_{sid}") / "model.pt" for s in seeds]
        if all(m.exists() for m in bm):
            rows.append({"label": "base_loso", "sid": sid, "lines": lines, "models": bm})
        rows.append({"label": "calib_loso", "sid": sid, "lines": lines,
                     "models": [V.run_dir(out, s, f"loso_{sid}") / "model.pt" for s in seeds]})
    decs = a.decoders.split(",")
    res = V.eval_rows(out, rows, decs, a.procs)
    has_base = all(f"base_loso|{s}|{decs[0]}" in res for s in hold)
    pool = {d: V.pooled(res, labels, hold, d, ref="base_loso" if has_base else "zs") for d in decs}
    V.print_res(res, pool, f"CV: leave-one-session-out on {hold} ({len(seeds)} seed(s), +{n_calib} calibration phrases)")
    f = out / "results" / "cv_ingest.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"holdout": hold, "seeds": seeds, "rows": res, "pooled": pool}, indent=1, default=float))
    print(f"[ingest] CV -> {f}")
    return pool


def cmd_ingest(a) -> int:
    out, base = Path(a.out_root), Path(a.base_root)
    if out.resolve() in (V.V3.resolve(), base.resolve()):
        raise SystemExit("--out-root must be a new version directory (never overwrite ctc_v3 / the base version)")
    sessions = V.load_data(base / "desk_data.json")
    man = Path(a.align_manifest) if a.align_manifest else base / "deploy" / "manifest.json"
    models, _ = V.load_manifest_models(man, a.align_group)
    print(f"[ingest] base data {base / 'desk_data.json'}: {sum(len(v['windows']) for v in sessions.values())} phrases; "
          f"alignment ensemble {man} [{a.align_group}] ({len(models)} models)", flush=True)
    n_calib, summary, confirmed_total = 0, {}, 0
    confirm, reject = parse_idx(getattr(a, "confirm", "")), parse_idx(getattr(a, "reject", ""))
    for sdir in map(Path, a.sessions):
        sid = sdir.name
        st = ensure_stream(sdir)
        prompts = prompt_rows(sdir)
        if confirm or reject:
            n = set_status(sdir, prompts, confirm, reject)
            print(f"[ingest] {sid}: {n} review statuses set from --confirm/--reject", flush=True)
        review = load_review(sdir, prompts)   # creates a pending review file on the first run
        L = V.ensemble_cont(out, models, sid)
        times = st.t[::2][:len(L)]
        rows, acc, n_conf, n_bad = [], [], 0, 0
        for i, t_sh, t_dn, phrase in prompts:
            try:
                text = legacy_training_text(phrase)
            except ValueError as e:
                text, err = "", str(e)
            r = score_window(L, times, t_sh - LEAD, t_dn + TAIL, text) if text else \
                {"status": "unsupported_prompt", "t0": t_sh, "t1": t_dn, "margin_per_char": float("-inf"),
                 "greedy_cer": 1.0, "greedy": ""}
            bad = unusable(r)
            ent = review[i]
            # A confirmed label is the user's statement of what they typed: the recognizer never overrides it.
            d = (decision(ent, r, a.min_margin, a.max_cer) if ent.get("status") == "confirmed"
                 else {"accepted": False, "reason": bad or ent.get("status", "pending")})
            r.update({"idx": i, "prompt": phrase, "text": text, "shown": t_sh, "done": t_dn, "unusable": bad,
                      "review_status": ent.get("status"), "accepted": bool(d["accepted"]), "reason": d["reason"],
                      "model_agreement": d.get("model_agreement")})
            if d["accepted"]:
                acc.append([r["t0"], r["t1"], d.get("model_text") or text])
                n_conf += 1
            n_bad += bool(bad)
            rows.append(r)
            mark = "USE " if d["accepted"] else ("DROP" if bad else "----")
            print(f"  {mark} #{i:<3} {r['reason']:<24} margin {r['margin_per_char']:+6.2f} "
                  f"{text!r} | model read {r.get('greedy', '')!r}", flush=True)
        annotate_review(sdir, rows)
        diag = {}
        if sid in sessions:
            old_w = {w[2]: w for w in sessions[sid]["windows"]}
            ious = [max(0.0, min(t1, old_w[t][1]) - max(t0, old_w[t][0])) / max(t1 - t0, 1e-6)
                    for t0, t1, t in acc if t in old_w]
            diag = {"replaced_windows": len(old_w), "frac_new_window_inside_old": float(np.mean(ious)) if ious else None}
            print(f"[ingest] WARNING {sid} already in base data ({len(old_w)} windows) -> replaced; "
                  f"mean fraction of each new window inside the stored window: {diag['frac_new_window_inside_old']}")
        if acc:
            sessions[sid] = {"kind": "calib", "source": f"calib_ingest: phrases.jsonl windows, human-confirmed in "
                             f"{sdir / 'calibration_review.json'}; model alignment ({man} [{a.align_group}]) is a "
                             f"quality flag only", "windows": acc}
        n_calib += len(acc)
        confirmed_total += n_conf
        summary[sid] = {"prompts": len(prompts), "confirmed": n_conf, "unusable": n_bad,
                        "awaiting_review": len(prompts) - n_conf - n_bad
                        - sum(r["review_status"] == "rejected" and not r["unusable"] for r in rows), **diag}
        fjson = out / "ingest" / f"{sid}.json"
        fjson.parent.mkdir(parents=True, exist_ok=True)
        fjson.write_text(json.dumps({"sid": sid, "manifest": str(man), "group": a.align_group,
                                     "min_margin": a.min_margin, "max_cer": a.max_cer, "rows": rows,
                                     "summary": summary[sid]}, indent=1, default=float))
        print(f"[ingest] {sid}: {n_conf} confirmed, {n_bad} unusable, {summary[sid]['awaiting_review']} awaiting "
              f"review -> {fjson}", flush=True)
    if getattr(a, "review_only", False) or not confirmed_total:
        for sdir in map(Path, a.sessions):
            print(f"[ingest] review file: {Path(sdir) / 'calibration_review.json'}")
        print("[ingest] no confirmed labels yet -> nothing trained. Confirm the prompts you typed correctly:\n"
              "         set \"status\": \"confirmed\" (or \"rejected\") per entry, correcting \"text\" where you\n"
              "         typed something else, or re-run with --confirm 0-19 --reject 7 ; then run this command again.",
              flush=True)
        return 0 if getattr(a, "review_only", False) else 2
    n = V.save_data(out / "desk_data.json", sessions, f"{a.version}: base {base} + calibration sessions {list(summary)}")
    print(f"[ingest] {out / 'desk_data.json'}: {n} phrases ({n_calib} from calibration)", flush=True)
    if a.no_train:
        return 0
    name = f"final_d{n}"
    train(out, name, a.seeds, a.steps, a.gpu_wait, compute=getattr(a, "compute", "mini"), sids=list(sessions),
          host=getattr(a, "mini_host", MINI), root=getattr(a, "mini_root", MINI_ROOT))
    mf = V.write_manifest(out, name, list(map(int, a.seeds.split(","))), a.version,
                          note=f"desk group fine-tuned on {n} labelled desk phrases ({out / 'desk_data.json'})")
    print(f"[ingest] manifest -> {mf}", flush=True)
    pool = {} if a.no_cv else run_cv(a, out, base, sessions, n_calib)
    print(json.dumps({"summary": summary, "phrases": n, "manifest": str(mf),
                      "cv_pooled_wer": {d: {k: round(v["wer"], 3) for k, v in p.items()} for d, p in pool.items()}},
                     indent=1))
    return 0


def cmd_calibrate(a) -> int:
    """Alignment scores of the true labels vs wrong labels on every labelled session, using the base LOSO models
    (each session scored by models that never saw it = the situation of a new calibration session)."""
    base = Path(a.base_root)
    sessions = V.load_data(base / "desk_data.json")
    seeds = list(map(int, a.cv_seeds.split(",")))
    rng = np.random.default_rng(0)
    rows = []
    for sid, v in sessions.items():
        models = [V.run_dir(base, s, f"loso_{sid}") / "model.pt" for s in seeds]
        L = V.ensemble_cont(base, models, sid)
        times = C.stream(sid).t[::2][:len(L)]
        wins = sorted(v["windows"])
        texts = [w[2] for w in wins]
        vocab = sorted({w for t in texts for w in t.split()})
        for j, (t0, t1, txt) in enumerate(wins):
            ws = txt.split()
            k = int(rng.integers(len(ws)))
            sw = ws[:k] + [str(rng.choice([x for x in vocab if x != ws[k]]))] + ws[k + 1:]
            cands = {"true": txt, "other_phrase": texts[(j + 1) % len(texts)],
                     "first_half": " ".join(ws[:max(1, len(ws) // 2)]), "one_word_swapped": " ".join(sw),
                     "one_word_dropped": " ".join(ws[:k] + ws[k + 1:]) if len(ws) > 1 else txt + " x"}
            tail = TAIL if v["kind"] == "prompted" else 0.0
            for kind, t in cands.items():
                r = score_window(L, times, t0 - LEAD, t1 + tail, t)
                rows.append({"sid": sid, "j": j, "kind": kind, "margin": r["margin_per_char"], "cer": r["greedy_cer"]})
    kinds = ["true", "one_word_dropped", "one_word_swapped", "first_half", "other_phrase"]
    print(f"alignment calibration: {len(sessions)} sessions, {sum(r['kind'] == 'true' for r in rows)} true windows")
    for kd in kinds:
        m = np.array([r["margin"] for r in rows if r["kind"] == kd])
        c = np.array([r["cer"] for r in rows if r["kind"] == kd])
        print(f"  {kd:<17} margin/char min {m.min():+.2f} p10 {np.percentile(m, 10):+.2f} median {np.median(m):+.2f} "
              f"max {m.max():+.2f} | greedy CER median {np.median(c):.2f} max {c.max():.2f}")
    grid = []
    for mm in (-0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0):
        for mc in (0.4, 0.5, 0.6, 0.8, 1.0):
            acc = {kd: float(np.mean([r["margin"] >= mm and r["cer"] <= mc for r in rows if r["kind"] == kd])) for kd in kinds}
            grid.append({"min_margin": mm, "max_cer": mc, **acc})
    print("  accept rates (min_margin, max_cer): " + " / ".join(kinds))
    for g in grid:
        print(f"   {g['min_margin']:+.1f} {g['max_cer']:.1f}  " + "  ".join(f"{g[k]:.2f}" for k in kinds))
    f = base / "results" / "align_calibration.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"rows": rows, "grid": grid, "lead": LEAD, "tail": TAIL, "edge": EDGE}, indent=1, default=float))
    print(f"-> {f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="*", help="calibration session directories under data/sessions/")
    p.add_argument("--calibrate", action="store_true", help="report alignment-threshold separation and exit")
    p.add_argument("--base-root", default=".cache/ctc_v4", help="version whose desk data / manifest / LOSO models are the base")
    p.add_argument("--out-root", default=".cache/ctc_v5")
    p.add_argument("--version", default="v5")
    p.add_argument("--align-manifest", default="", help="default <base-root>/deploy/manifest.json")
    p.add_argument("--align-group", default="desk")
    p.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    p.add_argument("--max-cer", type=float, default=MAX_CER)
    p.add_argument("--confirm", default="", help="idx spec (e.g. 0-19,21) marked CONFIRMED in the review file")
    p.add_argument("--reject", default="", help="idx spec marked REJECTED in the review file")
    p.add_argument("--review-only", action="store_true", help="write/refresh the review files and stop")
    p.add_argument("--compute", default="mini", choices=["mini", "local"],
                   help="where training runs (default: the Mac mini; 'local' is not for heavy runs)")
    p.add_argument("--mini-host", default=MINI)
    p.add_argument("--mini-root", default=MINI_ROOT)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=V.RECIPE["steps"])
    p.add_argument("--gpu-wait", type=float, default=float("inf"))
    p.add_argument("--no-train", action="store_true", help="stop after writing desk_data.json")
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--cv-holdout", default=f"{V.BLIND1},{V.BLIND2}")
    p.add_argument("--cv-seeds", default="0,1,2")
    p.add_argument("--decoders", default="qwen,char")
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args(argv)
    if a.calibrate:
        return cmd_calibrate(a)
    if not a.sessions:
        p.error("give at least one session directory (or --calibrate)")
    return cmd_ingest(a)


if __name__ == "__main__":
    raise SystemExit(main())
