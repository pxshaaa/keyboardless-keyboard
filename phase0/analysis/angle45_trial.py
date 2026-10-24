"""One-command ingest + scoring for a real 45-degree trial session.

Takes a freshly recorded 45-deg keyboard session, runs the existing landmark + stream pipeline on it,
fine-tunes the 45-deg-pretrained CTC model, and scores it against a top-down baseline session with the
SAME code path and the SAME metrics, so the two numbers are comparable.

  python -m phase0.analysis.angle45_trial run --session <45deg-session-id> [--baseline <topdown-session-id>]

Metrics, per session (all within-session, blocked in time, so no cross-angle transfer is assumed):
  (a) key-down detection AUC from the kinematic feature stack  (pv_probe protocol, 10 s block bootstrap)
  (b) CTC character error rate on held-out typing windows, fine-tuned from the matching pretrained model
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SESS = Path("data/sessions")
RES = Path("results/angle45")
PRE = Path(".cache/hwt/pre45")
PY = ".venv/bin/python"


def sh(cmd: list[str], **kw) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def ensure_landmarks(sid: str) -> None:
    s = SESS / sid
    if not (s / "landmarks.parquet").exists():
        sh([PY, "-W", "ignore", "-m", "phase0.analysis.extract_landmarks", str(s)])
    if not (s / "keys.jsonl").exists():
        raise SystemExit(f"{sid}: no keys.jsonl — the trial must be a KEYBOARD session so the keylogger labels it")


# ---------------------------------------------------------------- (a) key-down detection
def keydown_auc(sid: str, half: float = 0.040, folds: int = 5) -> dict:
    """pv_probe's protocol on one session: kinematic features -> P(key-down within +-40 ms), time-blocked CV."""
    from phase0.analysis import pv_common as pv, pv_probe as pp, taps_gb as gb, touch_feats as tf
    kfr, t, P = gb.load_frames(SESS / sid)
    kt = pv.keydowns(SESS / sid)
    y = np.abs(t[:, None] - kt[None, :]).min(1) <= half
    Xg = np.hstack([gb.build_groups(P)[g] for g in gb.GROUPS]).astype(np.float32)
    per0, pool0 = tf.hand_feats(P[:, 0])
    per1, pool1 = tf.hand_feats(P[:, 1])
    n = len(kfr)
    X = np.hstack([Xg, per0.reshape(n, -1), per1.reshape(n, -1), pool0.reshape(n, -1), pool1.reshape(n, -1)]).astype(np.float32)
    oof = pp.oof_lgb(X, y, t, folds=folds)
    blk = pp.block_ids(t)
    ok = np.isfinite(oof)
    auc = pp.auc(y[ok], oof[ok])
    ci = pp.boot_auc(y[ok], oof[ok], blk[ok])
    shuf = pp.auc(y[ok], np.roll(oof[ok], int(30 * 60)))
    det = np.isfinite(P[:, :, 0, 0])
    return {"session": sid, "frames": int(n), "keydowns": int(len(kt)), "positives": int(y.sum()),
            "keydown_auc": auc, "ci95": ci, "auc_timeshuffled": shuf,
            "landmark_rate_any_hand": float(det.any(1).mean()), "landmark_rate_both": float(det.all(1).mean())}


# ---------------------------------------------------------------- (b) CTC fine-tune + CER
def ctc_cer(sid: str, init: Path | None, steps: int, seed: int, holdout: float = 0.25) -> dict:
    """Fine-tune from `init` on the first (1-holdout) of the session, report CER on the held-out tail."""
    import torch
    from phase0.analysis.seqctc import (Cfg, KbdSource, build, train, eval_windows, sym_text, infer,
                                        greedy, rows_cer, load_session)
    st = load_session(sid)
    cut = st.t[0] + (1 - holdout) * (st.t[-1] - st.t[0])
    wins = [w for w in eval_windows(st) if w[0] >= cut]
    refs = [sym_text(x[2]) for x in wins]
    cfg = Cfg(steps=steps, lr=5e-4, seed=seed)
    m = build(cfg)
    if init and init.exists():
        ck = torch.load(init, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["sd"])
    m = train(m, [(KbdSource(st, t_range=(st.t[0], cut)), 1.0)], cfg, log=f"[ft {sid} s{seed}]")
    hyp = [greedy(lp) for lp in infer(m, st, wins)]
    return {"session": sid, "init": str(init) if init else None, "windows": len(wins),
            "cer": float(rows_cer(list(zip(refs, hyp)))), "seed": seed, "steps": steps}


def _worker(args: list[str]) -> dict:
    """Run one metric in a fresh process and read its JSON back (see the note in cmd_run)."""
    f = Path(".cache/angle45") / f"worker_{abs(hash(tuple(args)))}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    sh([PY, "-W", "ignore", "-u", "-m", "phase0.analysis.angle45_trial", *args, "--out", str(f)])
    return json.loads(f.read_text())


def cmd_keydown(a) -> int:
    r = keydown_auc(a.session)
    Path(a.out).write_text(json.dumps(r, indent=1, default=float))
    return 0


def cmd_ctc(a) -> int:
    init = Path(a.init) if a.init else None
    r = ctc_cer(a.session, init, a.steps, a.seed)
    Path(a.out).write_text(json.dumps(r, indent=1, default=float))
    return 0


def cmd_run(a) -> int:
    RES.mkdir(parents=True, exist_ok=True)
    out = {"trial_session": a.session, "baseline_session": a.baseline, "when": time.strftime("%F %T"), "arms": {}}
    pairs = [("trial_45deg", a.session, PRE / f"hwt_el45_s{a.seed}.pt")]
    if a.baseline:
        pairs.append(("baseline_topdown", a.baseline, PRE / f"hwt_el90_s{a.seed}.pt"))
    for tag, sid, init in pairs:
        ensure_landmarks(sid)
        # the two metrics run in SEPARATE processes on purpose: lightgbm's OpenMP runtime and torch/MPS
        # segfault when loaded together on this machine.
        r = {"keydown": _worker(["keydown", "--session", sid])}
        print(f"[{tag}] {sid}: key-down AUC {r['keydown']['keydown_auc']:.3f} "
              f"[{r['keydown']['ci95'][0]:.3f},{r['keydown']['ci95'][1]:.3f}] "
              f"(shuffled {r['keydown']['auc_timeshuffled']:.3f}), landmarks {r['keydown']['landmark_rate_any_hand']:.3f}", flush=True)
        if not a.skip_ctc:
            r["ctc_pretrained"] = _worker(["ctc", "--session", sid, "--init", str(init), "--steps", str(a.steps), "--seed", str(a.seed)])
            r["ctc_scratch"] = _worker(["ctc", "--session", sid, "--steps", str(a.steps), "--seed", str(a.seed)])
            print(f"[{tag}] {sid}: CER {r['ctc_pretrained']['cer']:.3f} from {init.name if init else 'scratch'}, "
                  f"{r['ctc_scratch']['cer']:.3f} from scratch", flush=True)
        out["arms"][tag] = r
    if len(out["arms"]) == 2:
        t_, b_ = out["arms"]["trial_45deg"], out["arms"]["baseline_topdown"]
        out["delta_45_minus_topdown"] = {
            "keydown_auc": t_["keydown"]["keydown_auc"] - b_["keydown"]["keydown_auc"],
            "landmark_rate": t_["keydown"]["landmark_rate_any_hand"] - b_["keydown"]["landmark_rate_any_hand"],
            "cer": (t_["ctc_pretrained"]["cer"] - b_["ctc_pretrained"]["cer"]) if not a.skip_ctc else None}
        print("delta (45deg - top-down):", json.dumps(out["delta_45_minus_topdown"], indent=None), flush=True)
    f = RES / f"trial_{a.session}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", f, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.angle45_trial", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--session", required=True, help="the new 45-degree session id under data/sessions/")
    s.add_argument("--baseline", default="20260911-164237-kbd", help="a top-down keyboard session to compare against")
    s.add_argument("--steps", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--skip-ctc", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("keydown"); s.add_argument("--session", required=True); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_keydown)
    s = sub.add_parser("ctc"); s.add_argument("--session", required=True); s.add_argument("--init", default="")
    s.add_argument("--steps", type=int, default=400); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_ctc)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
