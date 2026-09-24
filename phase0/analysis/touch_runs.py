"""Runner: one arm x seeds -> kbd held-out F1, desk count error, oracle CER, end-to-end CER."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import lightgbm  # noqa: F401

from phase0.analysis import touch_common as tc


def feat_fn(arm: str):
    if arm in ("base", "selftrain", "selftrain_deskonly", "selftrain_w1", "selftrain_w10"):
        return tc.base_X
    if arm == "touch":
        from phase0.analysis import touch_feats as tf
        return tf.feat_base_touch
    if arm == "touchdyn":
        from phase0.analysis import touch_feats as tf
        return tf.feat_base_touchdyn
    if arm == "ego":
        from phase0.analysis import contact_ego as ce
        return ce.feat_base_ego
    if arm == "touch_ego":
        from phase0.analysis import touch_feats as tf
        from phase0.analysis import contact_ego as ce
        return lambda s: np.hstack([tf.feat_base_touch(s), ce.ego_feats(s)])
    raise SystemExit(arm)


def desk_file(arm, seed):
    return tc.CACHE / f"desk_probs_{arm}_s{seed}.npz"


def kbd_file(arm, seed):
    return tc.CACHE / f"kbd_{arm}_s{seed}.json"


def run_seed(arm: str, seed: int) -> dict:
    if kbd_file(arm, seed).exists() and desk_file(arm, seed).exists():
        k = json.loads(kbd_file(arm, seed).read_text())
        z = np.load(desk_file(arm, seed))
        return {"kbd": k, "p": z["p"], "gate": z["gate"]}
    fn = feat_fn(arm)
    if arm.startswith("selftrain"):
        from phase0.analysis import desk_selftrain as st
        from phase0.analysis import taps_gb as gb
        base = np.load(desk_file("base", seed))
        kw = {"selftrain": {}, "selftrain_deskonly": {"desk_only": True},
              "selftrain_w1": {"w_desk": 1.0}, "selftrain_w10": {"w_desk": 10.0}}[arm]
        r = st.selftrain(fn, base["p"], seed, **kw)
        gate = np.where(np.isfinite(r["gate_new"]), r["gate_new"], base["gate"])
        bk = json.loads(kbd_file("base", seed).read_text())
        m, g = r["models"][0]
        k = tc.kbd_score(m, g, bk["cfg"], fn)
        k["cfg"] = bk["cfg"]
        k["note"] = "fold-0 self-trained model, event cfg from the base arm's keyboard OOF tuning"
        n_pseudo = [len(x) for x in r["taps"]]
        k["pseudo_taps"] = int(sum(n_pseudo))
        p = r["p"]
    else:
        c0 = kbd_file(arm, 0)
        if seed == 0 or not c0.exists():
            res = tc.kbd_protocol(fn, seed=seed)
        else:  # seeds > 0 reuse this arm's seed-0 OOF-tuned event parameters
            res = tc.kbd_protocol(fn, seed=seed, oof=False, cfg=json.loads(c0.read_text())["cfg"])
        d = tc.desk_probs(res["model"], res["gate"], fn)
        np.savez(tc.CACHE / f"kbdtest_probs_{arm}_s{seed}.npz", p=res["test_p"], gate=res["test_gate"])
        import joblib
        joblib.dump({"model": res["model"], "gate": res["gate"], "cfg": res["cfg"]},
                    tc.CACHE / f"model_{arm}_s{seed}.joblib")
        k = {x: v for x, v in res.items() if x not in ("model", "gate", "test_p", "test_gate")}
        p, gate = d["p"], d["gate"]
    kbd_file(arm, seed).write_text(json.dumps(k, default=float))
    np.savez(desk_file(arm, seed), p=p, gate=gate)
    return {"kbd": k, "p": p, "gate": gate}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--oracle-dens", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    out_p = tc.Path("results/contact") / f"arm_{a.arm}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for seed in a.seeds:
        t0 = time.time()
        r = run_seed(a.arm, seed)
        row = out.get(str(seed), {})
        row["kbd"] = {k: r["kbd"][k] for k in ("f1", "recall", "precision", "taps", "still_fp")
                      if k in r["kbd"]}
        row["count"] = tc.count_error(r["p"], r["gate"])
        print(f"[{a.arm} s{seed}] kbd F1={100*row['kbd']['f1']:.1f} "
              f"desk count_err={row['count']['count_err']:.3f} r={row['count']['count_r']:.3f}",
              flush=True)
        if a.oracle and "oracle" not in row:
            oc = tc.oracle_cer(r["p"], r["gate"], procs=a.procs)
            row["oracle"] = {k: v for k, v in oc.items() if k != "rows"}
            print(f"[{a.arm} s{seed}] oracle CER={oc['oracle_cer']:.3f} "
                  f"[{oc['ci'][0]:.3f}, {oc['ci'][1]:.3f}] dens={oc['density_med']:.2f}", flush=True)
        if a.oracle_dens and "oracle_dens" not in row:
            od = tc.oracle_density(r["p"], r["gate"], procs=a.procs)
            row["oracle_dens"] = od
            print(f"[{a.arm} s{seed}] density-matched oracle CER " + " ".join(
                f"{d}:{od[d]['cer']:.3f}" for d in od if d != "mean") + f" mean={od['mean']['cer']:.3f}", flush=True)
        if a.e2e and "e2e" not in row:
            e = tc.e2e(r["p"], r["gate"], f"{a.arm}_s{seed}", procs=a.procs)
            row["e2e"] = e
            print(f"[{a.arm} s{seed}] e2e CER={e['cer']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}] "
                  f"snap exact={e['snap_exact']:.2f} <=1={e['snap_le1']:.2f}", flush=True)
        out[str(seed)] = row
        out_p.write_text(json.dumps(out, indent=1, default=float))
        print(f"[{a.arm} s{seed}] done ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
