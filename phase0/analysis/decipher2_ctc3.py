"""decipher2 on ctc_v3 posteriors: a 'zz-ctc3[st]-<sid>' twin session (zs + desk v3 ensembles, optional capped self-training).
decipher2.py stays byte-identical (md5 pinned by the frozen configs); it skips its own CTC step because decipher.json exists."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from phase0.analysis import decipher as D
from phase0.analysis import decipher2 as D2

V3 = Path(".cache/ctc_v3")


def prepare(src: Path, a) -> Path:
    dj = src / "decipher.json"
    if dj.exists() and json.loads(dj.read_text()).get("ctc") == "v3":   # CV set built by ctcv3_eval mkdecipher
        return src
    tgt = D.SESS / f"zz-ctc3{'st' if a.selftrain else ''}-{src.name}"
    if not a.rerun and (tgt / "decipher.json").exists():
        return tgt
    t0 = time.time()
    if not (src / "landmarks.parquet").exists():
        D2.sh([sys.executable, "-m", "phase0.analysis.extract_landmarks", str(src)])
    D2.sh([sys.executable, "-W", "ignore", str(V3 / "predict.py"), str(src)])
    post = V3 / "out" / src.name / "posteriors.npz"
    desk = "desk"
    if a.selftrain:
        r = subprocess.run([sys.executable, "-W", "ignore", str(V3 / "selftrain.py"), str(src),
                            "--max-minutes", str(a.selftrain_minutes)])
        import numpy as np
        if r.returncode == 0 and "selftrained" in np.load(post).files:
            desk = "selftrained"
        else:
            print(f"[decipher2_ctc3] self-training rc={r.returncode}: falling back to the desk ensemble", flush=True)
    cmd = [sys.executable, "-W", "ignore", "-m", "phase0.analysis.ctcv3_eval", "mkdecipher", "--lpfile", str(post),
           "--src", src.name, "--out-sid", tgt.name, "--map", f"zs=zeroshot,desk={desk}"]
    if (src / "truth.txt").exists():
        cmd += ["--truth", str(src / "truth.txt")]
    D2.sh(cmd)
    print(f"[decipher2_ctc3] ctc_v3 posteriors ({'zs + ' + desk}) in {time.time() - t0:.0f}s", flush=True)
    return tgt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_dir", type=Path)
    p.add_argument("--score", type=Path, default=None)
    p.add_argument("--rerun", action="store_true", help="recompute the v3 posteriors (never decipher2's v2 CTC)")
    p.add_argument("--config", default=None)
    p.add_argument("--version", default=None, help="passed to decipher2 (its default when omitted); v4 = decipher2_v4 pipeline")
    p.add_argument("--manifest", type=Path, default=None,
                   help="v4 only: CTC deploy manifest (default .cache/ctc_v3/deploy/manifest.json), e.g. a retrained .cache/ctc_v4 one")
    p.add_argument("--selftrain", action="store_true", help="test-time self-training on the session before decoding")
    p.add_argument("--selftrain-minutes", type=float, default=15.0)
    a = p.parse_args(argv)
    if a.version == "v4":
        if a.selftrain or a.config not in (None, "recommended"):
            p.error("--version v4 is frozen: no --selftrain, --config recommended only")
        from phase0.analysis import decipher2_v4 as D4
        return D4.main(a.session_dir, manifest=a.manifest, truth=a.score, rerun=a.rerun)
    if a.manifest is not None:
        p.error("--manifest is only supported with --version v4")
    tgt = prepare(a.session_dir, a)
    D2.sh(["ssh", D2.MINI, f"mkdir -p {D2.ROOT}/data/sessions/{tgt.name} {D2.ROOT}/.cache/decipher/out"])
    D2.sh(["rsync", "-a", f".cache/decipher/out/{tgt.name}_lp.npz", f"{D2.MINI}:{D2.ROOT}/.cache/decipher/out/"])
    args = [str(tgt)]
    for flag, v in (("--version", a.version), ("--config", a.config), ("--score", a.score)):
        if v is not None:
            args += [flag, str(v)]
    return D2.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
