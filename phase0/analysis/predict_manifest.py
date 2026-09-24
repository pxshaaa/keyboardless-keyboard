"""Run the (unmodified) .cache/ctc_v3/predict.py with another deploy manifest, e.g. a retrained desk ensemble published under
.cache/ctc_v4/. Model paths in the manifest resolve against: absolute; the manifest's folder; its parent (the ctc_v3 layout
'<root>/deploy/manifest.json' + 'deploy/models/..'); .cache/ctc_v3. Output goes to --out-dir (posteriors.npz, segments.json).
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.predict_manifest data/sessions/<id> --manifest <path> --out-dir <dir> [predict.py args]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PREDICT = Path(".cache/ctc_v3/predict.py")
SUBS = [('man = json.loads((HERE / "deploy" / "manifest.json").read_text())', "man = json.loads(__MANIFEST__.read_text())"),
        ('HERE / m["model"]', '__model_path__(m["model"])'),
        ('out = HERE / "out" / sid', "out = __OUTDIR__"),
        ('"manifest": str(HERE / "deploy" / "manifest.json")', '"manifest": str(__MANIFEST__)')]


def model_path(manifest: Path, rel: str) -> Path:
    p = Path(rel)
    for c in ([p] if p.is_absolute() else [manifest.parent / p, manifest.parent.parent / p, PREDICT.parent.resolve() / p]):
        if c.exists():
            return c
    raise FileNotFoundError(f"model {rel} not found relative to {manifest}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a, rest = ap.parse_known_args(argv)
    src = PREDICT.read_text()
    for old, new in SUBS:
        assert src.count(old) == 1, f"predict.py changed: {old!r}"
        src = src.replace(old, new)
    man = a.manifest.resolve()
    g = {"__name__": "__main__", "__file__": str(PREDICT.resolve()), "__MANIFEST__": man, "__OUTDIR__": a.out_dir,
         "__model_path__": lambda rel: model_path(man, rel)}
    sys.argv = [str(PREDICT), a.session_dir, *rest]
    exec(compile(src, str(PREDICT), "exec"), g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
