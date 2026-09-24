"""Freeze two llmdec v2 configs from TUNING results only (never blind sessions).
generic : camera/decoder weights (method, llm, prompt w/o profile, K, lam, wb, cb, mu_lex, context) chosen on the joint
          objective mean(kbd WER, old-desk WER), tiebreak CER sum; no personal terms.
personal: the generic config's camera/decoder weights held fixed; personal-context weights (profile prompt, mu_pers, personal
          word-LM nu / nu_ng) chosen on the user-written keyboard windows only (kbd WER, tiebreak kbd CER then old WER),
          because blind sessions contain the user's own phrases while the old desk session is generic prompted text.
Writes results/llmdec/frozen_config_v2.json (refuses to overwrite; LLMDEC_FROZEN_OUT for dry runs)."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

RES = Path("results/llmdec")
FROZEN = Path(os.environ.get("LLMDEC_FROZEN_OUT", str(RES / "frozen_config_v2.json")))
PERSONAL_KEYS = {"mu_pers", "nu", "nu_ng", "profile", "lam_p", "plm"}
PROFILE_OF = {"p3": "p5", "p4": "p5", "p6": "p7"}   # prompt without -> with profile summary


def parse_cfg_name(name: str) -> dict:
    return dict(kv.split("=") for kv in name.split(",")) if name != "default" else {}


def is_personal(run: dict) -> bool:
    if run["method"] == "gv":
        return run["prompt"] in ("p5", "p7") or run.get("mu_pers", 0) or run.get("nu", 0) or run.get("lam_p", 0)
    return any(k in run and str(run[k]) not in ("0", "0.0", "False") for k in PERSONAL_KEYS)


def collect():
    cands = []
    t = json.loads((RES / "tuning.json").read_text())
    for m, b in t["best"].items():
        cands.append({"label": f"gv_{m}_v1", "run": {"method": "gv", "llm": m, "groups": "zs",
                      **{k: b[k] for k in ("prompt", "K", "lam", "wb", "cb")}, "mu_lex": 0.0, "mu_pers": 0.0, "nu": 0.0}, "tuning": b})
    for f2 in sorted(RES.glob("tuning_v2_8b*.json")):
        plm = f2.stem[len("tuning_v2_8b_"):] if f2.stem != "tuning_v2_8b" else ""
        for tr in json.loads(f2.read_text())["trials"]:
            run = {"method": "gv", "llm": "8b", "groups": "zs",
                   **{k: tr[k] for k in ("prompt", "K", "lam", "wb", "cb", "mu_lex", "mu_pers", "nu")}}
            if tr.get("lam_p"):
                run.update(lam_p=tr["lam_p"], plm=plm)
            cands.append({"label": "gv_8b_v2" + (f"+{plm}" if tr.get("lam_p") else ""), "run": run, "tuning": tr})
    for f in sorted(RES.glob("dec2_*.json")):
        if "val" in f.stem:
            continue
        llm = f.stem.split("_")[-1]
        for name, v in json.loads(f.read_text()).items():
            if isinstance(v, dict) and "obj" in v:
                cands.append({"label": f"dec2_{llm}", "run": {"method": "dec2", "llm": llm, "grid": name, **parse_cfg_name(name)},
                              "tuning": v})
    return cands


def camera_key(run: dict):
    r = {k: v for k, v in run.items() if k not in PERSONAL_KEYS and k != "grid"}
    if r["method"] == "gv":
        r["prompt"] = {"p5": "p3", "p7": "p6"}.get(r["prompt"], r["prompt"])
    return json.dumps({k: str(v) for k, v in sorted(r.items())})


def main() -> int:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite")
    cands = collect()
    obj = lambda c: (c["tuning"]["obj"], c["tuning"]["kbd"]["cer"] + c["tuning"]["old"]["cer"])  # noqa: E731
    gen = sorted([c for c in cands if not is_personal(c["run"])], key=obj)
    generic = gen[0]
    ck = camera_key(generic["run"])
    same_cam = [c for c in cands if camera_key(c["run"]) == ck]
    # profile prompt p5 pairs with p3 or p4 base; accept either base when the generic prompt is p3/p4
    if generic["run"]["method"] == "gv" and generic["run"]["prompt"] in ("p3", "p4"):
        alt = dict(generic["run"], prompt="p3" if generic["run"]["prompt"] == "p4" else "p4")
        same_cam += [c for c in cands if camera_key(c["run"]) == camera_key(alt) and c["run"]["prompt"] == "p5"]
    pers = sorted(same_cam, key=lambda c: (c["tuning"]["kbd"]["wer"], c["tuning"]["kbd"]["cer"], c["tuning"]["old"]["wer"]))
    personal = pers[0]
    rows = {}
    for c in sorted(cands, key=obj):
        fam = c["label"] + (":personal" if is_personal(c["run"]) else ":generic")
        rows.setdefault(fam, c)
    rows["FROZEN generic"] = generic
    rows["FROZEN personal"] = personal
    cfg = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tuned_on": ["keyboard windows (user-written text; hwt LOSO/held out-of-fold posteriors, 50 windows)",
                        "OLD desk session 20260910-202149-desk (generic prompted phrases), zero-shot hwtmix, 21 segments"],
           "protocol": __doc__,
           "generic": generic, "personal": personal,
           "recommended_for_blind2": "personal" if personal["tuning"]["kbd"]["wer"] <= generic["tuning"]["kbd"]["wer"] + 0.01 else "generic",
           "run": personal["run"] if personal["tuning"]["kbd"]["wer"] <= generic["tuning"]["kbd"]["wer"] + 0.01 else generic["run"],
           "report_rows": rows, "n_candidates": len(cands),
           "blind1_disclosure": ("blind session 1 (20260912-174542-desk) truth was visible in the task brief (e.g. 'codex', 'div'); "
                                 "no decoder in this study was scored on it before this freeze; personal corpus decontaminated "
                                 "against its truth (4-gram) by the builder and again in llmdec_personal.py"),
           "code_md5": {n: hashlib.md5(Path(f"phase0/analysis/{n}").read_bytes()).hexdigest()
                        for n in ("llmdec.py", "llmdec_mlx.py", "llmdec2.py", "decipher2.py")}}
    cfg["chosen"] = cfg["recommended_for_blind2"]
    # CTC posterior group for blind sessions (decided before any blind scoring): tuning used zero-shot posteriors only (the
    # desk-fine-tuned model was trained on the OLD session, so it cannot be tuned there). Evidence from results/seqctc2:
    # desk fine-tuning halves desk CER (0.157 -> 0.083 cross-validated). Deploy groups = "ens" (mean CTC ll of zs + desk
    # models, union of candidates, merged guesses); zs rows are reported alongside on blind-1.
    for k in ("generic", "personal"):
        cfg[k] = json.loads(json.dumps(cfg[k]))
        if cfg[k]["run"]["method"] == "gv":
            cfg[k]["run"]["groups"] = "ens"
    cfg["run"] = cfg[cfg["recommended_for_blind2"]]["run"]
    cfg["groups_decision"] = ("blind sessions decode with groups=ens (hwtmix zero-shot + hwtmix+desk fine-tuned, seeds 0,1); "
                              "weights tuned on zs posteriors; untuned for ens (no legal tuning data); zs variants reported too")
    v3 = Path(".cache/ctc_v3/deploy/manifest.json")
    cfg["ctc_v3_option"] = (str(v3) + " present at freeze: its posteriors may be decoded as an extra, untuned option"
                            if v3.exists() else "no ctc_v3 deploy manifest at freeze time")
    FROZEN.write_text(json.dumps(cfg, indent=1, default=float))
    for nm in ("generic", "personal"):
        c = cfg[nm]
        print(f"{nm:<9} {c['label']:<10} kbd {c['tuning']['kbd']['wer']:.3f} old {c['tuning']['old']['wer']:.3f} {json.dumps(c['run'])}")
    print("recommended for blind-2:", cfg["recommended_for_blind2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
