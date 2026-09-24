"""Render the angle sweep + occlusion + shadow JSONs into one markdown table.
Run: python -m phase0.analysis.contact_angle_report"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path("results/angle")


def main() -> int:
    sw = json.loads((RES / "angle_sweep.json").read_text())
    oc = json.loads((RES / "occlusion_typing.json").read_text())
    rows = []
    for name, r in sw["arms"].items():
        if "el" not in r:
            continue
        o = oc["arms"].get(name, {})
        rows.append((r["az"], -r["el"], name, r))
        rows[-1] = (r["az"], -r["el"], name, r, o)
    rows.sort()
    print("| view | elevation | contact AUC [95% CI] | occlusion rate (typing) | "
          "planar error at contact, median / p90 |")
    print("|---|---|---|---|---|")
    for az, negel, name, r, o in rows:
        view = "front" if az == 0 else "side"
        print(f"| {view} | {int(-negel)} deg | {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
              f"| {o.get('occlusion_rate', float('nan')):.2f} "
              f"| {o.get('loc_err_mm_median_contact', float('nan')):.1f} / "
              f"{o.get('loc_err_mm_p90_contact', float('nan')):.1f} mm |")
    for k in ("ref_cam4", "el90_front_postjit"):
        if k in sw["arms"]:
            r = sw["arms"][k]
            print(f"| ({k}) | - | {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | - | - |")
    for k, r in sw["arms"].items():
        if k.startswith("mirror"):
            print(f"| ({k}) | - | {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | - | - |")
    t = sw["true_height_auc"]
    print(f"| true fingertip height (upper bound) | - | {t['auc']:.3f} "
          f"[{t['ci'][0]:.3f}, {t['ci'][1]:.3f}] | - | - |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
