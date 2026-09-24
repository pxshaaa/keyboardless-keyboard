"""Human review is label provenance; recognizer agreement is only a quality flag."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phase0.analysis.text_contract import legacy_training_text


def load_review(sdir: Path, prompts: list) -> dict:
    digest = hashlib.sha256((sdir / "phrases.jsonl").read_bytes()).hexdigest()
    path = sdir / "calibration_review.json"
    if not path.exists():
        data = {"version": 1, "session": sdir.name, "prompts_sha256": digest,
                "entries": [{"idx": i, "status": "pending", "text": text} for i, _, _, text in prompts]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    data = json.loads(path.read_text())
    if data.get("version") != 1 or data.get("session") != sdir.name or data.get("prompts_sha256") != digest:
        raise ValueError(f"{path}: review does not match this recording's prompts")
    entries = data["entries"]
    ids = [e["idx"] for e in entries]
    if len(ids) != len(set(ids)) or set(ids) != {i for i, *_ in prompts}:
        raise ValueError(f"{path}: expected exactly one review entry per prompt")
    if any(e.get("status") not in {"pending", "confirmed", "rejected"} for e in entries):
        raise ValueError(f"{path}: invalid review status")
    return {e["idx"]: e for e in entries}


def decision(entry: dict, alignment: dict, min_margin: float, max_cer: float) -> dict:
    if entry.get("status") != "confirmed":
        return {"accepted": False, "reason": entry.get("status", "pending")}
    try:
        text = legacy_training_text(entry["text"])
    except (ValueError, KeyError) as e:
        return {"accepted": False, "reason": f"unsupported_label: {e}"}
    if alignment.get("status") != "ok" or alignment.get("cut_at_end", False):
        return {"accepted": False, "reason": "review_window", "model_text": text}
    agreement = alignment["margin_per_char"] >= min_margin and alignment["greedy_cer"] <= max_cer
    return {"accepted": True, "reason": "confirmed" if agreement else "confirmed_model_disagrees",
            "model_text": text, "model_agreement": agreement}
