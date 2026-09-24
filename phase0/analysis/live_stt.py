"""Rolling Whisper over the last WIN s of data/live/<ts>/audio.wav: heard.json (live tail) + transcript.jsonl (stable segments).
Run with the Python that has openai-whisper: /usr/bin/python3 -m phase0.analysis.live_stt data/live/<ts> [model]"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR, WIN, HOP, STABLE = 16000, 15.0, 2.0, 2.5
JUNK = {"thank you", "thanks", "you", "bye", "thanks for watching", "thank you for watching"}


def main() -> int:
    d = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    import whisper
    m = whisper.load_model(model)
    t0 = json.loads((d / "audio_t0.json").read_text())["t0"]
    out = open(d / "transcript.jsonl", "a")
    committed: list[tuple[float, float]] = []
    print("stt ready", flush=True)
    while True:
        time.sleep(HOP)
        try:
            with wave.open(str(d / "audio.wav"), "rb") as w:
                n = w.getnframes()
                k = min(n, int(WIN * SR))
                w.setpos(n - k)
                x = np.frombuffer(w.readframes(k), np.int16).astype(np.float32) / 32767
        except (wave.Error, EOFError):
            continue
        if len(x) < SR or float(np.abs(x).max()) < 0.004:
            continue
        x = x / max(float(np.abs(x).max()), 1e-3) * 0.6  # quiet room mic: normalise level
        t_win = t0 + (n - k) / SR
        t_end = t0 + n / SR
        tic = time.time()
        r = m.transcribe(x, language="en", fp16=False, condition_on_previous_text=False, no_speech_threshold=0.5)
        tail = []
        for s in r["segments"]:
            text = s["text"].strip()
            if not text or s.get("no_speech_prob", 0) > 0.6 or text.lower().strip(" .!?") in JUNK:
                continue
            a, b = t_win + s["start"], t_win + s["end"]
            tail.append(text)
            if b < t_end - STABLE and not any(min(b, cb) - max(a, ca) > 0.5 * (b - a) for ca, cb in committed):
                committed.append((a, b))
                out.write(json.dumps({"t0": a, "t1": b, "text": text}) + "\n"), out.flush()
                print(f"[{time.strftime('%H:%M:%S')}] heard: {text}", flush=True)
        (d / "heard.json").write_text(json.dumps({"text": " ".join(tail[-2:]), "lag_s": round(time.time() - tic, 1)}))


if __name__ == "__main__":
    raise SystemExit(main())
