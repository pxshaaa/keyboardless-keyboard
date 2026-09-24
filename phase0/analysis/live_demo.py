"""Live desk typing: WideCam -> MediaPipe -> desk CTC (rolling window) -> raw letters now, qwen word decode after a pause.
Serves http://localhost:8765. Run: PYTHONPATH=. .venv/bin/python -m phase0.analysis.live_demo"""
from __future__ import annotations

import argparse
import functools
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

torch.backends.mps.is_available = functools.lru_cache()(lambda: False)  # CPU: small models, no MPS warm-up stalls

from phase0.analysis import ctcv3 as C  # noqa: E402
from phase0.analysis import seqctc as S  # noqa: E402
from phase0.analysis import seqctc2 as S2  # noqa: E402
from phase0.analysis.predict_manifest import model_path  # noqa: E402
from phase0.capture.netsource import NetVideoSource  # noqa: E402

CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
QWEN_CFG = json.loads(Path("results/seqctc2/tuning/tune_qwen_lex_hwtmix.json").read_text())["best"]

HTML = """<!doctype html><title>live desk typing</title>
<style>body{margin:0;background:#111;color:#eee;font:16px -apple-system,sans-serif;display:flex;height:100vh}
#v{height:100vh;object-fit:contain;background:#000}#p{flex:1;padding:28px;display:flex;flex-direction:column;gap:18px;overflow:auto}
.l{color:#888;font-size:13px;letter-spacing:.08em;text-transform:uppercase}#raw{font:28px ui-monospace,Menlo,monospace;color:#9fd;min-height:40px}
#dec{font-size:44px;font-weight:700;line-height:1.2;min-height:60px}#hist div{font-size:22px;color:#bbb;margin:4px 0}#heard{font-size:26px;color:#fc9;min-height:36px}#st{color:#777;font-size:13px}</style>
<img id=v><div id=p><div><div class=l>raw letters (live)</div><div id=raw></div></div>
<div><div class=l>decoded phrase</div><div id=dec></div><div id=st></div></div><div><div class=l>heard (speech to text)</div><div id=heard></div></div>
<div><div class=l>earlier phrases</div><div id=hist></div></div></div>
<script>(async function loop(){try{const b=await (await fetch('/frame?'+Date.now())).blob();const u=URL.createObjectURL(b);
v.onload=()=>URL.revokeObjectURL(u);v.src=u}catch(e){}setTimeout(loop,60)})();
setInterval(async()=>{const s=await (await fetch('/state')).json();
raw.textContent=s.raw;dec.textContent=s.decoded;st.textContent=s.status;heard.textContent=s.heard;
hist.innerHTML=s.history.map(h=>'<div>'+h+'</div>').join('')},150)</script>"""


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = ""
        self.decoded = ""
        self.status = "loading models…"
        self.history: list[str] = []
        self.jpeg: bytes | None = None
        self.heard = ""


ST = State()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/state":
            with ST.lock:
                body = json.dumps({"raw": ST.raw, "decoded": ST.decoded, "status": ST.status, "history": ST.history[-6:][::-1],
                                   "heard": ST.heard})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path.startswith("/frame"):
            with ST.lock:
                j = ST.jpeg or b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(j)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(j)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                last = None
                while True:
                    with ST.lock:
                        j = ST.jpeg
                    if j is not None and j is not last:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                        last = j
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())


def load_models(manifest: Path, group: str):
    man = json.loads(manifest.read_text())
    return [C.load(m["recipe"], model_path(manifest.resolve(), m["model"])) for m in man["groups"][group]["members"]]


def draw(frame, hands):
    for pts in hands:
        p = pts[:, :2]
        for a, b in CONN:
            cv2.line(frame, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (235, 245, 255), 2, cv2.LINE_AA)
        for k in (4, 8, 12, 16, 20):
            cv2.circle(frame, tuple(p[k].astype(int)), 6, (80, 220, 255), -1, cv2.LINE_AA)


class Live:
    def __init__(self, a):
        self.a = a
        self.buf: deque[tuple[float, np.ndarray]] = deque()  # (t, P[2,21,3]) raw px
        self.blk = threading.Lock()
        self.models = load_models(Path(a.manifest), a.group)
        self.done_until = -1.0  # end time of the last finalised segment
        self.last_active = None
        self.t_ref = time.monotonic()
        self.qwen = None
        self.log = None
        if a.log:
            self.log = Path("data/live") / time.strftime("%Y%m%d-%H%M%S")
            self.log.mkdir(parents=True, exist_ok=True)
            threading.Thread(target=self.record_audio, daemon=True).start()
            threading.Thread(target=self.tail_transcript, daemon=True).start()

    def tail_transcript(self):
        f = self.log / "heard.json"
        while True:
            time.sleep(0.3)
            try:
                h = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            with ST.lock:
                ST.heard = h["text"]

    def record_audio(self, sr: int = 16000):
        import sounddevice as sd
        import wave
        wf = wave.open(str(self.log / "audio.wav"), "wb")
        wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(sr)
        t_start = time.monotonic() - self.t_ref
        (self.log / "audio_t0.json").write_text(json.dumps({"t0": t_start, "sr": sr}))

        def cb(x, n, ti, status):
            wf.writeframes((np.clip(x[:, 0], -1, 1) * 32767).astype(np.int16).tobytes())
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=(int(self.a.mic) if self.a.mic and self.a.mic.isdigit() else self.a.mic), callback=cb):
            while True:
                time.sleep(1)

    # ---------------------------------------------------------------- capture + landmarks
    def capture(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        from phase0.analysis.extract_landmarks import ensure_model
        opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ensure_model())),
                                     running_mode=RunningMode.VIDEO, num_hands=2)
        cap = NetVideoSource(self.a.camera)
        if not cap.isOpened():
            raise SystemExit("WideCam not reachable: open the app on the phone and keep it in the foreground")
        with ST.lock:
            ST.status = f"camera {cap.name} connected"
        last_ms, nf, t_fps = -1, 0, time.monotonic()
        with HandLandmarker.create_from_options(opts) as lm:
            while True:
                ok, fr = cap.read(timeout=2.0)
                if not ok or fr is None:
                    continue
                t = time.monotonic() - self.t_ref
                h, w = fr.shape[:2]
                ms = int(t * 1000)
                ms = ms if ms > last_ms else last_ms + 1
                last_ms = ms
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ms)
                P = np.full((2, 21, 3), np.nan, np.float32)
                hands = []
                for k, hl in enumerate((res.hand_landmarks or [])[:2]):
                    P[k] = [(q.x * w, q.y * h, q.z * w) for q in hl]
                    hands.append(P[k])
                with self.blk:
                    self.buf.append((t, P))
                    while self.buf and t - self.buf[0][0] > 20.0:
                        self.buf.popleft()
                nf += 1
                if nf % 3 == 0:  # ~20 fps preview, portrait like the demo clips
                    draw(fr, hands)
                    view = cv2.rotate(cv2.resize(fr, (960, 540)), cv2.ROTATE_90_COUNTERCLOCKWISE)
                    okj, j = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with ST.lock:
                        ST.jpeg = j.tobytes()
                if nf % 120 == 0:
                    now = time.monotonic()
                    self.fps = 120 / (now - t_fps)
                    t_fps = now

    # ---------------------------------------------------------------- rolling inference
    def window(self, span: float):
        with self.blk:
            items = list(self.buf)
        if len(items) < 60:
            return None
        t_now = items[-1][0]
        ts = np.array([t for t, _ in items])
        grid = np.arange(max(ts[0], t_now - span), t_now, 1 / 60)
        j = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
        j = np.where((j > 0) & (np.abs(ts[j - 1] - grid) < np.abs(ts[j] - grid)), j - 1, j)
        P = np.stack([items[k][1] for k in j])
        P[np.abs(ts[j] - grid) > 0.05] = np.nan
        A, M = S.normalise(P)
        return grid, A, M

    @torch.no_grad()
    def posteriors(self, grid, A, M):
        st = S.Stream("live", grid, A, M, np.zeros(0), np.zeros(0, np.int64))
        return C.logmean([S2.infer_cont(m, st) for m in self.models]), grid[::2][: (len(grid) + 1) // 2]

    def infer_loop(self):
        while True:
            time.sleep(self.a.tick)
            w = self.window(self.a.span)
            if w is None:
                continue
            try:
                lp, times = self.posteriors(*w)
            except Exception as e:  # noqa: BLE001
                with ST.lock:
                    ST.status = f"inference error: {e}"
                continue
            t_now = w[0][-1]
            segs = S2.segments(lp, times, gap=self.a.gap, pad=0.5, thr=0.5)
            segs = [(s0, s1) for s0, s1 in segs if times[min(s1, len(times) - 1)] > self.done_until + 0.3]
            if not segs:
                with ST.lock:
                    ST.raw, ST.status = "", f"listening · {getattr(self, 'fps', 0):.0f} fps tracking"
                continue
            s0, s1 = segs[-1]
            s1 = min(s1, len(times))
            raw = S.greedy(lp[s0:s1])
            pnb = 1 - np.exp(np.logaddexp(lp[s0:s1, S.BLANK], lp[s0:s1, S.OTHER]))
            act = np.where(pnb > 0.5)[0]
            t_last = float(times[s0 + act[-1]]) if len(act) else t_now
            with ST.lock:
                ST.raw = raw
                ST.status = f"typing… ({t_now - t_last:.1f}s since last tap)"
            if t_now - t_last >= self.a.pause and len(act) >= 3:
                self.done_until = float(times[s1 - 1])
                self.finalize(lp[s0:s1].copy(), raw, t_last)

    # ---------------------------------------------------------------- word decode after a pause
    def finalize(self, lp, raw, t_last):
        t0 = time.monotonic()
        if self.qwen is None:
            from phase0.analysis import autocorrect as AC
            S2._SC["qwen"] = self.qwen = AC.NLM("Qwen/Qwen2.5-0.5B", device="cpu")
        try:
            hyp = S2.run_decoder({"kind": "qwen", "cfg": QWEN_CFG}, lp)
        except Exception as e:  # noqa: BLE001
            hyp = f"[decode error: {e}] {raw}"
        shown_after = (time.monotonic() - self.t_ref) - t_last
        with ST.lock:
            if ST.decoded:
                ST.history.append(ST.decoded)
            ST.decoded = hyp
            ST.raw = ""
            ST.status = f"decoded {shown_after:.1f}s after your last tap (decoder {time.monotonic() - t0:.1f}s)"
        print(f"[{time.strftime('%H:%M:%S')}] raw='{raw}' -> '{hyp}'  ({shown_after:.1f}s after last tap)", flush=True)
        if self.log:
            k = len(list(self.log.glob("lp_*.npy")))
            np.save(self.log / f"lp_{k:03d}.npy", lp)
            with open(self.log / "events.jsonl", "a") as f:
                f.write(json.dumps({"k": k, "t_last_tap": t_last, "t_shown": time.monotonic() - self.t_ref, "raw": raw, "decoded": hyp}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="auto")
    ap.add_argument("--manifest", default=".cache/ctc_v4/deploy/manifest.json")
    ap.add_argument("--group", default="desk")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tick", type=float, default=0.4, help="s between rolling inferences")
    ap.add_argument("--span", type=float, default=15.0, help="rolling window, s")
    ap.add_argument("--gap", type=float, default=1.5, help="silence that separates phrases, s")
    ap.add_argument("--pause", type=float, default=1.2, help="silence after the last tap before decoding, s")
    ap.add_argument("--log", action="store_true", help="record mic + per-phrase posteriors to data/live/<ts>/ for review")
    ap.add_argument("--mic", default=None, help="sounddevice input device index/name (default: system default)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    live = Live(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=live.infer_loop, daemon=True).start()
    print(f"open http://localhost:{a.port}", flush=True)
    with ST.lock:
        ST.status = "loading word decoder…"
    from phase0.analysis import autocorrect as AC  # warm the 0.5B decoder before the first phrase
    S2._SC["qwen"] = live.qwen = AC.NLM("Qwen/Qwen2.5-0.5B", device="cpu")
    with ST.lock:
        ST.status = "connecting camera…"
    live.capture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
