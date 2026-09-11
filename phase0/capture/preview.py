"""Live preview: see what the camera sees, what MediaPipe finds, and taps fire.

Run this BEFORE recording a session. It exists to answer four questions that
are expensive to get wrong after the fact:

  1. Is the camera actually live? (a frozen feed still reads at full rate)
  2. Are both hands being detected, at your real hand position?
  3. Is the framing/angle good, or are fingers leaving frame?
  4. Do taps fire when you actually tap, and stay quiet when you hover?

Controls:  q or ESC = quit,  f = freeze-check,  SPACE = pause
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from phase0.analysis.online_taps import AHEAD
from phase0.analysis.online_taps import DEFAULT_MODEL as DEFAULT_TAP_MODEL
from phase0.analysis.online_taps import OnlineTapModel, assign_sides
from phase0.capture.devices import resolve


ROTATIONS = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def apply_rotation(frame, name):
    """Rotate in software so the phone's physical orientation doesn't matter.

    Applied before landmark detection. Raw footage on disk stays untouched, so
    a wrong choice here is re-runnable without re-recording.
    """
    code = ROTATIONS.get(name)
    return frame if code is None else cv2.rotate(frame, code)


FINGERTIPS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
# MediaPipe hand skeleton edges.
EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16), (13,17),(17,18),(18,19),(19,20), (0,17)]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmarker model -> {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


class OnlineTapDetector:
    """Minimal real-time tap detector, for VISUAL confirmation only.

    Deliberately simpler than phase0.analysis.detect_taps (which is the one
    that produces the actual measurements offline). This just needs to flash
    at roughly the right moment so a human can sanity-check the idea.
    Same physics: fingertip descends (y increases downward), velocity crosses
    zero, then rebounds.
    """

    def __init__(self, min_descent=60.0, refractory=0.08):
        self.hist: dict[tuple[int, int], collections.deque] = {}
        self.last_tap: dict[tuple[int, int], float] = {}
        self.min_descent = min_descent
        self.refractory = refractory

    def update(self, key, t, y) -> bool:
        h = self.hist.setdefault(key, collections.deque(maxlen=5))
        h.append((t, y))
        if len(h) < 4:
            return False
        (t0, y0), (t1, y1), (t2, y2) = h[-4], h[-3], h[-2]
        (t3, y3) = h[-1]
        if t2 - t0 <= 0 or t3 - t1 <= 0:
            return False
        v_before = (y2 - y0) / (t2 - t0)   # +ve = descending
        v_after = (y3 - y1) / (t3 - t1)    # -ve = rebounding
        if v_before < self.min_descent or v_after > -self.min_descent * 0.6:
            return False
        if t3 - self.last_tap.get(key, -9e9) < self.refractory:
            return False
        self.last_tap[key] = t3
        return True


def draw_hand(img, pts, conf, taps_now):
    for a, b in EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], (90, 200, 90), 2, cv2.LINE_AA)
    for j, p in enumerate(pts):
        if j in FINGERTIPS:
            hot = j in taps_now
            cv2.circle(img, p, 14 if hot else 8, (0, 0, 255) if hot else (255, 120, 0),
                       -1 if hot else 2, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 3, (200, 200, 200), -1, cv2.LINE_AA)


def resolve_detector(name, model_path, ahead=AHEAD):
    """-> (detector name, OnlineTapModel|None); the heuristic is the fallback when no model exists."""
    if name == "model" and not Path(model_path).exists():
        print(f"WARNING: {model_path} not found -- falling back to the crude heuristic detector, "
              "which fires while your hands rest. Train one: python -m phase0.analysis.taps_gb train ...",
              file=sys.stderr)
        return "heuristic", None
    return (name, OnlineTapModel(Path(model_path), ahead=ahead)) if name == "model" else (name, None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="iPhone 15 Pro Camera")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--rotate", default="none", choices=list(ROTATIONS),
                   help="rotate frames before detection: none|cw90|ccw90|180")
    p.add_argument("--scale", type=float, default=0.5, help="display scale")
    p.add_argument("--model", default="models/hand_landmarker.task")
    p.add_argument("--save", default=None, help="also write an annotated mp4 here")
    p.add_argument("--seconds", type=float, default=0.0, help="0 = until quit")
    p.add_argument("--headless", action="store_true",
                   help="no window; use with --save to produce a video to inspect")
    p.add_argument("--detector", choices=["heuristic", "model"], default="model",
                   help="model = the trained taps_gb classifier (what the offline numbers measure)")
    p.add_argument("--tap-model", default=str(DEFAULT_TAP_MODEL))
    p.add_argument("--ahead", type=int, default=AHEAD,
                   help="forward frames the model's centred filters see; this is the tap latency")
    p.add_argument("--backend", choices=["cv2", "av", "net"], default="cv2",
                   help="av = AVFoundation via pyobjc (iPhone Desk View); "
                        "net = WideCam iPhone stream over WiFi (--camera auto|IP|URL)")
    args = p.parse_args(argv)

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    if args.backend == "av":
        from phase0.capture.avsource import AVVideoSource
        cap = AVVideoSource(args.camera, args.width, args.height)
        idx, name = -1, cap.name
    elif args.backend == "net":
        from phase0.capture.netsource import NetVideoSource
        cap = NetVideoSource(args.camera)
        idx, name = -1, cap.name
    else:
        idx, name = resolve(args.camera)
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {idx} ({name}). Grant Camera permission "
              "to your terminal in System Settings > Privacy & Security.", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(ensure_model(Path(args.model)))),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))

    detector, model = resolve_detector(args.detector, args.tap_model, args.ahead)

    print(f"camera {idx}: {name}")
    print(f"detector: {detector}" + (f" (thr={model.thr:.2f}, gate={model.gate_thr:.2f}, "
                                     f"latency={model.latency_frames}-{model.max_latency_frames} frames)"
                                     if model else ""))
    print("q/ESC quit | SPACE pause")

    writer = None
    tapper = OnlineTapDetector()
    fps_hist = collections.deque(maxlen=30)
    prev_gray = None
    frozen_run = 0
    tap_flash: dict[tuple[int, int], float] = {}
    tap_total = 0
    frame_i = 0
    t_start = time.monotonic()
    last_t = t_start
    paused = False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("read failed", file=sys.stderr)
            break
        frame = apply_rotation(frame, args.rotate)
        now = time.monotonic()
        if args.seconds and now - t_start > args.seconds:
            break
        if not paused:
            fps_hist.append(now - last_t)
        last_t = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and np.array_equal(gray, prev_gray):
            frozen_run += 1
        else:
            frozen_run = 0
        prev_gray = gray

        res = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int((now - t_start) * 1000))

        h, w = frame.shape[:2]
        hl = list(res.hand_landmarks or [])
        labels = [res.handedness[i][0].category_name if res.handedness else "Unknown"
                  for i in range(len(hl))]
        # flash keys: side (Left/Right) for the model, raw slot for the heuristic
        sides = assign_sides(labels)
        if model is not None:
            wl = res.hand_world_landmarks or []
            packed = []
            for hi, hand in enumerate(hl[:2]):
                arr = np.full((21, 7), np.nan)
                arr[:, 0] = [l.x * w for l in hand]
                arr[:, 1] = [l.y * h for l in hand]
                arr[:, 2] = res.handedness[hi][0].score if res.handedness else np.nan
                arr[:, 3] = [l.z * w for l in hand]
                if hi < len(wl):
                    arr[:, 4:7] = [[q.x, q.y, q.z] for q in wl[hi]]
                packed.append((labels[hi], arr))
            for rec in model.push_hands(frame_i, now, packed):
                tap_flash[(rec["hand"], rec["finger"])] = now
                tap_total += 1
        taps_now: set[int] = set()
        for hi, hand in enumerate(hl):
            pts = [(int(l.x * w), int(l.y * h)) for l in hand]
            if model is None:
                for j in FINGERTIPS:
                    if tapper.update((hi, j), now, pts[j][1]):
                        tap_flash[(hi, j)] = now
                        tap_total += 1
                fkey = hi
            else:
                fkey = sides[hi]
            hot = {j for j in FINGERTIPS if now - tap_flash.get((fkey, j), -9e9) < 0.18}
            taps_now |= hot
            conf = res.handedness[hi][0].score if res.handedness else 0.0
            draw_hand(frame, pts, conf, hot)
            label = f"{res.handedness[hi][0].category_name} {conf:.2f}" if res.handedness else "?"
            cv2.putText(frame, label, (pts[0][0] - 40, pts[0][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        frame_i += 1
        fps = len(fps_hist) / sum(fps_hist) if sum(fps_hist) > 0 else 0.0
        nh = len(hl)
        prob = model.score if model is not None else 0.0
        bar = [f"{fps:5.1f} fps", f"hands: {nh}", f"taps: {tap_total}"]
        if model is not None:
            bar.append(f"score: {prob:.2f}")
        colour = (0, 255, 0) if nh else (0, 200, 255)
        if frozen_run > 15:
            bar.append("!! CAMERA FROZEN !!")
            colour = (0, 0, 255)
        if nh == 0:
            bar.append("no hands - check framing/lighting")
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "   ".join(bar), (16, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, colour, 3, cv2.LINE_AA)
        if model is not None:
            x0, x1, y0 = 16, min(w - 16, 336), 82
            cv2.rectangle(frame, (x0, y0), (x1, y0 + 22), (0, 0, 0), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * min(prob, 1.0)), y0 + 22),
                          (0, 0, 255) if prob >= model.thr else (0, 200, 255), -1)
            xt = x0 + int((x1 - x0) * model.thr)
            cv2.line(frame, (xt, y0 - 3), (xt, y0 + 25), (255, 255, 255), 2)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"avc1"),
                                         30.0, (w, h))
            writer.write(frame)

        if not args.headless:
            disp = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            cv2.imshow("phase0 preview", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"wrote {args.save}")
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"total taps detected: {tap_total}  detector: {detector}"
          + (f"  mean score: {model.mean_score:.3f}" if model is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
