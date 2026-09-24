"""Phase 0 session recorder: camera video + keystrokes in one process on one
time.monotonic() clock (see phase0/CONTRACT.md)."""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2

from phase0.capture.devices import preflight
from pynput import keyboard

# PRIVACY (hard requirement): a global key hook would otherwise capture passwords
# and messages. Only these keys are ever named; everything else -> "unknown".
UNKNOWN = "unknown"
_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_SPECIAL_KEYS: dict[str, str] = {
    "space": "space",
    "enter": "enter",
    "tab": "tab",
    "backspace": "backspace",
    "esc": "esc",
    "shift": "shift",
    "shift_r": "shift",
    "shift_l": "shift",
    "ctrl": "ctrl",
    "ctrl_r": "ctrl",
    "ctrl_l": "ctrl",
    "alt": "alt",
    "alt_r": "alt",
    "alt_l": "alt",
    "alt_gr": "alt",
    "cmd": "cmd",
    "cmd_r": "cmd",
    "cmd_l": "cmd",
}
MODIFIER_NAMES = frozenset({"shift", "ctrl", "alt", "cmd"})
# With any of these held the keystroke is a shortcut; shortcut sequences are as
# identifying as text, so the content is suppressed too.
_SUPPRESSING_MODIFIERS = frozenset({"ctrl", "alt", "cmd"})


def key_name(key: Any) -> str:
    """Raw pynput name ("shift_r", "f5") or character, before the allowlist."""
    name = getattr(key, "name", None)
    if isinstance(name, str):
        return name
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        return char
    return UNKNOWN


def normalize_key(key: Any, held_modifiers: Optional[set[str]] = None) -> str:
    """Contract-legal name: a-z0-9 / space enter tab backspace shift ctrl alt
    cmd esc / "unknown". Applies the privacy allowlist."""
    raw = key_name(key)

    special = _SPECIAL_KEYS.get(raw)
    if special is not None:
        return special

    # Multi-char raw name that is not a known special => function/media key.
    if len(raw) != 1:
        return UNKNOWN

    if held_modifiers and (held_modifiers & _SUPPRESSING_MODIFIERS):
        return UNKNOWN

    lowered = raw.lower()
    if lowered in _ALLOWED_CHARS:
        return lowered
    return UNKNOWN


class JsonlWriter:
    """Append-only JSONL sink, thread-safe, flushed every `flush_every` rows so
    a crash loses at most a fraction of a second."""

    def __init__(self, path: Path, flush_every: int = 30) -> None:
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        self._lock = threading.Lock()
        self._since_flush = 0
        self.count = 0
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self.count += 1
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._fh.flush()
                self._since_flush = 0

    def flush(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def list_mac_cameras() -> list[str]:
    """Camera names in AVFoundation order (position == OpenCV index)."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except Exception:
        return []
    names: list[str] = []
    for line in out.splitlines():
        m = re.match(r"^\s{4,}([^:]+):\s*$", line)
        if m:
            name = m.group(1).strip()
            if name and name != "Camera":
                names.append(name)
    return names


def resolve_camera(spec: str) -> tuple[int, str]:
    """Resolve --camera (integer index or name substring) to (index, name)."""
    names = list_mac_cameras() if sys.platform == "darwin" else []
    if spec.strip().lstrip("+-").isdigit():
        idx = int(spec)
        name = names[idx] if 0 <= idx < len(names) else f"index:{idx}"
        return idx, name
    needle = spec.lower()
    for i, name in enumerate(names):
        if needle in name.lower():
            return i, name
    raise SystemExit(
        f"No camera matching {spec!r}. Detected cameras: {names or '(none)'}\n"
        "Pass an integer index instead, or check System Settings > Privacy & "
        "Security > Camera."
    )


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    # These are hints only; the driver picks the nearest supported mode, so what
    # we actually got is measured and stored in meta.json.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


_QUEUE_MAX = 240  # ~4 s at 60 fps; beyond this we drop rather than stall capture


class Recorder:
    def __init__(self, session_dir: Path, args: argparse.Namespace) -> None:
        self.dir = session_dir
        self.args = args
        self.stop = threading.Event()
        self.q: "queue.Queue[Optional[tuple[float, Any]]]" = queue.Queue(_QUEUE_MAX)
        self.frames_written = 0
        self.frames_dropped = 0
        self.first_frame_t: Optional[float] = None
        self.last_frame_t: Optional[float] = None
        self._held: set[str] = set()
        self._held_lock = threading.Lock()
        self.advance = threading.Event()
        self.frames_jsonl = JsonlWriter(self.dir / "frames.jsonl")
        self.keys_jsonl = JsonlWriter(self.dir / "keys.jsonl", flush_every=1)

    def _log_key(self, key: Any, event: str) -> None:
        # Stamp first, classify second. Adds only the OS event-tap dispatch
        # latency (sub-ms) between the physical press and `t`.
        t = time.monotonic()
        raw = key_name(key)
        mod = _SPECIAL_KEYS.get(raw)
        if mod in MODIFIER_NAMES:
            with self._held_lock:
                if event == "down":
                    self._held.add(mod)
                else:
                    self._held.discard(mod)
            held: set[str] = set()
        else:
            with self._held_lock:
                held = set(self._held)
            if event == "down" and normalize_key(key, held) in Prompter.ADVANCE_KEYS:
                self.advance.set()
        self.keys_jsonl.write({"t": t, "event": event, "key": normalize_key(key, held)})

    def start_keyboard(self) -> keyboard.Listener:
        listener = keyboard.Listener(
            on_press=lambda k: self._log_key(k, "down"),
            on_release=lambda k: self._log_key(k, "up"),
        )
        listener.daemon = True
        listener.start()
        return listener

    def writer_loop(self, writer: cv2.VideoWriter) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            t, frame = item
            writer.write(frame)
            i = self.frames_written
            self.frames_written += 1
            # Row emitted only after a successful write, so `i` matches the
            # frame index inside video.mp4 exactly (contract).
            self.frames_jsonl.write({"i": i, "t": t})

    def capture_loop(self, cap: cv2.VideoCapture) -> None:
        deadline = time.monotonic() + self.args.seconds if self.args.seconds else None
        while not self.stop.is_set():
            ok, frame = cap.read()
            # Stamped IMMEDIATELY after read(), before anything touches `frame`;
            # encoding/disk happen on the writer thread so they cannot skew `t`.
            t = time.monotonic()
            if not ok:
                print("[recorder] camera read failed; stopping", file=sys.stderr)
                break
            if self.first_frame_t is None:
                self.first_frame_t = t
            self.last_frame_t = t
            try:
                self.q.put_nowait((t, frame))
            except queue.Full:
                # Never block the sensor loop on disk: drop and count.
                self.frames_dropped += 1
            if deadline is not None and t >= deadline:
                break
        self.stop.set()


def load_phrases(path: Path) -> list[str]:
    """Non-empty, non-'#' lines of a phrases file, in order."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.startswith("#")]


class Prompter:
    """Shows phrases one at a time, logging their windows to phrases.jsonl on the
    shared process clock (CONTRACT Amendment 1). Own thread: never stalls capture."""

    # Advance is time-based (--phrase-seconds) because the desk condition has no
    # keyboard to press; space/enter still advance early if one is available.
    ADVANCE_KEYS = frozenset({"space", "enter"})

    def __init__(
        self,
        phrases: list[str],
        writer: JsonlWriter,
        stop: threading.Event,
        advance: threading.Event,
        phrase_seconds: float = 12.0,
        clock=time.monotonic,
        out=sys.stdout,
    ) -> None:
        self.phrases = phrases
        self.writer = writer
        self.stop = stop
        self.advance = advance
        self.phrase_seconds = phrase_seconds
        self.clock = clock
        self.out = out
        self.completed = 0

    def _show(self, text: str) -> None:
        bar = "=" * 72
        print(f"\n{bar}\n  {text}\n{bar}", file=self.out, flush=True)

    def run(self) -> None:
        for idx, phrase in enumerate(self.phrases):
            if self.stop.is_set():
                return
            self.advance.clear()
            self.writer.write(
                {"t": self.clock(), "event": "shown", "phrase": phrase, "idx": idx}
            )
            self._show(f"[{idx + 1}/{len(self.phrases)}]  {phrase}")
            end = self.clock() + self.phrase_seconds
            while True:
                remaining = end - self.clock()
                if remaining <= 0 or self.stop.is_set() or self.advance.is_set():
                    break
                print(f"  {remaining:5.1f}s ", end="\r", file=self.out, flush=True)
                if self.advance.wait(min(0.2, remaining)):
                    break
            self.writer.write(
                {"t": self.clock(), "event": "done", "phrase": phrase, "idx": idx}
            )
            self.completed += 1
        print("\n[recorder] phrase list finished", file=self.out, flush=True)
        self.stop.set()


def _permission_watchdog(rec: Recorder, grace: float = 10.0) -> None:
    """pynput's Listener fails SILENTLY without macOS Input Monitoring: frames
    arrive, key events never do. Warn loudly and keep recording."""
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if rec.stop.wait(0.25):
            return
    if rec.keys_jsonl.count == 0 and rec.frames_written > 0 and rec.args.condition != "pad":
        print(
            "\n" + "!" * 72 + "\n"
            "!!  NO KEY EVENTS after 10s while frames ARE arriving.\n"
            "!!  pynput's keyboard hook fails SILENTLY without permission.\n"
            "!!  Grant your TERMINAL app Input Monitoring:\n"
            "!!    System Settings > Privacy & Security > Input Monitoring\n"
            "!!    (also check Accessibility), then RESTART the terminal.\n"
            "!!  Recording continues, but keys.jsonl will be empty.\n" + "!" * 72 + "\n",
            file=sys.stderr,
            flush=True,
        )


def _touchpad_watchdog(rec: "Recorder", logger, grace: float = 20.0) -> None:
    """A pad session with no trackpad frames has no labels; say so while it can still be fixed."""
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if rec.stop.wait(0.25):
            return
    if logger.frames == 0 and rec.frames_written > 0:
        print(
            "\n" + "!" * 72 + "\n!!  NO TRACKPAD CONTACTS after 20s. Tap the trackpad glass to check;\n"
            "!!  touches.jsonl stays empty until a finger touches the pad.\n" + "!" * 72 + "\n",
            file=sys.stderr,
            flush=True,
        )


def record(args: argparse.Namespace) -> int:
    backend = getattr(args, "backend", "cv2")
    if backend == "av":
        # AVFoundation via pyobjc: reaches devices cv2 refuses (iPhone Desk View).
        from phase0.capture.avsource import AVVideoSource

        try:
            cap = AVVideoSource(args.camera, args.width, args.height, args.fps)
        except ValueError as exc:
            raise SystemExit(str(exc))
        cam_index, cam_name = -1, cap.name
    elif backend == "net":
        # WideCam iPhone MJPEG over WiFi; --camera is "auto" (Bonjour), an IP, or a URL.
        from phase0.capture.netsource import NetVideoSource

        try:
            cap = NetVideoSource(args.camera)
        except ValueError as exc:
            raise SystemExit(str(exc))
        cam_index, cam_name = -1, cap.name
    else:
        cam_index, cam_name = resolve_camera(args.camera)
        cap = open_camera(cam_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        cap.release()
        print(
            f"ERROR: could not open camera {cam_index} ({cam_name}).\n"
            "On macOS, grant your terminal Camera access:\n"
            "  System Settings > Privacy & Security > Camera, then restart it.",
            file=sys.stderr,
        )
        return 2
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print(
            f"ERROR: camera {cam_index} ({cam_name}) opened but the first read "
            "failed.\nThis is almost always a macOS Camera permission problem: "
            "System Settings > Privacy & Security > Camera -> enable your "
            "terminal, then restart it.",
            file=sys.stderr,
        )
        return 2

    # Preflight: a camera can return ok=True at a high rate while handing back
    # the SAME frozen frame every time (measured on this machine: the built-in
    # FaceTime camera with the lid closed, ~56 reads/s, 100% byte-identical).
    # Nothing downstream can detect that, so refuse to record it.
    if not args.skip_preflight:
        pf = preflight(cap, seconds=args.preflight_seconds)
        print(
            f"preflight: {pf['reads_per_sec']:.1f} reads/s, "
            f"{100 * pf['frac_identical']:.0f}% byte-identical consecutive frames"
        )
        if pf["frozen"]:
            cap.release()
            print(
                f"ERROR: camera {cam_index} ({cam_name}) is NOT delivering live "
                f"video -- {pf['identical_pairs']}/{pf['pairs']} consecutive frames "
                "were byte-identical.\n"
                "The read rate looks fine but the image never changes, so the "
                "recording would be worthless.\n"
                "Common cause: a closed MacBook lid, a covered lens, or a virtual "
                "camera with no source. Pick another camera with --list-cameras, "
                "or pass --skip-preflight to override.",
                file=sys.stderr,
            )
            return 2
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            print("ERROR: camera stopped delivering frames after preflight.", file=sys.stderr)
            return 2

    height, width = frame.shape[:2]
    fps_claimed = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    started = datetime.now()
    session_id = f"{started.strftime('%Y%m%d-%H%M%S')}-{args.condition}"
    session_dir = Path(args.out_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # VideoWriter fps only sets playback rate; real timing lives in frames.jsonl.
    writer_fps = fps_claimed if fps_claimed > 1 else args.fps
    writer = cv2.VideoWriter(
        str(session_dir / "video.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"),
        writer_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print("ERROR: could not open VideoWriter with 'avc1' (H.264).", file=sys.stderr)
        return 3

    touch_logger = None
    touches_jsonl: Optional[JsonlWriter] = None
    touch_info: dict[str, Any] = {}
    if args.condition == "pad" and not getattr(args, "touchpad", False):
        print("[recorder] --condition pad implies --touchpad", flush=True)
        args.touchpad = True
    if getattr(args, "touchpad", False):
        from phase0.capture.touchpad import TouchpadError, TouchpadLogger

        touches_jsonl = JsonlWriter(session_dir / "touches.jsonl", flush_every=10)
        touch_logger = TouchpadLogger(touches_jsonl)
        try:
            touch_info = touch_logger.start()
        except TouchpadError as exc:
            touches_jsonl.close()
            cap.release()
            writer.release()
            print(f"ERROR: --touchpad could not start the built-in trackpad: {exc}", file=sys.stderr)
            return 5
        print(
            f"[recorder] trackpad {touch_info['width_mm']}x{touch_info['height_mm']} mm "
            "-> touches.jsonl (disable Tap to click)",
            flush=True,
        )

    rec = Recorder(session_dir, args)
    t0 = time.monotonic()
    print(
        f"[recorder] session {session_id}\n"
        f"[recorder] camera {cam_index} '{cam_name}'  {width}x{height} "
        f"(requested {args.width}x{args.height}), driver fps={fps_claimed:.1f}\n"
        f"[recorder] writing to {session_dir}\n"
        f"[recorder] Ctrl-C to stop"
        + (f" (auto-stop in {args.seconds:.0f}s)" if args.seconds else ""),
        flush=True,
    )

    prompter: Optional[Prompter] = None
    phrases_jsonl: Optional[JsonlWriter] = None
    if args.phrases:
        phrases = load_phrases(Path(args.phrases))
        if not phrases:
            cap.release()
            writer.release()
            print(f"ERROR: no phrases in {args.phrases}", file=sys.stderr)
            return 4
        phrases_jsonl = JsonlWriter(session_dir / "phrases.jsonl", flush_every=1)
        prompter = Prompter(
            phrases, phrases_jsonl, rec.stop, rec.advance, args.phrase_seconds
        )

    listener = rec.start_keyboard()
    wt = threading.Thread(target=rec.writer_loop, args=(writer,), daemon=True)
    wt.start()
    threading.Thread(target=_permission_watchdog, args=(rec,), daemon=True).start()
    if touch_logger is not None:
        threading.Thread(target=_touchpad_watchdog, args=(rec, touch_logger), daemon=True).start()
    pt: Optional[threading.Thread] = None
    if prompter is not None:
        pt = threading.Thread(target=prompter.run, daemon=True)
        pt.start()

    try:
        rec.capture_loop(cap)
    except KeyboardInterrupt:
        print("\n[recorder] Ctrl-C -- shutting down cleanly", flush=True)
    finally:
        rec.stop.set()
        try:
            listener.stop()
        except Exception:
            pass
        # Let the prompter close its in-flight phrase window before we close the file.
        if pt is not None:
            pt.join(timeout=5)
        if touch_logger is not None:
            touch_info = touch_logger.stop()
        rec.q.put(None)
        wt.join(timeout=30)
        cap.release()
        writer.release()

        span = (
            (rec.last_frame_t - rec.first_frame_t)
            if rec.first_frame_t is not None and rec.last_frame_t is not None
            else 0.0
        )
        # Measured empirically over the run, never the driver's claim.
        fps_actual = (
            (rec.frames_written - 1) / span
            if span > 0 and rec.frames_written > 1
            else 0.0
        )

        meta = {
            "session_id": session_id,
            "condition": args.condition,
            "started_at_iso": started.isoformat(),
            "t0_monotonic": t0,
            "camera_name": cam_name,
            "width": width,
            "height": height,
            "fps_requested": float(args.fps),
            "fps_actual": round(fps_actual, 4),
            "notes": (
                f"driver_fps_claimed={fps_claimed:.3f}; frames_dropped="
                f"{rec.frames_dropped}; requested={args.width}x{args.height}; "
                f"keys_logged={rec.keys_jsonl.count}; phrases_shown="
                f"{prompter.completed if prompter else 0}; "
                + (f"touches_down={touch_info.get('downs', 0)}; " if touch_logger else "")
                + f"{args.notes}"
            ).strip(),
        }
        if touch_logger is not None:
            meta["touchpad"] = touch_info
        (session_dir / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        rec.frames_jsonl.close()
        rec.keys_jsonl.close()
        if phrases_jsonl is not None:
            phrases_jsonl.close()
        if touches_jsonl is not None:
            touches_jsonl.close()

        print(
            "\n[recorder] done\n"
            f"  frames written : {rec.frames_written} (dropped {rec.frames_dropped})\n"
            f"  keys logged    : {rec.keys_jsonl.count}\n"
            f"  phrases done   : {prompter.completed if prompter else 0}\n"
            + (f"  trackpad       : {touch_info.get('downs', 0)} contacts, "
               f"{touch_info.get('frames_dropped', 0)} frames dropped, "
               f"clock={touch_info.get('clock_mode')}\n" if touch_logger else "")
            +
            f"  measured fps   : {fps_actual:.2f} over {span:.1f}s\n"
            f"  resolution     : {width}x{height}\n"
            f"  output dir     : {session_dir}",
            flush=True,
        )
        if rec.keys_jsonl.count == 0 and args.condition != "pad":
            print(
                "  WARNING: zero key events -- check Input Monitoring permission.",
                flush=True,
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m phase0.capture.recorder",
        description="Record a Phase 0 session: camera video + keystrokes on one "
        "time.monotonic() clock.",
    )
    p.add_argument("--condition", required=True, choices=["kbd", "desk", "pad"])
    p.add_argument(
        "--touchpad",
        action="store_true",
        help="also log built-in trackpad contacts to touches.jsonl (implied by --condition pad)",
    )
    p.add_argument(
        "--camera",
        default="0",
        help="camera index (e.g. 0) or a case-insensitive substring of the "
        'device name (e.g. "iPhone")',
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="auto-stop after N seconds (0 = until Ctrl-C)",
    )
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument(
        "--backend",
        choices=["cv2", "av", "net"],
        default="cv2",
        help="av = AVFoundation via pyobjc (iPhone Desk View); "
        "net = WideCam iPhone stream over WiFi (--camera auto|IP|URL)",
    )
    p.add_argument(
        "--out-root",
        default="data/sessions",
        help="root directory for session folders (default: data/sessions)",
    )
    p.add_argument(
        "--phrases",
        default=None,
        help="path to a phrases file; enables the prompter and phrases.jsonl "
        "(required for desk sessions, which produce no key events)",
    )
    p.add_argument(
        "--phrase-seconds",
        type=float,
        default=12.0,
        help="seconds each phrase is shown before auto-advancing; space/enter "
        "advances early where a keyboard exists (default: 12)",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the frozen-camera check (not recommended)",
    )
    p.add_argument("--preflight-seconds", type=float, default=2.0)
    p.add_argument("--notes", default="", help="free-text note stored in meta.json")
    p.add_argument(
        "--list-cameras", action="store_true", help="print detected cameras and exit"
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cameras:
        for i, name in enumerate(list_mac_cameras()):
            print(f"{i}: {name}")
        return 0
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
