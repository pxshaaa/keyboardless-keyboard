"""Demo clip of blind-2 desk session (skeleton + greedy CTC emissions + decoded text) -> demo/blind2_demo.mp4.
Ring fingertip is a display heuristic (QWERTY side -> hand, most-moved tip); the model outputs no finger."""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

SRC = Path("data/sessions/20260913-125733-desk")
CTC = Path("data/sessions/zz-ctc3-20260913-125733-desk")
LP = Path(".cache/decipher/out/zz-ctc3-20260913-125733-desk_lp.npz")
OUT = Path("demo")
SYMS = "_abcdefghijklmnopqrstuvwxyz #"  # seqctc.SYMS: blank, letters, space, other
BLANK, OTHER = 0, 28
RAW_W, RAW_H = 1280, 720  # raw phone frame; pipeline view is ccw90 -> 720x1280
SCALE = 1.5               # -> 1080x1920 output
OW, OH = int(RAW_H * SCALE), int(RAW_W * SCALE)
FPS = 60

CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIPS = (4, 8, 12, 16, 20)
LEFT_KEYS = set("qwertasdfgzxcvb")

ACCENT = (80, 220, 255)     # BGR warm yellow
BONE = (255, 245, 235)      # near-white
RING = (120, 255, 170)      # mint green (BGR)


def font(size: int, weight: str | None = None, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/SFNS.ttf"
    try:
        f = ImageFont.truetype(path, size)
        if weight:
            try:
                f.set_variation_by_name(weight)
            except Exception:  # noqa: BLE001
                pass
        return f
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)


def load_frames_t() -> np.ndarray:
    ft = np.array([json.loads(line)["t"] for line in open(SRC / "frames.jsonl")])
    return ft - ft[0]


def load_landmarks(n: int) -> np.ndarray:
    """(n, 2, 21, 2) output-pixel coords in the rotated (ccw90) + scaled view, NaN where missing."""
    tb = pq.read_table(SRC / "landmarks.parquet", columns=["i", "hand", "joint", "x", "y"])
    i, h, j = (tb.column(c).to_numpy().astype(int) for c in ("i", "hand", "joint"))
    x, y = tb.column("x").to_numpy(), tb.column("y").to_numpy()
    P = np.full((n, 2, 21, 2), np.nan, np.float32)
    ok = i < n
    # cv2.ROTATE_90_COUNTERCLOCKWISE: (x, y) in WxH -> (y, W-1-x) in HxW
    P[i[ok], h[ok], j[ok], 0] = y[ok] * SCALE
    P[i[ok], h[ok], j[ok], 1] = (RAW_W - 1 - x[ok]) * SCALE
    # light centred 3-frame smoothing (offline, no lag); keep NaN where the frame itself has no detection
    stack = np.stack([np.roll(P, 1, 0), P, np.roll(P, -1, 0)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(stack, 0)
    return np.where(np.isfinite(P), m, np.nan)


def emissions(seg_idx: int) -> tuple[list[tuple[int, str]], dict]:
    """Greedy CTC emission points of the ensemble posteriors: (video frame, char)."""
    d = json.loads((CTC / "decipher.json").read_text())
    seg = next(s for s in d["segments"] if s["i"] == seg_idx)
    s0, s1 = seg["frames30"]
    z = np.load(LP)
    lp = np.mean([z["zs"][s0:s1].astype(np.float64), z["desk"][s0:s1].astype(np.float64)], 0)
    a = lp.argmax(-1)
    out, prev = [], BLANK
    for k, c in enumerate(a):
        if c != prev and c not in (BLANK, OTHER):
            out.append((2 * (s0 + k), SYMS[c]))  # lp is 30 Hz = every 2nd video frame (times = ft[::2])
        prev = c
    return out, seg


def pick_tip(P: np.ndarray, f: int, ch: str, win: int = 8) -> tuple[float, float] | None:
    cur = P[f]
    hands = [hh for hh in (0, 1) if np.isfinite(cur[hh, 0]).all()]
    if not hands:
        return None
    if len(hands) == 2:  # typist's left hand = smaller wrist x in the rotated view
        left = min(hands, key=lambda hh: cur[hh, 0, 0])
        right = 1 - left
        if ch != " ":
            hands = [left] if ch in LEFT_KEYS else [right]
    tips = (4,) if ch == " " else (8, 12, 16, 20)
    best, bxy = -1.0, None
    seg = P[max(0, f - win):f + 1]
    for hh in hands:
        for t in tips:
            tr = seg[:, hh, t]
            tr = tr[np.isfinite(tr).all(1)]
            if len(tr) == 0:
                continue
            path = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0
            if path > best:
                best, bxy = path, (float(cur[hh, t, 0]), float(cur[hh, t, 1]))
    return bxy


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for w in text.split(" "):
        cand = (line + " " + w).strip()
        if draw.textlength(cand, font=f) <= width or not line:
            line = cand
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def stext(dr: ImageDraw.ImageDraw, xy, s: str, f, fill, sh: int = 3) -> None:
    a = fill[3] if len(fill) == 4 else 255
    for dx, dy in ((sh, sh), (0, sh), (sh, 0)):
        dr.text((xy[0] + dx, xy[1] + dy), s, font=f, fill=(0, 0, 0, int(a * 0.55)))
    dr.text(xy, s, font=f, fill=fill)


def follow(P: np.ndarray, fi: int, xy: tuple[float, float], max_d: float = 70.0) -> tuple[float, float]:
    tips = P[fi][:, list(TIPS)].reshape(-1, 2)
    tips = tips[np.isfinite(tips).all(1)]
    if len(tips) == 0:
        return xy
    d = np.linalg.norm(tips - np.array(xy), axis=1)
    k = int(d.argmin())
    return (float(tips[k, 0]), float(tips[k, 1])) if d[k] < max_d else xy


def draw_skeleton(img: np.ndarray, pts: np.ndarray) -> None:
    ov = img.copy()
    for hh in (0, 1):
        p = pts[hh]
        if not np.isfinite(p).all():
            continue
        q = p.astype(int)
        for a, b in CONN:
            cv2.line(ov, tuple(q[a]), tuple(q[b]), BONE, 3, cv2.LINE_AA)
        for j in range(21):
            if j not in TIPS:
                cv2.circle(ov, tuple(q[j]), 5, BONE, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for hh in (0, 1):
        p = pts[hh]
        if not np.isfinite(p).all():
            continue
        for j in TIPS:
            c = tuple(p[j].astype(int))
            cv2.circle(img, c, 9, ACCENT, -1, cv2.LINE_AA)
            cv2.circle(img, c, 9, (40, 40, 40), 1, cv2.LINE_AA)



def wrong_words(hyp: str, truth: str) -> set[str]:
    """Decoded words not matched to the typed text (order-preserving alignment)."""
    import difflib
    h, t = hyp.split(), truth.lower().split()
    ok = set()
    for blk in difflib.SequenceMatcher(a=h, b=t, autojunk=False).get_matching_blocks():
        ok.update(range(blk.a, blk.a + blk.size))
    return {w for i, w in enumerate(h) if i not in ok and w not in {h[j] for j in ok}}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=4)
    ap.add_argument("--lead", type=float, default=1.4)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--stills", default="66.9,70.2,73.2", help="session seconds to export as PNG")
    ap.add_argument("--out", default="blind2_demo.mp4")
    ap.add_argument("--hyp", default=None, help="override the displayed decoded text (NOT the model output)")
    ap.add_argument("--truth", default=None, help="typed text; after decoding, show it and mark differing words")
    ap.add_argument("--session", default=None, help="source session id (default: blind-2)")
    ap.add_argument("--ctc-sid", default=None, help="twin decode session id, e.g. zz-ctc3-<id> or zz-ctcm...-<id>")
    ap.add_argument("--hyp-json", default="decipher2_v3_recommended.json",
                    help="decode result in the twin dir: decipher2_v3_recommended.json (records) or decipher2_v4.json (segments)")
    ap.add_argument("--crop-top", type=int, default=0, help="output px to cut from the top (e.g. clutter above the hands)")
    a = ap.parse_args()
    global SRC, CTC, LP, OH
    if a.session:
        SRC = Path("data/sessions") / a.session
        CTC = Path("data/sessions") / (a.ctc_sid or f"zz-ctc3-{a.session}")
        LP = Path(".cache/decipher/out") / f"{CTC.name}_lp.npz"

    ft = load_frames_t()
    n = len(ft)
    P = load_landmarks(n)
    crop = a.crop_top - a.crop_top % 2
    P[..., 1] -= crop
    OH -= crop
    em, seg = emissions(a.seg)
    hj = json.loads((CTC / a.hyp_json).read_text())
    if "records" in hj:
        hyp = next(r["hyp"] for r in hj["records"] if r["id"] == f"seg{a.seg}")
    else:
        hyp = next(r["text"] for r in hj["segments"] if r["id"] == f"seg{a.seg}")
    if a.hyp:
        print(f"WARNING: displaying edited text {a.hyp!r} instead of model output {hyp!r}")
        hyp = a.hyp
    t_start, t_end, t_dec = seg["t0"] - a.lead, seg["t1"] + a.hold, seg["t1"]
    f0, f1 = int(np.searchsorted(ft, t_start)), int(np.searchsorted(ft, t_end))
    stills = {int(np.searchsorted(ft, float(s))): k for k, s in enumerate(a.stills.split(","), 1)}

    pops = []  # (frame, char, (x, y))
    for f, ch in em:
        xy = pick_tip(P, f, ch)
        if xy is not None:
            pops.append((f, ch, xy))
    print(f"seg{a.seg}: {len(em)} emissions, raw='{''.join(c for _, c in em)}', decoded='{hyp}'")
    print(f"clip {t_start:.2f}-{t_end:.2f}s = frames {f0}-{f1} ({(f1 - f0) / FPS:.2f}s)")

    F_CAP, F_LAB = font(34, "Semibold"), font(24, "Bold")
    F_POP, F_RAW, F_DEC = font(52, "Bold"), font(36, mono=True), font(64, "Bold")
    OUT.mkdir(exist_ok=True)
    out_mp4 = OUT / a.out
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-profile:v", "high", "-movflags", "+faststart", str(out_mp4)], stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(str(SRC / "video.mp4"))
    fi = 0
    while fi < f0:  # sequential read: exact frame indices (seeking in H.264 can be off)
        cap.grab()
        fi += 1
    POP_S = 0.4
    track: dict[int, tuple[float, float]] = {}
    for fi in range(f0, f1):
        ok, raw = cap.read()
        if not ok:
            break
        t = ft[fi]
        img = cv2.resize(cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE), (OW, OH + crop), interpolation=cv2.INTER_CUBIC)[crop:]
        draw_skeleton(img, P[fi])

        live = []
        for pf, ch, xy in pops:  # ring rides along with its fingertip while it fades
            if pf <= fi and t - ft[pf] < POP_S:
                track[pf] = follow(P, fi, track.get(pf, xy))
                live.append((pf, ch, track[pf], (t - ft[pf]) / POP_S))
        for _, _, (x, y), u in live:  # expanding, fading ring at the fingertip
            ov = img.copy()
            cv2.circle(ov, (int(x), int(y)), int(16 + 34 * u), RING, 4, cv2.LINE_AA)
            cv2.addWeighted(ov, 1 - u, img, u, 0, img)

        # bottom gradient bar
        bar_h = 520
        g = np.linspace(0, 0.88, bar_h)[:, None, None] ** 0.8
        img[OH - bar_h:] = (img[OH - bar_h:] * (1 - g)).astype(np.uint8)
        top_h = 260
        g2 = np.linspace(0.72, 0, top_h)[:, None, None] ** 0.9
        img[:top_h] = (img[:top_h] * (1 - g2)).astype(np.uint8)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        txt = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(txt)
        # letter pops above the ringed fingertip
        for _, ch, (x, y), u in live:
            s = "space" if ch == " " else ch
            fnt = F_POP if ch != " " else F_LAB
            alpha = int(255 * (1 - u) ** 0.7)
            w = dr.textlength(s, font=fnt)
            yy = y - 70 - 40 * u
            stext(dr, (x - w / 2, yy), s, fnt, (190, 255, 215, alpha))
        # top-left caption + legend
        M = 48
        stext(dr, (M, 44), "bare desk · 1 phone camera · no keyboard", F_CAP, (255, 255, 255, 250), 2)
        cy = 104
        dr.ellipse((M + 2, cy + 2, M + 24, cy + 24), outline=(170, 255, 120, 230), width=3)
        stext(dr, (M + 36, cy - 1), "raw keystroke signal", F_LAB, (225, 225, 225, 240), 2)

        # bottom text: raw greedy stream + decoded sentence
        raw_txt = "".join(ch for pf, ch in em if pf <= fi)
        y_raw = OH - 420
        decoded_now = fi >= int(np.searchsorted(ft, t_dec))
        if a.truth and decoded_now:
            # the typed text replaces the raw stream once the decoding is shown
            stext(dr, (M, y_raw), "YOU TYPED", F_LAB, (185, 185, 185, 240), 2)
            for k, line in enumerate(wrap(dr, a.truth, F_RAW, OW - 2 * M)[:2]):
                stext(dr, (M, y_raw + 34 + k * 46), line, F_RAW, (220, 220, 220, 240), 2)
        else:
            stext(dr, (M, y_raw), "RAW", F_LAB, (185, 185, 185, 240), 2)
            rl = wrap(dr, raw_txt, F_RAW, OW - 2 * M)
            for k, line in enumerate(rl[-2:]):
                stext(dr, (M, y_raw + 34 + k * 46), line, F_RAW, (200, 200, 200, 240), 2)
        if fi >= int(np.searchsorted(ft, t_dec)):
            u = min(1.0, (t - t_dec) / 0.45)
            al = int(255 * u)
            y_dec = OH - 250 + int(12 * (1 - u))
            stext(dr, (M, y_dec), "DECODED", F_LAB, (120, 230, 160, al), 2)
            bad = wrong_words(hyp, a.truth) if a.truth else set()
            for k, line in enumerate(wrap(dr, hyp, F_DEC, OW - 2 * M)[:2]):
                x = M
                for w in line.split(" "):
                    col = (255, 190, 70, al) if (k, w) in bad or w in bad else (255, 255, 255, al)
                    stext(dr, (x, y_dec + 34 + k * 78), w, F_DEC, col, 3)
                    x += int(dr.textlength(w + " ", font=F_DEC))

        pil = Image.alpha_composite(pil, txt).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        # short fade in / out
        k_in, k_out = (fi - f0) / (0.25 * FPS), (f1 - 1 - fi) / (0.35 * FPS)
        fade = min(1.0, k_in, k_out)
        if fade < 1.0:
            img = (img * max(fade, 0.0)).astype(np.uint8)
        if fi in stills:
            cv2.imwrite(str(OUT / f"frame_{stills[fi]}_{t:.1f}s.png"), img)
        ff.stdin.write(img.tobytes())
    cap.release()
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out_mp4}")
    return ff.returncode


if __name__ == "__main__":
    raise SystemExit(main())
