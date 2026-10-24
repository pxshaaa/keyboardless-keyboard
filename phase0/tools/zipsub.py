"""Fetch a SUBSET of a remote zip over HTTP range requests, without downloading the whole archive.

Written for PressureVisionDB (MIT licence), whose smallest published bundle is a 19.4 GB zip while this
machine has a ~12 GB budget: read the central directory from the tail, list members, pick a few, fetch only
their byte ranges.  Usage:
  python -m phase0.tools.zipsub list <url> [--out list.json] [--grep PATTERN]
  python -m phase0.tools.zipsub get  <url> --dest DIR --grep PATTERN [--max-mb 2000] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

UA = {"User-Agent": "curl/8.4"}


def _get(url: str, start: int, end: int) -> bytes:
    r = Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    with urlopen(r, timeout=120) as f:
        return f.read()


def size_of(url: str) -> int:
    r = Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urlopen(r, timeout=60) as f:
        cr = f.headers.get("Content-Range")
    return int(cr.split("/")[-1])


def central_directory(url: str, total: int, tail: int = 1 << 20) -> list[dict]:
    """-> [{name, offset, csize, usize, method}] read from the zip's central directory (zip64 aware)."""
    tail = min(tail, total)
    buf = _get(url, total - tail, total - 1)
    i = buf.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("no EOCD found in tail")
    cd_size, cd_off = struct.unpack("<II", buf[i + 12:i + 20])
    n = struct.unpack("<H", buf[i + 10:i + 12])[0]
    j = buf.rfind(b"PK\x06\x06")
    if j >= 0:                                   # zip64 EOCD
        n = struct.unpack("<Q", buf[j + 32:j + 40])[0]
        cd_size = struct.unpack("<Q", buf[j + 40:j + 48])[0]
        cd_off = struct.unpack("<Q", buf[j + 48:j + 56])[0]
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    for _ in range(n):
        if cd[p:p + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, off):     # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack(f"<{hsz // 8}Q", extra[q + 4:q + 4 + (hsz // 8) * 8]))
                    k = 0
                    if usize == 0xFFFFFFFF: usize = vals[k]; k += 1
                    if csize == 0xFFFFFFFF: csize = vals[k]; k += 1
                    if off == 0xFFFFFFFF: off = vals[k]; k += 1
                q += 4 + hsz
        out.append({"name": name, "offset": off, "csize": csize, "usize": usize, "method": method})
        p += 46 + nlen + elen + clen
    return out


def fetch_member(url: str, m: dict) -> bytes:
    head = _get(url, m["offset"], m["offset"] + 29)
    if head[:4] != b"PK\x03\x04":
        raise SystemExit(f"bad local header for {m['name']}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = m["offset"] + 30 + nlen + elen
    raw = _get(url, start, start + m["csize"] - 1)
    if m["method"] == 0:
        return raw
    return zlib.decompress(raw, -15)


def bulk_get(url: str, sel: list[dict], dest: Path, budget: float, gap: float) -> int:
    """Fetch contiguous runs of members in ONE range request each (members of a directory sit together
    in the archive), then slice the local headers out of the buffer. ~100x fewer HTTP round trips."""
    sel = sorted(sel, key=lambda m: m["offset"])
    runs, cur = [], []
    for m in sel:
        if cur and m["offset"] - (cur[-1]["offset"] + cur[-1]["csize"] + 30 + len(cur[-1]["name"])) > gap:
            runs.append(cur); cur = []
        cur.append(m)
    if cur:
        runs.append(cur)
    got = 0
    for r in runs:
        lo = r[0]["offset"]
        hi = r[-1]["offset"] + 30 + len(r[-1]["name"]) + 64 + r[-1]["csize"]
        if got + (hi - lo) > budget:
            print(f"budget reached at {got/1e6:.0f} MB", flush=True)
            break
        buf = _get(url, lo, hi)
        got += hi - lo
        for m in r:
            f = dest / m["name"]
            if f.is_dir() or m["name"].endswith("/"):
                continue
            p0 = m["offset"] - lo
            if buf[p0:p0 + 4] != b"PK\x03\x04":
                continue
            nlen, elen = struct.unpack("<HH", buf[p0 + 26:p0 + 30])
            st = p0 + 30 + nlen + elen
            raw = buf[st:st + m["csize"]]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(raw if m["method"] == 0 else zlib.decompress(raw, -15))
        print(f"  run {len(r):4d} files {(hi-lo)/1e6:7.1f} MB  {r[0]['name'][:60]}", flush=True)
    print(f"fetched {got/1e6:.0f} MB into {dest}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "get"))
    ap.add_argument("url")
    ap.add_argument("--grep", default="")
    ap.add_argument("--dest", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-mb", type=float, default=2000.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bulk", action="store_true")
    ap.add_argument("--gap-mb", type=float, default=4.0)
    a = ap.parse_args(argv)
    total = size_of(a.url)
    print(f"archive {total/1e9:.1f} GB", flush=True)
    cd = central_directory(a.url, total)
    print(f"{len(cd)} members", flush=True)
    sel = [m for m in cd if not a.grep or re.search(a.grep, m["name"])]
    if a.cmd == "list":
        pre = {}
        for m in cd:
            k = "/".join(m["name"].split("/")[:3])
            e = pre.setdefault(k, {"n": 0, "bytes": 0})
            e["n"] += 1; e["bytes"] += m["csize"]
        for k, v in sorted(pre.items())[:60]:
            print(f"  {k:<60} {v['n']:6d} files {v['bytes']/1e9:7.2f} GB", flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps({"total": total, "members": len(cd),
                                               "prefixes": pre, "sample": [m["name"] for m in cd[:40]],
                                               "matched": [m["name"] for m in sel[:200]]}, indent=1))
        return 0
    if a.limit:
        sel = sel[: a.limit]
    if a.bulk:
        return bulk_get(a.url, sel, Path(a.dest), a.max_mb * 1e6, a.gap_mb * 1e6)
    budget = a.max_mb * 1e6
    got = 0
    dest = Path(a.dest)
    for m in sel:
        if got + m["csize"] > budget:
            print(f"budget reached at {got/1e6:.0f} MB", flush=True)
            break
        f = dest / m["name"]
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(fetch_member(a.url, m))
        got += m["csize"]
        if len(list(dest.rglob("*"))) % 50 == 0:
            print(f"  {got/1e6:.0f} MB ...", flush=True)
    print(f"fetched {got/1e6:.0f} MB into {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
