"""Run a command while holding an exclusive per-machine GPU lock, so only one GPU job runs at a time.
Usage: python -m phase0.tools.gpulock [--wait SECONDS] -- <command...>"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(os.environ.get("CVT_GPU_LOCK", str(Path.home() / ".cvt_gpu.lock")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=float("inf"), help="give up after this many seconds")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    f = open(LOCK, "a+")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > a.wait:
                print(f"gpulock: timed out waiting for {LOCK}", file=sys.stderr)
                return 75
            time.sleep(5)
    f.seek(0), f.truncate(), f.write(f"{os.getpid()} {time.strftime('%F %T')} {' '.join(cmd)[:200]}\n"), f.flush()
    try:
        return subprocess.call(cmd)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
