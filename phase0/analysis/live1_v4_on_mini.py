"""Run the frozen decipher2_v4 pipeline entirely on the Mac mini: its `ssh macmini` / `rsync macmini:` hops become local calls."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from phase0.analysis import decipher2 as D2
from phase0.analysis import decipher2_v4 as D4

HOME = str(Path.home())


def sh(cmd: list[str]) -> None:
    if cmd[0] == "ssh" and cmd[1] == D2.MINI:
        cmd = ["bash", "-lc", cmd[2]]
    elif cmd[0] == "rsync":
        cmd = [c.replace(f"{D2.MINI}:", HOME + "/") for c in cmd]
        cmd = [c.replace(f"{D2.MINI}:", HOME + "/") for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


D2.sh = sh
D4.D2 = D2
if __name__ == "__main__":
    raise SystemExit(D4.main(Path(sys.argv[1])))
