#!/usr/bin/env python3
"""
Unified entry point — switch between the YOLO detector and the CSRNet
density heatmap engine from one command, for easy side-by-side testing.

Usage
-----
    python run.py yolo --source video.mp4 --mode both
    python run.py csrnet --source video.mp4 --weights weights/csrnet_shanghaiA.pth

Every flag after 'yolo' or 'csrnet' is passed straight through to that
engine's own script (detector.py / csrnet_detector.py) — run either one
with --help to see its full option list, e.g.:

    python run.py yolo --help
    python run.py csrnet --help
"""

import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ENGINES = {
    "yolo": HERE / "detector.py",
    "csrnet": HERE / "csrnet_detector.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENGINES:
        print("Usage: python run.py <yolo|csrnet> [options]\n")
        print("Examples:")
        print("  python run.py yolo --source video.mp4 --mode both")
        print("  python run.py csrnet --source video.mp4 --weights weights/csrnet_shanghaiA.pth\n")
        print("Run 'python run.py yolo --help' or 'python run.py csrnet --help' "
              "for full options.")
        sys.exit(1)

    engine = sys.argv[1]
    passthrough_args = sys.argv[2:]
    script = ENGINES[engine]

    result = subprocess.run([sys.executable, str(script), *passthrough_args])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
