#!/usr/bin/env python3
"""Sequential runner for all modal decomposition scripts.

Running ``python pyModal.py`` is equivalent to executing each analysis script
individually::

    python pod.py
    python dmd.py
    python spod.py
    python bmsd.py

Optional ``--prep``, ``--compute`` and ``--plot`` flags are forwarded to each
script to allow staged execution.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

SCRIPT_ORDER = [
    ("pod", "pod.py"),
    ("dmd", "dmd.py"),
    ("spod", "spod.py"),
    ("bsmd", "bmsd.py"),
]


def run_script(script: str, flags: list[str]) -> None:
    """Execute a script with ``subprocess`` and propagate errors."""

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), script), *flags]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run modal analyses in sequence")
    parser.add_argument("--pod", action="store_true", help="Run POD only")
    parser.add_argument("--dmd", action="store_true", help="Run DMD only")
    parser.add_argument("--spod", action="store_true", help="Run SPOD only")
    parser.add_argument("--bsmd", action="store_true", help="Run BSMD only")
    parser.add_argument("--prep", action="store_true", help="Preprocess data")
    parser.add_argument("--compute", action="store_true", help="Compute analysis")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    selected = [name for name, _ in SCRIPT_ORDER if getattr(args, name)]
    if not selected:
        selected = [name for name, _ in SCRIPT_ORDER]

    flags = []
    if args.prep:
        flags.append("--prep")
    if args.compute:
        flags.append("--compute")
    if args.plot:
        flags.append("--plot")

    for name, script in SCRIPT_ORDER:
        if name in selected:
            run_script(script, flags)
