#!/usr/bin/env python3
"""Superseded: the order lint lives in `seam_inertness_probe.py order`.

Kept as a pointer rather than deleted so a reader who knows this name is not left
guessing. Two lints over the same span would drift apart, and the one with the
stale anchor text fails against the current seam; this one therefore forwards.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "seam_inertness_probe.py")
raise SystemExit(subprocess.call([sys.executable, PROBE, "order"] + sys.argv[1:]))
