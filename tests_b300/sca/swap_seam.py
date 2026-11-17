#!/usr/bin/env python3
"""Superseded: the swap lives in `seam_inertness_probe.py swap --to base`.

Kept as a pointer rather than deleted so a reader who knows this name is not left
guessing. The old version replaced one of two *stored* bodies; the copy of the
seam could drift from the file it stood for, and a pair that swaps a stale copy
measures the wrong tree. The probe now takes the seam side from
`csrc/flash_sca/native/host.cc` itself, so only the pre-change body is stored.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "seam_inertness_probe.py")
raise SystemExit(subprocess.call([sys.executable, PROBE, "swap"] + sys.argv[1:]))
