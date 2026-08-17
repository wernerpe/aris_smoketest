#!/usr/bin/env python3
"""Build the static meshcat scene from atlases in out/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm.viz.scene import build  # noqa: E402
from aris_sixarm.fleet import H_INV_DEFAULT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).parents[1] / "out"))
ap.add_argument("--html", default=str(Path(__file__).parents[1] / "out/reach_atlas.html"))
ap.add_argument("--h-inv", type=float, default=H_INV_DEFAULT)
a = ap.parse_args()
build(a.out, a.html, a.h_inv)
