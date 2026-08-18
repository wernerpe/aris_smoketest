"""Polyline letterforms for the word ARIS, and their placement on the sheet.

Pure geometry — no IK, no drake — so this imports under both the system
python3.12 and the pydrake venv python3.10.

Each glyph is a list of STROKES; a stroke is an (N, 2) polyline in a unit box
centred on the origin, x, y in [-0.5, 0.5].  `place()` scales it to a physical
height (y) and height*aspect width (x) and translates it to a paper-frame
centre.  Single-width forms only — the pen has no width control.

Glyph inventory (kept deliberately cheap: 6 strokes for the whole word, so
the lattice work is ~3 m of stroke in total):
  A  2 strokes: the two legs as one inverted V, then the crossbar
  R  2 strokes: the stem, then bowl + diagonal leg in one continuous move
  I  1 stroke : the vertical
  S  1 stroke : two tangent-blended half-ellipses (one smooth curve)
"""
import numpy as np

DEFAULT_HEIGHT = 0.38     # m, cap height on the paper
DEFAULT_ASPECT = 0.72     # width / height of the unit box

# where each letter goes and which arm owns it (arms 2 and 71 are inactive)
PLACEMENT = [
    ("A", 13, (0.45, 1.00)),
    ("R", 31, (1.30, 1.05)),
    ("I", 97, (2.15, 0.95)),
    ("S", 17, (3.15, 1.00)),
]


def _arc(cx, cy, rx, ry, a0, a1, n=48):
    """Elliptical arc, angles in radians, swept a0 -> a1 (sign = direction)."""
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])


def _A():
    # legs: bottom-left -> apex -> bottom-right (one polyline, sharp apex)
    legs = np.array([[-0.5, -0.5], [0.0, 0.5], [0.5, -0.5]])
    # crossbar at y = -0.10; the legs are at x = +-0.30 there
    bar = np.array([[-0.30, -0.10], [0.30, -0.10]])
    return [legs, bar]


def _R():
    stem = np.array([[-0.40, 0.5], [-0.40, -0.5]])
    # bowl: top horizontal -> half ellipse down the right -> bottom horizontal,
    # then straight on into the diagonal leg (one continuous stroke, one cusp
    # where the bowl closes back onto the stem).
    bowl = np.vstack([
        np.array([[-0.40, 0.5], [0.05, 0.5]]),
        _arc(0.05, 0.275, 0.30, 0.225, np.pi / 2, -np.pi / 2, 40),
        np.array([[-0.40, 0.05], [0.45, -0.5]]),
    ])
    return [stem, bowl]


def _I():
    return [np.array([[0.0, 0.5], [0.0, -0.5]])]


def _S():
    # upper half-ellipse: start upper-right, sweep CCW over the top and down
    # the left side to the waist; lower half continues CW out to the right,
    # around the bottom and back left.  Tangents match at the waist (0, 0).
    up = _arc(0.0, 0.25, 0.36, 0.25, np.radians(25), np.radians(270), 56)
    lo = _arc(0.0, -0.25, 0.36, 0.25, np.radians(90), np.radians(-155), 56)
    return [np.vstack([up, lo[1:]])]


UNIT = {"A": _A(), "R": _R(), "I": _I(), "S": _S()}


def place(name, center, height=DEFAULT_HEIGHT, aspect=DEFAULT_ASPECT):
    """Unit glyph -> list of (N,2) polylines in paper/world xy."""
    cx, cy = center
    w, h = height * aspect, height
    return [np.column_stack([cx + w * s[:, 0], cy + h * s[:, 1]])
            for s in UNIT[name]]


def word_strokes(placement=None, height=DEFAULT_HEIGHT, aspect=DEFAULT_ASPECT):
    """-> list of (letter, arm_id, [stroke polylines]) in writing order."""
    out = []
    for name, arm_id, ctr in (placement or PLACEMENT):
        out.append((name, arm_id, place(name, ctr, height, aspect)))
    return out


def bbox(strokes):
    p = np.vstack(strokes)
    return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()


def stroke_length(pts):
    return float(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1).sum())
