#!/usr/bin/env python3
"""Read a SolidWorks 2024/25 .SLDPRT / .SLDASM without SolidWorks.

    python3 scripts/read_solidworks.py list  <file>            # the streams
    python3 scripts/read_solidworks.py dump  <file> [outdir]   # all of them
    python3 scripts/read_solidworks.py bbox  <file>            # the part's box
    python3 scripts/read_solidworks.py asm   <file.SLDASM>     # placed parts

WHY THIS IS IN THE REPO.  The pen holder's whole tool transform turns on one
file — `Natural hold assembly - closed.SLDASM`, nested two zips deep inside
`raw_slack_file_dump/Pen holder cad(1).zip` — and the raw CAD is deliberately
NOT committed.  Without this, every number in docs/SYSTEM_MODEL.md 7a is a
claim nobody can check.  With it, `asm` reprints them in about a second.

THE FORMAT, because it is not documented anywhere and cost an afternoon.  A
2024/25-era SolidWorks file is a flat sequence of records, each:

    14 00 06 00 08 00   6 bytes, constant, and the only way in
    <4>                 a per-file tag
    <u32> crc  <u32> compressed  <u32> uncompressed  <u32> name length
    <name>              the name, with EVERY BYTE'S NIBBLES SWAPPED
    <payload>           RAW DEFLATE — no zlib header, so a magic-byte scan
                        for 78 9c finds nothing and the file looks encrypted

Swap the name's nibbles and an OPC package appears: `[Content_Types].xml`,
`docProps/*.xml`, `Contents/*`, `PreviewPNG`.  The two that matter:

  * `swXmlContents/COMPINSTANCETREE` — plain XML.  Every component with a
    `swTransform` of 16 doubles laid out ROW-major with the translation in
    the LAST ROW (so p_asm = p_local @ A + t, i.e. the column-vector rotation
    is A-transpose).  Getting that convention wrong gives boxes 1.3 mm wide.
  * `Contents/DisplayLists` — binary, but it opens with ten float64 in
    METRES: centre(3), bbox max(3), bbox min(3), bounding-sphere radius.
    Self-checking, and `bbox` verifies it rather than trusting it.

Both were cross-checked against the STLs of the parts the 2026.08.19 delivery
ships: the 22-deg housing comes back 80.1 x 33.781 x 50.0 mm either way.
"""
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# every byte's nibbles swapped, as a 256-byte translation table
NIBBLE = bytes(((b & 0x0F) << 4) | (b >> 4) for b in range(256))
MARK = b"\x14\x00\x06\x00\x08\x00"
HDR = struct.Struct("<IIII")          # crc, compressed, uncompressed, namelen


def streams(path):
    """Every record of the file -> [(name, compressed, uncompressed, bytes)].

    Walks the marker rather than a directory, because there is no directory:
    a record that fails to inflate, or inflates to the wrong length, is not a
    record and the scan steps one byte on instead of trusting the header.
    """
    d = Path(path).read_bytes()
    out, seen, i = [], set(), 0
    while True:
        i = d.find(MARK, i)
        if i < 0:
            return out
        p = i + len(MARK) + 4                      # skip the per-file tag
        if p + HDR.size > len(d):
            return out
        _, csz, usz, nl = HDR.unpack_from(d, p)
        p += HDR.size
        if not (0 < nl < 260 and 0 < csz and p + nl + csz <= len(d)):
            i += 1
            continue
        raw = d[p:p + nl].translate(NIBBLE)
        if not all(32 <= c < 127 for c in raw):
            i += 1
            continue
        try:
            dec = zlib.decompressobj(-15).decompress(d[p + nl:p + nl + csz])
        except zlib.error:
            i += 1
            continue
        if usz and len(dec) != usz:                # header disagrees: not one
            i += 1
            continue
        name = raw.decode("ascii")
        if (name, csz, usz) not in seen:           # the tail repeats the table
            seen.add((name, csz, usz))
            out.append((name, csz, usz, dec))
        i = p + nl + csz


def _one(path, want):
    for name, _, _, dec in streams(path):
        if name == want:
            return dec
    raise SystemExit(f"{Path(path).name}: no {want!r} stream")


def bbox(path):
    """The part's own bounding box -> (min(3), max(3)) in METRES.

    `Contents/DisplayLists` opens with centre, max, min, sphere radius.  Both
    identities are checked here: a layout that has drifted would otherwise be
    read as geometry and quietly mis-scale everything downstream.
    """
    a = np.frombuffer(_one(path, "Contents/DisplayLists")[8:88],
                      dtype=np.float64)
    c, hi, lo, r = a[0:3], a[3:6], a[6:9], float(a[9])
    if not (np.allclose(c, (hi + lo) / 2, atol=1e-9)
            and abs(r - np.linalg.norm(hi - c)) < 1e-9):
        raise SystemExit(f"{Path(path).name}: DisplayLists header is not "
                         "centre/max/min/radius — the format has moved")
    return lo, hi


def components(path):
    """Placed TOP-LEVEL components -> [(name, A (3,3), t (3,))], metres.

    `A` and `t` are as SolidWorks writes them: p_asm = p_local @ A + t.

    Sub-assemblies appear in the same stream with transforms in their OWN
    frames, so the top-level model is identified rather than guessed at: the
    `swFile` whose path ends in this file's own name gives an id, the
    `swModel` carrying that `swFileRef` is the assembly, and only the
    references inside that element are placed in assembly coordinates.
    """
    xml = _one(path, "swXmlContents/COMPINSTANCETREE").decode("utf-8",
                                                              "replace")
    stem = Path(path).name
    fid = None
    for m in re.finditer(r'<swFile id="(\d+)"[^>]*swPath="([^"]*)"', xml):
        if m.group(2).replace("\\", "/").rsplit("/", 1)[-1] == stem:
            fid = m.group(1)
    if fid is None:
        raise SystemExit(f"{stem}: no swFile names this document")
    blocks = [b for b in xml.split("<swModel ")[1:]
              if re.match(r'[^>]*swFileRef="%s"' % fid, b)]
    if not blocks:
        raise SystemExit(f"{stem}: no swModel references swFile {fid}")
    out = []
    for m in re.finditer(r'swComponentName="([^"]+)"[^>]*?'
                         r'swTransform="([^"]+)"', max(blocks, key=len)):
        v = [float(x) for x in m.group(2).split()]
        out.append((m.group(1),
                    np.array([v[0:3], v[4:7], v[8:11]]), np.array(v[12:15])))
    return out


def main(argv):
    if len(argv) < 2 or argv[0] not in ("list", "dump", "bbox", "asm"):
        raise SystemExit(__doc__)
    cmd, path = argv[0], argv[1]
    if cmd == "list":
        for name, csz, usz, _ in streams(path):
            print(f"{usz:>10} (c={csz:>9})  {name}")
    elif cmd == "dump":
        out = Path(argv[2] if len(argv) > 2 else "out/sldstreams")
        out.mkdir(parents=True, exist_ok=True)
        for name, _, _, dec in streams(path):
            (out / name.replace("/", "__")).write_bytes(dec)
        print(f"wrote {len(streams(path))} streams to {out}")
    elif cmd == "bbox":
        lo, hi = bbox(path)
        print(json.dumps(dict(min_mm=list(np.round(lo * 1000, 4)),
                              max_mm=list(np.round(hi * 1000, 4)),
                              extent_mm=list(np.round((hi - lo) * 1000, 4))),
                         indent=1))
    else:
        here = Path(path).parent
        for name, A, t in components(path):
            row = f"{name[:52]:<54} t_mm {np.round(t * 1000, 3)}"
            src = here / f"{name}.SLDPRT"
            if src.is_file():                   # place its box if we have it
                lo, hi = bbox(src)
                C = np.array([[x, y, z] for x in (lo[0], hi[0])
                              for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                W = (C @ A + t) * 1000
                row += ("  asm box mm x[%8.3f,%8.3f] y[%8.3f,%8.3f] "
                        "z[%8.3f,%8.3f]" % (W[:, 0].min(), W[:, 0].max(),
                                            W[:, 1].min(), W[:, 1].max(),
                                            W[:, 2].min(), W[:, 2].max()))
            print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
