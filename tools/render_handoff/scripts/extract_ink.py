#!/usr/bin/env python3
"""Fully decode drake static meshcat HTML; extract ink strokes + their animation timing."""
import base64
import json
import re
import struct
import sys

import msgpack

SRC = "/home/peter/Documents/aris_proj/aris_writing.html"
OUT = "/tmp/claude-1000/-home-peter-Documents-aris-proj/f19cb860-9890-4964-af59-c280c149f61d/scratchpad/ink_data.json"

with open(SRC, "rb") as f:
    text = f.read().decode("utf-8", errors="ignore")

blobs = re.findall(r'data:application/octet-binary;base64,([A-Za-z0-9+/=]+)', text)

cmds = []
fails = 0
for b in blobs:
    raw = base64.b64decode(b)
    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False,
                                unicode_errors="surrogateescape",
                                max_buffer_size=len(raw) + 1024)
    unpacker.feed(raw)
    got = False
    try:
        for obj in unpacker:
            if isinstance(obj, dict) and "type" in obj:
                cmds.append(obj)
            got = True
    except Exception as e:
        if not got:
            fails += 1
            print("still failing:", type(e).__name__, e, file=sys.stderr)

print(f"blobs={len(blobs)} cmds={len(cmds)} fails={fails}", file=sys.stderr)

from collections import Counter
types = Counter(c.get("type") for c in cmds)
print("command types:", dict(types), file=sys.stderr)


def decode_typed_array(arr):
    """Meshcat packs arrays as {itemSize, type, array} where array may be bytes or ext."""
    if isinstance(arr, dict):
        item_size = arr.get("itemSize", 3)
        dtype = arr.get("type", "Float32Array")
        data = arr.get("array")
        if isinstance(data, (bytes, bytearray)):
            fmt = {"Float32Array": "f", "Float64Array": "d", "Uint32Array": "I",
                   "Uint16Array": "H", "Uint8Array": "B", "Int32Array": "i"}[dtype]
            n = len(data) // struct.calcsize(fmt)
            vals = struct.unpack(f"<{n}{fmt}", bytes(data[:n * struct.calcsize(fmt)]))
        elif isinstance(data, msgpack.ExtType):
            fmt = {"Float32Array": "f", "Float64Array": "d", "Uint32Array": "I",
                   "Uint16Array": "H", "Uint8Array": "B", "Int32Array": "i"}[dtype]
            payload = data.data
            n = len(payload) // struct.calcsize(fmt)
            vals = struct.unpack(f"<{n}{fmt}", payload[:n * struct.calcsize(fmt)])
        else:
            vals = list(data)
            item_size = arr.get("itemSize", 3)
        return [list(vals[i:i + item_size]) for i in range(0, len(vals), item_size)]
    return arr


ink = {}
for c in cmds:
    if c.get("type") != "set_object":
        continue
    p = c.get("path", "")
    if not p.startswith("/ink/"):
        continue
    o = c["object"]
    top = o.get("object", {})
    geoms = {g["uuid"]: g for g in o.get("geometries", [])}
    mats = {m["uuid"]: m for m in o.get("materials", [])}
    geom = geoms.get(top.get("geometry"))
    mat = mats.get(top.get("material"), {})
    pos_attr = geom.get("data", geom).get("attributes", {}).get("position")
    points = decode_typed_array(pos_attr)
    color = mat.get("color")
    ink[p] = {
        "points": points,
        "color": color,
        "linewidth": mat.get("linewidth"),
        "object_type": top.get("type"),
        "matrix": top.get("matrix"),
    }

print(f"ink strokes: {len(ink)}", file=sys.stderr)

# transforms on /ink paths
ink_tf = {}
for c in cmds:
    if c.get("type") == "set_transform" and c.get("path", "").startswith("/ink"):
        ink_tf[c["path"]] = list(c.get("matrix", []))
print(f"ink transforms: {len(ink_tf)} -> {list(ink_tf)[:5]}", file=sys.stderr)

# properties on /ink paths
ink_props = [c for c in cmds if c.get("type") == "set_property" and c.get("path", "").startswith("/ink")]
print(f"ink set_property: {len(ink_props)}", file=sys.stderr)
for c in ink_props[:10]:
    print("  ", c["path"], c["property"], str(c["value"])[:80], file=sys.stderr)

# animation tracks for ink
anim_info = {}
anims = [c for c in cmds if c.get("type") == "set_animation"]
for a in anims:
    for t in a.get("animations", []):
        p = t.get("path")
        clip = t.get("clip", {})
        fps = clip.get("fps")
        for tr in clip.get("tracks", []):
            name = tr.get("name")
            times = tr.get("keys") or tr.get("times")
            if p.startswith("/ink"):
                keys = tr.get("keys")
                if keys is not None:
                    ktimes = [k.get("time") for k in keys]
                    kvals = [k.get("value") for k in keys]
                else:
                    ktimes = list(tr.get("times", []))
                    kvals = list(tr.get("values", []))
                anim_info.setdefault(p, []).append(
                    {"prop": name, "fps": fps, "times": ktimes, "values_sample": kvals[:6],
                     "n_keys": len(ktimes)})

print(f"animated ink paths: {len(anim_info)}", file=sys.stderr)
for p in list(anim_info)[:3]:
    print("  ", p, json.dumps(anim_info[p])[:400], file=sys.stderr)

# non-ink animated path sample for fps reference
for a in anims:
    for t in a.get("animations", [])[:1]:
        clip = t.get("clip", {})
        print("sample clip fps:", clip.get("fps"), "duration:", clip.get("duration"),
              "path:", t.get("path"), file=sys.stderr)
        for tr in clip.get("tracks", [])[:1]:
            keys = tr.get("keys")
            if keys:
                print("  track name:", tr.get("name"), "n_keys:", len(keys),
                      "t0:", keys[0].get("time"), "t_end:", keys[-1].get("time"), file=sys.stderr)
            else:
                times = tr.get("times", [])
                print("  track name:", tr.get("name"), "n_keys:", len(times),
                      "t0:", times[0] if times else None,
                      "t_end:", times[-1] if times else None, file=sys.stderr)

with open(OUT, "w") as f:
    json.dump({"ink": ink, "ink_transforms": ink_tf, "ink_anim": anim_info}, f)
print("wrote", OUT, file=sys.stderr)

# also dump the full command inventory of failed-before types now that decode works
so = Counter(c.get("path", "").split("/")[1] for c in cmds if c.get("type") == "set_object")
print("set_object by root:", dict(so), file=sys.stderr)
