import bpy
import json

out = {}

# ink-like objects
ink_objs = [o.name for o in bpy.data.objects if "_c0" in o.name or o.name.lower().startswith(("l0_", "l1_", "l2_", "l3_", "ink"))]
out["ink_like_objects"] = ink_objs[:20]
out["ink_like_count"] = len(ink_objs)

# ink collection?
out["collections_named_ink"] = [c.name for c in bpy.data.collections if "ink" in c.name.lower()]

# default Collection contents
col = bpy.data.collections.get("Collection")
if col:
    out["default_collection_sample"] = [o.name for o in col.objects[:15]]
    out["default_collection_count"] = len(col.objects)

# table/paper world transforms for frame mapping
for name_part in ["table", "paper", "base13", "pen13"]:
    matches = [o for o in bpy.data.objects if name_part in o.name.lower()]
    for o in matches[:2]:
        m = o.matrix_world
        out[f"obj_{o.name}"] = {
            "loc": [round(v, 4) for v in o.matrix_world.translation],
            "dim": [round(v, 4) for v in o.dimensions],
            "type": o.type,
            "hide_render": o.hide_render,
        }

# any object with visibility keyframes? (Blender 5.x layered actions)
def iter_fcurves(action):
    try:
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    yield from cb.fcurves
    except AttributeError:
        yield from getattr(action, "fcurves", [])

vis_kf = 0
for o in bpy.data.objects:
    ad = o.animation_data
    if ad and ad.action:
        for fc in iter_fcurves(ad.action):
            if "hide" in fc.data_path:
                vis_kf += 1
                break
out["objects_with_visibility_keyframes"] = vis_kf

# total animated objects
out["animated_objects"] = sum(1 for o in bpy.data.objects if o.animation_data and o.animation_data.action)

# empties
out["empty_count"] = sum(1 for o in bpy.data.objects if o.type == "EMPTY")
out["mesh_count"] = sum(1 for o in bpy.data.objects if o.type == "MESH")

print(json.dumps(out, indent=1))
