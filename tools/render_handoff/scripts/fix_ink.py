import bpy
import json

INK_JSON = "/tmp/claude-1000/-home-peter-Documents-aris-proj/f19cb860-9890-4964-af59-c280c149f61d/scratchpad/ink_data.json"
BEVEL_RADIUS = 0.0035

with open(INK_JSON) as f:
    data = json.load(f)
ink = data["ink"]
anim = data["ink_anim"]

ink_col = bpy.data.collections.get("ink")
if ink_col is None:
    ink_col = bpy.data.collections.new("ink")
    bpy.context.scene.collection.children.link(ink_col)

# Remove the broken triangulated ink meshes
removed = 0
for obj in list(bpy.data.objects):
    if obj.name.startswith("ink_L") and obj.type == "MESH":
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        removed += 1

# Shared ink material (near-black, slightly glossy like wet ink)
mat = bpy.data.materials.get("ink_material")
if mat is None:
    mat = bpy.data.materials.new("ink_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.004, 0.004, 0.004, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65

created = 0
keyed = 0
for path, stroke in ink.items():
    name = "ink_" + path.split("/")[-1]
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = BEVEL_RADIUS
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    pts = stroke["points"]
    spline.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        spline.points[i].co = (p[0], p[1], p[2], 1.0)
    curve.materials.append(mat)

    obj = bpy.data.objects.new(name, curve)
    ink_col.objects.link(obj)
    created += 1

    tracks = anim.get(path, [])
    reveal = None
    for t in tracks:
        if t["prop"] == ".visible" and t["times"]:
            reveal = int(t["times"][-1])
    if reveal is not None:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=0)
        obj.keyframe_insert("hide_render", frame=0)
        obj.keyframe_insert("hide_viewport", frame=max(0, reveal - 1))
        obj.keyframe_insert("hide_render", frame=max(0, reveal - 1))
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=reveal)
        obj.keyframe_insert("hide_render", frame=reveal)
        keyed += 1

print(f"removed {removed} broken meshes; created {created} ink curves; keyframed {keyed}")
