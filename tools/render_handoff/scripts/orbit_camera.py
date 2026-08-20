import bpy
import math
from mathutils import Matrix, Vector

scene = bpy.context.scene
cam = bpy.data.objects["Camera"]

target = Vector((1.8035, 0.98, 0.35))
direction = Vector((0.0, 3.43, -2.2)).normalized()

pivot = bpy.data.objects.get("CameraOrbit")
if pivot is None:
    pivot = bpy.data.objects.new("CameraOrbit", None)
    scene.collection.objects.link(pivot)
pivot.animation_data_clear()
pivot.parent = None
pivot.location = target
pivot.rotation_euler = (0.0, 0.0, 0.0)
pivot.scale = (1.0, 1.0, 1.0)

# Bind camera in the pivot's LOCAL space (pivot unrotated at bind time,
# so local = world - pivot origin). No matrix_parent_inverse involved.
cam.animation_data_clear()
cam.parent = pivot
cam.matrix_parent_inverse = Matrix.Identity(4)
local_offset = -direction * 5.2  # camera sits opposite the view direction
cam.location = local_offset
world_loc = target + local_offset
cam.rotation_euler = (target - world_loc).normalized().to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 40

# One slow full revolution, constant speed
f0, f1 = scene.frame_start, scene.frame_end
pivot.rotation_euler = (0.0, 0.0, 0.0)
pivot.keyframe_insert("rotation_euler", index=2, frame=f0)
pivot.rotation_euler = (0.0, 0.0, math.tau)
pivot.keyframe_insert("rotation_euler", index=2, frame=f1)

for layer in pivot.animation_data.action.layers:
    for strip in layer.strips:
        for cb in strip.channelbags:
            for fc in cb.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

# Sanity check: evaluated camera world position at a few frames
bpy.context.view_layer.update()
for f in (0, 450, 900):
    scene.frame_set(f)
    m = cam.matrix_world
    loc = m.translation
    fwd = -(m.col[2].to_3d().normalized())
    to_target = (target - loc).normalized()
    print(f"f{f}: cam=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) aim_dot={fwd.dot(to_target):.4f}")

scene.frame_set(f0)
bpy.ops.wm.save_mainfile()
print("saved")
