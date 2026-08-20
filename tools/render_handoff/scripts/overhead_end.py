import bpy
import math

scene = bpy.context.scene
cam = bpy.data.objects["Camera"]
pivot = bpy.data.objects["CameraOrbit"]

F_HOLD_START = 1590   # last orbit-only frame
F_ARRIVE = 1740       # overhead reached (writing done at 1723)
F_END = 1788

TOTAL_DEG = 60.0

def pivot_angle(frame):
    return TOTAL_DEG * frame / F_END

# Overhead pose (world): straight above the pivot point, looking down,
# image-up = +y so ARIS reads upright. Local = R_z(-pivot_angle) applied,
# which leaves (0,0,h) unchanged; local z-rot must counter the pivot spin
# so the overhead view holds perfectly still in world space.
H_LOCAL = 4.55  # camera z = 0.35 + 4.55 = 4.9

cam.animation_data_clear()
cam.rotation_mode = 'XYZ'

# Key 1: current orbit pose, constant until F_HOLD_START
orbit_loc = tuple(cam.location)
orbit_rot = tuple(cam.rotation_euler)
cam.keyframe_insert("location", frame=F_HOLD_START)
cam.keyframe_insert("rotation_euler", frame=F_HOLD_START)

# Key 2: overhead arrival
cam.location = (0.0, 0.0, H_LOCAL)
cam.rotation_euler = (0.0, 0.0, -math.radians(pivot_angle(F_ARRIVE)))
cam.keyframe_insert("location", frame=F_ARRIVE)
cam.keyframe_insert("rotation_euler", frame=F_ARRIVE)

# Key 3: end — location unchanged, z-rot keeps countering pivot (linear)
cam.rotation_euler = (0.0, 0.0, -math.radians(pivot_angle(F_END)))
cam.keyframe_insert("location", frame=F_END)
cam.keyframe_insert("rotation_euler", frame=F_END)

# Interpolation: smooth swoop into overhead, linear counter-spin during hold
for layer in cam.animation_data.action.layers:
    for strip in layer.strips:
        for cb in strip.channelbags:
            for fc in cb.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER' if kp.co.x < F_ARRIVE else 'LINEAR'
                fc.update()

bpy.context.view_layer.update()
print(f"orbit pose loc={tuple(round(v,3) for v in orbit_loc)} "
      f"rot={tuple(round(math.degrees(v),1) for v in orbit_rot)}")
for f in (1500, 1665, 1740, 1788):
    scene.frame_set(f)
    m = cam.matrix_world
    print(f"f{f}: world cam=({m.translation.x:.2f},{m.translation.y:.2f},{m.translation.z:.2f})")
scene.frame_set(0)
bpy.ops.wm.save_mainfile()
print("saved")
