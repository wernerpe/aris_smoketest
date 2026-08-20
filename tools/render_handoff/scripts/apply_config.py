import bpy
import os
from mathutils import Vector

scene = bpy.context.scene
render = scene.render

# --- Render engine (universal settings from blendervids) ---
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
gpus = []
for device in prefs.devices:
    if device.type in {'OPTIX', 'CUDA', 'HIP', 'METAL'}:
        device.use = True
        gpus.append(device.name)
scene.cycles.device = 'GPU'
scene.cycles.samples = 25
scene.cycles.preview_samples = 2
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'
scene.cycles.denoising_use_gpu = True
scene.render.use_persistent_data = True

# --- Resolution / output ---
render.resolution_x = 1920
render.resolution_y = 1080
render.resolution_percentage = 100
render.fps = 30
render.film_transparent = False
render.image_settings.media_type = 'VIDEO'
render.ffmpeg.format = 'MPEG4'
render.ffmpeg.codec = 'H264'
render.ffmpeg.constant_rate_factor = 'MEDIUM'
render.ffmpeg.gopsize = 18
render.ffmpeg.audio_codec = 'NONE'
os.makedirs('/home/peter/Documents/aris_proj/output', exist_ok=True)
render.filepath = '/home/peter/Documents/aris_proj/output/aris_writing.mp4'

# --- HDRI lighting with white camera-visible background ---
hdri_path = '/home/peter/Documents/manipulation_station/blendervids/brown_photostudio_02.hdr'
world = scene.world
if not world:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
tree = world.node_tree
tree.nodes.clear()

tex_coord = tree.nodes.new('ShaderNodeTexCoord')
mapping = tree.nodes.new('ShaderNodeMapping')
env_tex = tree.nodes.new('ShaderNodeTexEnvironment')
env_tex.image = bpy.data.images.load(hdri_path)
hdri_bg = tree.nodes.new('ShaderNodeBackground')
hdri_bg.inputs['Strength'].default_value = 1.0
white_bg = tree.nodes.new('ShaderNodeBackground')
white_bg.inputs['Color'].default_value = (1, 1, 1, 1)
white_bg.inputs['Strength'].default_value = 1.0
light_path = tree.nodes.new('ShaderNodeLightPath')
mix_shader = tree.nodes.new('ShaderNodeMixShader')
out_node = tree.nodes.new('ShaderNodeOutputWorld')

tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
tree.links.new(env_tex.outputs['Color'], hdri_bg.inputs['Color'])
tree.links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
tree.links.new(hdri_bg.outputs['Background'], mix_shader.inputs[1])
tree.links.new(white_bg.outputs['Background'], mix_shader.inputs[2])
tree.links.new(mix_shader.outputs['Shader'], out_node.inputs['Surface'])

# --- Shadow catcher just under the table slab ---
# Table spans z -0.054..-0.004, centered at (1.8035, 0.98)
if 'ShadowCatcher_Floor' not in bpy.data.objects:
    mesh = bpy.data.meshes.new('ShadowCatcher_Floor')
    s = 10.0
    cx, cy, cz = 1.8035, 0.98, -0.0545
    mesh.from_pydata(
        [(cx - s, cy - s, cz), (cx + s, cy - s, cz), (cx + s, cy + s, cz), (cx - s, cy + s, cz)],
        [], [(0, 1, 2, 3)])
    mesh.update()
    floor = bpy.data.objects.new('ShadowCatcher_Floor', mesh)
    scene.collection.objects.link(floor)
    floor.is_shadow_catcher = True
    fmat = bpy.data.materials.new('ShadowCatcher_Material')
    fmat.use_nodes = True
    fb = fmat.node_tree.nodes['Principled BSDF']
    fb.inputs['Base Color'].default_value = (1, 1, 1, 1)
    fb.inputs['Roughness'].default_value = 1.0
    floor.data.materials.append(fmat)

# --- Camera from the meshcat viewpoint (pulled back to fit 50mm) ---
target = Vector((1.8035, 0.98, 0.15))
direction = Vector((0.0, 3.43, -2.2)).normalized()  # meshcat camera->target direction
cam_loc = target - direction * 5.9

if 'Camera' in bpy.data.objects:
    cam = bpy.data.objects['Camera']
else:
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    scene.collection.objects.link(cam)
cam.location = cam_loc
look = (target - cam_loc).normalized()
cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 50
cam.data.sensor_width = 36
cam.data.clip_start = 0.1
cam.data.clip_end = 1000
scene.camera = cam

print(f"engine={scene.render.engine} gpus={gpus} frames={scene.frame_start}-{scene.frame_end} "
      f"cam_loc={tuple(round(v,3) for v in cam.location)} out={render.filepath}")
