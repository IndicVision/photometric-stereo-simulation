import bpy
import csv
import json
import math
import mathutils
import os
import numpy as np

def cm_to_m(x): return x / 100.0
def mm_to_m(x): return x / 1000.0
def deg_to_rad(x): return math.radians(x)

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.lights: bpy.data.lights.remove(block)
    for block in bpy.data.cameras: bpy.data.cameras.remove(block)

def setup_core_scene(cfg):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = cfg["render"]["device"]
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = cfg["render"]["compute_device"]
        for dev in prefs.devices: dev.use = True
    except: pass

    scene.cycles.samples = cfg["render"]["samples"]
    scene.cycles.max_bounces = cfg["render"]["max_bounces"]
    scene.render.resolution_x = cfg["render"]["resolution_x"]
    scene.render.resolution_y = cfg["render"]["resolution_y"]
    scene.view_settings.view_transform = cfg["photometric_settings"]["view_transform"]
    scene.view_settings.gamma = cfg["photometric_settings"]["gamma"]

    pc = cfg["plane_configs"][0]
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane = bpy.context.object
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    
    n_zen, n_azi = deg_to_rad(90 - pc["elevation_deg"]), deg_to_rad(pc["azimuth_deg"])
    target_normal = mathutils.Vector((math.sin(n_zen)*math.cos(n_azi), math.sin(n_zen)*math.sin(n_azi), math.cos(n_zen)))
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(target_normal)
    
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = cfg["camera"]["focal_length_mm"]
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.location = plane.location + (plane.rotation_quaternion @ mathutils.Vector((0, 0, cm_to_m(cfg["camera"]["distance_cm"]))))
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = plane.rotation_quaternion

def build_lights_from_csv(config_rows, spec):
    lights = []
    ids = []
    
    for row in config_rows:
        lid = str(row['Light_ID'])
        ld = bpy.data.lights.new(f"light_{lid}", 'AREA')
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        
        # STRICT METER CONVERSION
        x_m = cm_to_m(float(row['Pos_X_cm']))
        y_m = cm_to_m(float(row['Pos_Y_cm']))
        z_m = cm_to_m(float(row['Pos_Z_cm']))
        obj.location = mathutils.Vector((x_m, y_m, z_m))
        
        dir_vec = mathutils.Vector((float(row['Dir_X']), float(row['Dir_Y']), float(row['Dir_Z'])))
        obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
        
        ld.shape = spec["shape"]
        ld.size = mm_to_m(spec["size_mm"][0])
        ld.size_y = mm_to_m(spec["size_mm"][1])
        ld.spread = deg_to_rad(spec["beam_angle_deg"])
        ld.energy = float(row['Power_W'])
        
        lights.append(obj)
        ids.append(lid)
        
    return lights, ids

def apply_exposure_16bit(scene, exr_path, exp, out_path):
    if not os.path.exists(exr_path): return
    img = bpy.data.images.load(exr_path, check_existing=False)
    pixels = np.array(img.pixels[:])
    w,h = img.size
    pixels = pixels.reshape((h,w,4))
    pixels[:,:,:3] *= exp
    pixels = np.clip(pixels,0,1)

    out_img = bpy.data.images.new("temp", w,h)
    out_img.pixels = pixels.flatten()

    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_depth='16'
    out_img.filepath_raw = out_path
    out_img.file_format='PNG'
    out_img.save()

    bpy.data.images.remove(img)
    bpy.data.images.remove(out_img)

def render_pipeline(scene, lights, ids, out_dir, cfg):
    exr_dir = os.path.join(out_dir, "exr_raw")
    png_dir = os.path.join(out_dir, "png_16bit")
    os.makedirs(exr_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    exp_start, exp_end, step = cfg["exposure"]["start"], cfg["exposure"]["end"], cfg["exposure"]["step"]
    if exp_start == exp_end: exposures = [exp_start]
    else: exposures = [round(exp_start + x * step, 6) for x in range(int(round((exp_end - exp_start) / step)) + 1)]

    for i, light in enumerate(lights):
        for l in lights: l.hide_render = True
        light.hide_render = False

        id_str = ids[i]
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")

        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = exr_path
        bpy.ops.render.render(write_still=True)

        for exp in exposures:
            png_path = os.path.join(png_dir, f"light_{id_str}_exp_{exp:.4f}.png")
            apply_exposure_16bit(scene, exr_path, exp, png_path)

def main():
    json_path = r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\scripts\multiple_config_rndr.json"
    
    with open(json_path, 'r') as f:
        cfg = json.load(f)

    csv_path = cfg["paths"]["csv_output"]
    base_out = cfg["paths"]["base_output_dir"]

    configs = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['Config_ID']
            if cid not in configs: configs[cid] = []
            configs[cid].append(row)

    for cid, rows in configs.items():
        print(f"\n======================================")
        print(f"Setting up and Rendering: {cid}")
        
        clean_scene()
        setup_core_scene(cfg)
        
        lights, ids = build_lights_from_csv(rows, cfg["light_spec"])
        
        out_dir = os.path.join(base_out, cid)
        render_pipeline(bpy.context.scene, lights, ids, out_dir, cfg)

    print("\nALL RENDER CONFIGURATIONS COMPLETED.")

if __name__ == "__main__":
    main()