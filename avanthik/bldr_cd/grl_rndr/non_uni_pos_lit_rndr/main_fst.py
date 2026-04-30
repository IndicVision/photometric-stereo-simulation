import bpy
import math
import os
import mathutils
import json
import numpy as np

class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            v = v[k]
        return v
    @property
    def base_output_dir(self):
        return self.get("paths", "base_output_dir")

def cm_to_m(x): return x / 100.0
def deg_to_rad(x): return math.radians(x)

def setup_gpu(cfg):
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.compute_device_type = cfg.get("render", "compute_device")
    for d in cprefs.devices:
        d.use = True
    bpy.context.scene.cycles.device = cfg.get("render", "device")

def setup_render_settings(cfg):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "Raw"
    render_cfg = cfg.get("render")
    scene.cycles.max_bounces = render_cfg.get("max_bounces", 0)
    scene.cycles.film_exposure = 1.0
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'

def create_plane(pc):
    if "Plane" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        bpy.context.object.name = "Plane"
    plane = bpy.data.objects["Plane"]
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    target = mathutils.Vector((
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.cos(deg_to_rad(pc["azimuth_deg"])),
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.sin(deg_to_rad(pc["azimuth_deg"])),
        math.cos(deg_to_rad(90 - pc["elevation_deg"]))
    ))
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = mathutils.Vector((0,0,1)).rotation_difference(target)

def ensure_lights(n, ltype):
    lights = [o for o in bpy.data.objects if o.type == "LIGHT" and o.name.startswith("light.")]
    if len(lights) == n and all(o.data.type == ltype for o in lights):
        return sorted(lights, key=lambda x: x.name)
    for o in lights:
        bpy.data.objects.remove(o, do_unlink=True)
    out = []
    for i in range(n):
        ld = bpy.data.lights.new(f"light.{i+1:03d}", ltype)
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        out.append(obj)
    return out

def get_manual_positions(center, vectors, dist_m):
    pos = []
    for v in vectors:
        vec = mathutils.Vector((v['lx'], v['ly'], v['lz']))
        pos.append(center + (vec * dist_m))
    return pos

def configure_lights_batch(lights, params, positions, center):
    for obj, pos in zip(lights, positions):
        ld = obj.data
        obj.location = pos
        ld.energy = params["energy"]
        ld.shape = params["shape"]
        ld.size = cm_to_m(params["dim_a_cm"])
        ld.size_y = cm_to_m(params["dim_b_cm"])
        ld.spread = deg_to_rad(params["light_prop_value"])
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()

def apply_exposure_blender(exr_path, exposure_value, output_path):
    img = bpy.data.images.load(exr_path, check_existing=False)
    width, height = img.size
    pixels = np.array(img.pixels[:]).reshape((height, width, 4))
    pixels[:, :, :3] *= exposure_value
    pixels = np.clip(pixels, 0, 1)
    output_img = bpy.data.images.new("temp_output", width, height, alpha=True)
    output_img.pixels = pixels.flatten().tolist()
    output_img.filepath_raw = output_path
    output_img.file_format = 'PNG'
    output_img.save()
    bpy.data.images.remove(img)
    bpy.data.images.remove(output_img)

def render_single_light_exr(scene, lights, light_idx, output_path):
    for i, light in enumerate(lights):
        light.hide_render = (i != light_idx)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def process_configuration(scene, lights, temp_dir, final_dir, exposure_values):
    exr_files = []
    for idx, light in enumerate(lights):
        exr_path = os.path.join(temp_dir, f"light_{idx+1:03d}.exr")
        render_single_light_exr(scene, lights, idx, exr_path)
        exr_files.append(exr_path)
    for idx, exr_path in enumerate(exr_files):
        for exposure in exposure_values:
            output_path = os.path.join(final_dir, f"{idx+1:03d}_{exposure:.3f}.png")
            apply_exposure_blender(exr_path, exposure, output_path)

def main():
    config_path = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\non_uni_pos_lit_rndr\main_fst_cfg.json"
    with open(config_path) as f:
        cfg = Config(json.load(f))
    setup_gpu(cfg)
    setup_render_settings(cfg)
    scene = bpy.context.scene
    loops = cfg.get("global_loops")
    manual_vectors = cfg.get("manual_light_setup", "vectors")
    
    start, end, step = cfg.get("exposure", "start"), cfg.get("exposure", "end"), cfg.get("exposure", "step")
    exposure_values = [round(start + i * step, 3) for i in range(int((end - start) / step) + 1)]

    light_params = {
        "energy": loops["energy_array"][0],
        "light_prop_value": cfg.get("light","area_light","spread_angles_deg")[0],
        "dim_a_cm": cfg.get("light","area_light","dimensions")[0]["dim_a_cm"],
        "dim_b_cm": cfg.get("light","area_light","dimensions")[0]["dim_b_cm"],
        "shape": cfg.get("light","area_light","dimensions")[0]["shape"]
    }

    for samples in loops["samples_array"]:
        scene.cycles.samples = samples
        for pc in cfg.get("plane_configs"):
            create_plane(pc)
            plane_obj = bpy.data.objects["Plane"]
            center = plane_obj.matrix_world @ mathutils.Vector((0,0,0))
            
            # CALCULATE AREA FOR NAMING
            area_cm2 = pc["length_cm"] * pc["breadth_cm"]
            conf_folder = f"{pc['azimuth_deg']:.2f}_{pc['elevation_deg']:.2f}_{pc['center_cm'][0]:.2f}_{pc['center_cm'][1]:.2f}_{pc['center_cm'][2]:.2f}_{area_cm2:.2f}"
            
            lights = ensure_lights(len(manual_vectors), "AREA")
            for dist_cm in loops["distance_array_cm"]:
                positions = get_manual_positions(center, manual_vectors, cm_to_m(dist_cm))
                configure_lights_batch(lights, light_params, positions, center)
                
                l_type = loops["light_type_array"][0].lower()
                l_shape = light_params["shape"].lower()
                l_spread = light_params["light_prop_value"]
                light_setup_name = f"{len(manual_vectors)}_{dist_cm:.2f}_{l_type}_{l_shape}_{l_spread}_Man_{light_params['energy']:.2f}"
                
                final_dir = os.path.join(cfg.base_output_dir, f"samples_{samples}", conf_folder, light_setup_name)
                temp_dir = os.path.join(final_dir, "_temp_exr")
                os.makedirs(final_dir, exist_ok=True)
                os.makedirs(temp_dir, exist_ok=True)
                
                process_configuration(scene, lights, temp_dir, final_dir, exposure_values)
                import shutil
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()