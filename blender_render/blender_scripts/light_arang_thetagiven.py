import bpy
import os
import json
import math
import numpy as np
from mathutils import Vector

class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            if isinstance(v, dict) and k in v: v = v[k]
            else: return None
        return v

# --- UPDATE THIS PATH TO YOUR SAVED JSON CONFIG ---
JSON_PATH = r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\blender_scripts\light_arang_thetagiven.json"

# ==========================================
#         GEOMETRIC SOLVER (YOUR EQ)
# ==========================================
def solve_for_z(r, phi_rad, theta_in_deg):
    """
    Solves your handwritten equation for the target height 'z'.
    Target Center is fixed at [0, 0, 2.7]. 
    Light Z (global) = r*cos(phi) + 2.7.
    """
    # The Vertical distance (h) in your equation's numerator
    # Let h = (r*cos(phi) + 2.7 - z)
    # cos(theta) = h / sqrt( (r*sin(phi))^2 + h^2 )
    
    theta_rad = math.radians(theta_in_deg)
    
    # Algebraically, z = (r*cos(phi) + 2.7) - (r*sin(phi) / tan(theta))
    # This ensures the vector from light to [0,0,z] has incidence theta.
    h = (r * math.sin(phi_rad)) / math.tan(theta_rad)
    z = (r * math.cos(phi_rad) + 2.7) - h
    
    return z / 100.0 # Convert cm to meters for Blender

def create_lighting_rig(cfg):
    logger_info = "Building Rig: Solving Geometric Incidence Equation"
    print(f"\n--- {logger_info} ---")
    
    sp_cfg = cfg.get("study_params") or {}
    l_spec = cfg.get("light_spec") or {}
    powers = cfg.get("calibrated_power_w") or [1.17] * 4
    
    # Your Origin offset (2.7 cm)
    origin_z = 2.7 / 100.0 
    theta_in = sp_cfg.get("theta_in_deg")
    
    dim_x, dim_y = l_spec.get("size_mm")[0]/1000, l_spec.get("size_mm")[1]/1000

    bpy.ops.object.select_all(action='DESELECT')
    [obj.select_set(True) for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
    bpy.ops.object.delete()

    target_lights = []
    for i, l_data in enumerate(cfg.get("lights") or []):
        az_rad = math.radians(l_data.get("azimuth", 0.0))
        phi_rad = math.radians(l_data.get("zenith", 0.0))
        r_cm = l_data.get("radius_cm", 8.0)
        r_m = r_cm / 100.0
        
        # 1. Position Light in Global Frame
        global_pos = Vector((
            r_m * math.sin(phi_rad) * math.cos(az_rad),
            r_m * math.sin(phi_rad) * math.sin(az_rad),
            (r_m * math.cos(phi_rad)) + origin_z
        ))

        # 2. Solve for Z using your Incident Equation
        if theta_in is not None:
            # Calculate the specific Z coordinate on the vertical axis [0,0,z] 
            # where the incidence will be exactly theta_in
            z_target_m = solve_for_z(r_cm, phi_rad, theta_in)
            target_pt = Vector((0, 0, z_target_m))
        else:
            # Fallback: Point to the sphere origin [0, 0, 2.7]
            target_pt = Vector((0, 0, origin_z))

        # 3. Geometric Direction Vector (Light -> Solved Target)
        dir_vec = (target_pt - global_pos).normalized()

        # 4. Create Blender Light
        light_data = bpy.data.lights.new(name=f"Light_{i+1}", type='AREA')
        light_data.size, light_data.size_y = dim_x, dim_y
        light_data.energy = powers[i] if i < len(powers) else 1.17
        
        obj = bpy.data.objects.new(name=f"Light_{i+1}", object_data=light_data)
        bpy.context.collection.objects.link(obj)
        obj.location = global_pos
        obj.rotation_euler = dir_vec.to_track_quat('-Z', 'Y').to_euler()
        
        target_lights.append(obj)
        print(f"  Light_{i+1}: Pos {global_pos}, Target Z: {target_pt.z*100:.2f}cm")

    bpy.context.view_layer.update()
    return target_lights
# ==========================================
#         RENDER SETTINGS
# ==========================================
def apply_render_settings(cfg):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples        = cfg.get("render", "samples") or 4096
    scene.render.resolution_x   = cfg.get("render", "resolution_x") or 2100
    scene.render.resolution_y   = cfg.get("render", "resolution_y") or 1400
    scene.render.resolution_percentage = 100

    prefs  = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = cfg.get("render", "compute_device") or 'CUDA'
        for d in cprefs.devices: d.use = True
    except: pass
    scene.cycles.device = cfg.get("render", "device") or 'GPU'

    photo = cfg.get("photometric_settings") or {}
    mb = photo.get("max_bounces", 0)
    scene.cycles.max_bounces          = mb
    scene.cycles.diffuse_bounces      = mb
    scene.cycles.glossy_bounces       = mb
    scene.cycles.transmission_bounces = mb
    if photo.get("disable_caustics", True):
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False

    scene.view_settings.view_transform = photo.get("view_transform") or 'Raw'
    scene.view_settings.look            = 'None'
    scene.view_settings.gamma           = photo.get("gamma") or 1.0
    scene.render.dither_intensity       = photo.get("dither_intensity", 0.0)

# ==========================================
#         EXPOSURE -> PNG
# ==========================================
def save_exposed_pngs(scene, exr_path, exposures, png_dir, id_str, bit_depth):
    if not os.path.exists(exr_path):
        print(f"  [WARN] EXR not found: {exr_path}")
        return

    img = bpy.data.images.load(exr_path, check_existing=False)
    w, h = img.size
    raw = np.array(img.pixels[:], dtype=np.float32)
    bpy.data.images.remove(img)

    buf_name = "__ExportBuffer__"
    if buf_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[buf_name])
    buf = bpy.data.images.new(buf_name, w, h, alpha=True)

    rs = scene.render.image_settings
    prev_fmt, prev_depth = rs.file_format, rs.color_depth
    rs.file_format = 'PNG'
    rs.color_depth = str(bit_depth)

    for exp in exposures:
        arr = raw.reshape(-1, 4).copy()
        arr[:, :3] *= exp
        np.clip(arr, 0.0, 1.0, out=arr)
        buf.pixels.foreach_set(arr.flatten())
        out_path = os.path.join(png_dir, f"light_{id_str}_exp_{exp:.4f}.png")
        buf.save_render(filepath=out_path, scene=scene)
        print(f"      PNG saved: {out_path}")

    bpy.data.images.remove(buf)
    rs.file_format = prev_fmt
    rs.color_depth = prev_depth

# ==========================================
#         MAIN RENDER LOOP
# ==========================================
def main():
    if not os.path.exists(JSON_PATH):
        print(f"CRITICAL: Config not found -> {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        cfg = Config(json.load(f))

    # 1. Create the lighting rig dynamically based on the JSON
    target_lights = create_lighting_rig(cfg)

    if not target_lights:
        print("CRITICAL: No target lights were created. Aborting.")
        return

    # 2. Apply render settings
    apply_render_settings(cfg)
    scene    = bpy.context.scene
    
    # 3. Handle Outputs
    out_cfg  = cfg.get("output") or {}
    do_render = out_cfg.get("do_render", False)
    
    if not do_render:
        print("\n=== Setup Complete. Rendering is skipped (do_render=false) ===")
        return

    save_exr = out_cfg.get("save_exr", True)
    save_png = out_cfg.get("save_png", False)
    bit_depth = out_cfg.get("png_bit_depth", 16)
    base_out  = out_cfg.get("base_output_dir") or "output"

    # Build exposure list for PNGs (if needed)
    start = cfg.get("exposure", "start") or 1.0
    end   = cfg.get("exposure", "end")   or start
    step  = cfg.get("exposure", "step")  or 1.0
    if start == end or step <= 0:
        exposures = [start]
    else:
        count     = int(round((end - start) / step)) + 1
        exposures = [round(start + i * step, 6) for i in range(count)]

    # Output folders
    exr_dir = os.path.join(base_out, "exr_raw")
    png_dir = os.path.join(base_out, "png_exposed")
    os.makedirs(exr_dir, exist_ok=True)
    if save_png: os.makedirs(png_dir, exist_ok=True)

    # Hide all target lights to start
    for l in target_lights:
        l.hide_render = True

    # Render one light at a time
    for light in target_lights:
        id_str = light.name.split('_')[-1]
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")

        light.hide_render = False
        print(f"\n--- Rendering: {light.name} ---")

        # Render to EXR
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = exr_path
        bpy.ops.render.render(write_still=True)
        print(f"    EXR saved: {exr_path}")

        # Generate PNGs from EXR 
        if save_png:
            save_exposed_pngs(scene, exr_path, exposures, png_dir, id_str, bit_depth)

        # Remove EXR if not requested
        if not save_exr and os.path.exists(exr_path):
            os.remove(exr_path)
            print(f"    EXR deleted (save_exr=false)")

        light.hide_render = True

    # Restore: show all lights
    for l in target_lights:
        l.hide_render = False

    print("\n=== Render Complete ===")

if __name__ == "__main__":
    main()