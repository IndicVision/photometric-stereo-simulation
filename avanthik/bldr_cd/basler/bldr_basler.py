import bpy
import math
import os
import mathutils
import json
import numpy as np

# ==========================================
#        CONFIGURATION & UTILITIES
# ==========================================
class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            if k in v: v = v[k]
            else: return None
        return v

def cm_to_m(x): return x / 100.0
def mm_to_m(x): return x / 1000.0
def deg_to_rad(x): return math.radians(x)
def rad_to_deg(x): return math.degrees(x)

def setup_scene_settings(cfg):
    """
    Configures the scene for Linear/Raw rendering.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Sampling & Light Paths
    scene.cycles.samples = cfg.get("render", "samples") or 128
    scene.cycles.max_bounces = cfg.get("render", "max_bounces") or 0
    
    # Resolution
    res_x = cfg.get("render", "resolution_x") or 1920
    res_y = cfg.get("render", "resolution_y") or 1080
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    
    # CRITICAL: Linear Workflow Settings (Gamma 1.0)
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.gamma = 1.0
    scene.cycles.film_exposure = 1.0  # We apply exposure math manually later
    
    # GPU Setup
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = cfg.get("render", "compute_device") or 'CUDA'
        for d in cprefs.devices: d.use = True
    except: pass
    scene.cycles.device = cfg.get("render", "device") or 'GPU'

# ==========================================
#               GEOMETRY SETUP
# ==========================================
def create_plane(pc):
    """
    Creates the sample plane based on the first code's logic.
    """
    if "Plane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Plane"], do_unlink=True)
        
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.context.object
    plane.name = "Plane"
    
    # Material: Standard White Diffuse (Lambertian)
    mat = bpy.data.materials.new(name="Standard_White")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    diff = nodes.new('ShaderNodeBsdfDiffuse')
    diff.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    diff.inputs['Roughness'].default_value = 0.0  # 0.0 = True Lambertian
    
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(diff.outputs['BSDF'], out.inputs['Surface'])
    
    if plane.data.materials: plane.data.materials[0] = mat
    else: plane.data.materials.append(mat)
    
    # Geometry
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    
    # Rotation (Normal Vector Logic)
    n_zen = deg_to_rad(90 - pc["elevation_deg"])
    n_azi = deg_to_rad(pc["azimuth_deg"])
    target_normal = mathutils.Vector((
        math.sin(n_zen) * math.cos(n_azi),
        math.sin(n_zen) * math.sin(n_azi),
        math.cos(n_zen)
    ))
    
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(target_normal)
    
    return plane

def setup_camera(plane_center, plane_rot, dist_cm=20):
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
        
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0 
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Place camera strictly relative to plane normal
    local_offset = mathutils.Vector((0, 0, cm_to_m(dist_cm)))
    global_offset = plane_rot @ local_offset
    cam_obj.location = plane_center + global_offset
    
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = plane_rot
    
    return cam_obj

# ==========================================
#               LIGHT SETUP
# ==========================================
def setup_lights(cfg, plane_center, plane_rot):
    """
    Sets up lights using the corrected relative coordinate logic.
    """
    coords = cfg.get("light_rig", "coordinates_cm")
    tilts = cfg.get("light_rig", "tilt_angles_deg")
    calibrated_powers = cfg.get("light_rig", "calibrated_power_w")
    light_ids = cfg.get("light_rig", "light_ids")
    spec = cfg.get("light_spec")
    
    # Cleanup old lights
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    lights = []
    metadata_list = []
    
    # Default ID generation if missing
    if not light_ids: light_ids = [i+1 for i in range(len(coords))]
    
    for i, (pos_cm, fixed_tilt, custom_id) in enumerate(zip(coords, tilts, light_ids)):
        ld = bpy.data.lights.new(f"light_{custom_id}", 'AREA')
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        lights.append(obj)
        
        # --- POSITION ---
        # We treat JSON coordinates as RELATIVE to the plane center 
        # to fix the "double height" bug from the first code.
        local_pos_vec = mathutils.Vector([cm_to_m(p) for p in pos_cm])
        global_offset = plane_rot @ local_pos_vec
        obj.location = plane_center + global_offset
        
        # --- SHAPE ---
        ld.shape = spec.get('shape', 'RECTANGLE')
        size_data = spec.get('size_mm', 50.0)
        
        if isinstance(size_data, list):
            ld.size = mm_to_m(size_data[0])
            ld.size_y = mm_to_m(size_data[1])
        else:
            ld.size = mm_to_m(size_data)
            ld.size_y = mm_to_m(size_data)
            
        ld.spread = deg_to_rad(spec.get('beam_angle_deg', 180.0))
        
        # --- POWER ---
        power = calibrated_powers[i] if calibrated_powers and i < len(calibrated_powers) else 1.0
        ld.energy = power
        
        # --- ORIENTATION ---
        # Logic to point light at origin (0,0,0) of the local frame
        yaw_rad = math.atan2(-local_pos_vec.y, -local_pos_vec.x)
        theta_rad = deg_to_rad(fixed_tilt)
        
        dir_x = math.cos(yaw_rad) * math.sin(theta_rad)
        dir_y = math.sin(yaw_rad) * math.sin(theta_rad)
        dir_z = -math.cos(theta_rad)
        
        vec_local = mathutils.Vector((dir_x, dir_y, dir_z))
        vec_global = plane_rot @ vec_local
        
        obj.rotation_euler = vec_global.to_track_quat('-Z', 'Y').to_euler()

        # Metadata collection
        metadata_list.append({
            "id": custom_id,
            "power_w": power,
            "tilt": fixed_tilt,
            "local_pos": pos_cm
        })
        
    return lights, light_ids, metadata_list

# ==========================================
#          POST-PROCESSING (EXPOSURE)
# ==========================================
def apply_exposure_16bit(scene, exr_path, exposure_value, output_path):
    """
    Loads EXR, applies linear exposure scaling, saves as 16-bit PNG.
    """
    if not os.path.exists(exr_path):
        print(f"Error: EXR not found {exr_path}")
        return

    try:
        # 1. Load Data
        img = bpy.data.images.load(exr_path, check_existing=False)
        pixels = np.array(img.pixels[:])
        w, h = img.size
        pixels = pixels.reshape((h, w, 4)) # RGBA
        
        # 2. Apply Exposure (Math)
        # Pixel_New = Pixel_Old * Exposure
        pixels[:, :, :3] *= exposure_value
        
        # 3. Clip to Valid Range [0.0 - 1.0]
        # Since we are saving to integer format (PNG), we must clip values > 1.0
        pixels = np.clip(pixels, 0.0, 1.0)
        
        # 4. Create Temp Image Block
        temp_name = "Temp_Export_Buffer"
        if temp_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[temp_name])
            
        out_img = bpy.data.images.new(temp_name, w, h, alpha=True)
        out_img.pixels = pixels.flatten().tolist()
        
        # 5. SAVE SETTINGS (Force 16-bit)
        # We must override scene settings momentarily to ensure .save() uses 16-bit
        prev_fmt = scene.render.image_settings.file_format
        prev_depth = scene.render.image_settings.color_depth
        
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '16' # Forces 16-bit
        
        out_img.filepath_raw = output_path
        out_img.file_format = 'PNG'
        out_img.save()
        
        # 6. Restore Settings
        scene.render.image_settings.file_format = prev_fmt
        scene.render.image_settings.color_depth = prev_depth
        
        # Cleanup
        bpy.data.images.remove(img)
        bpy.data.images.remove(out_img)
        
    except Exception as e:
        print(f"Failed to save PNG {output_path}: {str(e)}")

# ==========================================
#             RENDER PIPELINE
# ==========================================
def render_pipeline(scene, lights, light_ids, exposures, output_dir):
    exr_dir = os.path.join(output_dir, "exr_raw")
    png_dir = os.path.join(output_dir, "png_16bit")
    
    os.makedirs(exr_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    print(f"\nStarting Batch Render...")
    print(f"Output Directory: {output_dir}")
    
    # --- PHASE 1: RENDER EXR MASTERS ---
    for i, light in enumerate(lights):
        # Isolate Light
        for l in lights: l.hide_render = True
        light.hide_render = False
        
        id_str = f"{light_ids[i]:03d}" if isinstance(light_ids[i], int) else str(light_ids[i])
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")
        
        print(f"[{i+1}/{len(lights)}] Rendering Light {id_str} (EXR)...")
        
        # Configure for EXR Output
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = exr_path
        
        # Render Frame
        bpy.ops.render.render(write_still=True)
        
        # --- PHASE 2: GENERATE EXPOSURES ---
        print(f"    > Generating {len(exposures)} PNGs...")
        
        for exp in exposures:
            png_name = f"light_{id_str}_exp_{exp:.4f}.png"
            png_path = os.path.join(png_dir, png_name)
            
            # Apply exposure and save as 16-bit
            apply_exposure_16bit(scene, exr_path, exp, png_path)

    print("Batch Complete.")

# ==========================================
#                   MAIN
# ==========================================
def main():
    # --- UPDATE THIS PATH TO YOUR JSON FILE ---
    json_path = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\basler\bldr_basler_cfg.json"
    
    print("Loading Configuration...")
    with open(json_path, 'r') as f: cfg = Config(json.load(f))
    
    setup_scene_settings(cfg)
    
    # Calculate Exposure List
    start = cfg.get("exposure", "start")
    end = cfg.get("exposure", "end")
    step = cfg.get("exposure", "step")
    
    if start == end: 
        exposures = [start]
    else:
        # Safe float range generation
        count = int(round((end - start) / step)) + 1
        exposures = [round(start + x * step, 6) for x in range(count)]
    
    # Process Plane Configurations
    for pc in cfg.get("plane_configs"):
        print(f"\n--- Processing Plane: Azi {pc['azimuth_deg']}, Ele {pc['elevation_deg']} ---")
        
        # 1. Setup Geometry
        plane = create_plane(pc)
        center = plane.matrix_world.translation
        rot = plane.matrix_world.to_quaternion()
        
        # 2. Setup Camera & Lights
        setup_camera(center, rot, dist_cm=20)
        lights, light_ids, meta = setup_lights(cfg, center, rot)
        
        # 3. Prepare Folders
        folder_name = f"Azi{pc['azimuth_deg']}_Ele{pc['elevation_deg']}"
        out_dir = os.path.join(cfg.get("paths", "base_output_dir"), folder_name)
        
        # 4. Save Metadata
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
            
        # 5. Execute Render
        render_pipeline(bpy.context.scene, lights, light_ids, exposures, out_dir)

if __name__ == "__main__":
    main()