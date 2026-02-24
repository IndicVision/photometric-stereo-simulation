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

def clean_scene():
    """Wipes the scene completely clean to prevent memory leaks."""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'CAMERA', 'LIGHT'}:
            obj.select_set(True)
    bpy.ops.object.delete() 
    
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)

def setup_scene_settings(cfg):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    scene.cycles.samples = cfg.get("render", "samples") or 128
    scene.cycles.max_bounces = cfg.get("photometric_settings", "max_bounces") or 0
    
    res_x = cfg.get("render", "resolution_x") or 1920
    res_y = cfg.get("render", "resolution_y") or 1080
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    
    # Photometric / Linear Workflow
    scene.view_settings.view_transform = cfg.get("photometric_settings", "view_transform") or 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.gamma = cfg.get("photometric_settings", "gamma") or 1.0
    scene.cycles.film_exposure = 1.0  
    
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
    if "Plane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Plane"], do_unlink=True)
        
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.context.object
    plane.name = "Plane"
    
    mat = bpy.data.materials.new(name="Standard_White")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    diff = nodes.new('ShaderNodeBsdfDiffuse')
    diff.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    diff.inputs['Roughness'].default_value = 0.0  
    
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(diff.outputs['BSDF'], out.inputs['Surface'])
    
    if plane.data.materials: plane.data.materials[0] = mat
    else: plane.data.materials.append(mat)
    
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    
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

def setup_camera(plane_center, plane_rot, dist_cm=20.0, cam_type='PERSP', lens=50.0, ortho_scale=1.0):
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
        
    cam_data = bpy.data.cameras.new("Camera")
    
    if cam_type == 'ORTHO':
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = ortho_scale
    else:
        cam_data.type = 'PERSP'
        cam_data.lens = lens

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    local_offset = mathutils.Vector((0, 0, cm_to_m(dist_cm)))
    global_offset = plane_rot @ local_offset
    cam_obj.location = plane_center + global_offset
    
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = plane_rot
    
    return cam_obj

# ==========================================
#        LIGHT SETUP (ABSOLUTE LOGIC)
# ==========================================
def setup_lights(cfg):
    """
    Sets up lights using explicit vectors and absolute positions.
    No automatic shifting math is applied here.
    """
    coords = cfg.get("light_rig", "coordinates_cm")
    directions = cfg.get("light_rig", "directions")
    powers = cfg.get("light_rig", "calibrated_power_w")
    light_ids = cfg.get("light_rig", "light_ids")
    spec = cfg.get("light_spec")
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    lights = []
    metadata_list = []
    
    if not light_ids: light_ids = [i+1 for i in range(len(coords))]
    
    for i in range(len(coords)):
        ld = bpy.data.lights.new(f"light_{light_ids[i]:03d}", 'AREA')
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        
        # 1. Absolute Position in 3D Space
        obj.location = mathutils.Vector([cm_to_m(p) for p in coords[i]])
        
        # 2. Explicit Direction Targeting (Normalize the vector and track)
        target_dir = mathutils.Vector(directions[i]).normalized()
        obj.rotation_euler = target_dir.to_track_quat('-Z', 'Y').to_euler()
        
        # 3. Shape and Specs
        ld.shape = spec.get('shape', 'RECTANGLE')
        size_data = spec.get('size_mm', 50.0)
        
        if isinstance(size_data, list):
            ld.size = mm_to_m(size_data[0])
            ld.size_y = mm_to_m(size_data[1])
        else:
            ld.size = mm_to_m(size_data)
            ld.size_y = mm_to_m(size_data)
            
        ld.spread = deg_to_rad(spec.get('beam_angle_deg', 180.0))
        power = powers[i] if powers and i < len(powers) else 1.0
        ld.energy = power
        
        lights.append(obj)
        metadata_list.append({
            "id": light_ids[i],
            "power_w": power,
            "global_pos": coords[i],
            "direction": directions[i]
        })
        
    return lights, light_ids, metadata_list

# ==========================================
#          POST-PROCESSING (EXPOSURE)
# ==========================================
def apply_exposure_16bit(scene, exr_path, exposure_value, output_path):
    if not os.path.exists(exr_path): return
    try:
        img = bpy.data.images.load(exr_path, check_existing=False)
        pixels = np.array(img.pixels[:])
        w, h = img.size
        pixels = pixels.reshape((h, w, 4)) 
        
        pixels[:, :, :3] *= exposure_value
        pixels = np.clip(pixels, 0.0, 1.0)
        
        temp_name = "Temp_Export_Buffer"
        if temp_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[temp_name])
            
        out_img = bpy.data.images.new(temp_name, w, h, alpha=True)
        out_img.pixels = pixels.flatten().tolist()
        
        prev_fmt = scene.render.image_settings.file_format
        prev_depth = scene.render.image_settings.color_depth
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '16' 
        
        out_img.filepath_raw = output_path
        out_img.file_format = 'PNG'
        out_img.save()
        
        scene.render.image_settings.file_format = prev_fmt
        scene.render.image_settings.color_depth = prev_depth
        
        bpy.data.images.remove(img)
        bpy.data.images.remove(out_img)
        
    except Exception as e:
        print(f"Failed to save PNG {output_path}: {str(e)}")

# ==========================================
#             RENDER PIPELINE
# ==========================================
def render_pipeline(scene, lights, light_ids, exposures, output_dir, save_png=True):
    exr_dir = os.path.join(output_dir, "exr_raw")
    png_dir = os.path.join(output_dir, "png_16bit")
    
    os.makedirs(exr_dir, exist_ok=True)
    if save_png: os.makedirs(png_dir, exist_ok=True)
    
    print(f"\nStarting Batch Render...")
    print(f"Output Directory: {output_dir}")
    
    for i, light in enumerate(lights):
        for l in lights: l.hide_render = True
        light.hide_render = False
        
        id_str = f"{light_ids[i]:03d}" if isinstance(light_ids[i], int) else str(light_ids[i])
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")
        
        print(f"    > Rendering Light {id_str} (EXR)...")
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = exr_path
        bpy.ops.render.render(write_still=True)
        
        if save_png:
            for exp in exposures:
                png_name = f"light_{id_str}_exp_{exp:.4f}.png"
                png_path = os.path.join(png_dir, png_name)
                apply_exposure_16bit(scene, exr_path, exp, png_path)

    print("Batch Complete.")

# ==========================================
#                   MAIN
# ==========================================
def main():
    print("\n" + "="*50)
    print("STARTING BLENDER AUTOMATION SCRIPT (PRE-SHIFT)")
    print("="*50)
    
    # Path explicitly checks your requested directory structure
    json_path = r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\config.json"
    
    print(f"Looking for JSON Configuration at: {json_path}")
    if not os.path.exists(json_path):
        print(f">>> FATAL ERROR: JSON file does not exist at {json_path} <<<")
        print("Please check your spelling and file path.")
        return
        
    with open(json_path, 'r') as f: cfg = Config(json.load(f))
    print("JSON Loaded Successfully. Proceeding to Render...")
    
    clean_scene()
    setup_scene_settings(cfg)
    
    cam_conf = cfg.get("camera") or {}
    cam_type = cam_conf.get("type", "PERSP")
    cam_dist = cam_conf.get("distance_cm", 20.0)
    current_lens = cam_conf.get("focal_length_mm", 50.0)
    
    res_x = bpy.context.scene.render.resolution_x
    res_y = bpy.context.scene.render.resolution_y
    aspect_ratio = res_x / res_y

    start = cfg.get("exposure", "start")
    end = cfg.get("exposure", "end")
    step = cfg.get("exposure", "step")
    
    if start == end: 
        exposures = [start]
    else:
        if step <= 0: step = 0.01 
        count = int(round((end - start) / step)) + 1
        exposures = [round(start + x * step, 6) for x in range(count)]
    
    save_png_flag = cfg.get("output", "save_png")
    if save_png_flag is None: save_png_flag = False 
    base_out = cfg.get("output", "base_output_dir") or "output"

    for pc in cfg.get("plane_configs"):
        print(f"\n=======================================================")
        print(f"Processing Plane: Azi {pc['azimuth_deg']}, Ele {pc['elevation_deg']}")
        
        # 1. Setup Geometry
        plane = create_plane(pc)
        center = plane.matrix_world.translation
        rot = plane.matrix_world.to_quaternion()
        
        # 2. Setup Camera (with Auto-Fit logic)
        current_scale = cam_conf.get("ortho_scale", 1.0)
        if cam_type == 'ORTHO' and cam_conf.get("auto_fit", False):
            w = cm_to_m(pc["length_cm"])
            h = cm_to_m(pc["breadth_cm"])
            margin = cam_conf.get("margin", 1.0)
            current_scale = max(w, h * aspect_ratio) * margin

        setup_camera(center, rot, 
                     dist_cm=cam_dist, 
                     cam_type=cam_type, 
                     lens=current_lens, 
                     ortho_scale=current_scale)
        
        # 3. Setup Lights (Executes exactly once per plane)
        lights, light_ids, meta = setup_lights(cfg)
        
        # 4. Prepare Output Folders
        folder_name = f"Azi{pc['azimuth_deg']}_Ele{pc['elevation_deg']}"
        out_dir = os.path.join(base_out, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 5. Save Metadata
        with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)
            
        # 6. Execute Render
        render_pipeline(bpy.context.scene, lights, light_ids, exposures, out_dir, save_png=save_png_flag)

    print("\nAll Render Operations Complete!")

if __name__ == "__main__":
    main()