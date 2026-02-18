import bpy
import math
import os
import mathutils
import json
import numpy as np

# ---------------- UTILS & SETUP ----------------
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
    """Configures GPU, Cycles, Res, and EXR format"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Render Settings
    scene.cycles.samples = cfg.get("render", "samples") or 128
    scene.cycles.max_bounces = cfg.get("render", "max_bounces") or 0
    
    # --- NEW: Resolution Setup ---
    res_x = cfg.get("render", "resolution_x") or 1920
    res_y = cfg.get("render", "resolution_y") or 1080
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    
    # Image Format
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.exr_codec = 'ZIP'
    
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.cycles.film_exposure = 1.0
    
    # GPU Setup
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = cfg.get("render", "compute_device")
        for d in cprefs.devices: d.use = True
    except: pass
    scene.cycles.device = cfg.get("render", "device")

def save_yaml(data, filepath):
    with open(filepath, 'w') as f:
        f.write("# Light Configuration Metadata\n")
        f.write("# 'light_id': Custom ID from JSON\n\n")
        for entry in data:
            f.write(f"- light_id: {entry['light_id']}\n")
            f.write(f"  coordinates_local_cm: {entry['coordinates_local_cm']}\n")
            f.write(f"  orientation_local_deg:\n")
            f.write(f"    roll: {entry['orientation_local_deg']['roll']}\n")
            f.write(f"    pitch: {entry['orientation_local_deg']['pitch']}\n")
            f.write(f"    yaw: {entry['orientation_local_deg']['yaw']}\n")
            f.write(f"  direction_vector_local: {entry['direction_vec_local']}\n")
            f.write(f"  position_tilt_deg: {entry['position_tilt_deg']}\n")
            f.write(f"  fixed_tilt_deg: {entry['fixed_tilt_deg']}\n")
            f.write(f"  radiant_power_w: {entry['radiant_power_w']}\n")
            f.write("\n")

# ---------------- CAMERA ----------------
def setup_camera(plane_center, plane_rot, dist_cm=20):
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = 'PERSP'       
    cam_data.lens = 50.0          
    cam_data.sensor_fit = 'HORIZONTAL'  
    cam_data.sensor_width = 36.0  
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_dist_m = cm_to_m(dist_cm)
    local_offset = mathutils.Vector((0, 0, cam_dist_m))
    global_offset = plane_rot @ local_offset
    cam_obj.location = plane_center + global_offset
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = plane_rot
    return cam_obj

# ---------------- GEOMETRY & MATERIAL ----------------
def create_diffuse_material():
    mat_name = "Standard_White"
    if mat_name in bpy.data.materials: return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    node_diffuse.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    node_diffuse.inputs['Roughness'].default_value = 1.0 
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(node_diffuse.outputs['BSDF'], node_output.inputs['Surface'])
    return mat

def create_plane(pc):
    if "Plane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Plane"], do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.context.object
    plane.name = "Plane"
    mat = create_diffuse_material()
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

def get_plane_data(plane_obj):
    return plane_obj.matrix_world.translation, plane_obj.matrix_world.to_quaternion()

# ---------------- LIGHTS ----------------
def setup_lights(cfg, plane_center, plane_rot):
    coords = cfg.get("light_rig", "coordinates_cm")
    tilts = cfg.get("light_rig", "tilt_angles_deg")
    calibrated_powers = cfg.get("light_rig", "calibrated_power_w")
    light_ids = cfg.get("light_rig", "light_ids")
    spec = cfg.get("light_spec")
    
    if not tilts or len(tilts) != len(coords):
        print("WARNING: 'tilt_angles_deg' missing/mismatch. Defaulting.")
        tilts = [39.0, 39.0, 39.0, 19.893] 
        
    if not light_ids or len(light_ids) != len(coords):
        print("WARNING: 'light_ids' missing/mismatch. Generating defaults.")
        light_ids = [i + 1 for i in range(len(coords))]

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    lights = []
    light_metadata = [] 
    
    print(f"\n{'='*20} LIGHT SETUP {'='*20}")
    
    for i, (pos_cm, fixed_tilt, custom_id) in enumerate(zip(coords, tilts, light_ids)):
        ld = bpy.data.lights.new(f"light_{custom_id}", 'AREA')
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        lights.append(obj)
        
        # Geometry
        local_pos_vec = mathutils.Vector([cm_to_m(p) for p in pos_cm])
        global_offset = plane_rot @ local_pos_vec
        obj.location = plane_center + global_offset
        
        ld.shape = spec['shape']
        ld.size = mm_to_m(spec['size_mm'])
        ld.spread = deg_to_rad(spec['beam_angle_deg'])
        
        # Power
        if calibrated_powers and i < len(calibrated_powers):
            final_energy = calibrated_powers[i]
            source_type = "Direct"
        else:
            final_energy = spec["luminous_flux_lm"] / spec["luminous_efficacy_lm_w"]
            source_type = "Calc"
        ld.energy = final_energy
        
        # Orientation
        yaw_rad = math.atan2(-local_pos_vec.y, -local_pos_vec.x)
        theta_rad = deg_to_rad(fixed_tilt)
        dir_x = math.cos(yaw_rad) * math.sin(theta_rad)
        dir_y = math.sin(yaw_rad) * math.sin(theta_rad)
        dir_z = -math.cos(theta_rad)
        vec_local = mathutils.Vector((dir_x, dir_y, dir_z))
        d_norm_local = vec_local.normalized()
        
        vec_global = plane_rot @ vec_local
        obj.rotation_euler = vec_global.to_track_quat('-Z', 'Y').to_euler()
        
        # Metadata Calcs
        light_quat_global = obj.rotation_euler.to_quaternion()
        light_quat_local = plane_rot.inverted() @ light_quat_global
        local_euler = light_quat_local.to_euler()
        r = round(rad_to_deg(local_euler.x), 2)
        p = round(rad_to_deg(local_euler.y), 2)
        y = round(rad_to_deg(local_euler.z), 2)
        
        dist = local_pos_vec.length
        pos_tilt_deg = round(math.degrees(math.acos(max(min(local_pos_vec.z / dist, 1.0), -1.0))), 4) if dist > 0 else 0.0
        
        print(f"ID {custom_id}: {source_type} P={final_energy:.2f}W | Tilt={fixed_tilt}°")
        
        light_metadata.append({
            "light_id": custom_id,
            "coordinates_local_cm": pos_cm,
            "orientation_local_deg": {"roll": r, "pitch": p, "yaw": y},
            "direction_vec_local": [round(d_norm_local.x, 4), round(d_norm_local.y, 4), round(d_norm_local.z, 4)],
            "position_tilt_deg": pos_tilt_deg,
            "fixed_tilt_deg": fixed_tilt,
            "radiant_power_w": final_energy
        })
        
    print(f"{'='*50}\n")
    return lights, light_metadata, light_ids

# ---------------- PROCESSING ----------------
def measure_and_save(exr_path, exposure, out_path):
    img = bpy.data.images.load(exr_path)
    pixels = np.array(img.pixels[:])
    w, h = img.size
    pixels = pixels.reshape((h, w, 4))
    pixels[:, :, :3] *= exposure
    
    center_y, center_x = h // 2, w // 2
    sample = pixels[center_y-2:center_y+3, center_x-2:center_x+3, :3]
    avg_intensity = np.mean(sample)
    
    pixels = np.clip(pixels, 0, 1)
    
    out_img = bpy.data.images.new("temp", w, h, alpha=True)
    out_img.pixels = pixels.flatten().tolist()
    out_img.filepath_raw = out_path
    out_img.file_format = 'JPEG'
    out_img.save()
    bpy.data.images.remove(img)
    bpy.data.images.remove(out_img)
    return avg_intensity

def render_pipeline(scene, lights, light_ids, exposures, output_dir):
    exr_dir = os.path.join(output_dir, "exr_raw")
    jpg_dir = os.path.join(output_dir, "jpg")
    os.makedirs(exr_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)
    
    exr_files = []
    
    print("Phase 1: Raytracing (EXR)...")
    for i, light in enumerate(lights):
        for l in lights: l.hide_render = True
        light.hide_render = False
        
        custom_id = light_ids[i]
        id_str = f"{custom_id:03d}" if isinstance(custom_id, int) else str(custom_id)
        
        path = os.path.join(exr_dir, f"light_{id_str}.exr")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        exr_files.append((path, id_str))

    print("Phase 2: Post-Processing (JPEG)...")
    for (exr_path, id_str) in exr_files:
        for exp in exposures:
            # Output Name Logic
            out_name = f"light_{id_str}.jpg" 
            if len(exposures) > 1:
                out_name = f"light_{id_str}_exp_{exp:.3f}.jpg"
                
            out_path = os.path.join(jpg_dir, out_name)
            intensity = measure_and_save(exr_path, exp, out_path)
            print(f"Light {id_str} | Exp {exp} | Intensity: {intensity:.6f}")

def main():
    json_path = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\protype_2\model_1_cfg.json"
    
    with open(json_path, 'r') as f: cfg = Config(json.load(f))
    setup_scene_settings(cfg)
    start, end, step = cfg.get("exposure", "start"), cfg.get("exposure", "end"), cfg.get("exposure", "step")
    exposures = [round(start + x * step, 3) for x in range(int((end-start)/step)+1)]
    
    for pc in cfg.get("plane_configs"):
        plane = create_plane(pc)
        center, rotation = get_plane_data(plane)
        setup_camera(center, rotation, dist_cm=20)
        
        lights, metadata, light_ids = setup_lights(cfg, center, rotation)
        
        folder_name = f"Azi{pc['azimuth_deg']}_Ele{pc['elevation_deg']}"
        out_dir = os.path.join(cfg.get("paths", "base_output_dir"), folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        json_file_path = os.path.join(out_dir, "metadata.json")
        with open(json_file_path, 'w') as f: json.dump(metadata, f, indent=2)
        
        yaml_file_path = os.path.join(out_dir, "metadata.yaml")
        save_yaml(metadata, yaml_file_path)
        print(f"Metadata saved to: {out_dir}")
        
        render_pipeline(bpy.context.scene, lights, light_ids, exposures, out_dir)

if __name__ == "__main__":
    main()