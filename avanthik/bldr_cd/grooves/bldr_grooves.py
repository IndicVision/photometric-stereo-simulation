import bpy
import math
import os
import mathutils
import json
import numpy as np

# ==========================================
#         CONFIGURATION & UTILITIES
# ==========================================
class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            if isinstance(v, dict) and k in v: v = v[k]
            else: return None
        return v

def cm_to_m(x): return x / 100.0
def mm_to_m(x): return x / 1000.0
def deg_to_rad(x): return math.radians(x)

def clean_scene():
    """Wipes the scene completely clean."""
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'CAMERA', 'LIGHT'}:
            obj.select_set(True)
    bpy.ops.object.delete() 
    
    # Purge orphan data blocks
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)

def setup_scene_settings(cfg):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Render Specs
    scene.cycles.samples = cfg.get("render", "samples") or 128
    scene.render.resolution_x = cfg.get("render", "resolution_x") or 1920
    scene.render.resolution_y = cfg.get("render", "resolution_y") or 1080
    scene.render.resolution_percentage = 100
    
    # Device
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = cfg.get("render", "compute_device") or 'CUDA'
        for d in cprefs.devices: d.use = True
    except: pass
    scene.cycles.device = cfg.get("render", "device") or 'GPU'

    # Photometric / Linear Workflow (CRITICAL)
    photo_cfg = cfg.get("photometric_settings") or {}
    max_bounces = photo_cfg.get("max_bounces", 0)
    
    scene.cycles.max_bounces = max_bounces
    scene.cycles.diffuse_bounces = max_bounces
    scene.cycles.glossy_bounces = max_bounces
    scene.cycles.transmission_bounces = max_bounces
    
    if photo_cfg.get("disable_caustics", True):
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False

    # Standard View Transform = Raw, Gamma = 1.0
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = photo_cfg.get("dither_intensity", 0.0)

# ==========================================
#             MATERIAL SETUP
# ==========================================
def setup_material(mat_cfg):
    mat_name = mat_cfg.get("name", "Dynamic_Material")
    if mat_name in bpy.data.materials: return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    mat_type = mat_cfg.get("type", "DIFFUSE").upper()
    
    if mat_type == "PRINCIPLED":
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = mat_cfg.get("base_color", [0.8, 0.8, 0.8, 1.0])
        shader.inputs['Roughness'].default_value = mat_cfg.get("roughness", 0.5)
        shader.inputs['Metallic'].default_value = mat_cfg.get("metallic", 0.0)
        spec = mat_cfg.get("specular_ior_level", 0.5)
        if 'Specular IOR Level' in shader.inputs: shader.inputs['Specular IOR Level'].default_value = spec
        elif 'Specular' in shader.inputs: shader.inputs['Specular'].default_value = spec
    else:
        # PURE LAMBERTIAN
        shader = nodes.new('ShaderNodeBsdfDiffuse')
        shader.inputs['Color'].default_value = mat_cfg.get("diffuse_color", [1.0, 1.0, 1.0, 1.0])
        shader.inputs['Roughness'].default_value = mat_cfg.get("roughness", 0.0)

    mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    return mat

# ==========================================
#          GEOMETRY (Blender 4.0 Fixed)
# ==========================================
def create_grooved_object(obj_cfg, mat_cfg):
    # Parse
    base_l = cm_to_m(obj_cfg["base_length_cm"])
    base_b = cm_to_m(obj_cfg["base_breadth_cm"])
    base_h = mm_to_m(obj_cfg.get("base_thickness_mm", 5.0))
    
    g_type = obj_cfg.get("groove_type", "DIP")
    g_rad, g_len = cm_to_m(obj_cfg["groove_radius_cm"]), cm_to_m(obj_cfg["groove_length_cm"])
    g_width = cm_to_m(obj_cfg["groove_width_cm"])
    if g_width >= 2 * g_rad: g_width = 1.99 * g_rad
    
    vert_offset = math.sqrt(g_rad**2 - (g_width/2)**2)
    
    # Base
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    base_obj = bpy.context.object
    base_obj.name = "Base_Plate"
    base_obj.dimensions = (base_l, base_b, base_h)
    base_obj.location = (0, 0, -base_h/2) # Top surface at Z=0
    
    # Tool
    bpy.ops.mesh.primitive_cylinder_add(radius=g_rad, depth=g_len, vertices=128)
    cyl = bpy.context.object
    cyl.rotation_euler = (math.pi/2, 0, 0)
    tool_parts = [cyl]

    if obj_cfg.get("use_spherical_caps", False):
        for y_pos in [g_len/2, -g_len/2]:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=g_rad, segments=64, ring_count=32)
            s = bpy.context.object
            s.location = (0, y_pos, 0)
            tool_parts.append(s)

    # BLENDER 4.0 FIX: Context Override for Join
    with bpy.context.temp_override(active_object=tool_parts[0], selected_editable_objects=tool_parts):
        bpy.ops.object.join()
        
    tool_obj = tool_parts[0]
    
    # Position Tool
    tool_obj.rotation_euler[2] = deg_to_rad(obj_cfg.get("groove_axis_angle_deg", 0.0))
    
    # NEW LOGIC: For a HILL, we submerge the cylinder slightly into the base
    # so that the 'hump' sits on top and the union results in a solid piece.
    if g_type == "HILL":
        # Move the cylinder down so its top arc is visible above Z=0
        z_loc = -vert_offset 
    else:
        # For DIP, move it up so it cuts into the plate
        z_loc = vert_offset
        
    tool_obj.location = (0, 0, z_loc)
    
    # Boolean
    mod = base_obj.modifiers.new(name="Groove_Bool", type='BOOLEAN')
    mod.object = tool_obj
    mod.solver = 'FAST'
    mod.operation = 'UNION' if g_type == "HILL" else 'DIFFERENCE'
    
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.object.modifier_apply(modifier="Groove_Bool")
    bpy.data.objects.remove(tool_obj, do_unlink=True)
    
    # Material & Smooth
    mat = setup_material(mat_cfg)
    if base_obj.data.materials: base_obj.data.materials[0] = mat
    else: base_obj.data.materials.append(mat)
    
    # Set the object as active and select it
    bpy.context.view_layer.objects.active = base_obj
    base_obj.select_set(True)

    # Blender 4.5+ Robust Smoothing
    if bpy.app.version >= (4, 1, 0):
        # This operator automatically handles the "Smooth by Angle" modifier logic
        bpy.ops.object.shade_smooth_by_angle(angle=deg_to_rad(30))
    else:
        # Compatibility for 4.0 and older
        base_obj.data.use_auto_smooth = True
        base_obj.data.auto_smooth_angle = deg_to_rad(30)
        bpy.ops.object.shade_smooth()

    # --- ORIENTATION LOGIC ---
    # Apply Center Shift
    c_cm = obj_cfg["center_cm"]
    base_obj.location = (base_obj.location.x + cm_to_m(c_cm[0]), 
                         base_obj.location.y + cm_to_m(c_cm[1]), 
                         base_obj.location.z + cm_to_m(c_cm[2]))
    
    # Apply Rotation (Azimuth/Elevation) to the Object
    ele, azi = obj_cfg["elevation_deg"], obj_cfg["azimuth_deg"]
    phi, theta = deg_to_rad(90 - ele), deg_to_rad(azi)
    
    # Standard Normal is Z+
    target_normal = mathutils.Vector((
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi)
    ))
    
    base_obj.rotation_mode = 'QUATERNION'
    base_obj.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(target_normal)
    
    return base_obj

# ==========================================
#          FIXED CAMERA & LIGHTS
# ==========================================
def setup_fixed_camera(cam_cfg):
    """Sets up a camera at a fixed world position looking at a fixed target."""
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
        
    cam_data = bpy.data.cameras.new("Camera")
    
    # Type
    if cam_cfg.get("type", "PERSP") == 'ORTHO':
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = cam_cfg.get("ortho_scale", 10.0) # Explicit scale required
    else:
        cam_data.type = 'PERSP'
        cam_data.lens = cam_cfg.get("focal_length_mm", 50.0)

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Fixed Position
    pos = cam_cfg.get("position_cm", [0, 0, 50])
    target = cam_cfg.get("look_at_cm", [0, 0, 0])
    
    cam_obj.location = mathutils.Vector([cm_to_m(p) for p in pos])
    look_at = mathutils.Vector([cm_to_m(p) for p in target])
    
    # Point Camera at Target
    direction = look_at - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    return cam_obj

def setup_fixed_lights(cfg):
    coords = cfg.get("light_rig", "coordinates_cm")
    directions = cfg.get("light_rig", "directions") # New field
    powers = cfg.get("light_rig", "calibrated_power_w")
    light_ids = cfg.get("light_rig", "light_ids")
    spec = cfg.get("light_spec")
    
    lights = []
    metadata = []
    
    for i in range(len(coords)):
        ld = bpy.data.lights.new(f"light_{light_ids[i]}", 'AREA')
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        
        # 1. Position
        obj.location = mathutils.Vector([cm_to_m(p) for p in coords[i]])
        
        # 2. Rotation using Unit Direction
        # We want the Light's -Z axis to point along the 'direction' vector
        target_dir = mathutils.Vector(directions[i]).normalized()
        obj.rotation_euler = target_dir.to_track_quat('-Z', 'Y').to_euler()
        
        # 3. Physical Specs
        ld.energy = powers[i]
        size_mm = spec.get('size_mm', [50, 50])
        ld.size = mm_to_m(size_mm[0])
        ld.size_y = mm_to_m(size_mm[1])
        
        lights.append(obj)
        metadata.append({"id": light_ids[i], "pos": coords[i], "dir": directions[i]})
        
    return lights, light_ids, metadata

# ==========================================
#        HIGH-SPEED IO & POST-PROCESS
# ==========================================
def process_exposures_fast(scene, exr_path, exposures, output_dir, id_str, png_depth):
    """Loads EXR once, applies multiple exposures, saves PNGs. Fast."""
    if not os.path.exists(exr_path): return

    try:
        # Load EXR
        img = bpy.data.images.load(exr_path, check_existing=False)
        # Setup Output Image Buffer
        w, h = img.size
        # Get pixels (Linear Float)
        # Pixels are RGBA flattened list
        raw_pixels = np.array(img.pixels[:], dtype=np.float32) 
        
        # Cleanup input image from Blender memory immediately to save RAM
        bpy.data.images.remove(img)

        # Temp Buffer for Saving
        temp_img_name = "Buffer_Export"
        if temp_img_name in bpy.data.images: bpy.data.images.remove(bpy.data.images[temp_img_name])
        out_img = bpy.data.images.new(temp_img_name, w, h, alpha=True)
        
        # Setup Scene for PNG saving
        rs = scene.render.image_settings
        prev_fmt = rs.file_format
        prev_depth = rs.color_depth
        rs.file_format = 'PNG'
        rs.color_depth = str(png_depth)
        
        png_dir = os.path.join(output_dir, "png_exposed")
        
        # Process Loop (Memory only)
        for exp in exposures:
            # Apply Exposure
            # Multipling RGBA. Alpha (index 3, 7, 11...) should technically not be multiplied if premultiplied
            # But for solid objects against black, usually fine. 
            # Ideally: reshape, multiply RGB, clip, flatten.
            
            # Reshape for easier math
            arr_view = raw_pixels.reshape(-1, 4)
            
            # Multiply RGB by Exposure
            exposed = arr_view.copy()
            exposed[:, :3] *= exp 
            
            # Clip 0..1
            np.clip(exposed, 0.0, 1.0, out=exposed)
            
            # Push to Blender Image
            out_img.pixels.foreach_set(exposed.flatten())
            
            # Save
            png_name = f"light_{id_str}_exp_{exp:.4f}.png"
            out_img.save_render(filepath=os.path.join(png_dir, png_name), scene=scene)

        # Cleanup
        bpy.data.images.remove(out_img)
        rs.file_format = prev_fmt
        rs.color_depth = prev_depth

    except Exception as e:
        print(f"Fast Exposure Error: {str(e)}")

def render_pipeline(scene, lights, light_ids, exposures, output_dir, out_cfg):
    save_exr = out_cfg.get("save_exr", True)
    save_png = out_cfg.get("save_png", False)
    png_depth = out_cfg.get("png_bit_depth", 16)
    
    exr_dir = os.path.join(output_dir, "exr_raw")
    png_dir = os.path.join(output_dir, "png_exposed")
    
    os.makedirs(exr_dir, exist_ok=True)
    if save_png: os.makedirs(png_dir, exist_ok=True)
    
    for i, light in enumerate(lights):
        # Isolate Light
        for l in lights: l.hide_render = True
        light.hide_render = False
        
        id_str = f"{light_ids[i]:03d}" if isinstance(light_ids[i], int) else str(light_ids[i])
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")
        
        # RENDER TO EXR (Always needed as Source)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32' 
        scene.render.filepath = exr_path
        
        print(f"    > Rendering Light {id_str}...")
        bpy.ops.render.render(write_still=True)
        
        # GENERATE PNGs (Fast Mode)
        if save_png:
            process_exposures_fast(scene, exr_path, exposures, output_dir, id_str, png_depth)
        
        # DELETE EXR if not requested
        if not save_exr and os.path.exists(exr_path):
            os.remove(exr_path)

# ==========================================
#                    MAIN
# ==========================================
def main():
    json_path = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grooves\bldr_grooves_cfg.json"
    
    if not os.path.exists(json_path):
        print(f"CRITICAL: Config not found at {json_path}")
        return

    print(f"Loading Config...")
    with open(json_path, 'r') as f: cfg = Config(json.load(f))
    
    setup_scene_settings(cfg)
    clean_scene()
    
    mat_cfg = cfg.get("material") or {}
    cam_cfg = cfg.get("camera") or {}
    out_cfg = cfg.get("output") or {}
    
    # Exposures
    start, end, step = cfg.get("exposure", "start"), cfg.get("exposure", "end"), cfg.get("exposure", "step")
    if start == end: exposures = [start]
    else:
        if step <= 0: step = 0.01 
        count = int(round((end - start) / step)) + 1
        exposures = [round(start + x * step, 6) for x in range(count)]
    
    base_out = out_cfg.get("base_output_dir") or "output"
    
    # SETUP FIXED RIG (Camera & Lights once)
    # Note: If object blocks lights, that's physics.
    
    for i, obj_cfg in enumerate(cfg.get("object_configs")):
        obj_name = obj_cfg.get('name', f'Obj_{i}')
        print(f"\n--- Processing {obj_name} ---")
        
        clean_scene()
        
        # 1. FIXED RIG SETUP
        setup_fixed_camera(cam_cfg)
        lights, light_ids, meta = setup_fixed_lights(cfg)
        
        # 2. OBJECT SETUP (Rotates based on Azimuth/Elevation)
        create_grooved_object(obj_cfg, mat_cfg)
        
        # 3. PATHS
        azi, ele = obj_cfg['azimuth_deg'], obj_cfg['elevation_deg']
        folder_name = f"{obj_name}_Azi{azi}_Ele{ele}"
        out_dir = os.path.join(base_out, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 4. METADATA (Includes Camera/Light info for PS)
        meta_full = {
            "lights": meta,
            "camera": {
                "pos_cm": cam_cfg.get("position_cm"),
                "look_at_cm": cam_cfg.get("look_at_cm")
            },
            "object_orientation": {"azimuth": azi, "elevation": ele}
        }
        with open(os.path.join(out_dir, "metadata.json"), 'w') as f: json.dump(meta_full, f, indent=2)
            
        # 5. RENDER
        render_pipeline(bpy.context.scene, lights, light_ids, exposures, out_dir, out_cfg)

if __name__ == "__main__":
    main()