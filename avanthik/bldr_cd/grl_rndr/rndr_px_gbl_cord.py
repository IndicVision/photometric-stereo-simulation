import bpy
import json
import os
import math
import mathutils

# ==========================================
# 1. SETUP & UTILITIES
# ==========================================

def cm_to_m(val):
    return val / 100.0

def deg_to_rad(val):
    return math.radians(val)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def setup_render_environment(cfg):
    """Configures the scene for exact coordinate data output."""
    scene = bpy.context.scene
    
    # --- Engine Settings ---
    scene.render.engine = 'CYCLES'
    scene.cycles.device = cfg['render']['device']
    scene.cycles.samples = cfg['render']['samples']  
    scene.cycles.use_denoising = False
    
    # --- CRITICAL: Coordinate Accuracy Fixes ---
    
    # 1. Disable Exposure Scaling
    # Ensures pixel values are not multiplied by a global exposure factor
    scene.cycles.film_exposure = 1.0
    
    # 2. Minimize Anti-Aliasing Smearing
    # 'BOX' filter with width 0.01 acts like a point sampler.
    # It prevents the object coordinate from blending with the background coordinate.
    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width = 0.01
    
    # 3. Enable Alpha/Transparency
    # Essential for distinguishing "Zero Coordinate" from "Empty Background"
    scene.render.film_transparent = True
    
    # --- GPU Setup ---
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.compute_device_type = "CUDA"
    for d in cprefs.devices:
        d.use = True

    # --- Resolution ---
    scene.render.resolution_x = cfg['render']['resolution_x']
    scene.render.resolution_y = cfg['render']['resolution_y']
    scene.render.resolution_percentage = 100

    # --- Color Management ---
    # 'Raw' ensures no gamma correction is applied to the coordinate data
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    
    # --- Output Format ---
    # OPEN_EXR Float (Full 32-bit) with RGBA to store transparency
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGBA' # Changed from RGB to RGBA
    scene.render.image_settings.exr_codec = 'ZIP' 

# ==========================================
# 2. SHADER MAGIC
# ==========================================

def create_world_position_material():
    """Creates a material that emits the World Position as color."""
    mat_name = "WorldPos_Override"
    
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])
        
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Geometry Node (Source of coordinates)
    node_geo = nodes.new(type='ShaderNodeNewGeometry')
    node_geo.location = (-400, 0)
    
    # Emission Node (Passes data purely without shading)
    node_emit = nodes.new(type='ShaderNodeEmission')
    node_emit.location = (-200, 0)
    
    # Output Node
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (0, 0)
    
    # Link: Geometry[Position] -> Emission[Color] -> Output[Surface]
    links.new(node_geo.outputs['Position'], node_emit.inputs['Color'])
    links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    
    return mat

# ==========================================
# 3. SCENE MANIPULATION
# ==========================================

def create_plane(pc):
    """Creates/Updates the plane geometry."""
    if "Plane" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        bpy.context.object.name = "Plane"
    
    plane = bpy.data.objects["Plane"]
    
    # Dimensions & Location
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    
    # Rotation
    target = mathutils.Vector((
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.cos(deg_to_rad(pc["azimuth_deg"])),
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.sin(deg_to_rad(pc["azimuth_deg"])),
        math.cos(deg_to_rad(90 - pc["elevation_deg"]))
    ))
    
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = mathutils.Vector((0,0,1)).rotation_difference(target)
    
    return plane

def get_conf_string(pc):
    """Generates the configuration folder name string."""
    area = pc["length_cm"] * pc["breadth_cm"]
    return f"{pc['azimuth_deg']:.2f}_{pc['elevation_deg']:.2f}_{pc['center_cm'][0]:.2f}_{pc['center_cm'][1]:.2f}_{pc['center_cm'][2]:.2f}_{area:.2f}"

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():
    # --- UPDATE PATH HERE ---
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\rndr_px_gbl_cord_cfg.json"
    
    cfg = load_config(CONFIG_PATH)
    setup_render_environment(cfg)
    
    # Create and apply the data shader
    pos_mat = create_world_position_material()
    bpy.context.scene.view_layers[0].material_override = pos_mat
    
    base_output = cfg['paths']['output_dir']
    res_x = cfg['render']['resolution_x']
    res_y = cfg['render']['resolution_y']
    
    # Create the resolution-specific folder name
    res_folder_name = f"{res_x}_{res_y}"
    
    print(f"\nSTARTING GEOMETRY EXPORT ({res_folder_name})...")
    print("="*60)
    
    for pc in cfg['plane_configs']:
        # 1. Setup Plane
        create_plane(pc)
        bpy.context.view_layer.update()
        
        # 2. Generate Paths
        conf_name = get_conf_string(pc)
        
        # Structure: Base / Resolution / Configuration / File
        output_folder = os.path.join(base_output, res_folder_name, conf_name)
        output_file = os.path.join(output_folder, "world_position.exr")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 3. Render
        print(f"Rendering: {conf_name}")
        bpy.context.scene.render.filepath = output_file
        bpy.ops.render.render(write_still=True)
        print(f"Saved to: .../{res_folder_name}/{conf_name}/world_position.exr")

    # Cleanup
    bpy.context.scene.view_layers[0].material_override = None
    print("="*60)
    print("GEOMETRY EXPORT COMPLETE")

if __name__ == "__main__":
    main()