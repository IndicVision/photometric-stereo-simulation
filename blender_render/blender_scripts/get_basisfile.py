import os
import json
import bpy
from mathutils import Vector

# =========================================================
# ⚙️ SCRIPT CONFIGURATION
# =========================================================

# The path where this complete JSON file will be saved
OUTPUT_PATH = r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\renders\config_test_5\config_test_5.json"

# The names of the lights in your Blender Outliner
LIGHT_NAMES = ["Light_1", "Light_2", "Light_3", "Light_4"]

# =========================================================
# STATIC JSON DATA (NON-LIGHT SETTINGS)
# =========================================================

final_json = {
    "paths": {
        "world_coordinate_csv": "D:\\Chandana\\Photometric_Stereo\\photometric_stereo_simulation\\blender_render\\renders\\config_test_5\\mask\\object_mapping.csv",
        "image_dir": "D:\\Chandana\\Photometric_Stereo\\photometric_stereo_simulation\\blender_render\\renders\\config_test_5\\exr_raw",
        "output_dir": "D:\\Chandana\\Photometric_Stereo\\photometric_stereo_simulation\\blender_render\\renders\\config_test_5\\analysis"
    },
    "global_settings": {
        "system_units": "m",
        "apply_dark_threshold": False,
        "dark_threshold_value": 0.02,
        "csv_from": "MskHom",
        "_options": ["Blender", "MskHom"]
    },
    "camera": {
        "use_auto_center": True,
        "manual_center_pixel": [1050, 700],
        "_comment": "Auto center assumes [W/2, H/2] is the optical axis."
    },
    "plane": {
        "dimensions_cm": [3.05, 2.15],
        "elevation_deg": 90.0,
        "azimuth_deg": 0.0
    },
    "resolution": {
        "width": 2100,
        "height": 1400
    },
    "lights": [] # We will populate this array below
}


# =========================================================
# EXTRACT AND FLIP LIGHT DATA
# =========================================================

def flip_axes(vec):
    """Applies the coordinate flip: New X = -X, New Y = -Y, New Z = Z"""
    return Vector((-vec.x, -vec.y, vec.z))

for name in LIGHT_NAMES:
    obj = bpy.data.objects.get(name)
    
    if not obj:
        print(f"Warning: Could not find object named '{name}'")
        continue
    if obj.type != 'LIGHT' or obj.data.type != 'AREA':
        print(f"Warning: '{name}' is not an Area Light!")
        continue

    # Get the 4x4 World Matrix
    mw = obj.matrix_world
    
    # 1. Position (Translation)
    pos = mw.to_translation()
    
    # 2. Extract rotation matrix (3x3) to find local axes in global space
    rot_mat = mw.to_3x3()
    
    # In Blender Area Lights:
    # Width is along Local X. Height is along Local Y. Light emits along Local -Z.
    ax_u = (rot_mat @ Vector((1.0, 0.0, 0.0))).normalized()
    ax_v = (rot_mat @ Vector((0.0, 1.0, 0.0))).normalized()
    norm = (rot_mat @ Vector((0.0, 0.0, -1.0))).normalized()
    
    # 3. APPLY THE FLIP
    pos_flipped = flip_axes(pos)
    ax_u_flipped = flip_axes(ax_u)
    ax_v_flipped = flip_axes(ax_v)
    norm_flipped = flip_axes(norm)

    # 4. Grab dimensions directly from Blender to be safe
    dim_x = obj.data.size
    dim_y = obj.data.size_y if obj.data.shape == 'RECTANGLE' else obj.data.size

    # 5. Format the dictionary for the specific light
    light_id_num = name.split('_')[-1] # Extracts "1" from "Light_1"
    
    light_cfg = {
        "id": f"L00{light_id_num}",
        "file_name": f"light_{light_id_num}.exr",
        "shape": "RECT",
        "pos_m": [round(pos_flipped.x, 6), round(pos_flipped.y, 6), round(pos_flipped.z, 6)],
        "norm_dir": [round(norm_flipped.x, 6), round(norm_flipped.y, 6), round(norm_flipped.z, 6)],
        "dims_m": [round(dim_x, 6), round(dim_y, 6)],
        "ax_u": [round(ax_u_flipped.x, 6), round(ax_u_flipped.y, 6), round(ax_u_flipped.z, 6)],
        "ax_v": [round(ax_v_flipped.x, 6), round(ax_v_flipped.y, 6), round(ax_v_flipped.z, 6)],
        "radius_m": 0.006,
        "sampling": [1000, 1000],
        "gamma": 1.0,
        "bit_depth": 1,
        "spread_deg": 180.0
    }
    
    final_json["lights"].append(light_cfg)


# =========================================================
# SAVE TO FILE
# =========================================================

print("\n--- JSON GENERATION COMPLETE ---")

with open(OUTPUT_PATH, 'w') as f:
    json.dump(final_json, f, indent=2) # Changed indent to 2 to match your formatting style

print(f"Full configuration saved successfully to: {OUTPUT_PATH}")