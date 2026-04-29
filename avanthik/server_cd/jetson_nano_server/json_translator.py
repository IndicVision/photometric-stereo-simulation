import json
import os
from pathlib import Path

# Static Rig Lights - These remain constant
RIG_LIGHTS = [
    {
      "id": "L001", "file_name": "light_001.png", "shape": "RECT",
      "pos_m": [0.09, 0.0, 0.0658], "dims_m": [0.0156, 0.0536],
      "norm_dir": [-1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],        
      "radius_m": 0.006, "sampling": [300, 1000], "gamma": 1.0, "bit_depth": 16, "spread_deg": 180.0
    },
    {
      "id": "L002", "file_name": "light_002.png", "shape": "RECT",
      "pos_m": [0, 0.09, 0.0658], "dims_m": [0.0536, 0.0156],
      "norm_dir": [0.0, -1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
      "radius_m": 0.006, "sampling": [1000, 300], "gamma": 1.0, "bit_depth": 16, "spread_deg": 180.0
    },
    {
      "id": "L003", "file_name": "light_003.png", "shape": "RECT",
      "pos_m": [-0.09, 0.0, 0.0658], "dims_m": [0.0156, 0.0536],
      "norm_dir": [1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
      "radius_m": 0.006, "sampling": [300, 1000], "gamma": 1.0, "bit_depth": 16, "spread_deg": 180.0
    },
    {
      "id": "L004", "file_name": "light_004.png", "shape": "RECT",
      "pos_m": [0.0, -0.09, 0.0658], "dims_m": [0.0536, 0.0156],
      "norm_dir": [0.0, 1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
      "radius_m": 0.006, "sampling": [1000, 300], "gamma": 1.0, "bit_depth": 16, "spread_deg": 180.0
    }
]

def generate_pipeline_configs(session_dir: str):
    session_path = Path(session_dir).resolve()
    
    # 1. Load Android Files (Fails if they don't exist)
    with open(session_path / "masking_meta.json", 'r') as f:
        m_meta = json.load(f)
    with open(session_path / "reconstruction_meta.json", 'r') as f:
        r_meta = json.load(f)

    # 2. Extract Data (No defaults - will crash if key is missing)
    img_w = r_meta["camera_settings"]["image_width"]
    img_h = r_meta["camera_settings"]["image_height"]
    f_mm = r_meta["camera_settings"]["focal_length_mm"]
    obj_dist = r_meta["camera_settings"]["object_distance_m"]
    
    # Android usually reports sensor width as the larger value
    # We ensure we use the correct dimension for the width
    dim1 = r_meta["camera_settings"]["sensor_width_mm"]
    dim2 = r_meta["camera_settings"]["sensor_height_mm"]
    
    # Logic: Match the physical sensor dimension to the pixel dimension
    if img_w > img_h:
        # Landscape: Width is the larger dimension
        actual_sensor_w = max(dim1, dim2)
    else:
        # Portrait: Width is the smaller dimension
        actual_sensor_w = min(dim1, dim2)

    # ---------------------------------------------------------
    # Generate Masking Config (trad_guidedFilter_u2netp_msk_cfg.json)
    # ---------------------------------------------------------
    mask_cfg = {
        "output_dir": str(session_path / "mask_output"),
        "image_paths": [str(session_path / img) for img in m_meta["image_paths"]],
        "world_coordinates": m_meta["world_coordinates"],
        "generate_csv": m_meta["generate_csv"]
    }

    # ---------------------------------------------------------
    # Generate Surface Config (rdm_obj_nrml_pls_surf_cfg.json)
    # ---------------------------------------------------------
    surf_cfg = {
        "paths": {
            "output_dir": str(session_path / "surf_output"),
            "world_coordinate_csv": str(session_path / "mask_output" / "valid_pixels_hybrid_u2netp.csv"),
            "image_dir": str(session_path)
        },
        "global_settings": {
            "csv_from": "MskHom",
            "apply_dark_threshold": True,
            "dark_threshold_value": 0.03,
            "min_valid_lights": 3,
            "detrending_mode": "quadratic",
            "z_exaggeration": 3.0,
            "max_iterations": 15,
            "convergence_threshold": 1e-5
        },
        "camera": {
            "use_auto_center": True,
            "manual_center_pixel": [img_w // 2, img_h // 2],
            "sensor_width_mm": actual_sensor_w,
            "focal_length_mm": f_mm,
            "object_distance_m": obj_dist
        },
        "resolution": {"width": img_w, "height": img_h},
        "lights": RIG_LIGHTS
    }

    # Write the files
    with open(session_path / "mask_cfg.json", 'w') as f:
        json.dump(mask_cfg, f, indent=2)
    with open(session_path / "surf_cfg.json", 'w') as f:
        json.dump(surf_cfg, f, indent=2)

    return True

if __name__ == "__main__":
    try:
        generate_pipeline_configs("./test_session")
        print("✅ SUCCESS: Configs generated with strict data check.")
    except KeyError as e:
        print(f"❌ ERROR: Missing data in JSON: {e}")