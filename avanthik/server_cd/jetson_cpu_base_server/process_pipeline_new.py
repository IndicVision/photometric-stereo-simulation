"""
process_pipeline.py
Glue between the Jetson server and the masking + reconstruction codes.

Flow:
  1. Phone sends: 4 PNGs + masking_meta.json + reconstruction_meta.json
  2. translate_phone_jsons converts them to mask_cfg and surf_cfg
  3. Masking runs using mask_cfg
  4. Reconstruction runs using surf_cfg
  5. HTML path returned to server, which sends it back to phone
"""

import os
import matplotlib
matplotlib.use('Agg') # Forces headless mode, preventing Tkinter thread crashes
import json
import gc
import cupy as cp
from pathlib import Path
import zipfile

# ── Static Rig Lights — matches your physical product ────────────────────────
RIG_LIGHTS = [
    {
        "id": "L001", "file_name": "light_001.png", "shape": "RECT",
        "pos_m": [0.09, 0.0, 0.03325], "dims_m": [0.0536, 0.0536],
        "norm_dir": [-1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": [100, 100], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L002", "file_name": "light_002.png", "shape": "RECT",
        "pos_m": [0, 0.09, 0.03325], "dims_m": [0.0536, 0.0536],
        "norm_dir": [0.0, -1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": [100, 100], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L003", "file_name": "light_003.png", "shape": "RECT",
        "pos_m": [-0.09, 0.0, 0.03325], "dims_m": [0.0536, 0.0536],
        "norm_dir": [1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": [100, 100], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L004", "file_name": "light_004.png", "shape": "RECT",
        "pos_m": [0.0, -0.09, 0.03325], "dims_m": [0.0536, 0.0536],
        "norm_dir": [0.0, 1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": [100, 100], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    }
]


# ── JSON Translator ───────────────────────────────────────────────────────────
def translate_phone_jsons(session_dir, m_meta, r_meta):
    session_path = Path(session_dir).resolve()

    mask_output_dir = str(session_path / "mask_output")
    surf_output_dir = str(session_path / "surf_output")
    csv_path        = str(session_path / "mask_output" / "valid_pixels_hybrid_u2netp.csv")

    # 1. ALWAYS create the Masking config (m_meta is always guaranteed)
    mask_cfg = {
        "output_dir":        mask_output_dir,
        "image_paths":       [str(session_path / img) for img in m_meta["image_paths"]],
        "world_coordinates": m_meta["world_coordinates"],
        "generate_csv":      m_meta["generate_csv"],
        "return_mask_image": m_meta.get("return_mask_image", True),
    }

    # 2. ONLY build the Surface config if r_meta exists
    surf_cfg = None
    if r_meta:
        cam      = r_meta["camera_settings"]
        img_w    = cam["image_width"]
        img_h    = cam["image_height"]
        f_mm     = cam["focal_length_mm"]
        obj_dist = cam["object_distance_m"]

        dim1 = cam["sensor_width_mm"]
        dim2 = cam["sensor_height_mm"]
        if img_w > img_h:
            actual_sensor_w = max(dim1, dim2)
        else:
            actual_sensor_w = min(dim1, dim2)

        surf_cfg = {
            "paths": {
                "output_dir":           surf_output_dir,
                "world_coordinate_csv": csv_path,
                "image_dir":            str(session_path),
            },
            "global_settings": {
                "csv_from":              "MskHom",
                "apply_dark_threshold":  True,
                "dark_threshold_value":  0.03,
                "min_valid_lights":      3,
                "detrending_mode":       "quadratic",
                "z_exaggeration":        3.0,
                "vis_median_kernel":     5,
                "mask_erosion_pixels":   3,
            },
            "camera": {
                "use_auto_center":     True,
                "manual_center_pixel": [img_w // 2, img_h // 2],
                "focal_length_mm":     f_mm,
                "sensor_width_mm":     actual_sensor_w,
                "object_distance_m":   obj_dist,
                "object_elevation_m":  m_meta["world_coordinates"][2],
            },
            "plane": {
                "dimensions_cm": [],
                "elevation_deg": 90.0,
                "azimuth_deg":   0.0,
            },
            "resolution": {
                "width":  img_w,
                "height": img_h,
            },
            "lights":                RIG_LIGHTS,
            "max_iterations":        15,
            "convergence_threshold": 1e-5,
        }

    return mask_cfg, surf_cfg


# ── Main pipeline entry point ─────────────────────────────────────────────────
def run_pipeline(png_files, json_files, save_dir):
    # ── Step 0: Parse the phone JSONs ─────────────────────────────────────────
    m_meta = None
    r_meta = None

    for f in json_files:
        with open(f, "r") as fp:
            data = json.load(fp)
        pt = data.get("pipeline_type", "")
        if pt == "background_removal":
            m_meta = data
        elif pt == "photometric_stereo":
            r_meta = data

    if m_meta is None:
        raise ValueError("masking_meta.json not found or missing pipeline_type='background_removal'")
    # if r_meta is None:
    #    raise ValueError("reconstruction_meta.json not found or missing pipeline_type='photometric_stereo'")

    # ── Step 1: Translate to pipeline configs ─────────────────────────────────
    print("\n[PIPELINE] -- Step 1: Translating phone JSONs ------------------")
    mask_cfg, surf_cfg = translate_phone_jsons(save_dir, m_meta, r_meta)
    if surf_cfg:
        print("[PIPELINE] surf_output_dir : {}".format(surf_cfg["paths"]["output_dir"]))
        print("[PIPELINE] sensor_width_mm : {}".format(surf_cfg["camera"]["sensor_width_mm"]))
        print("[PIPELINE] resolution      : {} x {}".format(
            surf_cfg["resolution"]["width"], surf_cfg["resolution"]["height"]))
    else:
        print("[PIPELINE] surf_output_dir : Skipped (Masking Only)")
    

    # ── Step 2: Masking ───────────────────────────────────────────────────────
    print("\n[PIPELINE] -- Step 2: Masking ----------------------------------")
    from masking import HybridMaskerU2NetP
    masker = HybridMaskerU2NetP(mask_cfg)
    masker.process()

    # ── FREE all memory before reconstruction ─────────────────────────────────
    print("[PIPELINE] Freeing VRAM and RAM after masking...")
    del masker
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("[PIPELINE] VRAM freed. Ready for reconstruction.")

    if surf_cfg:
        csv_path = surf_cfg["paths"]["world_coordinate_csv"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Masking finished but CSV not found at: {}".format(csv_path)
            )
    print("[PIPELINE] Masking complete")

    # ── Step 3: Reconstruction (ONLY IF REQUESTED) ────────────────────────────
    html_path = None
    if r_meta:
        print("\n[PIPELINE] -- Step 3: Reconstruction ---------------------------")
        from reconstruction_pyamg import AutoIterativePipeline
        pipeline  = AutoIterativePipeline(surf_cfg)
        html_path = pipeline.run()
    else:
        print("\n[PIPELINE] -- Skipping Reconstruction (Masking Only) -----------")

    # ── Step 4: Zip Requested Outputs ─────────────────────────────────────────
    import zipfile
    zip_path = os.path.join(save_dir, "results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        
        # 1. Always check for Mask Image
        if m_meta.get("return_mask_image", False):
            mask_img = os.path.join(mask_cfg["output_dir"], "2_hybrid_final_mask.png")
            if os.path.exists(mask_img): zipf.write(mask_img, arcname="mask.png")
            
        # 2. Check for Reconstruction Outputs (If they exist)
        if r_meta:
            req_outs = r_meta.get("requested_outputs", {})
            if html_path and os.path.exists(html_path):
                iter_dir = os.path.dirname(html_path)
                
                if req_outs.get("3d_surface_html", False):
                    zipf.write(html_path, arcname="surface_3d.html")
                if req_outs.get("depth_map_png", False):
                    depth_img = os.path.join(iter_dir, "2D_depth_map.png")
                    if os.path.exists(depth_img): zipf.write(depth_img, arcname="depth_map.png")
                if req_outs.get("normal_map_png", False):
                    normal_img = os.path.join(iter_dir, "normal_map.png")
                    if os.path.exists(normal_img): zipf.write(normal_img, arcname="normal_map.png")

    print("[PIPELINE] Zipped results to: {}".format(zip_path))
    return str(zip_path)
