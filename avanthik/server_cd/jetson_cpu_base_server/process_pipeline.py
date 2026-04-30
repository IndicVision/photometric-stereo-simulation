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
import json
import gc
import cupy as cp
from pathlib import Path


# ── Static Rig Lights — matches your physical product ────────────────────────
RIG_LIGHTS = [
    {
        "id": "L001", "file_name": "light_001.png", "shape": "RECT",
        "pos_m": [0.09, 0.0, 0.0658], "dims_m": [0.0156, 0.0536],
        "norm_dir": [-1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": [300, 1000], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L002", "file_name": "light_002.png", "shape": "RECT",
        "pos_m": [0, 0.09, 0.0658], "dims_m": [0.0536, 0.0156],
        "norm_dir": [0.0, -1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": [1000, 300], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L003", "file_name": "light_003.png", "shape": "RECT",
        "pos_m": [-0.09, 0.0, 0.0658], "dims_m": [0.0156, 0.0536],
        "norm_dir": [1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": [300, 1000], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L004", "file_name": "light_004.png", "shape": "RECT",
        "pos_m": [0.0, -0.09, 0.0658], "dims_m": [0.0536, 0.0156],
        "norm_dir": [0.0, 1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": [1000, 300], "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    }
]


# ── JSON Translator ───────────────────────────────────────────────────────────
def translate_phone_jsons(session_dir, m_meta, r_meta):
    session_path = Path(session_dir).resolve()

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

    mask_output_dir = str(session_path / "mask_output")
    surf_output_dir = str(session_path / "surf_output")
    csv_path        = str(session_path / "mask_output" / "valid_pixels_hybrid_u2netp.csv")

    mask_cfg = {
        "output_dir":        mask_output_dir,
        "image_paths":       [str(session_path / img) for img in m_meta["image_paths"]],
        "world_coordinates": m_meta["world_coordinates"],
        "generate_csv":      m_meta["generate_csv"],
        "return_mask_image": m_meta.get("return_mask_image", True),
    }

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
    if r_meta is None:
        raise ValueError("reconstruction_meta.json not found or missing pipeline_type='photometric_stereo'")

    # ── Step 1: Translate to pipeline configs ─────────────────────────────────
    print("\n[PIPELINE] -- Step 1: Translating phone JSONs ------------------")
    mask_cfg, surf_cfg = translate_phone_jsons(save_dir, m_meta, r_meta)
    print("[PIPELINE] mask_output_dir : {}".format(mask_cfg["output_dir"]))
    print("[PIPELINE] surf_output_dir : {}".format(surf_cfg["paths"]["output_dir"]))
    print("[PIPELINE] sensor_width_mm : {}".format(surf_cfg["camera"]["sensor_width_mm"]))
    print("[PIPELINE] resolution      : {} x {}".format(
        surf_cfg["resolution"]["width"], surf_cfg["resolution"]["height"]))

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

    csv_path = surf_cfg["paths"]["world_coordinate_csv"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Masking finished but CSV not found at: {}".format(csv_path)
        )
    print("[PIPELINE] Masking complete")

    # ── Step 3: Reconstruction ────────────────────────────────────────────────
    print("\n[PIPELINE] -- Step 3: Reconstruction ---------------------------")
    from reconstruction_pyamg import AutoIterativePipeline
    pipeline  = AutoIterativePipeline(surf_cfg)
    html_path = pipeline.run()

    if html_path is None or not os.path.exists(str(html_path)):
        raise FileNotFoundError(
            "Reconstruction finished but HTML was not produced."
        )

    print("[PIPELINE] Reconstruction complete.  HTML: {}".format(html_path))
    return str(html_path)
