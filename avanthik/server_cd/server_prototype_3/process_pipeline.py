"""
process_pipeline.py
Glue between the laptop server and the masking + reconstruction codes.

Flow:
  1. Phone sends: 4 PNGs + masking_meta.json + reconstruction_meta.json
  2. translate_phone_jsons converts them to mask_cfg and surf_cfg
  3. Masking runs using mask_cfg
  4. Reconstruction runs using surf_cfg
  5. ZIP of requested outputs returned to server, which sends it back to phone

NOTE: This file was previously called process_pipeline_new.py on the Jetson.
      Renamed to process_pipeline.py to match the import in laptop_server.py.
      The only other change from the Jetson version: the reconstruction import
      now references  recon_w_fallback_iter2  (supports both GPU and CPU).

--- DYNAMIC RIG GEOMETRY (added) ---
When the object is elevated (on a platform) or thick, three things change:

  1. camera-to-surface distance
       object_distance_m = camera_to_base_m - platform_elevation_m - object_thickness_m
       This feeds directly into the pinhole-model dx_meters calculation in
       reconstruction_upd.py (line: dx = sensor_px_size * obj_dist / focal_len).

  2. light z-coordinate relative to the base stays fixed (3.325 cm), but the
     object surface has risen, so the effective light height above the surface is:
       light_z_above_surface = LIGHT_CENTER_Z_M - object_surface_z_m
     We keep pos_m[2] as the absolute z from base (reconstruction_upd.py adds
     object_elevation_m to pixel z_world before the kernel, so the light positions
     must remain in the same absolute frame).  We pass object_surface_z_m as
     object_elevation_m so the kernel correctly shifts pixel coordinates.

  3. visible panel height (dims_m[1]) shrinks because the elevated platform
     physically occludes the bottom portion of each area-light panel.

     Geometry (side view, one light):
       - Panel spans z = [PANEL_BOTTOM_Z, PANEL_TOP_Z] at horizontal dist = LIGHT_XY_M
       - Platform top is at z = platform_elevation_m
       - The platform edge directly blocks everything on the panel below
         z = platform_elevation_m (since the panel is at the same horizontal
         position as the platform edge from the object's perspective).
       - Effective panel bottom  = max(PANEL_BOTTOM_Z, platform_elevation_m)
       - Effective panel height  = PANEL_TOP_Z - effective_panel_bottom
       - New panel center z      = (PANEL_TOP_Z + effective_panel_bottom) / 2
       - New dims_m[1]           = effective panel height  (width dims_m[0] unchanged)
       - New sampling[1]         = round(BASE_SAMPLING * new_height / PANEL_HEIGHT_M)
         (keeps sample density constant regardless of clipping)

Phone JSON new fields (in reconstruction_meta):
  camera_to_base_m      : float  — fixed physical distance from camera to base (m)
  platform_elevation_m  : float  — height of the lifting platform (m), 0 if none
  object_thickness_m    : float  — thickness of the object (m), 0 for flat stamps
"""

import os
import matplotlib
matplotlib.use('Agg')  # Forces headless mode, preventing Tkinter thread crashes
import json
import gc
import zipfile
from pathlib import Path

# CuPy is optional — only imported for the memory-free call after masking
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ── Static Rig Physical Constants ─────────────────────────────────────────────
# These never change — they describe the physical hardware.

LIGHT_XY_M        = 0.09          # horizontal distance of each light from origin (m)
LIGHT_CENTER_Z_M  = 0.0675       # z of each light panel center from base (m)
PANEL_HEIGHT_M    = 0.045        # actual light-emitting panel height = width (m)
PANEL_WIDTH_M     = 0.045        # horizontal panel dimension (m), never changes
CASING_MARGIN_M   = 0.000        # dead border between casing edge and panel edge (m)
                                   # (6.65 - 5.36) / 2 = 0.645 cm ≈ 0.0065 m
PANEL_TOP_Z_M     = LIGHT_CENTER_Z_M + PANEL_HEIGHT_M / 2.0   # 0.03325 + 0.0268 = 0.06005 m
PANEL_BOTTOM_Z_M  = LIGHT_CENTER_Z_M - PANEL_HEIGHT_M / 2.0   # 0.03325 - 0.0268 = 0.00645 m
BASE_SAMPLING     = [100, 100]    # default sampling for full unclipped panel


# ── Static Rig Lights — matches your physical product ────────────────────────
# Used as the TEMPLATE; compute_dynamic_rig() produces the actual per-session list.
RIG_LIGHTS_TEMPLATE = [
    {
        "id": "L001", "file_name": "light_001.png", "shape": "RECT",
        "pos_m": [ -LIGHT_XY_M,  0.0,          LIGHT_CENTER_Z_M],
        "dims_m": [PANEL_WIDTH_M, PANEL_HEIGHT_M],
        "norm_dir": [1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": list(BASE_SAMPLING), "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L002", "file_name": "light_002.png", "shape": "RECT",
        "pos_m": [0.0,          -LIGHT_XY_M,   LIGHT_CENTER_Z_M],
        "dims_m": [PANEL_WIDTH_M, PANEL_HEIGHT_M],
        "norm_dir": [0.0, 1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": list(BASE_SAMPLING), "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L003", "file_name": "light_003.png", "shape": "RECT",
        "pos_m": [LIGHT_XY_M,  0.0,          LIGHT_CENTER_Z_M],
        "dims_m": [PANEL_WIDTH_M, PANEL_HEIGHT_M],
        "norm_dir": [-1.0, 0.0, 0.0], "ax_u": [0.0, 0.0, 1.0], "ax_v": [0.0, 1.0, 0.0],
        "radius_m": 0.006, "sampling": list(BASE_SAMPLING), "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    },
    {
        "id": "L004", "file_name": "light_004.png", "shape": "RECT",
        "pos_m": [0.0,         LIGHT_XY_M,   LIGHT_CENTER_Z_M],
        "dims_m": [PANEL_WIDTH_M, PANEL_HEIGHT_M],
        "norm_dir": [0.0, -1.0, 0.0], "ax_u": [1.0, 0.0, 0.0], "ax_v": [0.0, 0.0, 1.0],
        "radius_m": 0.006, "sampling": list(BASE_SAMPLING), "gamma": 1.0,
        "bit_depth": 16, "spread_deg": 180.0
    }
]


# ── Dynamic Rig Geometry ──────────────────────────────────────────────────────
def compute_dynamic_rig(camera_to_base_m, platform_elevation_m, object_thickness_m):
    """
    Compute all geometry that changes when the object is elevated or thick.

    Parameters
    ----------
    camera_to_base_m     : float  Fixed distance from camera lens to the base (m).
    platform_elevation_m : float  Height of the lifting platform from base (m). 0 = no platform.
    object_thickness_m   : float  Thickness of the object sitting on the platform (m). 0 = flat stamp.

    Returns
    -------
    object_distance_m    : float  Camera-to-object-surface distance for pinhole model (m).
    object_surface_z_m   : float  Absolute z of the object surface from base (m).
                                  Passed as object_elevation_m in the camera config so
                                  reconstruction_upd.py shifts pixel z_world correctly
                                  before feeding to the area-light kernel.
    lights               : list   Deep-copied and adjusted RIG_LIGHTS with corrected
                                  pos_m[2], dims_m[1], and sampling[1] for each light.
    """
    import copy
    import math

    object_surface_z_m = platform_elevation_m + object_thickness_m

    # 1. Camera-to-surface distance (for pinhole dx_meters in reconstruction_upd.py)
    object_distance_m = camera_to_base_m - object_surface_z_m
    if object_distance_m <= 0:
        raise ValueError(
            f"[RIG] Object surface (z={object_surface_z_m*100:.2f} cm) is at or above the "
            f"camera (camera_to_base={camera_to_base_m*100:.2f} cm). "
            f"Check platform_elevation_m and object_thickness_m."
        )

    # 2. Clipped panel geometry
    #    The platform top at z=platform_elevation_m physically blocks the lower
    #    portion of every light panel (panels are at the same horizontal ring as
    #    the platform edge, so the sight-line from the object surface is directly
    #    occluded below z=platform_elevation_m on the panel).
    effective_panel_bottom_z = max(PANEL_BOTTOM_Z_M, object_surface_z_m)
    effective_panel_height   = PANEL_TOP_Z_M - effective_panel_bottom_z
    effective_panel_center_z = (PANEL_TOP_Z_M + effective_panel_bottom_z) / 2.0

    # Clamp: if the platform somehow reaches or exceeds the panel top, something
    # is physically wrong — raise early rather than produce a zero-height panel.
    if effective_panel_height <= 0:
        raise ValueError(
            f"[RIG] Object surface ({object_surface_z_m*100:.2f} cm) reaches or "
            f"exceeds the top of the light panel (panel top = {PANEL_TOP_Z_M*100:.2f} cm). "
            f"The lights would be fully occluded."
        )

    # Proportionally reduce vertical sampling to keep sample density constant.
    clip_ratio           = effective_panel_height / PANEL_HEIGHT_M   # 0 < ratio <= 1.0
    effective_sampling_v = max(1, round(BASE_SAMPLING[1] * clip_ratio))

    # 3. Build adjusted light list (deep copy so the template is never mutated)
    lights = copy.deepcopy(RIG_LIGHTS_TEMPLATE)
    for light in lights:
        light["pos_m"][2]    = effective_panel_center_z   # shift center z up with clip
        light["dims_m"][1]   = effective_panel_height      # shrink vertical extent
        light["sampling"][1] = effective_sampling_v        # proportional sample count

    # ── Diagnostic printout ───────────────────────────────────────────────────
    print("[RIG] Dynamic rig geometry computed:")
    print(f"  camera_to_base_m        = {camera_to_base_m*100:.3f} cm")
    print(f"  platform_elevation_m    = {platform_elevation_m*100:.3f} cm")
    print(f"  object_thickness_m      = {object_thickness_m*100:.3f} cm")
    print(f"  object_surface_z_m      = {object_surface_z_m*100:.3f} cm  (= platform + thickness)")
    print(f"  object_distance_m       = {object_distance_m*100:.3f} cm  (camera to surface)")
    print(f"  panel_bottom_z (orig)   = {PANEL_BOTTOM_Z_M*100:.3f} cm")
    print(f"  panel_top_z             = {PANEL_TOP_Z_M*100:.3f} cm")
    print(f"  effective_panel_bottom  = {effective_panel_bottom_z*100:.3f} cm")
    print(f"  effective_panel_height  = {effective_panel_height*100:.3f} cm  "
          f"(clipped from {PANEL_HEIGHT_M*100:.3f} cm, ratio={clip_ratio:.3f})")
    print(f"  effective_panel_center  = {effective_panel_center_z*100:.3f} cm")
    print(f"  effective_sampling_v    = {effective_sampling_v}  "
          f"(from base {BASE_SAMPLING[1]})")

    return object_distance_m, object_surface_z_m, lights


# ── JSON Translator ───────────────────────────────────────────────────────────
def translate_phone_jsons(session_dir, m_meta, r_meta):
    session_path = Path(session_dir).resolve()

    mask_output_dir = str(session_path / "mask_output")
    surf_output_dir = str(session_path / "surf_output_dd")
    csv_path        = str(session_path / "mask_output" / "valid_pixels_hybrid_u2netp.csv")

    # 1. ALWAYS create the Masking config
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

        dim1 = cam["sensor_width_mm"]
        dim2 = cam["sensor_height_mm"]
        if img_w > img_h:
            actual_sensor_w = max(dim1, dim2)
        else:
            actual_sensor_w = min(dim1, dim2)

        # ── NEW: read elevation/thickness inputs from phone JSON ──────────────
        camera_to_base_m     = float(r_meta.get("camera_to_base_m",     0.10))  # default 10 cm if not provided
        platform_elevation_m = float(r_meta.get("platform_elevation_m", 0.0))
        object_thickness_m   = float(r_meta.get("object_thickness_m",   0.0))

        # ── Compute all dynamic geometry in one place ─────────────────────────
        object_distance_m, object_surface_z_m, dynamic_lights = compute_dynamic_rig(
            camera_to_base_m     = camera_to_base_m,
            platform_elevation_m = platform_elevation_m,
            object_thickness_m   = object_thickness_m,
        )

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
                "saturation_threshold_value": 0.95,
                "min_valid_lights":      3,
                "detrending_mode": r_meta.get("detrending_mode", "none"),

                "z_exaggeration":        1.0,
                "vis_median_kernel":     0,
                "mask_erosion_pixels":   0,
            },
            "camera": {
                "use_auto_center":     True,
                "manual_center_pixel": [img_w // 2, img_h // 2],
                "focal_length_mm":     f_mm,
                "sensor_width_mm":     actual_sensor_w,
                # Pinhole model distance: camera to OBJECT SURFACE (not base)
                "object_distance_m":   object_distance_m,
                # Elevation offset passed to reconstruction kernel so pixel
                # z_world values are shifted to absolute coordinates before
                # the area-light direction vectors are computed.
                "object_elevation_m":  object_surface_z_m,
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
            # Dynamic lights: z-position and vertical dims adjusted for elevation
            "lights":                dynamic_lights,
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

    # ── Step 1: Translate to pipeline configs ─────────────────────────────────
    print("\n[PIPELINE] -- Step 1: Translating phone JSONs ------------------")
    mask_cfg, surf_cfg = translate_phone_jsons(save_dir, m_meta, r_meta)
    with open(os.path.join(save_dir, "mask_cfg.json"), "w") as f: json.dump(mask_cfg, f, indent=4)
    with open(os.path.join(save_dir, "surf_cfg.json"), "w") as f: json.dump(surf_cfg, f, indent=4)
    if surf_cfg:
        print("[PIPELINE] surf_output_dir  : {}".format(surf_cfg["paths"]["output_dir"]))
        print("[PIPELINE] sensor_width_mm  : {}".format(surf_cfg["camera"]["sensor_width_mm"]))
        print("[PIPELINE] object_distance_m: {:.4f} m  (camera to surface)".format(
            surf_cfg["camera"]["object_distance_m"]))
        print("[PIPELINE] object_elevation : {:.4f} m  (surface z from base)".format(
            surf_cfg["camera"]["object_elevation_m"]))
        print("[PIPELINE] resolution       : {} x {}".format(
            surf_cfg["resolution"]["width"], surf_cfg["resolution"]["height"]))
        print("[PIPELINE] light L001 pos_m : {}".format(surf_cfg["lights"][0]["pos_m"]))
        print("[PIPELINE] light L001 dims_m: {}".format(surf_cfg["lights"][0]["dims_m"]))
        print("[PIPELINE] light L001 samp  : {}".format(surf_cfg["lights"][0]["sampling"]))
    else:
        print("[PIPELINE] Reconstruction skipped (Masking Only)")

    # ── Step 2: Masking ───────────────────────────────────────────────────────
    print("\n[PIPELINE] -- Step 2: Masking ----------------------------------")
    from masking import HybridMaskerU2NetP
    masker = HybridMaskerU2NetP(mask_cfg)
    masker.process()

    # ── FREE all memory before reconstruction ─────────────────────────────────
    print("[PIPELINE] Freeing RAM after masking...")
    del masker
    gc.collect()
    if HAS_CUPY:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("[PIPELINE] VRAM freed.")

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
        from reconstruction_upd import AutoIterativePipeline
        pipeline  = AutoIterativePipeline(surf_cfg)
        html_path = pipeline.run()
    else:
        print("\n[PIPELINE] -- Skipping Reconstruction (Masking Only) -----------")

    # ── Step 4: Zip Requested Outputs (Split for Phone vs Desktop) ────────────
    phone_zip   = os.path.join(save_dir, "results_phone.zip")
    desktop_zip = os.path.join(save_dir, "results_desktop.zip")

    with zipfile.ZipFile(phone_zip, 'w') as z_phone, zipfile.ZipFile(desktop_zip, 'w') as z_desk:

        # 0. Inputs (DESKTOP ONLY)
        for png_path in png_files:
            if os.path.exists(png_path):
                z_desk.write(png_path, arcname=os.path.join("inputs", os.path.basename(png_path)))

        # 1. Mask Image (BOTH) & Mask CSV (DESKTOP ONLY)
        if m_meta.get("return_mask_image", False):
            mask_img = os.path.join(mask_cfg["output_dir"], "2_hybrid_final_mask.png")
            if os.path.exists(mask_img):
                z_phone.write(mask_img, arcname="mask.png")
                z_desk.write(mask_img, arcname="mask.png")

        mask_csv = os.path.join(mask_cfg["output_dir"], "valid_pixels_hybrid_u2netp.csv")
        if os.path.exists(mask_csv):
            z_desk.write(mask_csv, arcname="valid_pixels.csv")

        # 2. Reconstruction Outputs
        if r_meta:
            req_outs = r_meta.get("requested_outputs", {})
            if html_path and os.path.exists(html_path):
                iter_dir = os.path.dirname(html_path)

                # 3D Mapping CSV (DESKTOP ONLY)
                import glob
                csv_files = glob.glob(os.path.join(iter_dir, "mapping_iter*.csv"))
                if csv_files:
                    z_desk.write(csv_files[0], arcname="3d_mapping.csv")

                # Visuals (BOTH)
                if req_outs.get("3d_surface_html", False):
                    z_phone.write(html_path, arcname="surface_3d.html")
                    z_desk.write(html_path, arcname="surface_3d.html")
                if req_outs.get("depth_map_png", False):
                    depth_img = os.path.join(iter_dir, "2D_depth_map.png")
                    if os.path.exists(depth_img):
                        z_phone.write(depth_img, arcname="depth_map.png")
                        z_desk.write(depth_img, arcname="depth_map.png")
                if req_outs.get("normal_map_png", False):
                    normal_img = os.path.join(iter_dir, "normal_map.png")
                    if os.path.exists(normal_img):
                        z_phone.write(normal_img, arcname="normal_map.png")
                        z_desk.write(normal_img, arcname="normal_map.png")

    print("[PIPELINE] Zipped lightweight results for phone: {}".format(phone_zip))
    print("[PIPELINE] Zipped heavy debug results for laptop: {}".format(desktop_zip))

    # Return both paths so the server knows how to handle them
    return phone_zip, desktop_zip