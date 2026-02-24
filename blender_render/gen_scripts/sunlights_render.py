import bpy
import os
import json
import numpy as np


class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            if isinstance(v, dict) and k in v: v = v[k]
            else: return None
        return v

JSON_PATH = r"D:\\Chandana\\Photometric_Stereo\\photometric_stereo_simulation\\blender_render\\scripts\\sunlights_render.json"


def apply_render_settings(cfg):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples        = cfg.get("render", "samples") or 128
    scene.render.resolution_x   = cfg.get("render", "resolution_x") or 1920
    scene.render.resolution_y   = cfg.get("render", "resolution_y") or 1080
    scene.render.resolution_percentage = 100

    prefs  = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = cfg.get("render", "compute_device") or 'CUDA'
        for d in cprefs.devices: d.use = True
    except: pass
    scene.cycles.device = cfg.get("render", "device") or 'GPU'

    photo = cfg.get("photometric_settings") or {}
    mb = photo.get("max_bounces", 0)
    scene.cycles.max_bounces          = mb
    scene.cycles.diffuse_bounces      = mb
    scene.cycles.glossy_bounces       = mb
    scene.cycles.transmission_bounces = mb
    if photo.get("disable_caustics", True):
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False

    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look            = 'None'
    scene.view_settings.gamma           = 1.0
    scene.render.dither_intensity       = photo.get("dither_intensity", 0.0)

# ==========================================
#         EXPOSURE → PNG
# ==========================================
def save_exposed_pngs(scene, exr_path, exposures, png_dir, id_str, bit_depth):
    if not os.path.exists(exr_path):
        print(f"  [WARN] EXR not found: {exr_path}")
        return

    img = bpy.data.images.load(exr_path, check_existing=False)
    w, h = img.size
    raw = np.array(img.pixels[:], dtype=np.float32)
    bpy.data.images.remove(img)

    buf_name = "__ExportBuffer__"
    if buf_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[buf_name])
    buf = bpy.data.images.new(buf_name, w, h, alpha=True)

    rs = scene.render.image_settings
    prev_fmt, prev_depth = rs.file_format, rs.color_depth
    rs.file_format = 'PNG'
    rs.color_depth = str(bit_depth)

    for exp in exposures:
        arr = raw.reshape(-1, 4).copy()
        arr[:, :3] *= exp
        np.clip(arr, 0.0, 1.0, out=arr)
        buf.pixels.foreach_set(arr.flatten())
        out_path = os.path.join(png_dir, f"light_{id_str}_exp_{exp:.4f}.png")
        buf.save_render(filepath=out_path, scene=scene)
        print(f"      PNG saved: {out_path}")

    bpy.data.images.remove(buf)
    rs.file_format = prev_fmt
    rs.color_depth = prev_depth

# ==========================================
#         MAIN RENDER LOOP
# ==========================================
def main():
    if not os.path.exists(JSON_PATH):
        print(f"CRITICAL: Config not found → {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        cfg = Config(json.load(f))

    apply_render_settings(cfg)

    scene    = bpy.context.scene
    out_cfg  = cfg.get("output") or {}
    save_exr = out_cfg.get("save_exr", True)
    save_png = out_cfg.get("save_png", True)
    bit_depth = out_cfg.get("png_bit_depth", 16)
    base_out  = out_cfg.get("base_output_dir") or "output"

    # Build exposure list
    start = cfg.get("exposure", "start") or 1.0
    end   = cfg.get("exposure", "end")   or start
    step  = cfg.get("exposure", "step")  or 1.0
    if start == end or step <= 0:
        exposures = [start]
    else:
        count     = int(round((end - start) / step)) + 1
        exposures = [round(start + i * step, 6) for i in range(count)]

    # Collect all SUN lights in the scene
    sun_lights = [obj for obj in scene.objects if obj.type == 'LIGHT' and obj.data.type == 'SUN']
    if not sun_lights:
        print("CRITICAL: No SUN lights found in the scene. Aborting.")
        return

    print(f"Found {len(sun_lights)} sun light(s): {[l.name for l in sun_lights]}")

    # Output folders
    exr_dir = os.path.join(base_out, "exr_raw_noise")
    png_dir = os.path.join(base_out, "png_exposed_noise")
    os.makedirs(exr_dir, exist_ok=True)
    if save_png: os.makedirs(png_dir, exist_ok=True)

    # Hide all lights to start
    for l in sun_lights:
        l.hide_render = True

    # Render one light at a time
    for i, light in enumerate(sun_lights):
        id_str   = f"{(i+1):03d}"          # 001, 002, … (or rename to match your IDs)
        exr_path = os.path.join(exr_dir, f"light_{id_str}.exr")

        # Isolate this light
        light.hide_render = False
        print(f"\n--- Rendering: {light.name} (id={id_str}) ---")

        # Render to EXR
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = exr_path
        bpy.ops.render.render(write_still=True)
        print(f"    EXR saved: {exr_path}")

        # Generate PNGs from EXR
        if save_png:
            save_exposed_pngs(scene, exr_path, exposures, png_dir, id_str, bit_depth)

        # Remove EXR if not requested
        if not save_exr and os.path.exists(exr_path):
            os.remove(exr_path)
            print(f"    EXR deleted (save_exr=false)")

        # Hide this light before moving to next
        light.hide_render = True

    # Restore: show all lights
    for l in sun_lights:
        l.hide_render = False

    print("\n=== Render Complete ===")

if __name__ == "__main__":
    main()