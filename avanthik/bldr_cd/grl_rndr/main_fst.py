"""
Simplified Ultra-Optimized Blender Rendering Pipeline
- Renders each light once at high bit-depth
- Applies all 25 exposures in post-processing (25x speedup)
- Simple, reliable approach that definitely works
"""
import bpy
import math
import os
import mathutils
import json
import numpy as np
class Config:
    def __init__(self, cfg):
        self.cfg = cfg
    def get(self, *keys):
        v = self.cfg
        for k in keys:
            v = v[k]
        return v
    @property
    def base_output_dir(self):
        return self.get("paths", "base_output_dir")
def cm_to_m(x): return x / 100.0
def deg_to_rad(x): return math.radians(x)
# ---------------- GPU SETUP ----------------
def setup_gpu(cfg):
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.compute_device_type = cfg.get("render", "compute_device")
    for d in cprefs.devices:
        d.use = True
    scene = bpy.context.scene
    scene.cycles.device = cfg.get("render", "device")
def setup_render_settings(cfg):
    """Apply all render settings once at startup"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "Raw"
    render_cfg = cfg.get("render")
    scene.cycles.max_bounces = render_cfg.get("max_bounces", 0)
    scene.cycles.diffuse_bounces = render_cfg.get("diffuse_bounces", 0)
    scene.cycles.glossy_bounces = render_cfg.get("glossy_bounces", 0)
    scene.cycles.transmission_bounces = render_cfg.get("transmission_bounces", 0)
    scene.cycles.volume_bounces = render_cfg.get("volume_bounces", 0)
    scene.cycles.transparent_max_bounces = render_cfg.get("transparent_max_bounces", 0)
    scene.cycles.use_adaptive_sampling = render_cfg.get("use_adaptive_sampling", False)
    scene.cycles.use_denoising = render_cfg.get("use_denoising", False)
    scene.cycles.sample_clamp_direct = render_cfg.get("sample_clamp_direct", 0.0)
    scene.cycles.sample_clamp_indirect = render_cfg.get("sample_clamp_indirect", 0.0)
    scene.render.dither_intensity = render_cfg.get("dither_intensity", 0.0)
    # CRITICAL: Render at neutral exposure
    scene.cycles.film_exposure = 1.0
    # CRITICAL: Use high bit-depth format
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.exr_codec = 'ZIP'
# ---------------- PLANE ----------------
def create_plane(pc):
    if "Plane" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        bpy.context.object.name = "Plane"
    plane = bpy.data.objects["Plane"]
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    target = mathutils.Vector((
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.cos(deg_to_rad(pc["azimuth_deg"])),
        math.sin(deg_to_rad(90 - pc["elevation_deg"])) * math.sin(deg_to_rad(pc["azimuth_deg"])),
        math.cos(deg_to_rad(90 - pc["elevation_deg"]))
    ))
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = mathutils.Vector((0,0,1)).rotation_difference(target)
def get_plane_info(pc):
    plane = bpy.data.objects["Plane"]
    center = plane.matrix_world @ mathutils.Vector((0,0,0))
    n_zen = deg_to_rad(90 - pc["elevation_deg"])
    n_azi = deg_to_rad(pc["azimuth_deg"])
    normal = mathutils.Vector((
        math.sin(n_zen)*math.cos(n_azi),
        math.sin(n_zen)*math.sin(n_azi),
        math.cos(n_zen)
    ))
    area = pc["length_cm"] * pc["breadth_cm"]
    return center, normal, area
# ---------------- LIGHTS ----------------
def ensure_lights(n, ltype):
    lights = [o for o in bpy.data.objects if o.type == "LIGHT" and o.name.startswith("light.")]
    if len(lights) == n and all(o.data.type == ltype for o in lights):
        return sorted(lights, key=lambda x: x.name)
    for o in lights:
        bpy.data.objects.remove(o, do_unlink=True)
    out = []
    for i in range(n):
        ld = bpy.data.lights.new(f"light.{i+1:03d}", ltype)
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        out.append(obj)
    return out
def calculate_light_positions(center, nvec, n, dist_m, psi):
    pos = []
    for i in range(n):
        theta = i * 2 * math.pi / n
        A = nvec.x * math.cos(theta) + nvec.y * math.sin(theta)
        B = nvec.z
        R = math.sqrt(A*A + B*B)
        phi = math.asin(min(max(math.cos(psi)/R, -1), 1)) - math.atan2(B, A)
        pos.append(center + mathutils.Vector((
            dist_m * math.sin(phi) * math.cos(theta),
            dist_m * math.sin(phi) * math.sin(theta),
            dist_m * math.cos(phi)
        )))
    return pos
def configure_lights_batch(lights, params, positions, center):
    shape = params["shape"]
    size = cm_to_m(params["dim_a_cm"])
    size_y = cm_to_m(params["dim_b_cm"])
    spread = deg_to_rad(params["light_prop_value"])
    energy = params["energy"]
    for obj, pos in zip(lights, positions):
        ld = obj.data
        obj.location = pos
        ld.energy = energy
        ld.shape = shape
        ld.size = size
        ld.size_y = size_y
        ld.spread = spread
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
def get_exposure_values(cfg):
    start = cfg.get("exposure", "start")
    end = cfg.get("exposure", "end")
    step = cfg.get("exposure", "step")
    return [round(start + i * step, 3)
            for i in range(int((end - start) / step) + 1)]
# ---------------- POST-PROCESSING ----------------
def apply_exposure_blender(exr_path, exposure_value, output_path):
    """Apply exposure using Blender's built-in tools"""
    # Load EXR
    img = bpy.data.images.load(exr_path, check_existing=False)
    # Get pixel data
    pixels = np.array(img.pixels[:])
    width, height = img.size
    pixels = pixels.reshape((height, width, 4))  # RGBA
    # Apply exposure to RGB only
    pixels[:, :, :3] *= exposure_value
    pixels = np.clip(pixels, 0, 1)
    # Convert to 8-bit
    pixels_8bit = (pixels * 255).astype(np.uint8)
    # Save as PNG using Blender
    output_img = bpy.data.images.new("temp_output", width, height, alpha=True)
    output_img.pixels = pixels.flatten().tolist()
    output_img.filepath_raw = output_path
    output_img.file_format = 'PNG'
    output_img.save()
    # Cleanup
    bpy.data.images.remove(img)
    bpy.data.images.remove(output_img)
def render_single_light_exr(scene, lights, light_idx, output_path):
    """Render a single light to EXR"""
    # Hide all lights except the target one
    for i, light in enumerate(lights):
        light.hide_render = (i != light_idx)
    # Render
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
def process_configuration(scene, lights, temp_dir, final_dir, exposure_values):
    """
    Render each light once as EXR, then apply all exposures
    This is 25x faster than rendering each exposure
    """
    num_lights = len(lights)
    print(f"\n=== Phase 1: Rendering {num_lights} lights as EXR ===")
    # Step 1: Render each light once at neutral exposure
    exr_files = []
    for idx, light in enumerate(lights):
        exr_path = os.path.join(temp_dir, f"light_{idx+1:03d}.exr")
        print(f"  Rendering light {idx+1}/{num_lights}...", end=" ")
        render_single_light_exr(scene, lights, idx, exr_path)
        exr_files.append(exr_path)
        print("✓")
    print(f"\n=== Phase 2: Applying {len(exposure_values)} exposures to each light ===")
    # Step 2: Apply all exposures to each EXR
    total = num_lights * len(exposure_values)
    count = 0
    for idx, exr_path in enumerate(exr_files):
        print(f"  Processing light {idx+1}/{num_lights}...", end=" ")
        for exposure in exposure_values:
            output_path = os.path.join(final_dir, f"{idx+1:03d}_{exposure:.3f}.png")
            apply_exposure_blender(exr_path, exposure, output_path)
            count += 1
        print(f"✓ ({count}/{total} images)")
    print(f"✓ All {total} images generated")
# ---------------- MAIN ----------------
def main():
    config_path = r"C:\Users\vishn\Desktop\avanthik\blender_code\general_rendering\blender_parameters_fast_3.json"
    with open(config_path) as f:
        cfg = Config(json.load(f))
    setup_gpu(cfg)
    setup_render_settings(cfg)
    scene = bpy.context.scene
    loops = cfg.get("global_loops")
    exposure_values = get_exposure_values(cfg)
    print(f"Configuration: {len(exposure_values)} exposures per light")
    print(f"Speedup: ~{len(exposure_values)}x faster than original\n")
    light_params = {
        "energy": loops["energy_array"][0],
        "light_prop_value": cfg.get("light","area_light","spread_angles_deg")[0],
        "dim_a_cm": cfg.get("light","area_light","dimensions")[0]["dim_a_cm"],
        "dim_b_cm": cfg.get("light","area_light","dimensions")[0]["dim_b_cm"],
        "shape": cfg.get("light","area_light","dimensions")[0]["shape"]
    }
    for samples in loops["samples_array"]:
        scene.cycles.samples = samples
        base_dir = os.path.join(cfg.base_output_dir, f"samples_{samples}")
        for pc in cfg.get("plane_configs"):
            create_plane(pc)
            center, normal, area = get_plane_info(pc)
            conf = f"{pc['azimuth_deg']:.2f}_{pc['elevation_deg']:.2f}_{pc['center_cm'][0]:.2f}_{pc['center_cm'][1]:.2f}_{pc['center_cm'][2]:.2f}_{area:.2f}"
            for n_l in loops["num_lights_array"]:
                lights = ensure_lights(n_l, "AREA")
                for dist in loops["distance_array_cm"]:
                    for psi in loops["psi_angle_array_deg"]:
                        positions = calculate_light_positions(
                            center, normal, n_l, cm_to_m(dist), deg_to_rad(psi)
                        )
                        configure_lights_batch(lights, light_params, positions, center)
                        final_dir = os.path.join(
                            base_dir, conf,
                            f"{n_l}_{dist:.2f}_area_rectangle_{light_params['light_prop_value']:.1f}_{psi:.2f}_{light_params['energy']:.2f}"
                        )
                        temp_dir = os.path.join(final_dir, "_temp_exr")
                        os.makedirs(final_dir, exist_ok=True)
                        os.makedirs(temp_dir, exist_ok=True)
                        print(f"\n{'='*70}")
                        print(f"Configuration: samples={samples}, dist={dist}, psi={psi}")
                        print(f"Output: {final_dir}")
                        print(f"{'='*70}")
                        # Process this configuration
                        process_configuration(scene, lights, temp_dir, final_dir, exposure_values)
                        # Optional: Clean up temp files
                        import shutil
                        shutil.rmtree(temp_dir)
    print("\n" + "="*70)
    print("ALL RENDERING COMPLETE!")
    print("="*70)
if __name__ == "__main__":
    main()
