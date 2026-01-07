import bpy
import math
import os
import mathutils
import json
import pandas as pd
import numpy as np


# =========================
# CONFIG HELPERS
# =========================

class Config:
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, *keys):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found at: {keys}")
        return value


class CustomConfig:
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, *keys):
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found at: {keys}")
        return value

    @property
    def csv_path(self):
        return self.get('paths', 'exposure_csv_path')

    @property
    def ev_methods(self):
        return self.get('exposure', 'ev_methods')

    @property
    def base_output_dir(self):
        return self.get('paths', 'base_output_dir')


# =========================
# UTILS
# =========================

def cm_to_m(cm):
    return cm / 100.0


def deg_to_rad(deg):
    return math.radians(deg)


# =========================
# RENDER SETUP (ONCE)
# =========================

def setup_render_settings_once():
    scene = bpy.context.scene
    scene.render.use_persistent_data = True

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.color_mode = 'RGB'


# =========================
# PLANE
# =========================

def create_plane(plane_config):
    length_m = cm_to_m(plane_config['length_cm'])
    breadth_m = cm_to_m(plane_config['breadth_cm'])
    center_m = [cm_to_m(c) for c in plane_config['center_cm']]

    if "Plane" in bpy.data.objects:
        plane = bpy.data.objects["Plane"]
    else:
        bpy.ops.mesh.primitive_plane_add(
            size=1.0,
            enter_editmode=False,
            align='WORLD'
        )
        plane = bpy.context.object
        plane.name = "Plane"

    plane.dimensions.x = length_m
    plane.dimensions.y = breadth_m
    plane.location = center_m

    zenith = deg_to_rad(90.0 - plane_config['elevation_deg'])
    azimuth = deg_to_rad(plane_config['azimuth_deg'])

    target_dir = mathutils.Vector((
        math.sin(zenith) * math.cos(azimuth),
        math.sin(zenith) * math.sin(azimuth),
        math.cos(zenith)
    ))

    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(target_dir)

    return plane


def get_plane_info(plane_config):
    plane = bpy.data.objects.get("Plane")
    plane_center = plane.matrix_world @ mathutils.Vector((0, 0, 0))

    n_zenith = deg_to_rad(90.0 - plane_config['elevation_deg'])
    n_azimuth = deg_to_rad(plane_config['azimuth_deg'])

    n_vector = mathutils.Vector((
        math.sin(n_zenith) * math.cos(n_azimuth),
        math.sin(n_zenith) * math.sin(n_azimuth),
        math.cos(n_zenith)
    ))

    area_cm2 = plane_config['length_cm'] * plane_config['breadth_cm']
    return plane_center, n_vector, area_cm2


# =========================
# LIGHTS
# =========================

def calculate_light_position(plane_center, n_vector, num_lights, light_distance_m, psi_rad):
    positions = []

    for i in range(1, num_lights + 1):
        theta = (i - 1) * (2 * math.pi / num_lights)

        A = n_vector.x * math.cos(theta) + n_vector.y * math.sin(theta)
        B = n_vector.z
        R = math.sqrt(A * A + B * B)

        clamped = min(max(math.cos(psi_rad) / R, -1.0), 1.0)
        phi = math.asin(clamped) - math.atan2(B, A)

        loc = mathutils.Vector((
            light_distance_m * math.sin(phi) * math.cos(theta),
            light_distance_m * math.sin(phi) * math.sin(theta),
            light_distance_m * math.cos(phi)
        ))

        positions.append(loc + plane_center)

    return positions


def create_lights_and_setup(light_type, plane_center, params, positions):
    for l in bpy.data.lights:
        bpy.data.lights.remove(l)

    for i, pos in enumerate(positions):
        light_data = bpy.data.lights.new(
            name=f"light.{i+1:03d}",
            type=light_type
        )

        light_data.energy = params['energy']
        val = params['light_prop_value']

        if light_type == 'AREA':
            light_data.shape = params['shape']
            light_data.size = cm_to_m(params['dim_a_cm'])
            light_data.size_y = cm_to_m(params['dim_b_cm'])
            if val is not None:
                light_data.spread = deg_to_rad(val)

        elif light_type == 'SUN' and val is not None:
            light_data.angle = deg_to_rad(val)

        elif light_type == 'SPOT' and val is not None:
            light_data.spot_size = deg_to_rad(val)

        elif light_type == 'POINT' and val is not None:
            light_data.shadow_soft_size = cm_to_m(val)

        obj = bpy.data.objects.new(light_data.name, light_data)
        bpy.context.collection.objects.link(obj)

        obj.location = pos
        obj.rotation_euler = (plane_center - obj.location).to_track_quat('-Z', 'Y').to_euler()


# =========================
# RENDERING
# =========================

def render_with_exposure(scene, lights, active_light, exposure, output_path):
    for l in lights:
        l.hide_render = True

    active_light.hide_render = False

    scene.cycles.film_exposure = exposure
    scene.render.filepath = output_path

    bpy.ops.render.render(write_still=True)


# =========================
# MAIN
# =========================

def main():
    with open(
        r"C:\Users\vishn\Desktop\avanthik\blender_code\general_rendering\blender_parameters_fast_3.json",
        'r'
    ) as f:
        cfg = Config(json.load(f))

    with open(
        r"C:\Users\vishn\Desktop\avanthik\blender_code\general_rendering\render_best_ev_fast_config.json",
        'r'
    ) as f:
        custom_cfg = CustomConfig(json.load(f))

    setup_render_settings_once()

    df = pd.read_csv(custom_cfg.csv_path)
    plane_configs = cfg.get('plane_configs')
    ev_methods = custom_cfg.ev_methods

    scene = bpy.context.scene

    print("\nDIRECT EXPOSURE RENDERING (BATCHED)")
    print("=" * 60)

    for idx, row in df.iterrows():
        samples = int(row['Samples'])
        config_name = row['Configuration']
        light_setup = row['Light_Setup']

        print(f"\n[{idx+1}/{len(df)}] {config_name} | {light_setup}")

        # ---- Plane ----
        p_parts = config_name.split('_')
        p_config = next(
            (
                c for c in plane_configs
                if f"{c['azimuth_deg']:.2f}" == p_parts[0]
                and f"{c['elevation_deg']:.2f}" == p_parts[1]
            ),
            None
        )

        if not p_config:
            print("  ⚠ Plane config not found")
            continue

        create_plane(p_config)
        p_center, n_vec, _ = get_plane_info(p_config)

        # ---- Light parsing ----
        parts = light_setup.split('_')
        num_l = int(parts[0])
        dist_cm = float(parts[1])
        l_type = parts[2].upper()
        psi_deg = float(parts[-2])
        energy = float(parts[-1])

        l_params = {
            'energy': energy,
            'light_prop_value': None,
            'dim_a_cm': None,
            'dim_b_cm': None,
            'shape': None
        }

        if l_type == 'AREA':
            l_params['shape'] = parts[3].upper()
            l_params['light_prop_value'] = float(parts[4])
            area_dim = cfg.get('light', 'area_light', 'dimensions')[0]
            l_params['dim_a_cm'] = area_dim['dim_a_cm']
            l_params['dim_b_cm'] = area_dim['dim_b_cm']

        elif l_type == 'POINT':
            l_params['light_prop_value'] = float(parts[4])

        elif l_type == 'SUN':
            l_params['light_prop_value'] = float(parts[3].replace('angle', ''))

        elif l_type == 'SPOT':
            l_params['light_prop_value'] = float(parts[3].replace('beam', ''))

        # ---- Lights ----
        positions = calculate_light_position(
            p_center,
            n_vec,
            num_l,
            cm_to_m(dist_cm),
            deg_to_rad(psi_deg)
        )

        create_lights_and_setup(l_type, p_center, l_params, positions)

        lights = sorted(
            [o for o in bpy.data.objects if o.type == 'LIGHT'],
            key=lambda x: x.name
        )

        scene.cycles.samples = samples

        dest_base = os.path.join(
            custom_cfg.base_output_dir,
            f"samples_{samples}",
            config_name,
            light_setup
        )
        os.makedirs(dest_base, exist_ok=True)

        # ---- Render ----
        for li, light in enumerate(lights):
            for method in ev_methods:
                ev = row[method]
                if pd.isna(ev):
                    continue

                out_path = os.path.join(
                    dest_base,
                    f"{li+1:03d}_{method}.png"
                )

                render_with_exposure(
                    scene,
                    lights,
                    light,
                    float(ev),
                    out_path
                )

        print(f"  ✓ {num_l} lights × {len(ev_methods)} exposures")

    print("\n" + "=" * 70)
    print("ALL RENDERING COMPLETE (SAFE MODE)")
    print("=" * 70)


if __name__ == "__main__":
    main()
