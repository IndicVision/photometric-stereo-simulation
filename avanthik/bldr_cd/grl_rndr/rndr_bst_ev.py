import bpy
import math
import os
import mathutils
import json
from pathlib import Path
import pandas as pd
import numpy as np

class Config:
    def __init__(self, config_dict):
        self._config = config_dict
    def get(self, *keys):
        value = self._config
        for key in keys:
            if isinstance(value, dict): value = value[key]
            else: raise KeyError(f"Key '{key}' not found at: {keys}")
        return value

class CustomConfig:
    def __init__(self, config_dict):
        self._config = config_dict
    def get(self, *keys):
        value = self._config
        for key in keys:
            if isinstance(value, dict): value = value[key]
            else: raise KeyError(f"Key '{key}' not found at: {keys}")
        return value
    @property
    def csv_path(self): return self.get('paths', 'exposure_csv_path')
    @property
    def ev_methods(self): return self.get('exposure', 'ev_methods')
    @property
    def base_output_dir(self): return self.get('paths', 'base_output_dir')

def cm_to_m(cm): return cm / 100.0
def deg_to_rad(deg): return math.radians(deg)

def create_plane(plane_config):
    length_m, breadth_m = cm_to_m(plane_config['length_cm']), cm_to_m(plane_config['breadth_cm'])
    center_m = [cm_to_m(c) for c in plane_config['center_cm']]
    if "Plane" in bpy.data.objects: plane = bpy.data.objects["Plane"]
    else:
        bpy.ops.mesh.primitive_plane_add(size=1.0, enter_editmode=False, align='WORLD')
        plane = bpy.context.object
        plane.name = "Plane"
    plane.dimensions.x, plane.dimensions.y = length_m, breadth_m
    plane.location = center_m
    # Match main.py's rotation logic
    zenith = deg_to_rad(90.0 - plane_config['elevation_deg'])
    azimuth = deg_to_rad(plane_config['azimuth_deg'])
    target_dir = mathutils.Vector((math.sin(zenith)*math.cos(azimuth), math.sin(zenith)*math.sin(azimuth), math.cos(zenith)))
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = mathutils.Vector((0,0,1)).rotation_difference(target_dir)
    return plane

def calculate_light_position(plane_center, n_vector, num_lights, light_distance_m, psi_rad):
    positions = []
    for i in range(1, num_lights + 1):
        theta_i = (i - 1) * (2 * math.pi / num_lights)
        A = n_vector.x * math.cos(theta_i) + n_vector.y * math.sin(theta_i)
        B, R = n_vector.z, math.sqrt((n_vector.x*math.cos(theta_i)+n_vector.y*math.sin(theta_i))**2 + n_vector.z**2)
        clamped = min(max(math.cos(psi_rad)/R, -1.0), 1.0)
        phi_i = math.asin(clamped) - math.atan2(B, A)
        loc = mathutils.Vector((light_distance_m*math.sin(phi_i)*math.cos(theta_i), light_distance_m*math.sin(phi_i)*math.sin(theta_i), light_distance_m*math.cos(phi_i)))
        positions.append(loc + plane_center)
    return positions

def create_lights_and_setup(current_light_type, plane_center, loop_params, light_positions):
    # Clear old lights
    for light in bpy.data.lights: bpy.data.lights.remove(light)
    for i, location in enumerate(light_positions):
        light_data = bpy.data.lights.new(name=f"light.{i+1:03d}", type=current_light_type)
        light_data.energy = loop_params['energy']
        val = loop_params['light_prop_value']
        
        if current_light_type == 'AREA':
            light_data.shape = loop_params['shape']
            light_data.size = cm_to_m(loop_params['dim_a_cm'])
            light_data.size_y = cm_to_m(loop_params['dim_b_cm'])
            if val is not None: light_data.spread = deg_to_rad(val)
        elif current_light_type == 'SUN' and val is not None: 
            light_data.angle = deg_to_rad(val)
        elif current_light_type == 'SPOT' and val is not None: 
            light_data.spot_size = deg_to_rad(val)
        elif current_light_type == 'POINT' and val is not None: 
            light_data.shadow_soft_size = cm_to_m(val)
        
        obj = bpy.data.objects.new(name=light_data.name, object_data=light_data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        # Re-point light to center
        obj.rotation_euler = (plane_center - obj.location).to_track_quat('-Z', 'Y').to_euler()

def main():
    # Load configs
    with open(r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\main_cfg.json", 'r') as f:
        cfg = Config(json.load(f))
    with open(r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\rndr_bst_ev_cfg.json", 'r') as f:
        custom_cfg = CustomConfig(json.load(f))

    df = pd.read_csv(custom_cfg.csv_path) #
    plane_configs = cfg.get('plane_configs')

    for _, row in df.iterrows():
        samples = int(row['Samples'])
        config_name, light_setup = row['Configuration'], row['Light_Setup']
        
        # 1. Reconstruct Plane
        p_parts = config_name.split('_')
        # Find matching config from parameters.json
        p_config = next((c for c in plane_configs if f"{c['azimuth_deg']:.2f}" == p_parts[0] and f"{c['elevation_deg']:.2f}" == p_parts[1]), None)
        if not p_config: continue

        plane = create_plane(p_config)
        p_center, n_vec, _ = get_plane_info(p_config)

        # 2. Extract specific light parameters from setup name string
        parts = light_setup.split('_')
        num_l = int(parts[0])
        dist_cm = float(parts[1])
        l_type = parts[2].upper()
        psi_deg = float(parts[-2])
        energy = float(parts[-1])
        
        # Robust parsing logic to handle different light types
        l_params = {'energy': energy, 'light_prop_value': None, 'dim_a_cm': None, 'dim_b_cm': None, 'shape': None}
        if l_type == 'AREA':
            l_params['shape'] = parts[3].upper()
            l_params['light_prop_value'] = float(parts[4]) # spread
            # Area lights use dimensions from blender_parameters.json
            area_dim = cfg.get('light', 'area_light', 'dimensions')[0] 
            l_params['dim_a_cm'] = area_dim['dim_a_cm']
            l_params['dim_b_cm'] = area_dim['dim_b_cm']
        elif l_type == 'POINT':
            l_params['light_prop_value'] = float(parts[4]) # radius
        elif l_type == 'SUN':
            l_params['light_prop_value'] = float(parts[3].replace('angle', ''))
        elif l_type == 'SPOT':
            l_params['light_prop_value'] = float(parts[3].replace('beam', ''))

        # 3. Setup Lights
        l_pos = calculate_light_position(p_center, n_vec, num_l, cm_to_m(dist_cm), deg_to_rad(psi_deg))
        create_lights_and_setup(l_type, p_center, l_params, l_pos)
        
        # 4. Final Render
        dest_base = os.path.join(custom_cfg.base_output_dir, f"samples_{samples}", config_name, light_setup)
        os.makedirs(dest_base, exist_ok=True)

        scene = bpy.context.scene
        scene.cycles.samples = samples
        
        for method in custom_cfg.ev_methods:
            ev = row[method]
            if pd.isna(ev): continue
            
            scene.cycles.film_exposure = float(ev)
            lights = sorted([o for o in bpy.data.objects if o.type=='LIGHT'], key=lambda x: x.name)
            
            for i, light in enumerate(lights):
                # Isolate light
                for l in lights: l.hide_render = True
                light.hide_render = False
                
                scene.render.filepath = os.path.join(dest_base, f"{i+1:03d}_{method}.png")
                bpy.ops.render.render(write_still=True)

def get_plane_info(plane_config):
    plane = bpy.data.objects.get("Plane")
    plane_center = plane.matrix_world @ mathutils.Vector((0,0,0))
    n_zenith = deg_to_rad(90.0 - plane_config['elevation_deg'])
    n_azimuth = deg_to_rad(plane_config['azimuth_deg'])
    n_vector = mathutils.Vector((math.sin(n_zenith)*math.cos(n_azimuth), math.sin(n_zenith)*math.sin(n_azimuth), math.cos(n_zenith)))
    return plane_center, n_vector, plane_config['length_cm'] * plane_config['breadth_cm']

if __name__ == "__main__":
    main()