import bpy
import math
import os
import mathutils
import json
from pathlib import Path

class Config:
    def __init__(self, config_dict):
        self._config = config_dict
    def get(self, *keys):
        value = self._config
        for key in keys:
            if isinstance(value, dict): value = value[key]
            else: raise KeyError(f"Key '{key}' not found at: {keys}")
        return value
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
    target_dir = mathutils.Vector((
        math.sin(deg_to_rad(90-plane_config['elevation_deg'])) * math.cos(deg_to_rad(plane_config['azimuth_deg'])),
        math.sin(deg_to_rad(90-plane_config['elevation_deg'])) * math.sin(deg_to_rad(plane_config['azimuth_deg'])),
        math.cos(deg_to_rad(90-plane_config['elevation_deg']))
    ))
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = mathutils.Vector((0,0,1)).rotation_difference(target_dir)
    return plane

def get_plane_info(plane_config):
    plane = bpy.data.objects.get("Plane")
    plane_center = plane.matrix_world @ mathutils.Vector((0,0,0))
    n_zenith = deg_to_rad(90.0 - plane_config['elevation_deg'])
    n_azimuth = deg_to_rad(plane_config['azimuth_deg'])
    n_vector = mathutils.Vector((math.sin(n_zenith)*math.cos(n_azimuth), math.sin(n_zenith)*math.sin(n_azimuth), math.cos(n_zenith)))
    return plane_center, n_vector, plane_config['length_cm'] * plane_config['breadth_cm']

def calculate_light_position(plane_center, n_vector, num_lights, light_distance_m, psi_rad):
    positions = []
    for i in range(1, num_lights + 1):
        theta_i = (i - 1) * (2 * math.pi / num_lights)
        A = n_vector.x * math.cos(theta_i) + n_vector.y * math.sin(theta_i)
        B, R = n_vector.z, math.sqrt((n_vector.x*math.cos(theta_i)+n_vector.y*math.sin(theta_i))**2 + n_vector.z**2)
        clamped = min(max(math.cos(psi_rad)/R, -1.0), 1.0)
        phi_i = math.asin(clamped) - math.atan2(B, A)
        loc = mathutils.Vector((light_distance_m*math.sin(phi_i)*math.cos(theta_i), light_distance_m*math.sin(phi_i)*math.sin(theta_i), light_distance_m*math.cos(phi_i)))
        positions.append((loc + plane_center, theta_i, phi_i))
    return positions

def create_lights_and_setup(cfg, current_light_type, plane_config, loop_params, light_positions):
    for light in bpy.data.lights: bpy.data.lights.remove(light)
    for i, (location, _, _) in enumerate(light_positions):
        light_data = bpy.data.lights.new(name=f"light.{i+1:03d}", type=current_light_type)
        light_data.energy = loop_params['energy']
        val = loop_params['light_prop_value']
        if current_light_type == 'AREA':
            light_data.shape = loop_params['shape']
            light_data.size = cm_to_m(loop_params['dim_a_cm'])
            light_data.size_y = cm_to_m(loop_params['dim_b_cm'])
            if val is not None: light_data.spread = deg_to_rad(val)
        elif current_light_type == 'SUN' and val: light_data.angle = deg_to_rad(val)
        elif current_light_type == 'SPOT' and val: light_data.spot_size = deg_to_rad(val)
        elif current_light_type == 'POINT' and val: light_data.shadow_soft_size = cm_to_m(val)
        
        obj = bpy.data.objects.new(name=light_data.name, object_data=light_data)
        bpy.context.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (plane_config['center_m_vec'] - obj.location).to_track_quat('-Z', 'Y').to_euler()

def render(cfg, output_path, exposure, samples):
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.engine = cfg.get('render', 'engine')
    scene.cycles.film_exposure = exposure
    lights = sorted([o for o in bpy.data.objects if o.type=='LIGHT' and o.name.startswith('light.')], key=lambda x: x.name)
    for i, light in enumerate(lights):
        for l in lights: l.hide_render = True
        light.hide_render = False
        scene.render.filepath = os.path.join(output_path, f"{i+1:03d}_{exposure:.3f}.png")
        bpy.ops.render.render(write_still=True)

def main():
    with open(r"C:\Users\vishn\Desktop\avanthik\blender_code\blender_parameters.json", 'r') as f:
        cfg = Config(json.load(f))
    
    loops = cfg.get('global_loops')
    for samples in loops['samples_array']:
        sample_dir = os.path.join(cfg.base_output_dir, f"samples_{samples}")
        for p_config in cfg.get('plane_configs'):
            create_plane(p_config)
            p_center, n_vec, area = get_plane_info(p_config)
            p_config['center_m_vec'] = p_center
            conf_name = f"{p_config['azimuth_deg']:.2f}_{p_config['elevation_deg']:.2f}_{p_config['center_cm'][0]:.2f}_{p_config['center_cm'][1]:.2f}_{p_config['center_cm'][2]:.2f}_{area:.2f}"
            
            for l_type in loops['light_type_array']:
                prop_loop = []
                if l_type == 'AREA':
                    for s in cfg.get('light','area_light','spread_angles_deg'):
                        for d in cfg.get('light','area_light','dimensions'):
                            prop_loop.append({'val':s, 'name':f"{d['shape'].lower()}_{s:.1f}", 'da':d['dim_a_cm'], 'db':d['dim_b_cm'], 'sh':d['shape']})
                elif l_type == 'POINT':
                    for r in cfg.get('light','point_light','radius_cm'):
                        prop_loop.append({'val':r, 'name':f"radius_{r:.2f}", 'da':None, 'db':None, 'sh':None})
                elif l_type == 'SUN':
                    angles = cfg.get('light', 'sun_light', 'angles_deg')
                    for angle in angles:
                        prop_loop.append({'val': angle, 'name': f"angle{angle:.3f}", 'da': None, 'db': None, 'sh': None})
                elif l_type == 'SPOT':
                    beams = cfg.get('light', 'spot_light', 'beam_angles_deg')
                    for beam in beams:
                        prop_loop.append({'val': beam, 'name': f"beam{beam:.1f}", 'da': None, 'db': None, 'sh': None})
                else:
                    prop_loop.append({'val': None, 'name': "default", 'da': None, 'db': None, 'sh': None})
                    
                for n_l in loops['num_lights_array']:
                    for dist in loops['distance_array_cm']:
                        for prop in prop_loop:
                            for psi in [deg_to_rad(a) for a in loops['psi_angle_array_deg']]:
                                for eng in loops['energy_array']:
                                    l_pos = calculate_light_position(p_center, n_vec, n_l, cm_to_m(dist), psi)
                                    params = {'energy':eng, 'light_prop_value':prop['val'], 'dim_a_cm':prop['da'], 'dim_b_cm':prop['db'], 'shape':prop['sh']}
                                    create_lights_and_setup(cfg, l_type, p_config, params, l_pos)
                                    dest = os.path.join(sample_dir, conf_name, f"{n_l}_{dist:.2f}_{l_type.lower()}_{prop['name']}_{math.degrees(psi):.2f}_{eng:.2f}")
                                    os.makedirs(dest, exist_ok=True)
                                    for ev in [round(cfg.get('exposure','start') + i*cfg.get('exposure','step'), 3) for i in range(int((cfg.get('exposure','end')-cfg.get('exposure','start'))/cfg.get('exposure','step'))+1)]:
                                        render(cfg, dest, ev, samples)

if __name__ == "__main__": main()