import bpy, math, os, mathutils, json
import pandas as pd
import numpy as np

class Config:
    def __init__(self, config_dict): self._config = config_dict
    def get(self, *keys):
        value = self._config
        for key in keys: value = value[key]
        return value

def cm_to_m(cm): return cm / 100.0
def deg_to_rad(deg): return math.radians(deg)

def create_plane(pc):
    if "Plane" not in bpy.data.objects: bpy.ops.mesh.primitive_plane_add(size=1.0)
    plane = bpy.data.objects["Plane"]
    plane.dimensions = (cm_to_m(pc["length_cm"]), cm_to_m(pc["breadth_cm"]), 0.0)
    plane.location = [cm_to_m(c) for c in pc["center_cm"]]
    target = mathutils.Vector((math.sin(deg_to_rad(90-pc["elevation_deg"]))*math.cos(deg_to_rad(pc["azimuth_deg"])), math.sin(deg_to_rad(90-pc["elevation_deg"]))*math.sin(deg_to_rad(pc["azimuth_deg"])), math.cos(deg_to_rad(90-pc["elevation_deg"]))))
    plane.rotation_mode, plane.rotation_quaternion = "QUATERNION", mathutils.Vector((0,0,1)).rotation_difference(target)
    return plane

def create_manual_lights(vectors, center, dist_m, params):
    for l in bpy.data.lights: bpy.data.lights.remove(l)
    lights = []
    for i, v in enumerate(vectors):
        ld = bpy.data.lights.new(name=f"light.{i+1:03d}", type='AREA')
        ld.energy, ld.shape, ld.size, ld.size_y, ld.spread = params['energy'], params['shape'], cm_to_m(params['dim_a_cm']), cm_to_m(params['dim_b_cm']), deg_to_rad(params['spread'])
        obj = bpy.data.objects.new(ld.name, ld)
        bpy.context.collection.objects.link(obj)
        obj.location = center + (mathutils.Vector((v['lx'], v['ly'], v['lz'])) * dist_m)
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        lights.append(obj)
    return lights

def main():
    cfg = Config(json.load(open(r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\non_uni_pos_lit_rndr\main_fst_cfg.json")))
    cust = Config(json.load(open(r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\non_uni_pos_lit_rndr\rndr_bst_ev_fst_cfg.json")))
    bpy.context.scene.render.use_persistent_data, bpy.context.scene.render.image_settings.file_format = True, 'PNG'
    df = pd.read_csv(cust.get('paths', 'exposure_csv_path'))
    
    for _, row in df.iterrows():
        p_parts = row['Configuration'].split('_')
        # MATCHING PLANE BASED ON AZIMUTH, ELEVATION, AND AREA
        pc = next(c for c in cfg.get('plane_configs') if f"{c['azimuth_deg']:.2f}" == p_parts[0] and f"{c['elevation_deg']:.2f}" == p_parts[1] and f"{c['length_cm']*c['breadth_cm']:.2f}" == p_parts[5])
        center = create_plane(pc).matrix_world @ mathutils.Vector((0,0,0))
        
        l_parts = row['Light_Setup'].split('_')
        l_params = {'energy': float(l_parts[-1]), 'shape': cfg.get('light', 'area_light', 'dimensions')[0]['shape'], 'spread': float(cfg.get('light', 'area_light', 'spread_angles_deg')[0]), 'dim_a_cm': cfg.get('light', 'area_light', 'dimensions')[0]['dim_a_cm'], 'dim_b_cm': cfg.get('light', 'area_light', 'dimensions')[0]['dim_b_cm']}
        
        lights = create_manual_lights(cfg.get("manual_light_setup", "vectors"), center, cm_to_m(float(l_parts[1])), l_params)
        dest = os.path.join(cust.get('paths', 'base_output_dir'), f"samples_{int(row['Samples'])}", row['Configuration'], row['Light_Setup'])
        os.makedirs(dest, exist_ok=True)
        
        method_name = cust.get('exposure', 'ev_methods')[0] #
        
        for li, obj in enumerate(lights):
            for l in lights: l.hide_render = True
            obj.hide_render = False
            bpy.context.scene.cycles.film_exposure = float(row[method_name])
            # RESTORED FILENAME STYLE: {ID}_{METHOD}.png
            bpy.context.scene.render.filepath = os.path.join(dest, f"{li+1:03d}_{method_name}.png")
            bpy.ops.render.render(write_still=True)

if __name__ == "__main__": main()