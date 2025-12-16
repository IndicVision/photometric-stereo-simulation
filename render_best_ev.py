# import necessary modules
import bpy
import math
import os
import mathutils
import json
from pathlib import Path
import pandas as pd  # <--- NEW IMPORT for CSV handling

# --- Configuration Class ---
class Config:
    """Configuration class to access JSON values directly"""
    def __init__(self, config_dict):
        self._config = config_dict
    
    def get(self, *keys):
        """Extract nested dictionary values using key path"""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found or path is incorrect at: {keys}")
        return value

# --- NEW Custom Configuration Class ---
class CustomConfig:
    """Configuration class for custom settings (CSV path, methods, etc.)"""
    def __init__(self, config_dict):
        self._config = config_dict

    def get(self, *keys):
        """Extract nested dictionary values using key path"""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found or path is incorrect at: {keys}")
        return value
    
    @property
    def csv_path(self):
        return self.get('paths', 'exposure_csv_path')

    @property
    def ev_methods(self):
        # Returns a list of the CSV column names to use for exposure values
        return self.get('exposure', 'ev_methods')
        
    @property # <--- NEW PROPERTY ADDED HERE
    def base_output_dir(self):
        return self.get('paths', 'base_output_dir')

# --- Helper Functions for Conversion ---
def cm_to_m(cm):
    """Converts centimeters to meters for Blender units"""
    return cm / 100.0

def deg_to_rad(deg):
    """Converts degrees to radians"""
    return math.radians(deg)

# --- Scene Setup Functions ---

# input: plane configuration dictionary
# output: created/updated plane object
def create_plane(plane_config):
    """Creates or updates a rectangular plane with specified size, center, and orientation."""
    
    length_m = cm_to_m(plane_config['length_cm'])
    breadth_m = cm_to_m(plane_config['breadth_cm'])
    center_m = [cm_to_m(c) for c in plane_config['center_cm']]
    azimuth_rad = deg_to_rad(plane_config['azimuth_deg'])
    elevation_rad = deg_to_rad(plane_config['elevation_deg'])
    
    # Check for existing plane
    if "Plane" in bpy.data.objects:
        plane = bpy.data.objects["Plane"]
    else:
        # Create a new plane mesh. Using size=1.0 for predictable initial state.
        bpy.ops.mesh.primitive_plane_add(size=1.0, enter_editmode=False, align='WORLD')
        plane = bpy.context.object
        plane.name = "Plane"

    # Set dimensions and location using plane.dimensions for direct size control
    plane.dimensions.x = length_m
    plane.dimensions.y = breadth_m
    plane.location = center_m
    
    # Set orientation based on normal vector azimuth/elevation
    
    # Calculate target normal vector (Zenith angle is 90 - elevation)
    zenith_angle_rad = deg_to_rad(90.0) - elevation_rad
    
    # Target direction vector (n)
    target_dir = mathutils.Vector((
        math.sin(zenith_angle_rad) * math.cos(azimuth_rad),
        math.sin(zenith_angle_rad) * math.sin(azimuth_rad),
        math.cos(zenith_angle_rad)
    ))
    
    # Vector of default plane normal (0, 0, 1)
    default_normal = mathutils.Vector((0.0, 0.0, 1.0))
    
    # Calculate rotation quaternion needed to align default_normal to target_dir
    rot_quat = default_normal.rotation_difference(target_dir)
    plane.rotation_mode = 'QUATERNION'
    plane.rotation_quaternion = rot_quat
    
    return plane

# input: plane configuration dictionary
# output: plane center (world space), normal vector (world space), area (cm^2)
def get_plane_info(plane_config):
    """Calculates the plane's center and the normalized normal vector in world space."""
    plane = bpy.data.objects.get("Plane")
    if not plane:
        raise RuntimeError("Plane object not found. Run create_plane first.")
        
    plane_center = plane.matrix_world @ mathutils.Vector((0.0, 0.0, 0.0))
    
    # Calculate target normal vector (Zenith angle is 90 - elevation)
    n_azimuth_rad = deg_to_rad(plane_config['azimuth_deg'])
    n_elevation_rad = deg_to_rad(plane_config['elevation_deg'])
    n_zenith_rad = deg_to_rad(90.0) - n_elevation_rad
    
    # Recalculate n_vector from angles for consistency with the derivation
    n_vector = mathutils.Vector((
        math.sin(n_zenith_rad) * math.cos(n_azimuth_rad),
        math.sin(n_zenith_rad) * math.sin(n_azimuth_rad),
        math.cos(n_zenith_rad)
    ))

    # Area calculation: L * B, converted to cm^2 (m^2 * 10000)
    plane_area_cm2 = plane_config['length_cm'] * plane_config['breadth_cm']

    return plane_center, n_vector, plane_area_cm2

# input: plane center (world space), normal vector (world space), number of lights, light distance (m), psi angle (rad)
# output: list of tuples (light world location, theta_i (rad), phi_i (rad))
def calculate_light_position(plane_center, n_vector, num_lights, light_distance_m, psi_rad):
    """
    Calculates the position and orientation of light sources to maintain a constant angle psi
    relative to the plane normal vector (n_vector).
    """
    
    positions = []
    
    for i in range(1, num_lights + 1):
        # 1. Azimuth (theta_i): Evenly distributed around the Z-axis (World Up)
        theta_i_rad = (i - 1) * (2 * math.pi / num_lights)
        
        # 2. Components for the R-Formula
        nx, ny, nz = n_vector.x, n_vector.y, n_vector.z
        
        # A and B depend on the light's azimuth (theta_i) and plane normal (n)
        A = nx * math.cos(theta_i_rad) + ny * math.sin(theta_i_rad)
        B = nz
        
        # Calculate R = sqrt(A^2 + B**2)
        R = math.sqrt(A**2 + B**2)
        
        # Check for geometric constraints
        if R < 1e-6:
             raise RuntimeError("R (sqrt(A^2+B^2)) is near zero. Plane normal is vertical and light is at origin.")
        
        cos_psi_over_r = math.cos(psi_rad) / R
        if abs(cos_psi_over_r) > 1.0 + 1e-6:
             raise RuntimeError(f"No valid light position exists. Requires cos(psi)/R <= 1. Current value: {cos_psi_over_r:.4f}. Try reducing psi or changing the plane normal.")

        # Clamp the value just in case of floating point inaccuracies
        clamped_value = min(max(cos_psi_over_r, -1.0), 1.0)
        
        # 3. Solve for Zenith Angle (phi_i) using the derived equation
        # phi_i = arcsin( cos(psi) / R ) - arctan( B / A )
        
        # Calculate eta = arctan(B / A)
        eta_rad = math.atan2(B, A) # safer than atan(B/A) near A=0
        
        # Calculate the arcsin term
        arcsin_term = math.asin(clamped_value)
        
        # Calculate the light's zenith angle (phi_i)
        phi_i_rad = arcsin_term - eta_rad
        
        # 4. Calculate Cartesian Coordinates
        # Light Source Vector (s_i) based on (phi_i, theta_i) in WORLD space
        light_location = mathutils.Vector((
            light_distance_m * math.sin(phi_i_rad) * math.cos(theta_i_rad),
            light_distance_m * math.sin(phi_i_rad) * math.sin(theta_i_rad),
            light_distance_m * math.cos(phi_i_rad)
        ))
        
        # Translate from origin-relative to plane-center-relative
        world_location = light_location + plane_center
        
        positions.append((world_location, theta_i_rad, phi_i_rad))
        
    return positions

# input: configuration, current light type, plane configuration, loop parameters, light positions
# output: creates lights in the scene
def create_lights_and_setup(cfg, current_light_type, plane_config, loop_params, light_positions):
    """
    Creates lights, applies properties, and positions them based on calculated data.
    """
    # Remove existing lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    light_type = current_light_type
    energy = loop_params['energy']
    light_prop_value = loop_params['light_prop_value']
    dim_a_cm = loop_params['dim_a_cm']
    dim_b_cm = loop_params['dim_b_cm']
    shape = loop_params['shape']
    
    
    for i, (location, theta_i, phi_i) in enumerate(light_positions):
        light_name = f"light.{i+1:03d}"
        light = bpy.data.lights.new(name=light_name, type=light_type)
        
        # Configure light based on type
        if light_type == 'AREA':
            light.shape = shape # Use the specific shape (RECTANGLE or ELLIPSE) from loop_params
            
            if shape == 'RECTANGLE':
                light.size = cm_to_m(dim_a_cm) # Size X
                light.size_y = cm_to_m(dim_b_cm) # Size Y
            elif shape == 'ELLIPSE':
                # Blender uses 'size' for the major diameter/radius in one axis and 'size_y' for the other.
                # For ELLIPSE/DISC, Blender uses 'size' for the radius along the light's X-axis and 'size_y' for Y.
                light.size_x = cm_to_m(dim_a_cm) # Major Axis A
                light.size_y = cm_to_m(dim_b_cm) # Minor Axis B
            
            # Apply spread if provided
            if light_prop_value is not None and hasattr(light, 'spread'):
                light.spread = deg_to_rad(light_prop_value) # spread is in radians
                
        elif light_type == 'SUN':
            if light_prop_value is not None:
                light.angle = deg_to_rad(light_prop_value) # angle is in radians
        elif light_type == 'SPOT':
            if light_prop_value is not None:
                light.spot_size = deg_to_rad(light_prop_value) # spot_size (beam angle) is in radians
        
        # Set common light properties
        light.energy = energy
        
        # Create light object and link to scene
        light_obj = bpy.data.objects.new(name=light_name, object_data=light)
        bpy.context.collection.objects.link(light_obj)
        
        # Set position
        light_obj.location = location
        
        # Point light towards plane center
        direction_vector = plane_config['center_m_vec'] - light_obj.location
        rot_quat = direction_vector.to_track_quat('-Z', 'Y') # Pointing negative Z axis toward the center
        light_obj.rotation_euler = rot_quat.to_euler()

# input: configuration, output path, film exposure, loop parameters, exposure_method_name (NEW)
# output: renders the scene for each light source sequentially
def render(cfg, output_path, film_exposure, loop_params, exposure_method_name):
    """Renders the scene for each light source sequentially."""
    os.makedirs(output_path, exist_ok=True)
    
    # Get current scene
    scene = bpy.context.scene
    
    # Setup Camera (assumes camera is set up externally)
    if not scene.camera:
        camera = bpy.data.objects.get('Camera')
        if camera: scene.camera = camera
    if not scene.camera:
        raise RuntimeError("No camera found in scene. Please add a Camera object.")
    
    # Apply Render and Color Management settings
    scene.render.engine = cfg.get('render', 'engine')
    scene.cycles.device = cfg.get('render', 'device')
    if 'cycles' in bpy.context.preferences.addons:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = cfg.get('render', 'compute_device')
    
    scene.cycles.max_bounces = cfg.get('render', 'max_bounces')
    scene.view_settings.view_transform = cfg.get('color_management', 'view_transform')
    scene.cycles.samples = cfg.get('render', 'samples')
    scene.render.dither_intensity = cfg.get('render', 'dither_intensity')
    scene.render.image_settings.file_format = cfg.get('image', 'format')
    scene.render.image_settings.color_mode = cfg.get('image', 'color_mode')
    scene.render.image_settings.color_depth = cfg.get('image', 'color_depth')
    
    # Set the current film exposure
    scene.cycles.film_exposure = film_exposure
    
    # Get all lights and sort
    lights_to_render = [obj for obj in bpy.data.objects if obj.type == 'LIGHT' and obj.name.startswith('light.')]
    try:
        # Sort by name: light.001, light.002, etc.
        lights_to_render = sorted(lights_to_render, key=lambda x: int(x.name.split('.')[1]))
    except:
        raise RuntimeError("Error sorting lights. Check light naming format (light.001, etc.).")
    
    if not lights_to_render:
        raise RuntimeError("No lights found in scene to render.")

    # Iterate over lights and render one by one
    for i, light in enumerate(lights_to_render):
        # Hide all lights
        for l in lights_to_render:
            l.hide_render = True
        
        # Unhide only the current light
        light.hide_render = False
        
        # Set output file path
        # Naming: <source number>_<method name> (MODIFIED)
        filename = f"{i+1:03d}_{exposure_method_name}.png"
        scene.render.filepath = os.path.join(output_path, filename)
        
        print(f"Rendering {filename} (Exposure: {film_exposure})")
        
        # Perform render (NOTE: uncomment this in actual Blender environment)
        bpy.ops.render.render(write_still=True) 
    
    # Restore lights' visibility after rendering is done
    for l in lights_to_render:
        l.hide_render = False


# --- Main Execution Function ---

def main():
    # --- 1. Load Configuration Files ---
    json_path = Path(r"C:\Users\vishn\Desktop\avanthik\blender_code\blender_parameters.json") # use appropriate path
    custom_json_path = Path(r"C:\Users\vishn\Desktop\avanthik\blender_code\render_best_ev_config.json") # NEW PATH
    
    try:
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        with open(custom_json_path, 'r') as f:
            custom_config_dict = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: JSON file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")
        return

    cfg = Config(config_dict)
    custom_cfg = CustomConfig(custom_config_dict) # NEW Custom config object

    # --- 2. Load and Prepare Exposure Data from CSV ---
    try:
        df = pd.read_csv(custom_cfg.csv_path)
        # Combine Configuration and Light_Setup to form a unique key
        df['Config_Key'] = df['Configuration'] + '__' + df['Light_Setup']
        # Convert the DataFrame to a dictionary for quick lookup: 
        # {'Configuration__Light_Setup': {'Method1': EV1, 'Method2': EV2, ...}, ...}
        ev_data_lookup = df.set_index('Config_Key')[custom_cfg.ev_methods].to_dict('index')
        print(f"Loaded {len(ev_data_lookup)} exposure configurations from CSV.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {custom_cfg.csv_path}")
        return
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return

    # 3. Get Loop Arrays (exposure_values range is no longer needed)
    loops = cfg.get('global_loops')
    light_type_array = loops['light_type_array']
    num_lights_array = loops['num_lights_array']
    psi_array_rad = [deg_to_rad(p) for p in loops['psi_angle_array_deg']]
    distance_array_m = [cm_to_m(d) for d in loops['distance_array_cm']]
    energy_array = loops['energy_array']
    plane_configs = cfg.get('plane_configs')
    
    # --- Start Nested Loops ---
    
    if not cfg.get('execute', 'render_scene') and not cfg.get('execute', 'create_lights'):
        print("Execution flags are set to False. Exiting.")
        return

    # Loop 1: Plane Configuration (Object Orientation)
    for p_config in plane_configs:
        
        # A. Setup the Plane (Object)
        plane = create_plane(p_config)
        plane_center_vec, n_vector, plane_area_cm2 = get_plane_info(p_config)
        
        # Augment config with calculated/converted values
        p_config['center_m_vec'] = plane_center_vec
        
        # Create output folder name for Plane Config
        plane_dir_name = (
            f"{p_config['azimuth_deg']:.2f}_"
            f"{p_config['elevation_deg']:.2f}_"
            f"{p_config['center_cm'][0]:.2f}_"
            f"{p_config['center_cm'][1]:.2f}_"
            f"{p_config['center_cm'][2]:.2f}_"
            f"{plane_area_cm2:.2f}"
        )
        plane_output_path = os.path.join(custom_cfg.base_output_dir, plane_dir_name)
        print(f"\n--- Plane Config: {plane_dir_name} ---")

        # Loop 2: Light Type
        for current_light_type in light_type_array:
            
            # Prepare light property loops based on current light type
            light_prop_loop = []
            
            if current_light_type == 'AREA':
                spreads = cfg.get('light', 'area_light', 'spread_angles_deg')
                dimensions = cfg.get('light', 'area_light', 'dimensions')
                
                for spread in spreads:
                    for dim in dimensions:
                        dim_a_cm = dim['dim_a_cm']
                        dim_b_cm = dim['dim_b_cm']
                        shape = dim['shape']
                        
                        if shape == 'RECTANGLE':
                            area_cm2 = dim_a_cm * dim_b_cm
                            area_name = f"{area_cm2:.2f}"
                        elif shape == 'ELLIPSE':
                            area_cm2 = math.pi * dim_a_cm * dim_b_cm # Area of Ellipse
                            area_name = f"{area_cm2:.2f}"
                        else:
                            area_cm2 = 0.0
                            area_name = f"{0.0}"

                        light_prop_loop.append({
                            'value': spread,
                            'name': f"{shape.lower()}_{spread:.1f}_{area_name}",
                            'dim_a_cm': dim_a_cm, 
                            'dim_b_cm': dim_b_cm,
                            'shape': shape
                        })
                        
            elif current_light_type == 'SUN':
                angles = cfg.get('light', 'sun_light', 'angles_deg')
                for angle in angles:
                    light_prop_loop.append({'value': angle, 'name': f"angle{angle:.3f}", 'dim_a_cm': None, 'dim_b_cm': None, 'shape': None})
            elif current_light_type == 'SPOT':
                beams = cfg.get('light', 'spot_light', 'beam_angles_deg')
                for beam in beams:
                    light_prop_loop.append({'value': beam, 'name': f"beam{beam:.1f}", 'dim_a_cm': None, 'dim_b_cm': None, 'shape': None})
            else:
                # Fallback for unsupported or default light types
                light_prop_loop.append({'value': None, 'name': "default", 'dim_a_cm': None, 'dim_b_cm': None, 'shape': None})
                
            # Loop 3: Number of Lights
            for num_lights in num_lights_array:
                
                # Loop 4: Distance from Plane Center
                for distance_m in distance_array_m:
                    distance_cm = distance_m * 100
                    
                    # Loop 5: Light-Specific Property (Spread/Angle/Beam/Area)
                    for light_prop in light_prop_loop:
                        
                        # Loop 6: Incident Angle (psi)
                        for psi_rad in psi_array_rad:
                            psi_deg = math.degrees(psi_rad)
                            
                            # B. Calculate Light Positions (The Logic from the Derivation)
                            try:
                                light_positions = calculate_light_position(
                                    plane_center_vec, n_vector, num_lights, distance_m, psi_rad
                                )
                            except RuntimeError as e:
                                print(f"Skipping configuration due to geometric constraint error: {e}")
                                continue
                                
                            # Loop 7: Light Energy
                            for energy in energy_array:
                                
                                # --- Final Loop Configuration Data and Key Generation ---
                                loop_params = {
                                    'num_lights': num_lights,
                                    'distance_m': distance_m,
                                    'light_prop_value': light_prop['value'],
                                    'dim_a_cm': light_prop['dim_a_cm'],
                                    'dim_b_cm': light_prop['dim_b_cm'],
                                    'shape': light_prop['shape'],
                                    'psi_rad': psi_rad,
                                    'energy': energy
                                }

                                # C. Create Lights and Setup Scene
                                if cfg.get('execute', 'create_lights'):
                                    create_lights_and_setup(cfg, current_light_type, p_config, loop_params, light_positions)
                                
                                # D. Create Final Output Folder Name
                                # Naming: <no. of source>_<distance (cm)>_<light type>_<prop>_<psi>_<energy>
                                light_prop_str = f"{current_light_type.lower()}_{light_prop['name']}"
                                
                                output_dir_name = (
                                    f"{num_lights}_"
                                    f"{distance_cm:.2f}_"
                                    f"{light_prop_str}_"
                                    f"{psi_deg:.2f}_"
                                    f"{energy:.2f}"
                                )
                                
                                # --- Check if this configuration exists in the CSV ---
                                config_key = f"{plane_dir_name}__{output_dir_name}"
                                
                                if config_key not in ev_data_lookup:
                                    print(f"Configuration not found in CSV: {config_key}. Skipping render.")
                                    continue # Skip to the next Energy loop iteration

                                # Configuration found, proceed with rendering
                                ev_values = ev_data_lookup[config_key]
                                
                                # Ensure the output path exists now that we know we will render
                                output_path = os.path.join(plane_output_path, output_dir_name)
                                os.makedirs(output_path, exist_ok=True)

                                # E. Render Loop (Exposure Method Loop - Replaces Loop 8)
                                if cfg.get('execute', 'render_scene'):
                                    
                                    for method_name in custom_cfg.ev_methods:
                                        exposure = ev_values.get(method_name)
                                        
                                        if exposure is not None:
                                            # Render calls render() for all lights at once for the given exposure
                                            # Pass the method name to the render function for file naming
                                            render(cfg, output_path, float(exposure), loop_params, method_name)
                                        else:
                                            print(f"Warning: Exposure value for method '{method_name}' is missing in CSV data for config {config_key}.")
                                    
                                    print(f"--- Completed: {output_dir_name} (Rendered {len(custom_cfg.ev_methods)} exposures)")

# Run main function
if __name__ == "__main__":
    main()