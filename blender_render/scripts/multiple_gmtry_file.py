import json
import csv
import math
import random
import os

def normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag == 0: return [0, 0, -1]
    return [v[0]/mag, v[1]/mag, v[2]/mag]

def main():
    # Path to your master config file
    json_path = r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\scripts\multiple_gmtry_file.json"
    
    with open(json_path, 'r') as f:
        cfg = json.load(f)

    base_coords = cfg["light_rig"]["coordinates_cm"]
    powers = cfg["light_rig"]["calibrated_power_w"]
    l_ids = cfg["light_rig"]["light_ids"]
    z_floor = cfg["light_rig"]["z_floor_cm"]
    
    num_configs = cfg["variations"]["num_random_configs"]
    azi_range = cfg["variations"]["azimuth_range_deg"]
    zen_range = cfg["variations"]["zenith_range_deg"]
    
    # Safely get the distance offset range with a fallback if missing
    dist_range = cfg["variations"].get("radius_offset_cm", [-2.0, 10.0])
    csv_path = cfg["paths"]["csv_output"]
    
    csv_data = []
    print(f"Generating {num_configs} large-variation configurations...")
    
    for config_idx in range(1, num_configs + 1):
        config_name = f"Config_{config_idx:03d}"
        
        for i in range(len(base_coords)):
            # 1. Generate random offsets for Azimuth, Zenith, and Distance (R)
            delta_azi = random.uniform(azi_range[0], azi_range[1])
            delta_zen = random.uniform(zen_range[0], zen_range[1])
            delta_r = random.uniform(dist_range[0], dist_range[1])
            
            x, y, z = base_coords[i]
            
            # 2. Calculate Base Spherical Coordinates
            r = math.sqrt(x**2 + y**2 + z**2)
            theta = math.acos(z / r) if r != 0 else 0.0
            phi = math.atan2(y, x)
            
            # 3. Apply Variations
            new_phi = phi + math.radians(delta_azi)
            
            # Cap zenith at 80 degrees so cos(theta) doesn't approach zero (preventing infinite radius)
            new_theta = max(0.01, min(math.radians(80.0), theta + math.radians(delta_zen)))
            
            # Add the absolute centimeter offset to the radius, ensure it doesn't go below 1cm
            scaled_r = max(1.0, r + delta_r) 
            
            temp_z = scaled_r * math.cos(new_theta)
            
            # 4. Z Boundary Logic: Ensure height never falls below z_floor_cm
            if temp_z < z_floor:
                new_z = z_floor
                # Recalculate radius if we had to push it up to the floor level
                new_r = z_floor / math.cos(new_theta) 
            else:
                new_z = temp_z
                new_r = scaled_r 
                
            # Convert back to Cartesian coordinates
            new_x = new_r * math.sin(new_theta) * math.cos(new_phi)
            new_y = new_r * math.sin(new_theta) * math.sin(new_phi)
            
            # 5. Direction Vector Math
            # Force the light to perfectly target the origin (0,0,0) based on its new position
            dir_x = -new_x
            dir_y = -new_y
            dir_z = -new_z
            
            dir_final = normalize([dir_x, dir_y, dir_z])
            
            csv_data.append({
                "Config_ID": config_name,
                "Light_ID": l_ids[i],
                "Pos_X_cm": round(new_x, 4),
                "Pos_Y_cm": round(new_y, 4),
                "Pos_Z_cm": round(new_z, 4),
                "Dir_X": round(dir_final[0], 5),
                "Dir_Y": round(dir_final[1], 5),
                "Dir_Z": round(dir_final[2], 5),
                "Power_W": powers[i]
            })

    # Ensure output directory exists before writing
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"Success! Non-uniform CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()