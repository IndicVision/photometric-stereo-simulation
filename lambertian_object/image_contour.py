import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path to your JSON config
config_path = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\image_contour.json"

def load_config(filename):
    with open(filename, "r") as f:
        return json.load(f)


def generate_contours(img, output_file, angle, set_num, light_num, p_start, p_center, p_end):
    # 1. Smooth to remove surface texture noise
    # img_smoothed = cv2.GaussianBlur(img, (31, 31), 0) 
    
    # 2. Level Logic: Ensure the peak is a specific, distinct line
    levels_low = np.linspace(p_start, p_center, 6)
    levels_high = np.linspace(p_center, p_end, 6)
    peak_levels = np.unique(np.concatenate([levels_low, levels_high]))

    # 3. Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray")
    
    # 4. Draw Contours (Object-Oriented ax.contour)
    # Using 'tab20' for high-contrast, distinct colors
    cp = ax.contour(img, levels=peak_levels, cmap="tab20", linewidths=2.5)
    
    # 5. Force Colorbar Height to match Plot Height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cp, cax=cax)
    
    # Force the ticks to match your peak_levels exactly
    cbar.set_ticks(peak_levels)
    cbar.set_ticklabels([f"{int(level)}" for level in peak_levels])
    cbar.set_label(f'Intensity Levels (Peak: {int(p_center)})', fontsize=8)
    
    ax.set_title(f"Contours: Orientation {angle}° | Set {set_num} | Light {light_num}\nRange: {int(p_start)} - {int(p_end)} (Peak) - {int(p_center)}")
    ax.axis("off")
    
    # 6. Save and clean up memory
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    


def process_sample(config):
    base_dir = Path(config["base_path"])
    out_dir = Path(config["output_path"])
    
    # Load CSVs
    sample_df = pd.read_csv(config["csv_path"])
    meas_df = pd.read_csv(config["csv_measurements"])
    
    # Normalize inputs to lists (Input normalization pattern)
    sample_list = config["sample_num"] if isinstance(config["sample_num"], list) else [config["sample_num"]]
    set_list = config["set_num"] if isinstance(config["set_num"], list) else [config["set_num"]]
    light_list = config["light_num"] if isinstance(config["light_num"], list) else [config["light_num"]]
    angle_list = config["orientation"]["angle_range"]

    for sample_id in sample_list:
        # Get metadata for folder naming
        meas_row = meas_df[meas_df["Sample Number"] == sample_id]
        if meas_row.empty:
            print(f"Error: Sample {sample_id} not found in measurements CSV.")
            continue

        length, breadth = meas_row.iloc[0]["Length"], meas_row.iloc[0]["Breadth"]
        sample_folder_name = f"sample_{sample_id}_{length}x{breadth}"
        sample_path = base_dir / sample_folder_name

        for s_num in set_list:
            for l_num in light_list:
                for angle in angle_list:
                    
                    # Filter CSV for matching parameters
                    match = sample_df[
                        (sample_df['sample_number'] == sample_id) &
                        (sample_df['set_number'] == s_num) &
                        (sample_df['light_number'] == l_num) &
                        (sample_df['orientation_angle_deg'] == angle) &
                        (sample_df['shutter_speed'].astype(str) == str(config['shutter_speed'])) &
                        (sample_df['aperture'] == config['aperture']) &
                        (sample_df['iso'] == config['iso'])
                    ]

                    if match.empty:
                        continue

                    row = match.iloc[0]
                    img_name = row['image_name']
                    
                    # Ensure extension is handled correctly
                    if not str(img_name).lower().endswith(('.jpg', '.jpeg')):
                        img_name = f"{img_name}.JPG"

                    # Construct path according to dataset structure
                    img_path = sample_path / f"set_{s_num}" / f"light_{l_num}" / f"{angle}_deg" / img_name

                    if img_path.exists():
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        
                        # Set Output Dir: sample_folder / set_x / light_x
                        target_out = out_dir / sample_folder_name / f"set_{s_num}" / f"light_{l_num}"
                        target_out.mkdir(parents=True, exist_ok=True)
                        
                        save_path = target_out / f"contour_{angle}deg.png"
                        
                        # Generate contour using the renamed 'peak' column
                        generate_contours(
                            img, save_path, angle, s_num, l_num, 
                            row['peak_start'], row['peak'], row['peak_end']
                        )
                        print(f"Generated: {sample_folder_name} | Set {s_num} | Light {l_num} | {angle}°")
                    else:
                        print(f"File Missing: {img_path}")

def main():
    config = load_config(config_path)
    process_sample(config)
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
    
    
    

