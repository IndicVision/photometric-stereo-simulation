import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\comparison_code\blender_sample_analysis_codes\blender_sample_analysis_config.json"

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_names(config_name, light_setup):
    """Parses parameters from folder names used in main_fast_3.py logic."""
    p_parts = config_name.split('_')
    l_parts = light_setup.split('_')
    return {
        "Azimuth": p_parts[0],
        "Elevation": p_parts[1],
        "Plane_Area": p_parts[5],
        "Num_Lights": l_parts[0],
        "Distance_cm": l_parts[1],
        "Light_Type": l_parts[2],
        "Light_Shape": l_parts[3],
        "Spread": l_parts[4],
        "Psi_Angle_Deg": l_parts[5],
        "Light_Energy": l_parts[6]
    }

def get_masked_light_setup(params, variation_param):
    """Reconstructs light setup string with 'None' for the varying parameter."""
    # Mapping keys to indices based on the folder naming convention
    # Format: {Num_Lights}_{Distance}_{Type}_{Shape}_{Spread}_{Psi}_{Energy}
    mapping = {
        "Distance_cm": 1,
        "Psi_Angle_Deg": 5,
        "Light_Energy": 6,
        "Spread": 4
    }
    
    parts = [
        params["Num_Lights"], params["Distance_cm"], params["Light_Type"],
        params["Light_Shape"], params["Spread"], params["Psi_Angle_Deg"], 
        params["Light_Energy"]
    ]
    
    if variation_param in mapping:
        parts[mapping[variation_param]] = "None"
        
    return "_".join(parts)

def extract_center_patch(image, patch_size):
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    half = patch_size // 2
    y1, y2 = max(0, cy - half), min(h, cy + half)
    x1, x2 = max(0, cx - half), min(w, cx + half)
    return image[y1:y2, x1:x2]

def get_patch_metrics(patch, color_mode):
    mask = (patch > 0).any(axis=2) if len(patch.shape) == 3 else (patch > 0)
    results = []
    if color_mode == "GRAY":
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        pixels = gray[mask]
        if pixels.size > 0:
            results.append([int(np.min(pixels)), int(np.max(pixels)), 
                            int(np.max(pixels) - np.min(pixels)), float(np.std(pixels))])
        else: results.append([0, 0, 0, 0.0])
    else:
        for i in [2, 1, 0]: # R, G, B
            pixels = patch[:, :, i][mask]
            if pixels.size > 0:
                results.append([int(np.min(pixels)), int(np.max(pixels)), 
                                int(np.max(pixels) - np.min(pixels)), float(np.std(pixels))])
            else: results.append([0, 0, 0, 0.0])
    return results

def run_analysis():
    cfg = load_config(CONFIG_PATH)
    source_df = pd.read_csv(cfg['paths']['source_csv'])
    renders_base = Path(cfg['paths']['renders_dir'])
    output_base = Path(cfg['paths']['output_dir'])
    output_base.mkdir(parents=True, exist_ok=True)
    
    settings = cfg['analysis_settings']
    exp_col, patch_size = settings['exposure_method'], settings['patch_size']
    mode, plot_metric = settings['color_mode'], settings['plotting_metric']
    threshold = settings['error_threshold']
    var_param = settings['variation_parameter']
    
    all_rows = []
    print(f"Aggregating data from renders for method: {exp_col}...")

    for _, row in source_df.iterrows():
        samples = int(row['Samples'])
        config_name, light_setup = row['Configuration'], row['Light_Setup']
        exposure_val = row[exp_col]
        params = parse_names(config_name, light_setup)
        folder_path = renders_base / f"samples_{samples}" / config_name / light_setup
        
        num_lights_int = int(params['Num_Lights'])
        for li in range(1, num_lights_int + 1):
            img_path = folder_path / f"{li:03d}_{exp_col}.png"
            if not img_path.exists(): continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            patch = extract_center_patch(img, patch_size)
            metrics = get_patch_metrics(patch, mode)
            
            channels = ["Gray"] if mode == "GRAY" else ["Red", "Green", "Blue"]
            for i, channel_name in enumerate(channels):
                mn, mx, diff, std = metrics[i]
                data_row = {
                    "Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                    "Light_ID": li, "Exposure": exposure_val, "Patch_Size": patch_size,
                    "Channel": channel_name, "Min": mn, "Max": mx, "Error_Diff": diff, 
                    "Std_Dev": std, "Threshold_Value": threshold
                }
                data_row.update(params)
                all_rows.append(data_row)

    new_results_df = pd.DataFrame(all_rows)

    # --- APPEND LOGIC ---
    csv_path = output_base / "noise_convergence_report.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        # Drop duplicates to prevent repeating the same render data if script is re-run
        combined_df = pd.concat([existing_df, new_results_df]).drop_duplicates(
            subset=["Samples", "Configuration", "Light_Setup", "Light_ID", "Channel"]
        )
        results_df = combined_df
    else:
        results_df = new_results_df

    # Calculate Min Samples Required
    print("Calculating optimal sample requirements...")
    group_keys = ["Configuration", "Light_Setup", "Light_ID", "Channel"]
    for name, group in results_df.groupby(group_keys):
        sorted_group = group.sort_values(by="Samples")
        qualified = sorted_group[sorted_group[plot_metric] <= threshold]
        min_s = int(qualified["Samples"].iloc[0]) if not qualified.empty else None
        results_df.loc[group.index, "Min_Samples_Required"] = min_s

    results_df.to_csv(csv_path, index=False)
    print(f"Master CSV updated (appended/merged) at {csv_path}")

    # --- PLOTTING ---
    print(f"Generating Plots...")
    # Variation param needs to be numeric for plotting
    results_df[var_param] = pd.to_numeric(results_df[var_param])
    
    # We group by plane and setup parameters EXCEPT the one we are varying
    plot_group_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]

    for name_vals, group in results_df.groupby(plot_group_keys):
        config_folder_name = name_vals[0] # Plane Configuration
        
        # Get one example row to parse the masked light setup name
        sample_row = parse_names(config_folder_name, group.iloc[0]["Light_Setup"])
        masked_light_folder = get_masked_light_setup(sample_row, var_param)
        
        # Create Directory structure: plane -> masked_light
        plot_dir = output_base / config_folder_name / masked_light_folder
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Within these folders, we further group by Light_ID and Channel to create individual files
        for (l_id, chan), sub_group in group.groupby(["Light_ID", "Channel"]):
            plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
            
            # Curves represent setup variations (e.g. different distances)
            for p_val, p_group in sub_group.groupby(var_param):
                p_group = p_group.sort_values(by="Samples")
                plt.plot(p_group["Samples"], p_group[plot_metric], marker='o', 
                         label=f"{var_param}: {p_val}", 
                         linewidth=cfg['plot_settings']['line_width'])

            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            
            plt.title(f"Convergence: ID {l_id} | {chan}\nMetric: {plot_metric}", fontsize=11, fontweight='bold')
            plt.xlabel("Number of Samples")
            plt.ylabel(plot_metric)
            plt.xscale('log')
            plt.legend(title=var_param)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # Filename: <light_ID>_<gray or red or green or blue>_<plotting_metric>_<error_threshold>_<patch_size>
            chan_clean = chan.lower()
            metric_clean = plot_metric.lower()
            file_name = f"{l_id:03d}_{chan_clean}_{metric_clean}_{threshold}_{patch_size}.png"
            
            plt.savefig(plot_dir / file_name, dpi=cfg['plot_settings']['dpi'])
            plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    run_analysis()