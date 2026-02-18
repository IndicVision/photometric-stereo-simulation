import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Absolute path to your JSON configuration
CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\comparison_code\blender_sample_analysis_codes\blender_sample_analysis_dummy_config.json"

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_names(config_name, light_setup):
    """Parses parameters from folder names. Returns strings to preserve folder integrity."""
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
    """Reconstructs light setup string with 'None' for the varying parameter for directory naming."""
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
        for i in [2, 1, 0]: # R, G, B order
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
    var_param = settings['variation_parameter']
    
    all_rows = []
    print(f"Aggregation phase starting for metric: {plot_metric}")

    # Process all image sets listed in source CSV
    for _, row in source_df.iterrows():
        samples = int(row['Samples'])
        config_name, light_setup = row['Configuration'], row['Light_Setup']
        exposure_val = row[exp_col]
        params = parse_names(config_name, light_setup)
        folder_path = renders_base / f"samples_{samples}" / config_name / light_setup
        
        num_l = int(params['Num_Lights'])
        for li in range(1, num_l + 1):
            img_path = folder_path / f"{li:03d}_{exp_col}.png"
            if not img_path.exists(): continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            patch = extract_center_patch(img, patch_size)
            metrics = get_patch_metrics(patch, mode)
            
            channels = ["Red", "Green", "Blue"] if mode == "RGB" else ["Gray"]
            for i, channel_name in enumerate(channels):
                mn, mx, diff, std = metrics[i]
                data_row = {
                    "Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                    "Light_ID": li, "Exposure": exposure_val, "Patch_Size": patch_size,
                    "Channel": channel_name, "Min": mn, "Max": mx, "Error_Diff": diff, "Std_Dev": std
                }
                data_row.update(params)
                all_rows.append(data_row)

    new_results_df = pd.DataFrame(all_rows)

    # --- Append and Merge Logic ---
    csv_path = output_base / "noise_convergence_report.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        # Drop duplicates to ensure unique configuration entries
        results_df = pd.concat([existing_df, new_results_df]).drop_duplicates(
            subset=["Samples", "Configuration", "Light_Setup", "Light_ID", "Channel"]
        )
    else:
        results_df = new_results_df

    results_df.to_csv(csv_path, index=False)
    print(f"Master CSV saved/updated at: {csv_path}")

    # --- Plotting: Parameter (X) vs Error (Y) ---
    print(f"Generating Plots (X={var_param}, Y={plot_metric})")
    results_df[var_param] = pd.to_numeric(results_df[var_param])
    
    # Grouping key for consistent plot grouping (Variation param is excluded from grouping)
    plot_group_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]

    for name_vals, group in results_df.groupby(plot_group_keys):
        config_folder_name = name_vals[0]
        
        # Determine the masked folder name for the light setup
        sample_params = parse_names(config_folder_name, group.iloc[0]["Light_Setup"])
        masked_light_folder = get_masked_light_setup(sample_params, var_param)
        
        # Establish directory hierarchy
        plot_dir = output_base / config_folder_name / masked_light_folder
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot one file per Light ID and Color Channel
        for (l_id, chan), sub_group in group.groupby(["Light_ID", "Channel"]):
            plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
            
            # Each curve represents a different Sample count
            for s_val, s_group in sub_group.groupby("Samples"):
                s_group = s_group.sort_values(by=var_param)
                plt.plot(s_group[var_param], s_group[plot_metric], marker='o', 
                         label=f"Samples: {s_val}", 
                         linewidth=cfg['plot_settings']['line_width'])

            plt.title(f"Parameter Variation | ID {l_id} | {chan}\nMetric: {plot_metric}", fontweight='bold')
            plt.xlabel(var_param.replace('_', ' '))
            plt.ylabel(plot_metric.replace('_', ' '))
            plt.legend(title="Sample Counts")
            plt.grid(True, which="both", ls="-", alpha=0.3)
            
            # Filename Format: <light_ID>_<channel>_<metric>_0_0_<patch_size>
            # (Using 0_0 as placeholders for Threshold/unused fields to match your format)
            chan_clean = chan.lower()
            metric_clean = plot_metric.lower()
            file_name = f"{l_id:03d}_{chan_clean}_{metric_clean}_0_0_{patch_size}.png"
            
            plt.savefig(plot_dir / file_name, dpi=cfg['plot_settings']['dpi'])
            plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    run_analysis()