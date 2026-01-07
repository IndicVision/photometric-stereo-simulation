import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the config file
CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\comparison_code\blender_sample_analysis_codes\upd_blender_sample_analysis_config.json"

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_names(config_name, light_setup):
    """Parses parameters from folder names used in main_fast_3.py logic."""
    p_parts = config_name.split('_')
    l_parts = light_setup.split('_')
    return {
        "Azimuth": p_parts[0], "Elevation": p_parts[1], "Plane_Area": p_parts[5],
        "Num_Lights": l_parts[0], "Distance_cm": l_parts[1], "Light_Type": l_parts[2],
        "Light_Shape": l_parts[3], "Spread": l_parts[4], "Psi_Angle_Deg": l_parts[5],
        "Light_Energy": l_parts[6]
    }

def get_masked_light_setup(params, variation_param):
    mapping = {"Distance_cm": 1, "Psi_Angle_Deg": 5, "Light_Energy": 6, "Spread": 4}
    parts = [params["Num_Lights"], params["Distance_cm"], params["Light_Type"],
             params["Light_Shape"], params["Spread"], params["Psi_Angle_Deg"], params["Light_Energy"]]
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
        results.append([int(np.min(pixels)), int(np.max(pixels)), int(np.max(pixels) - np.min(pixels)), float(np.std(pixels))] if pixels.size > 0 else [0,0,0,0.0])
    else:
        for i in [2, 1, 0]: # R, G, B
            pixels = patch[:, :, i][mask]
            results.append([int(np.min(pixels)), int(np.max(pixels)), int(np.max(pixels) - np.min(pixels)), float(np.std(pixels))] if pixels.size > 0 else [0,0,0,0.0])
    return results

def run_analysis():
    cfg = load_config(CONFIG_PATH)
    source_df = pd.read_csv(cfg['paths']['source_csv'])
    renders_base = Path(cfg['paths']['renders_dir'])
    recon_base = Path(cfg['paths']['recon_results_dir'])
    noise_out = Path(cfg['paths']['noise_plot_output_dir'])
    normal_out = Path(cfg['paths']['normal_plot_output_dir'])
    
    settings = cfg['analysis_settings']
    var_param = settings['variation_parameter']
    exp_col = settings['exposure_method']
    plot_metric = settings['plotting_metric']
    threshold = settings['error_threshold']
    
    all_rows = []
    recon_rows = []

    print("Extracting metrics and reconstruction data...")

    for _, row in source_df.iterrows():
        samples = int(row['Samples'])
        config_name, light_setup = row['Configuration'], row['Light_Setup']
        params = parse_names(config_name, light_setup)
        
        # 1. Intensity Noise Convergence (Toggleable)
        if settings['plot_intensity_convergence']:
            folder_path = renders_base / f"samples_{samples}" / config_name / light_setup
            num_lights = int(params['Num_Lights'])
            for li in range(1, num_lights + 1):
                img_path = folder_path / f"{li:03d}_{exp_col}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        patch = extract_center_patch(img, settings['patch_size'])
                        metrics = get_patch_metrics(patch, settings['color_mode'])
                        channels = ["Gray"] if settings['color_mode'] == "GRAY" else ["Red", "Green", "Blue"]
                        for i, chan in enumerate(channels):
                            mn, mx, diff, std = metrics[i]
                            d_row = {"Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                                     "Light_ID": li, "Channel": chan, "Error_Diff": diff, "Std_Dev": std}
                            d_row.update(params)
                            all_rows.append(d_row)

        # 2. Normal Reconstruction Convergence (Toggleable)
        if settings['plot_normal_convergence']:
            recon_csv = recon_base / f"samples_{samples}" / config_name / light_setup / "recon_report.csv"
            if recon_csv.exists():
                rdf = pd.read_csv(recon_csv)
                if 'mean_error' in rdf.columns:
                    r_row = {"Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                             "Mean_Angular_Error": rdf["mean_error"].iloc[0]}
                    r_row.update(params)
                    recon_rows.append(r_row)

    # --- INTENSITY PLOTTING ---
    if settings['plot_intensity_convergence'] and all_rows:
        print("Generating Intensity Noise Convergence Plots...")
        noise_out.mkdir(parents=True, exist_ok=True)
        df_noise = pd.DataFrame(all_rows)
        df_noise[var_param] = pd.to_numeric(df_noise[var_param])
        plot_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]
        
        for name_vals, group in df_noise.groupby(plot_keys):
            c_name = name_vals[0]
            masked_name = get_masked_light_setup(parse_names(c_name, group.iloc[0]["Light_Setup"]), var_param)
            save_path = noise_out / c_name / masked_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            for (l_id, chan), sub_group in group.groupby(["Light_ID", "Channel"]):
                fig = plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
                ax = plt.subplot(111)
                
                for p_val, p_group in sub_group.groupby(var_param):
                    p_group = p_group.sort_values(by="Samples")
                    x = p_group["Samples"]
                    y = p_group[plot_metric]
                    line, = ax.plot(x, y, marker='o', label=f"{var_param}: {p_val}", linewidth=cfg['plot_settings']['line_width'])
                    # Label at the end of the curve
                    if len(x) > 0:
                        ax.text(x.iloc[-1], y.iloc[-1], f' {p_val}', color=line.get_color(), va='center', fontsize=9)
                
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
                ax.set_title(f"Intensity Noise: {c_name} | ID {l_id} | {chan}", fontweight='bold')
                ax.set_xlabel("Samples (Log Scale)")
                ax.set_ylabel(plot_metric)
                ax.set_xscale('log')
                ax.grid(True, which="both", alpha=0.2)
                
                # Legend outside right
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(title=var_param, loc='upper left', bbox_to_anchor=(1, 1))
                
                plt.savefig(save_path / f"{l_id:03d}_{chan.lower()}_{plot_metric.lower()}.png", dpi=cfg['plot_settings']['dpi'], bbox_inches='tight')
                plt.close()

    # --- NORMAL PLOTTING ---
    if settings['plot_normal_convergence'] and recon_rows:
        print("Generating Normal Convergence Plots...")
        normal_out.mkdir(parents=True, exist_ok=True)
        df_normal = pd.DataFrame(recon_rows)
        df_normal[var_param] = pd.to_numeric(df_normal[var_param])
        n_thresh = settings['normal_error_threshold_deg']
        
        plot_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]
        for name_vals, group in df_normal.groupby(plot_keys):
            c_name = name_vals[0]
            masked_name = get_masked_light_setup(parse_names(c_name, group.iloc[0]["Light_Setup"]), var_param)
            save_path = normal_out / c_name / masked_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            fig = plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
            ax = plt.subplot(111)
            
            for p_val, p_group in group.groupby(var_param):
                p_group = p_group.sort_values(by="Samples")
                x = p_group["Samples"]
                y = p_group["Mean_Angular_Error"]
                line, = ax.plot(x, y, marker='s', label=f"{var_param}: {p_val}", linewidth=cfg['plot_settings']['line_width'])
                # Label at the end of the curve
                if len(x) > 0:
                    ax.text(x.iloc[-1], y.iloc[-1], f' {p_val}', color=line.get_color(), va='center', fontsize=9)
            
            ax.axhline(y=n_thresh, color='r', linestyle='--', label=f'Threshold ({n_thresh}°)')
            ax.set_title(f"Normal Convergence: {c_name}", fontweight='bold')
            ax.set_xlabel("Samples (Log Scale)")
            ax.set_ylabel("Mean Angular Error (Deg)")
            ax.set_xscale('log')
            ax.grid(True, which="both", alpha=0.2)
            
            # Legend outside right
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(title=var_param, loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.savefig(save_path / f"normal_convergence_thresh_{n_thresh}.png", dpi=cfg['plot_settings']['dpi'], bbox_inches='tight')
            plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    run_analysis()