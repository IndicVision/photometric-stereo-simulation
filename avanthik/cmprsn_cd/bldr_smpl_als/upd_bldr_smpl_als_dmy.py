import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\comparison_code\blender_sample_analysis_codes\upd_blender_sample_analysis_dummy_config.json"

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_names(config_name, light_setup):
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
    return image[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]

def get_patch_metrics(patch, color_mode):
    mask = (patch > 0).any(axis=2) if len(patch.shape) == 3 else (patch > 0)
    results = []
    if color_mode == "GRAY":
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        p = gray[mask]
        results.append([int(np.min(p)), int(np.max(p)), int(np.max(p)-np.min(p)), float(np.std(p))] if p.size > 0 else [0,0,0,0.0])
    else:
        for i in [2, 1, 0]: # R, G, B
            p = patch[:, :, i][mask]
            results.append([int(np.min(p)), int(np.max(p)), int(np.max(p)-np.min(p)), float(np.std(p))] if p.size > 0 else [0,0,0,0.0])
    return results

def run_analysis():
    cfg = load_config(CONFIG_PATH)
    source_df = pd.read_csv(cfg['paths']['source_csv'])
    renders_base = Path(cfg['paths']['renders_dir'])
    recon_base = Path(cfg['paths']['recon_results_dir'])
    
    settings = cfg['analysis_settings']
    var_param = settings['variation_parameter']
    plot_metric = settings['plotting_metric']
    
    all_intensity_rows = []
    all_normal_rows = []

    print(f"Aggregation phase starting...")

    for _, row in source_df.iterrows():
        samples = int(row['Samples'])
        config_name, light_setup = row['Configuration'], row['Light_Setup']
        params = parse_names(config_name, light_setup)
        
        # 1. Process Intensity Noise (only if bool is true)
        if settings['plot_intensity_convergence']:
            folder_path = renders_base / f"samples_{samples}" / config_name / light_setup
            for li in range(1, int(params['Num_Lights']) + 1):
                img_path = folder_path / f"{li:03d}_{settings['exposure_method']}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        patch = extract_center_patch(img, settings['patch_size'])
                        metrics = get_patch_metrics(patch, settings['color_mode'])
                        channels = ["Red", "Green", "Blue"] if settings['color_mode'] == "RGB" else ["Gray"]
                        for i, chan in enumerate(channels):
                            mn, mx, diff, std = metrics[i]
                            d_row = {"Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                                     "Light_ID": li, "Channel": chan, "Error_Diff": diff, "Std_Dev": std}
                            d_row.update(params)
                            all_intensity_rows.append(d_row)

        # 2. Process Normal Error (only if bool is true)
        if settings['plot_normal_convergence']:
            recon_csv = recon_base / f"samples_{samples}" / config_name / light_setup / "recon_report.csv"
            if recon_csv.exists():
                rdf = pd.read_csv(recon_csv)
                if 'mean_error' in rdf.columns:
                    n_row = {"Samples": samples, "Configuration": config_name, "Light_Setup": light_setup,
                             "Mean_Angular_Error": rdf['mean_error'].iloc[0]}
                    n_row.update(params)
                    all_normal_rows.append(n_row)

    # --- Plotting Intensity Metric ---
    if settings['plot_intensity_convergence'] and all_intensity_rows:
        print(f"Generating Intensity Plots (Y={plot_metric})")
        df_int = pd.DataFrame(all_intensity_rows)
        df_int[var_param] = pd.to_numeric(df_int[var_param])
        int_out_base = Path(cfg['paths']['intensity_plot_dir'])
        int_out_base.mkdir(parents=True, exist_ok=True)
        
        plot_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]
        for name_vals, group in df_int.groupby(plot_keys):
            masked_folder = get_masked_light_setup(parse_names(name_vals[0], group.iloc[0]["Light_Setup"]), var_param)
            plot_dir = int_out_base / name_vals[0] / masked_folder
            plot_dir.mkdir(parents=True, exist_ok=True)
            for (l_id, chan), sub_group in group.groupby(["Light_ID", "Channel"]):
                plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
                for s_val, s_group in sub_group.groupby("Samples"):
                    s_group = s_group.sort_values(by=var_param)
                    plt.plot(s_group[var_param], s_group[plot_metric], marker='o', label=f"Samples: {s_val}")
                plt.title(f"Intensity Variation | ID {l_id} | {chan}\nMetric: {plot_metric}", fontweight='bold')
                plt.xlabel(var_param); plt.ylabel(plot_metric); plt.legend(title="Samples"); plt.grid(True, alpha=0.3)
                plt.savefig(plot_dir / f"{l_id:03d}_{chan.lower()}_{plot_metric.lower()}.png", dpi=cfg['plot_settings']['dpi'])
                plt.close()

    # --- Plotting Normal Error ---
    if settings['plot_normal_convergence'] and all_normal_rows:
        print(f"Generating Normal Error Plots (Y=Mean_Angular_Error)")
        df_norm = pd.DataFrame(all_normal_rows)
        df_norm[var_param] = pd.to_numeric(df_norm[var_param])
        norm_out_base = Path(cfg['paths']['normal_plot_dir'])
        norm_out_base.mkdir(parents=True, exist_ok=True)
        
        plot_keys = ["Configuration", "Num_Lights", "Light_Type", "Psi_Angle_Deg", "Light_Energy"]
        for name_vals, group in df_norm.groupby(plot_keys):
            masked_folder = get_masked_light_setup(parse_names(name_vals[0], group.iloc[0]["Light_Setup"]), var_param)
            plot_dir = norm_out_base / name_vals[0] / masked_folder
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(cfg['plot_settings']['figure_width'], cfg['plot_settings']['figure_height']))
            for s_val, s_group in group.groupby("Samples"):
                s_group = s_group.sort_values(by=var_param)
                plt.plot(s_group[var_param], s_group["Mean_Angular_Error"], marker='s', label=f"Samples: {s_val}")
            plt.title(f"Normal Reconstruction Error vs {var_param}\nConfig: {name_vals[0]}", fontweight='bold')
            plt.xlabel(var_param); plt.ylabel("Mean Angular Error (Degrees)"); plt.legend(title="Samples"); plt.grid(True, alpha=0.3)
            plt.savefig(plot_dir / f"normal_error_vs_{var_param.lower()}.png", dpi=cfg['plot_settings']['dpi'])
            plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    run_analysis()