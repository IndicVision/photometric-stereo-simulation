import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import json

class HybridExposureOptimizer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.input_dir = Path(self.config['paths']['input_base_dir'])
        self.output_dir = Path(self.config['paths']['output_base_dir'])
        self.matrix_dir = Path(self.config['paths']['light_matrix_base_dir'])
        self.report_path = self.output_dir / self.config['paths']['report_filename']
        
        # Configuration Constants
        C = self.config['constants']
        self.TARGET_MEAN = C['target_mean']
        self.PIXEL_MIN = C['pixel_min']
        self.PIXEL_MAX = C['pixel_max']
        self.HIST_BINS = C['hist_bins']
        self.EPSILON = C['epsilon_weight']
        
        # Plotting Flags
        self.SAVE_HIST_PLOTS = self.config['plotting']['save_hist_plots']
        self.SAVE_GD_PLOTS = self.config['plotting']['save_gd_plots']
        
        self.summary_data = [] 
        self.existing_configs = set()
        
        self._ensure_dir(self.output_dir)
        self._ensure_dir(self.matrix_dir)
        self._load_existing_report()
        
        print(f"[SYSTEM] Initialized Hybrid Optimizer. Report path: {self.report_path}")

    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _ensure_dir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

    def _load_existing_report(self):
        """Loads existing report and tracks 'Samples' to prevent duplicates."""
        if self.report_path.exists():
            try:
                df_existing = pd.read_csv(self.report_path)
                # Key now includes Samples to distinguish results properly
                self.existing_configs = set(
                    df_existing['Samples'].astype(str) + '__' + 
                    df_existing['Configuration'] + '__' + 
                    df_existing['Light_Setup']
                )
                self.summary_data = [
                    r for r in df_existing.to_dict('records') 
                    if r.get('Source_ID') != 'AGGREGATE'
                ]
            except Exception as e:
                print(f"[ERROR] Could not load existing report: {e}")
                
    def calculate_custom_skewness(self, img_array: np.ndarray) -> float:
        """Calculates Normalized Skewness around Middle Gray (128)."""
        if len(img_array) == 0: return float('inf')
        sigma = np.std(img_array)
        if sigma == 0: return float('inf') 
        deviations = img_array - self.TARGET_MEAN
        return np.mean(deviations ** 3) / (sigma ** 3)

    def _generate_histogram_plot(self, img: np.ndarray, source_id: int, ev: float, save_path: Path):
        if not self.SAVE_HIST_PLOTS: return
        plt.figure(figsize=(8, 5))
        plt.hist(img.flatten(), bins=self.HIST_BINS, range=(self.PIXEL_MIN, self.PIXEL_MAX+1), color='gray')
        plt.axvline(self.TARGET_MEAN, color='g', linestyle='--', label=f'Target ({self.TARGET_MEAN})')
        plt.title(f"Source {source_id} | EV {ev}")
        plt.savefig(save_path / f"hist_{source_id:03d}_{ev:.3f}.png")
        plt.close()

    def _plot_gd_result(self, x_vals, y_vals, f_interp, opt_ev, opt_skew, src_id, save_dir):
        if not self.SAVE_GD_PLOTS: return
        plt.figure(figsize=(8, 5))
        plt.scatter(x_vals, y_vals, color='red', label='Measured')
        x_dense = np.linspace(min(x_vals), max(x_vals), 500)
        plt.plot(x_dense, f_interp(x_dense), color='blue', label='Interpolated')
        plt.plot(opt_ev, opt_skew, 'g*', markersize=15, label=f'Opt EV: {opt_ev:.4f}')
        plt.title(f"GD Curve: Source {src_id}")
        plt.xlabel("Exposure Value"); plt.ylabel("Skewness")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / f"GD_Curve_{src_id:03d}.png")
        plt.close()

    def _parse_config_from_name(self, config_name: str, light_setup: str) -> Dict:
        """Parses physical parameters from folder names."""
        try:
            p_parts = config_name.split('_')
            plane_data = {
                'Plane_Azimuth_Deg': float(p_parts[0]),
                'Plane_Elevation_Deg': float(p_parts[1]),
                'Plane_Area_cm2': float(p_parts[5])
            }
            l_parts = light_setup.split('_')
            light_data = {
                'Num_Lights': int(l_parts[0]),
                'Distance_cm': float(l_parts[1]),
                'Light_Type': l_parts[2].upper(),
                'Psi_Angle_Deg': float(l_parts[-2]),
                'Light_Energy': float(l_parts[-1]),
                'Light_Prop_Name': "_".join(l_parts[3:-2])
            }
            return {**plane_data, **light_data}
        except: return {}

    def calculate_light_vectors(self, config_name: str, light_setup: str) -> List[Dict]:
        """Calculates unit vectors L for the light matrix CSV."""
        p_info = self._parse_config_from_name(config_name, light_setup)
        if not p_info: return []
        
        elev = np.radians(p_info['Plane_Elevation_Deg'])
        azim = np.radians(p_info['Plane_Azimuth_Deg'])
        zenith = np.radians(90.0) - elev
        n = [np.sin(zenith)*np.cos(azim), np.sin(zenith)*np.sin(azim), np.cos(zenith)]
        
        psi_rad = np.radians(p_info['Psi_Angle_Deg'])
        vectors = []
        for i in range(1, p_info['Num_Lights'] + 1):
            theta_i = (i - 1) * (2 * np.pi / p_info['Num_Lights'])
            A = n[0] * np.cos(theta_i) + n[1] * np.sin(theta_i)
            B, R = n[2], np.sqrt(A**2 + n[2]**2)
            phi_i = np.arcsin(np.clip(np.cos(psi_rad)/R, -1, 1)) - np.arctan2(B, A)
            vectors.append({
                'Configuration_Key': f"{config_name}__{light_setup}",
                'Light_ID': f"{i:03d}",
                'L_x': np.sin(phi_i)*np.cos(theta_i),
                'L_y': np.sin(phi_i)*np.sin(theta_i),
                'L_z': np.cos(phi_i)
            })
        return vectors

    def _aggregate_and_report(self, samples: int, config_name: str, light_setup: str, config_rows: List[Dict]):
        parsed = self._parse_config_from_name(config_name, light_setup)
        ev_c = np.array([r['Classical_Best_EV'] for r in config_rows])
        skew_c = np.array([r['Classical_Skew_128'] for r in config_rows])
        ev_gd = np.array([r['Final_GD_EV'] for r in config_rows])
        skew_gd = np.array([r['Final_GD_Skew_128'] for r in config_rows])
        
        w_c = 1.0 / (np.abs(skew_c) + self.EPSILON)
        w_gd = 1.0 / (np.abs(skew_gd) + self.EPSILON)

        agg_row = {
            **parsed, 'Samples': samples, 'Configuration': config_name, 'Light_Setup': light_setup, 'Source_ID': 'AGGREGATE',
            'Aggregated_Mean_EV_Classical': np.mean(ev_c),
            'Aggregated_WMean_EV_Classical': np.sum(ev_c * w_c) / np.sum(w_c),
            'Aggregated_Mean_EV_GD': np.mean(ev_gd),
            'Aggregated_WMean_EV_GD': np.sum(ev_gd * w_gd) / np.sum(w_gd),
        }
        # Clean existing entries for this specific sample/config combo
        self.summary_data = [r for r in self.summary_data if not (r.get('Configuration') == config_name and r.get('Light_Setup') == light_setup and r.get('Samples') == samples)]
        for row in config_rows: self.summary_data.append({**parsed, 'Samples': samples, **row})
        self.summary_data.append(agg_row)

    def run_pipeline(self):
        sample_dirs = [x for x in self.input_dir.iterdir() if x.is_dir() and x.name.startswith("samples_")]
        for s_dir in sample_dirs:
            samples_count = int(s_dir.name.split('_')[1])
            for config_dir in [x for x in s_dir.iterdir() if x.is_dir()]:
                for light_dir in [x for x in config_dir.iterdir() if x.is_dir()]:
                    
                    config_key = f"{samples_count}__{config_dir.name}__{light_dir.name}"
                    if config_key in self.existing_configs: continue
                    
                    # Geometry matrix saving
                    matrix_file = self.matrix_dir / f"light_matrix_{config_dir.name}__{light_dir.name}.csv"
                    if not matrix_file.exists():
                        vec_data = self.calculate_light_vectors(config_dir.name, light_dir.name)
                        pd.DataFrame(vec_data).to_csv(matrix_file, index=False)

                    target_plot_dir = self.output_dir / s_dir.name / config_dir.name / light_dir.name
                    if self.SAVE_HIST_PLOTS or self.SAVE_GD_PLOTS: self._ensure_dir(target_plot_dir)

                    all_images = list(light_dir.glob("*.png"))
                    source_groups = {}
                    for img in all_images:
                        sid = int(img.stem.split('_')[0])
                        if sid not in source_groups: source_groups[sid] = []
                        source_groups[sid].append(img)

                    current_config_rows = []
                    for sid, files in source_groups.items():
                        phase_data = []
                        for f in files:
                            ev = float(f.stem.split('_')[-1])
                            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                            if img is None: continue
                            if self.SAVE_HIST_PLOTS: self._generate_histogram_plot(img, sid, ev, target_plot_dir)
                            flat = img.flatten()
                            phase_data.append({'ev': ev, 'skew_filtered': self.calculate_custom_skewness(flat[flat>0]), 'skew_full': self.calculate_custom_skewness(flat)})
                        
                        if not phase_data: continue
                        win_b = min(phase_data, key=lambda x: abs(x['skew_filtered']))
                        
                        evs, skews = np.array([d['ev'] for d in phase_data]), np.array([d['skew_full'] for d in phase_data])
                        sort_idx = np.argsort(evs)
                        evs, skews = evs[sort_idx], skews[sort_idx]
                        
                        f_skew = interp1d(evs, skews, kind='linear', fill_value="extrapolate")
                        res = minimize_scalar(lambda e: abs(f_skew(e)), bounds=(evs.min(), evs.max()), method='bounded')
                        
                        if self.SAVE_GD_PLOTS: self._plot_gd_result(evs, skews, f_skew, res.x, f_skew(res.x), sid, target_plot_dir)
                        
                        current_config_rows.append({
                            'Configuration': config_dir.name, 'Light_Setup': light_dir.name, 'Source_ID': sid,
                            'Classical_Best_EV': win_b['ev'], 'Classical_Skew_128': win_b['skew_filtered'],
                            'Final_GD_EV': float(res.x), 'Final_GD_Skew_128': float(f_skew(res.x))
                        })

                    if current_config_rows:
                        self._aggregate_and_report(samples_count, config_dir.name, light_dir.name, current_config_rows)
                        self.existing_configs.add(config_key)
                        print(f"[SUCCESS] Processed {s_dir.name}: {light_dir.name}")

        if self.summary_data:
            df = pd.DataFrame(self.summary_data)
            cols = ['Samples', 'Configuration', 'Plane_Azimuth_Deg', 'Plane_Elevation_Deg', 'Plane_Area_cm2', 
                    'Light_Setup', 'Num_Lights', 'Distance_cm', 'Light_Type', 'Light_Prop_Name', 'Psi_Angle_Deg', 'Light_Energy',
                    'Aggregated_Mean_EV_Classical', 'Aggregated_WMean_EV_Classical', 'Aggregated_Mean_EV_GD', 'Aggregated_WMean_EV_GD', 
                    'Source_ID', 'Classical_Best_EV', 'Final_GD_EV']
            df[[c for c in cols if c in df.columns]].to_csv(self.report_path, index=False)

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\blender_code\general_rendering\upd_opt_exp_config.json"
    HybridExposureOptimizer(CONFIG_PATH).run_pipeline()