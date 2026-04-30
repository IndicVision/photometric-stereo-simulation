import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import json

class ManualHybridOptimizer:
    def __init__(self, config_path: str, blender_params_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(blender_params_path, 'r') as f:
            self.blender_params = json.load(f)
            
        self.input_dir = Path(self.config['paths']['input_base_dir'])
        self.output_dir = Path(self.config['paths']['output_base_dir'])
        self.matrix_dir = Path(self.config['paths']['light_matrix_base_dir'])
        self.report_path = self.output_dir / self.config['paths']['report_filename']
        self.TARGET_MEAN = self.config['constants']['target_mean']
        self.summary_data = []
        self._ensure_dir(self.output_dir)
        self._ensure_dir(self.matrix_dir)

    def _ensure_dir(self, path: Path): path.mkdir(parents=True, exist_ok=True)

    def calculate_custom_skewness(self, img_array: np.ndarray) -> float:
        if len(img_array) == 0: return float('inf')
        sigma = np.std(img_array)
        if sigma == 0: return float('inf') 
        return np.mean((img_array - self.TARGET_MEAN)**3) / (sigma**3)

    def _parse_manual_config(self, config_name: str, light_setup: str) -> Dict:
        try:
            p_parts = config_name.split('_')
            plane_data = {
                'Plane_Azimuth_Deg': float(p_parts[0]),
                'Plane_Elevation_Deg': float(p_parts[1]),
                'Plane_Area_cm2': float(p_parts[5]) # Area at index 5
            }
            l_parts = light_setup.split('_')
            light_data = {
                'Num_Lights': int(l_parts[0]),
                'Distance_cm': float(l_parts[1]),
                'Light_Type': l_parts[2].upper(),
                'Light_Shape': l_parts[3].upper(),
                'Light_Spread': float(l_parts[4]),
                'Light_Energy': float(l_parts[-1]),
                'Setup_Type': 'MANUAL' if l_parts[5] == "Man" else 'ORBITAL'
            }
            return {**plane_data, **light_data}
        except: return {}

    def save_manual_light_matrix(self, config_key: str):
        vectors = self.blender_params['manual_light_setup']['vectors']
        matrix_data = [{'Configuration_Key': config_key, 'Light_ID': f"{i+1:03d}", 'L_x': v['lx'], 'L_y': v['ly'], 'L_z': v['lz']} for i, v in enumerate(vectors)]
        pd.DataFrame(matrix_data).to_csv(self.matrix_dir / f"light_matrix_{config_key.replace('::', '__')}.csv", index=False)

    def run_pipeline(self):
        sample_dirs = [x for x in self.input_dir.iterdir() if x.is_dir() and x.name.startswith("samples_")]
        for s_dir in sample_dirs:
            samples_count = int(s_dir.name.split('_')[1])
            for config_dir in [x for x in s_dir.iterdir() if x.is_dir()]:
                for light_dir in [x for x in config_dir.iterdir() if x.is_dir()]:
                    parsed = self._parse_manual_config(config_dir.name, light_dir.name)
                    if not parsed: continue
                    self.save_manual_light_matrix(f"{config_dir.name}::{light_dir.name}")
                    source_groups = {}
                    for img in list(light_dir.glob("*.png")):
                        sid = int(img.stem.split('_')[0])
                        if sid not in source_groups: source_groups[sid] = []
                        source_groups[sid].append(img)
                    current_config_rows = []
                    for sid, files in source_groups.items():
                        phase_data = []
                        for f in files:
                            ev, img = float(f.stem.split('_')[-1]), cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                flat = img.flatten()
                                phase_data.append({'ev': ev, 'skew': self.calculate_custom_skewness(flat[flat>0])})
                        if not phase_data: continue
                        win_b = min(phase_data, key=lambda x: abs(x['skew']))
                        evs, skews = np.array([d['ev'] for d in phase_data]), np.array([d['skew'] for d in phase_data])
                        idx = np.argsort(evs)
                        f_skew = interp1d(evs[idx], skews[idx], kind='linear', fill_value="extrapolate")
                        res = minimize_scalar(lambda e: abs(f_skew(e)), bounds=(evs.min(), evs.max()), method='bounded')
                        current_config_rows.append({'Samples': samples_count, 'Configuration': config_dir.name, 'Light_Setup': light_dir.name, 'Source_ID': sid, 'Classical_Best_EV': win_b['ev'], 'Final_GD_EV': float(res.x)})
                    if current_config_rows:
                        agg_row = {**parsed, 'Samples': samples_count, 'Configuration': config_dir.name, 'Light_Setup': light_dir.name, 'Source_ID': 'AGGREGATE', 
                                   'Aggregated_Mean_EV_Classical': np.mean([r['Classical_Best_EV'] for r in current_config_rows]), 
                                   'Aggregated_Mean_EV_GD': np.mean([r['Final_GD_EV'] for r in current_config_rows])}
                        self.summary_data.extend(current_config_rows + [agg_row])
        if self.summary_data: pd.DataFrame(self.summary_data).to_csv(self.report_path, index=False)

if __name__ == "__main__":
    ManualHybridOptimizer(r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\non_uni_pos_lit_rndr\upd_opt_exp_cfg.json", 
                          r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\non_uni_pos_lit_rndr\main_fst_cfg.json").run_pipeline()