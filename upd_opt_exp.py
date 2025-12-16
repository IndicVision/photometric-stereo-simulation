import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import shutil
import json

class HybridExposureOptimizer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.input_dir = Path(self.config['paths']['input_base_dir'])
        self.output_dir = Path(self.config['paths']['output_base_dir'])
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
        self._load_existing_report()
        
        print(f"[SYSTEM] Initialized Hybrid Optimizer. Report path: {self.report_path}")

    def _load_config(self, path: str) -> Dict:
        """Loads configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"[FATAL ERROR] Failed to load configuration from {path}: {e}")

    def _ensure_dir(self, path: Path):
        """Creates directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)

    def _load_existing_report(self):
        """Loads existing report to prevent re-processing and duplicates."""
        if self.report_path.exists():
            try:
                df_existing = pd.read_csv(self.report_path)
                
                # Identify already processed configurations using Configuration + Light_Setup
                self.existing_configs = set(
                    df_existing['Configuration'] + '__' + df_existing['Light_Setup']
                )
                
                # Load existing rows to re-save them later, removing the aggregated rows initially
                # We rebuild the aggregated rows later.
                self.summary_data = [
                    r for r in df_existing.to_dict('records') 
                    if r.get('Source_ID') != 'AGGREGATE'
                ]

            except pd.errors.EmptyDataError:
                pass
            except Exception as e:
                print(f"[ERROR] Could not load existing report {self.report_path}: {e}")
                
    def calculate_custom_skewness(self, img_array: np.ndarray) -> float:
        """
        Calculates Normalized Skewness strictly around 128 (Middle Gray).
        Formula: E[(x - 128)^3] / sigma^3
        """
        if len(img_array) == 0: return float('inf')
        sigma = np.std(img_array)
        if sigma == 0: return float('inf') 

        deviations = img_array - self.TARGET_MEAN
        third_moment_fixed = np.mean(deviations ** 3)

        return third_moment_fixed / (sigma ** 3)

    def _generate_histogram_plot(self, img: np.ndarray, config_name: str, light_setup: str, source_id: int, ev: float, save_path: Path):
        """Helper to generate and save a histogram plot on demand."""
        if not self.SAVE_HIST_PLOTS: return
        
        try:
            flat_pixels = img.flatten()
            plt.figure(figsize=(10, 6))
            counts, _ = np.histogram(flat_pixels, bins=self.HIST_BINS, range=(self.PIXEL_MIN, self.PIXEL_MAX+1))
            plt.bar(np.arange(self.HIST_BINS), counts, color='gray', width=1.0)
            plt.axvline(self.TARGET_MEAN, color='g', linestyle='--', label=f'Target ({self.TARGET_MEAN:.0f})')
            plt.title(f"Source {source_id} | EV {ev}")
            plt.legend()
            
            filename = f"hist_{source_id:03d}_{ev:.3f}.png"
            plt.savefig(save_path / filename)
            plt.close()
        except Exception as e:
            print(f"[ERROR] Failed to generate histogram plot: {e}")

    def _generate_histogram_plot_final(self, img: np.ndarray, source_id: int, ev: float, save_path: Path, title_suffix: str):
        """Helper to generate and save a final histogram plot for the winner."""
        if not self.SAVE_GD_PLOTS: return
        try:
            flat_pixels = img.flatten()
            plt.figure(figsize=(10, 6))
            counts, _ = np.histogram(flat_pixels, bins=self.HIST_BINS, range=(self.PIXEL_MIN, self.PIXEL_MAX+1))
            plt.bar(np.arange(self.HIST_BINS), counts, color='gray', width=1.0)
            plt.axvline(self.TARGET_MEAN, color='g', linestyle='--', label=f'Target ({self.TARGET_MEAN:.0f})')
            plt.title(f"{title_suffix}: Source {source_id} | Best EV {ev:.3f}")
            plt.legend()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"[ERROR] Failed to generate final histogram plot: {e}")
            
    def _parse_config_from_name(self, config_name: str, light_setup: str) -> Dict:
        """
        Parses configuration parameters from the standardized directory names.
        """
        try:
            # 1. Parse Configuration Name (Plane Properties)
            # Format: <azimuth_deg>_<elevation_deg>_<center_x>_<center_y>_<center_z>_<area_cm2>
            plane_parts = config_name.split('_')
            plane_data = {
                'Plane_Azimuth_Deg': float(plane_parts[0]),
                'Plane_Elevation_Deg': float(plane_parts[1]),
                'Plane_Center_X_cm': float(plane_parts[2]),
                'Plane_Center_Y_cm': float(plane_parts[3]),
                'Plane_Center_Z_cm': float(plane_parts[4]),
                'Plane_Area_cm2': float(plane_parts[5])
            }

            # 2. Parse Light Setup Name
            # Format: <num_lights>_<distance_cm>_<light_type>_<prop_part>_<psi_deg>_<energy>
            light_parts = light_setup.split('_')
            
            num_lights = int(light_parts[0])
            distance_cm = float(light_parts[1])
            light_type = light_parts[2].upper() 
            psi_deg = float(light_parts[-2])
            energy = float(light_parts[-1])
            
            # The Light Property Name is the part between light_type and psi_deg
            light_prop_name = "_".join(light_parts[3:-2])

            light_data = {
                'Num_Lights': num_lights,
                'Distance_cm': distance_cm,
                'Light_Type': light_type,
                'Psi_Angle_Deg': psi_deg,
                'Light_Energy': energy,
                'Light_Prop_Name': light_prop_name
            }
            
            return {**plane_data, **light_data}
        except Exception as e:
            print(f"[ERROR] Failed to parse config names: {e} (Config: {config_name}, Light: {light_setup})")
            return {}


    # -------------------------------------------------------------------------
    # PHASE A: Data Extraction & Artifact Generation
    # -------------------------------------------------------------------------
    def run_phase_a(self, config_name: str, light_setup: str, source_id: int, image_files: List[Path], save_dir: Path) -> List[dict]:
        # ... (Phase A logic remains the same)
        phase_data = []

        for img_path in image_files:
            try:
                parts = img_path.stem.split('_')
                if len(parts) < 2: continue
                ev = float(parts[-1])
                
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                flat_pixels = img.flatten()

                self._generate_histogram_plot(img, config_name, light_setup, source_id, ev, save_dir)
                    
                pixels_nonzero = flat_pixels[flat_pixels > 0]
                skew_filtered = self.calculate_custom_skewness(pixels_nonzero)
                skew_full = self.calculate_custom_skewness(flat_pixels)

                phase_data.append({
                    'ev': ev,
                    'skew_filtered': skew_filtered,
                    'skew_full': skew_full,
                    'img_path': img_path, 
                })

            except Exception as e:
                print(f"[ERROR] Processing {img_path.name} in Phase A: {e}")
                continue
                
        return phase_data

    # -------------------------------------------------------------------------
    # PHASE B: Classical Optimization (Discrete Selection)
    # -------------------------------------------------------------------------
    def run_phase_b(self, phase_data: List[dict], source_id: int, save_dir: Path) -> dict:
        valid_candidates = [d for d in phase_data if d['skew_filtered'] != float('inf')]
        if not valid_candidates:
            return None

        winner = min(valid_candidates, key=lambda x: abs(x['skew_filtered']))
        
        if self.SAVE_GD_PLOTS:
            dst_plot = save_dir / f"Classical_Best_Plot_{source_id:03d}_{winner['ev']:.3f}.png"
            img = cv2.imread(str(winner['img_path']), cv2.IMREAD_GRAYSCALE)
            self._generate_histogram_plot_final(img, source_id, winner['ev'], dst_plot, "Classical Best")
            
        return winner


    # -------------------------------------------------------------------------
    # PHASE C: Gradient Descent (Fine-Tuning)
    # -------------------------------------------------------------------------
    def run_phase_c(self, phase_data: List[dict], source_id: int, save_dir: Path) -> Tuple[float, float]:
        evs = [d['ev'] for d in phase_data]
        skews = [d['skew_full'] for d in phase_data]
        
        sorted_pairs = sorted(zip(evs, skews))
        x_vals = np.array([p[0] for p in sorted_pairs])
        y_vals = np.array([p[1] for p in sorted_pairs])
        
        if len(x_vals) < 2: return 0.0, 0.0

        kind = 'cubic' if len(x_vals) > 3 else 'linear'
        try:
            f_skew = interp1d(x_vals, y_vals, kind=kind, fill_value="extrapolate")
        except:
            f_skew = interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
        
        res = minimize_scalar(lambda ev: abs(f_skew(ev)), bounds=(x_vals.min(), x_vals.max()), method='bounded')
        
        optimal_ev = float(res.x)
        optimal_skew = float(f_skew(optimal_ev))
        
        if self.SAVE_GD_PLOTS:
            self._plot_gd_result(x_vals, y_vals, f_skew, optimal_ev, optimal_skew, source_id, save_dir)

        return optimal_ev, optimal_skew

    def _plot_gd_result(self, x_vals, y_vals, f_interp, opt_ev, opt_skew, src_id, save_dir):
        """Generates the optimization curve plot."""
        if not self.SAVE_GD_PLOTS: return
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_vals, y_vals, color='red', s=60, zorder=5, label='Measured Data')
            x_dense = np.linspace(min(x_vals), max(x_vals), 500)
            y_dense = f_interp(x_dense)
            plt.plot(x_dense, y_dense, color='blue', alpha=0.6, label='Interpolated Model')
            
            plt.plot(opt_ev, opt_skew, 'g*', markersize=18, zorder=10, label=f'Optimal EV: {opt_ev:.4f}')
            
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.title(f"Optimization Curve: Source {src_id}")
            plt.xlabel("Exposure Value (EV)")
            plt.ylabel("Skewness")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"GD_Curve_Optimal_{src_id:03d}_{opt_ev:.3f}.png"
            plt.savefig(save_dir / filename)
            plt.close()
        except Exception as e:
            print(f"[ERROR] Failed to generate GD curve plot: {e}")

    # -------------------------------------------------------------------------
    # AGGREGATION AND REPORTING
    # -------------------------------------------------------------------------
    def _aggregate_and_report(self, config_name: str, light_setup: str, config_rows: List[Dict]):
        """
        Calculates mean and weighted mean EV for all light sources (N) in a configuration.
        Includes parsed physical data in the report rows.
        """
        
        # 1. Parse Physical Data
        parsed_data = self._parse_config_from_name(config_name, light_setup)
        if not parsed_data:
            print(f"[ERROR] Cannot aggregate due to parsing failure for {config_name}/{light_setup}")
            return
            
        # 2. Extract Data for Aggregation
        ev_c = np.array([r['Classical_Best_EV'] for r in config_rows])
        skew_c = np.array([r['Classical_Skew_128'] for r in config_rows])
        
        ev_gd = np.array([r['Final_GD_EV'] for r in config_rows])
        skew_gd = np.array([r['Final_GD_Skew_128'] for r in config_rows])
        
        # 3. CLASSICAL AGGREGATION
        mean_c_sm = np.mean(ev_c)
        weights_c = 1.0 / (np.abs(skew_c) + self.EPSILON)
        mean_c_wm = np.sum(ev_c * weights_c) / np.sum(weights_c)
        
        # 4. GD AGGREGATION
        mean_gd_sm = np.mean(ev_gd)
        weights_gd = 1.0 / (np.abs(skew_gd) + self.EPSILON)
        mean_gd_wm = np.sum(ev_gd * weights_gd) / np.sum(weights_gd)

        # 5. Define Aggregated Summary Row
        config_summary_row = {
            **parsed_data, # Includes all parsed physical parameters
            'Configuration': config_name,
            'Light_Setup': light_setup,
            'Source_ID': 'AGGREGATE', # Mark this row as the summary row
            'Aggregated_Mean_EV_Classical': mean_c_sm,
            'Aggregated_WMean_EV_Classical': mean_c_wm,
            'Aggregated_Mean_EV_GD': mean_gd_sm,
            'Aggregated_WMean_EV_GD': mean_gd_wm,
        }
        
        # 6. Update Summary Data
        
        # Remove existing rows for this config (both aggregate and individual) before re-appending
        self.summary_data = [
            r for r in self.summary_data 
            if not (r.get('Configuration') == config_name and r.get('Light_Setup') == light_setup)
        ]

        # Append detailed source data, including parsed physical data
        for row in config_rows:
            self.summary_data.append({**parsed_data, **row})

        # Append the new aggregated row
        self.summary_data.append(config_summary_row)


    def process_light_source(self, config_name: str, light_setup: str, source_id: int, files: List[Path], target_dir: Path) -> Optional[dict]:
        
        # 1. PHASE A
        phase_data = self.run_phase_a(config_name, light_setup, source_id, files, target_dir)
        if not phase_data: return None

        # 2. PHASE B
        winner_b = self.run_phase_b(phase_data, source_id, target_dir)
        if not winner_b: 
            return None

        # 3. PHASE C
        opt_ev_gd, opt_skew_gd = self.run_phase_c(phase_data, source_id, target_dir)

        # Return Data Row for individual source
        return {
            'Configuration': config_name,
            'Light_Setup': light_setup,
            'Source_ID': source_id,
            'Classical_Best_EV': winner_b['ev'],
            'Classical_Skew_128': winner_b['skew_filtered'],
            'Final_GD_EV': opt_ev_gd,
            'Final_GD_Skew_128': opt_skew_gd
        }

    def run_pipeline(self):
        if not self.input_dir.exists():
            print(f"[CRITICAL] Input directory does not exist: {self.input_dir}")
            return

        print(f"[PIPELINE] Scanning input structure...")
        config_dirs = [x for x in self.input_dir.iterdir() if x.is_dir()]
        
        # Determine if we need to create nested folders at all
        create_nested_dirs = self.SAVE_HIST_PLOTS or self.SAVE_GD_PLOTS

        for config_dir in config_dirs:
            light_dirs = [x for x in config_dir.iterdir() if x.is_dir()]
            
            for light_dir in light_dirs:
                
                config_key = f"{config_dir.name}__{light_dir.name}"
                
                if config_key in self.existing_configs:
                    continue

                relative_path = light_dir.relative_to(self.input_dir)
                target_path = self.output_dir / relative_path
                
                # --- MODIFICATION: Conditional Directory Creation ---
                if create_nested_dirs:
                    self._ensure_dir(target_path)
                # ----------------------------------------------------
                
                all_images = list(light_dir.glob("*.png"))
                if not all_images: continue

                source_groups = {}
                for img in all_images:
                    try:
                        src_id = int(img.stem.split('_')[0])
                        if src_id not in source_groups: source_groups[src_id] = []
                        source_groups[src_id].append(img)
                    except Exception as e: 
                        print(f"[ERROR] Could not parse Source ID from {img.name}: {e}")
                        continue

                current_config_rows = []
                for src_id, files in source_groups.items():
                    result_row = self.process_light_source(
                        config_dir.name, light_dir.name, src_id, files, target_path
                    )
                    
                    if result_row:
                        current_config_rows.append(result_row)
                
                if len(current_config_rows) == len(source_groups) and len(source_groups) > 0:
                    self.existing_configs.add(config_key)
                    self._aggregate_and_report(config_dir.name, light_dir.name, current_config_rows)
                    print(f"[SUCCESS] Processed and aggregated: {config_dir.name}/{light_dir.name} ({len(source_groups)} sources)")
                else:
                    if len(source_groups) > 0:
                        print(f"[WARNING] Skipping aggregation for {config_dir.name}/{light_dir.name}. Processed {len(current_config_rows)}/{len(source_groups)} sources.")


        # Final Summary Export
        if self.summary_data:
            df = pd.DataFrame(self.summary_data)
            
            # Define final column order (Parsed Physical Data -> Aggregated -> Individual)
            cols = [
                # Parsed Physical Data (Point 1)
                'Configuration', 'Plane_Azimuth_Deg', 'Plane_Elevation_Deg', 'Plane_Area_cm2', 
                'Light_Setup', 'Num_Lights', 'Distance_cm', 'Light_Type', 'Light_Prop_Name',
                'Psi_Angle_Deg', 'Light_Energy',
                
                # Aggregated Results
                'Aggregated_Mean_EV_Classical', 'Aggregated_WMean_EV_Classical',
                'Aggregated_Mean_EV_GD', 'Aggregated_WMean_EV_GD', 
                
                # Individual Source Data
                'Source_ID', 'Classical_Best_EV', 'Classical_Skew_128', 
                'Final_GD_EV', 'Final_GD_Skew_128'
            ]
            
            # Filter and reorder columns
            df_final = df[[c for c in cols if c in df.columns]]
            
            df_final.to_csv(self.report_path, index=False)
            print("\n" + "="*80)
            print(f"[REPORT] Pipeline Finished. Consolidated Report saved to: {self.report_path.name}")
            print("="*80)

# --- Execution ---
if __name__ == "__main__":
    # --- Configuration Path ---
    CONFIG_FILE_PATH = r"C:\Users\vishn\Desktop\avanthik\rafad\upd_opt_exp_config.json"
    
    try:
        optimizer = HybridExposureOptimizer(CONFIG_FILE_PATH)
        optimizer.run_pipeline()
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] {e}")
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")