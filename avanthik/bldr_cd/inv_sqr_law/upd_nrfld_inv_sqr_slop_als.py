import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import math
import numpy as np
from pathlib import Path
import re

class SlopeComparatorBatch:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.base_dir = Path(self.cfg['paths']['base_search_dir'])
        self.ev_report_path = Path(self.cfg['paths']['ev_report_path'])
        self.csv_suffix = self.cfg['processing']['target_csv_suffix']
        
        # Physics Constants
        self.rho = self.cfg['physics_constants']['rho']
        self.bit_depth = self.cfg['physics_constants']['bit_depth']
        
        # Area Calculation (cm -> m conversion)
        dim_a_m = self.cfg['physics_constants']['light_dim_a_cm'] * 0.01
        dim_b_m = self.cfg['physics_constants']['light_dim_b_cm'] * 0.01
        self.area_m2 = dim_a_m * dim_b_m
        
        # Load EV Report
        self.ev_df = None
        if self.ev_report_path.exists():
            print(f"Loading EV Report from: {self.ev_report_path}")
            self.ev_df = pd.read_csv(self.ev_report_path)
        else:
            print(f"[Error] EV Report not found at {self.ev_report_path}")

    def parse_folder_params(self, folder_name):
        """
        Extracts parameters from folder name.
        Expected Format: '4_6.00_area_rectangle_180.0_45.00_1.17'
        """
        try:
            parts = folder_name.split('_')
            params = {
                'num_lights': int(parts[0]),
                'dist': float(parts[1]),
                'spread': float(parts[4]),
                'psi': float(parts[5]),
                'energy': float(parts[6])
            }
            return params
        except Exception as e:
            print(f"[Warning] Could not parse folder params for '{folder_name}': {e}")
            return None

    def extract_strategy_name(self, filename):
        """
        Extracts the strategy/method from the CSV filename to use in titles.
        Example: 'slopes_summary_Gray_Binned_Median_Poly1.csv' -> 'Gray Binned Median'
        """
        # Remove extension and known prefixes/suffixes
        name = filename.replace('.csv', '')
        name = name.replace('slopes_summary_', '')
        name = name.replace('_Poly1', '') # Remove polynomial suffix if consistent
        return name.replace('_', ' ') # Return clean readable string

    def get_exposure_value(self, sample_folder, config_folder, light_setup_folder):
        """
        Look up 'Aggregated_Mean_EV_Classical' from the EV report.
        """
        if self.ev_df is None: return 1.0
        
        method = self.cfg['processing']['exposure_method']
        
        try:
            samples_val = int(sample_folder.split('_')[1])
        except:
            samples_val = 4096 
            
        subset = self.ev_df[
            (self.ev_df['Samples'] == samples_val) & 
            (self.ev_df['Configuration'] == config_folder) & 
            (self.ev_df['Light_Setup'] == light_setup_folder)
        ]
        
        if subset.empty:
            print(f"[Warning] No EV entry found for {config_folder}/{light_setup_folder}. Using E=1.0")
            return 1.0
            
        val = subset.iloc[0].get(method)
        if val is None or pd.isna(val):
            return 1.0
            
        return float(val)

    def calculate_theoretical_slope(self, phi, E, sigma_deg):
        """
        Equation: (rho * Phi * E * (2^bit - 1)) / (pi^2 * A * sin^2(sigma))
        """
        quant_scale = (2 ** self.bit_depth) - 1
        sigma_rad = math.radians(sigma_deg / 2.0)
        sin_sq_sigma = math.sin(sigma_rad) ** 2
        
        if sin_sq_sigma < 1e-9: sin_sq_sigma = 1e-9
        
        numerator = self.rho * phi * E * quant_scale
        denominator = (math.pi ** 2) * self.area_m2 * sin_sq_sigma
        
        return numerator / denominator

    def process_file(self, csv_path):
        # 1. Context
        light_setup_dir = csv_path.parent
        config_dir = light_setup_dir.parent
        sample_dir = config_dir.parent
        
        # 2. Parameters
        l_params = self.parse_folder_params(light_setup_dir.name)
        
        if not l_params: return 
        
        # 3. EV & Physics
        E = self.get_exposure_value(sample_dir.name, config_dir.name, light_setup_dir.name)
        theo_slope = self.calculate_theoretical_slope(
            phi=l_params['energy'], E=E, sigma_deg=l_params['spread']
        )
        
        # 4. Strategy Extraction (for Plotting distinction)
        strategy_name = self.extract_strategy_name(csv_path.name)
        
        # 5. Load and Update CSV
        try:
            df = pd.read_csv(csv_path)
            
            # --- CRITICAL UPDATE: Calculate Ratio ---
            df['Theoretical_Slope'] = theo_slope
            df['Slope_Ratio'] = df['Slope'] / theo_slope
            
            df.to_csv(csv_path, index=False)
            print(f"Updated: {csv_path.name} | Strategy: {strategy_name}")
            
        except Exception as e:
            print(f"[Error] Failed processing {csv_path}: {e}")
            return

        # 6. Generate Plot
        self.plot_comparison(df, csv_path, l_params, strategy_name, theo_slope)

    def plot_comparison(self, df, csv_path, l_params, strategy_name, theo_slope):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        df = df.sort_values('Light_ID')
        
        # Plot Ratio
        ax.plot(df['Light_ID'], df['Slope_Ratio'], 
                marker='o', linestyle='-', linewidth=2, 
                color=self.cfg['plot_settings'].get('data_color', 'blue'),
                label=f'Ratio ({strategy_name})')
        
        # Reference Line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal (1.0)')
        
        # Dynamic Title including STRATEGY
        title_str = (f"Slope Ratio Analysis: {strategy_name}\n"
                     f"Dist: {l_params['dist']}cm | Spread: {l_params['spread']}° | Energy: {l_params['energy']}W")
        
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_xlabel("Light Source ID", fontsize=10)
        ax.set_ylabel("Slope Ratio (Measured / Theoretical)", fontsize=10)
        
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
        
        # Info Box
        info_text = (f"Theoretical Slope: {theo_slope:.2f}\n"
                     f"Mean Ratio: {df['Slope_Ratio'].mean():.4f}\n"
                     f"Std Dev: {df['Slope_Ratio'].std():.4f}")
        
        plt.gca().annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        
        # --- SAFE SAVING ---
        # Use the original CSV filename stem to create the image name.
        # Example: slopes_summary_Gray_Binned_Median_Poly1.csv 
        # Becomes: slope_ratio_plot_slopes_summary_Gray_Binned_Median_Poly1.png
        # This prevents 'Binned_Mean' results from overwriting 'Binned_Median' results.
        
        out_name = f"slope_ratio_plot_{csv_path.stem}.png"
        out_path = csv_path.parent / out_name
        plt.savefig(out_path, dpi=300)
        plt.close()

    def run(self):
        print(f"Scanning for files ending in '{self.csv_suffix}'...")
        files = list(self.base_dir.rglob(f"*{self.csv_suffix}"))
        
        if not files:
            print("[Warning] No matching CSV files found.")
            return

        print(f"Found {len(files)} files. Processing batch...")
        
        for f in files:
            self.process_file(f)

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\upd_nrfld_inv_sqr_slop_als_cfg.json"
    SlopeComparatorBatch(CONFIG_PATH).run()