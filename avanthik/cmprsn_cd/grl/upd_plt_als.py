import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

class ReconstructionComparator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.base_dir = Path(self.cfg['paths']['normal_map_base_dir'])
        self.output_dir = Path(self.cfg['paths']['plot_output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_data(self):
        """Recursively finds all recon_report.csv files and combines them."""
        all_reports = []
        print(f"Searching for reports in: {self.base_dir}")
        
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.base_dir):
            if "recon_report.csv" in files:
                csv_path = Path(root) / "recon_report.csv"
                try:
                    df = pd.read_csv(csv_path)
                    all_reports.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
        
        if not all_reports:
            raise FileNotFoundError("No recon_report.csv files found in the base directory.")
            
        return pd.concat(all_reports, ignore_index=True)

    def run_analysis(self):
        master_df = self.aggregate_data()
        
        for spec in self.cfg['plot_specs']:
            print(f"Generating Plot: {spec['plot_name']}")
            
            # 1. Filter data based on fixed parameters if provided
            plot_df = master_df.copy()
            for param, value in spec.get('fixed_params', {}).items():
                plot_df = plot_df[plot_df[param] == value]
            
            if plot_df.empty:
                print(f"No data matched fixed_params for {spec['plot_name']}. Skipping.")
                continue

            # 2. Setup Plot
            plt.figure(figsize=self.cfg['style_settings']['figure_size'])
            
            x_var = spec['x_axis']
            y_var = spec['y_axis']
            group_var = spec['group_by']
            
            # 3. Loop through the "Third Variation" groups
            groups = sorted(plot_df[group_var].unique())
            for group_val in groups:
                group_data = plot_df[plot_df[group_var] == group_val].sort_values(by=x_var)
                
                plt.plot(
                    group_data[x_var], 
                    group_data[y_var], 
                    marker='o', 
                    label=f"{group_var}: {group_val}",
                    markersize=self.cfg['style_settings']['marker_size'],
                    linewidth=2
                )

            # 4. Final Styling
            plt.title(f"Comparison: {y_var.replace('_', ' ').title()} vs {x_var.title()}", fontsize=14, fontweight='bold')
            plt.xlabel(x_var.replace('_', ' ').title(), fontsize=12)
            plt.ylabel(y_var.replace('_', ' ').title(), fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title=group_var.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            save_path = self.output_dir / f"{spec['plot_name']}.png"
            plt.savefig(save_path, dpi=self.cfg['style_settings']['dpi'], bbox_inches='tight')
            plt.close()
            print(f"Saved to: {save_path}")

if __name__ == "__main__":
    # Ensure this points to your config file
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\comparison_code\upd_plot_analysis_config.json"
    comparator = ReconstructionComparator(CONFIG_PATH)
    comparator.run_analysis()