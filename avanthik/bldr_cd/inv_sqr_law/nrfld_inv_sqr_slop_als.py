import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from pathlib import Path

class SlopeComparator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.csv_path = Path(self.cfg['data_settings']['input_csv_path'])
        self.output_path = Path(self.cfg['data_settings']['output_plot_path'])
        self.theoretical_val = self.cfg['data_settings']['theoretical_slope_value']

    def run(self):
        # 1. Validation
        if not self.csv_path.exists():
            print(f"[Error] Input CSV not found at: {self.csv_path}")
            return

        # 2. Load Data
        try:
            df = pd.read_csv(self.csv_path)
            # Ensure required columns exist
            if 'Light_ID' not in df.columns or 'Slope' not in df.columns:
                print(f"[Error] CSV must contain 'Light_ID' and 'Slope' columns.")
                return
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            return

        # 3. Calculate Ratios
        # Ratio = Experimental Slope / Theoretical Value
        df['Ratio'] = df['Slope'] / self.theoretical_val

        # 4. Generate Plot
        self.plot_ratios(df)

    def plot_ratios(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by Light ID to ensure line connects sequentially
        df = df.sort_values('Light_ID')
        
        x = df['Light_ID']
        y = df['Ratio']

        # Plot Data Line & Points
        opts = self.cfg['plot_settings']
        ax.plot(x, y, marker='o', linestyle='-', 
                color=opts.get('data_point_color', 'blue'), 
                label='Slope Ratio', zorder=3)

        # Reference Line at y=1.0 (Perfect Match)
        ax.axhline(y=1.0, color=opts.get('reference_line_color', 'red'), 
                   linestyle='--', linewidth=2.0, label='Ideal Match (1.0)', zorder=2)

        # Formatting
        ax.set_title(opts.get('title', 'Slope Comparison'))
        ax.set_xlabel(opts.get('x_label', 'Light ID'))
        ax.set_ylabel(opts.get('y_label', 'Ratio'))
        
        # Force Integer Ticks on X-Axis (1, 2, 3...)
        # This prevents labels like "1.5" or "2.5"
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Optional Y-Limits to zoom in if ratios are close
        if opts.get('y_limits'):
            ax.set_ylim(opts['y_limits'])

        if opts.get('show_grid', True):
            ax.grid(True, which='both', linestyle='--', alpha=0.6, zorder=1)

        ax.legend()
        plt.tight_layout()

        # Save
        # Create parent directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        
        print(f"Comparison complete.")
        print(f"Theoretical Value: {self.theoretical_val}")
        print(f"Plot saved to: {self.output_path}")

if __name__ == "__main__":
    # --- Update the path to your config file here ---
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\nrfld_inv_sqr_slop_als_cfg.json"
    
    comparator = SlopeComparator(CONFIG_PATH)
    comparator.run()