import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

class UniversalPlotter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
    def calculate_metrics(self, x, y):
        """
        Calculates m, c, R-squared, and RMSE.
        """
        if self.cfg['fit_settings']['pass_through_origin']:
            # Forced through origin: y = m*x
            # m = sum(x*y) / sum(x^2)
            m = np.sum(x * y) / np.sum(x**2)
            c = 0.0
            y_pred = m * x
        else:
            # Standard fit: y = m*x + c
            m, c = np.polyfit(x, y, 1)
            y_pred = m * x + c

        # Residuals
        residuals = y - y_pred
        
        # RMSE: Square root of the mean of squared residuals
        rmse = np.sqrt(np.mean(residuals**2))
        
        # R-squared: 1 - (Sum of Squared Residuals / Total Sum of Squares)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return m, c, r_squared, rmse

    def run(self):
        # Load Data
        csv_path = Path(self.cfg['data_settings']['csv_path'])
        if not csv_path.exists():
            print(f"Error: File {csv_path} not found.")
            return
            
        df = pd.read_csv(csv_path)
        x_col = self.cfg['data_settings']['x_column']
        y_col = self.cfg['data_settings']['y_column']
        
        data = df[[x_col, y_col]].dropna()
        x = data[x_col].values
        y = data[y_col].values

        # Setup Plot
        plt.figure(figsize=(12, 7))
        
        # Plot Raw Data
        if self.cfg['plot_options'].get('show_raw_points', True):
            plt.scatter(x, y, alpha=0.6, label='Data Points', color='blue', s=20)
        
        if self.cfg['plot_options'].get('show_raw_curve', False):
            sort_idx = np.argsort(x)
            plt.plot(x[sort_idx], y[sort_idx], alpha=0.4, label='Data Curve', color='gray')

        # Linear Fit Analysis
        if self.cfg['fit_settings']['use_linear_fit']:
            m, c, r2, rmse = self.calculate_metrics(x, y)
            
            # Line generation for plotting
            start_x = 0 if self.cfg['fit_settings']['pass_through_origin'] else min(x)
            x_plot = np.linspace(start_x, max(x), 100)
            y_plot = m * x_plot + c
            
            plt.plot(x_plot, y_plot, 
                     color=self.cfg['fit_settings']['line_color'], 
                     label=f"{self.cfg['fit_settings']['fit_label']}",
                     linewidth=2)
            
            # Construct display string
            stats_list = [
                f"Equation: y = {m:.4f}x + {c:.4f}",
                f"Slope (m): {m:.5f}",
                f"Intercept (c): {c:.5f}"
            ]
            
            if self.cfg['plot_options'].get('show_r_squared', True):
                stats_list.append(f"$R^2$: {r2:.5f}")
            
            if self.cfg['plot_options'].get('show_rmse', True):
                stats_list.append(f"RMSE: {rmse:.5f}")

            stats_text = "\n".join(stats_list)

            # Annotate text box on plot
            plt.gca().annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                               verticalalignment='top', fontsize=10, fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        # Final Polish
        plt.title(self.cfg['plot_options']['title'], fontsize=14, fontweight='bold')
        plt.xlabel(self.cfg['plot_options']['x_label'])
        plt.ylabel(self.cfg['plot_options']['y_label'])
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save output
        save_path = self.cfg['data_settings']['output_plot_path']
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Analysis complete. Plot saved to: {save_path}")

if __name__ == "__main__":
    # Ensure plot_config.json is in the same directory
    plotter = UniversalPlotter(r"C:\Users\vishn\Desktop\avanthik\comparison_code\plot_from_csv_config.json")
    plotter.run()