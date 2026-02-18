# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Define the main plotting class
class UniversalPlotter:
    
    # Initialize with configuration path
    def __init__(self, config_path): # input: path to configuration JSON file
        with open(config_path, 'r') as f: # Load configuration from JSON file in read mode
            self.cfg = json.load(f) # Store configuration in an instance variable

    # Calculate metrics for linear fit
    def calculate_metrics(self, x, y): # input: x and y data arrays
        """
        Calculates m, c, R-squared, and RMSE.
        """
        if self.cfg['fit_settings']['pass_through_origin']: # Check if fit should pass through origin
            # Forced through origin: y = m*x
            # m = sum(x*y) / sum(x^2)
            m = np.sum(x * y) / np.sum(x**2) # Calculate slope which minimizes squared errors
            c = 0.0 
            y_pred = m * x 

        else: # If not passing through origin
            # Standard fit: y = m*x + c
            m, c = np.polyfit(x, y, 1) # Use numpy's polyfit to get slope and intercept
            y_pred = m * x + c 

        # Residuals
        residuals = y - y_pred
        
        # RMSE: Square root of the mean of squared residuals
        rmse = np.sqrt(np.mean(residuals**2))
        
        # R-squared: 1 - (Sum of Squared Residuals / Total Sum of Squares)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return m, c, r_squared, rmse # output: slope, intercept, R-squared, RMSE

    # Main run method
    def run(self): 
        # Load CSV Data
        csv_path = Path(self.cfg['data_settings']['csv_path']) # Get CSV path from configuration
        if not csv_path.exists(): # check if file exists
            print(f"Error: File {csv_path} not found.") # Print error if file does not exist
            return

        # Read CSV data    
        df = pd.read_csv(csv_path) # Load CSV data into a DataFrame
        x_col = self.cfg['data_settings']['x_column'] # Get x-axis column name from configuration
        y_col = self.cfg['data_settings']['y_column'] # Get y-axis column name from configuration
        
        data = df[[x_col, y_col]].dropna() # Take only relevant columns and drop rows with NaN values
        x = data[x_col].values # Extract x values as numpy array
        y = data[y_col].values # Extract y values as numpy array

        # Setup Plot
        plt.figure(figsize=(12, 7)) # Create a new figure with specified size
        
        # Plot Raw Data
        """
        Default setup is to plot raw data points without raw curve
        """
        if self.cfg['plot_options'].get('show_raw_points', True): # Check if raw points should be shown
            plt.scatter(x, y, alpha=0.6, label='Data Points', color='blue', s=20) # Scatter plot of raw data points

        if self.cfg['plot_options'].get('show_raw_curve', False): # Check if raw curve should be shown
            sort_idx = np.argsort(x) # Get indices that would sort x
            plt.plot(x[sort_idx], y[sort_idx], alpha=0.4, label='Data Curve', color='gray') # Plot raw data curve

        # --- Linear Fit Analysis (Optional via bool) ---
        if self.cfg['fit_settings'].get('use_linear_fit', False): # Check boolean to decide if curve fitting is needed
            m, c, r2, rmse = self.calculate_metrics(x, y) # Calculate fit metrics
            
            # Line generation for plotting
            start_x = 0 if self.cfg['fit_settings']['pass_through_origin'] else min(x) # Determine starting x value according to fit type
            x_plot = np.linspace(start_x, max(x), 100)
            y_plot = m * x_plot + c
            
            plt.plot(x_plot, y_plot, 
                     color=self.cfg['fit_settings']['line_color'], 
                     label=f"{self.cfg['fit_settings']['fit_label']}",
                     linewidth=2)
            
            # Construct display string
            stats_list = [
                f"Equation: y = {m:.5f}x + {c:.5f}",
                f"Slope (m): {m:.5f}",
                f"Intercept (c): {c:.5f}"
            ]
            
            # Add R-squared and RMSE in plot if enabled in config
            if self.cfg['plot_options'].get('show_r_squared', True):
                stats_list.append(f"$R^2$: {r2:.5f}")
            
            if self.cfg['plot_options'].get('show_rmse', True):
                stats_list.append(f"RMSE: {rmse:.5f}")

            stats_text = "\n".join(stats_list) # Create multi-line string for stats

            # Annotate text box on plot
            plt.gca().annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                               verticalalignment='top', fontsize=10, fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # --- Threshold Lines ---
        thresholds = self.cfg.get('thresholds', {})
        x_thresh = thresholds.get('x_value')
        y_thresh = thresholds.get('y_value')
        t_color = thresholds.get('line_color', 'green')
        t_style = thresholds.get('line_style', '--')
        t_width = thresholds.get('line_width', 1.5)

        if x_thresh is not None:
            plt.axvline(x=x_thresh, color=t_color, linestyle=t_style, linewidth=t_width, label=f'X Threshold ({x_thresh})')
            
        if y_thresh is not None:
            plt.axhline(y=y_thresh, color=t_color, linestyle=t_style, linewidth=t_width, label=f'Y Threshold ({y_thresh})')

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


# main execution
if __name__ == "__main__":
    """Ensure csvTOplot_cfg.json is in the same directory or update path"""
    # Replace the path below with your actual config file path
    plotter = UniversalPlotter(r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\grl\csvTOplot_cfg.json") 
    plotter.run()