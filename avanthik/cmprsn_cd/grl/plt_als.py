import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional # <-- FIX: Added necessary imports for type hints
import re

# --- Data Loading and Aggregation ---

def calculate_mean_error(plot_spec: Dict, config_key: str, analysis_base_dir: Path) -> Optional[float]:
    """
    Finds the pixel data CSV for a given configuration and calculates the mean angular error.
    """
    ps_method = plot_spec['ps_method']
    ev_method = plot_spec['ev_method']
    metric_file = plot_spec['metric_file']
    
    # Example key: 45.00_45.00_0.00_0.00_0.00_9.57__4_7.00_area_rectangle_0.0_30.25_45.00_1.17
    plane_dir, light_dir = config_key.split('__')
    
    # Construct the path to the pixel-wise CSV file
    csv_path = analysis_base_dir / plane_dir / light_dir / ev_method / f"{ps_method}_{metric_file}"
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # The metric name is the first column after X_Pixel and Y_Pixel
        # For Angular_Error_Deg_Pixel_Data.csv, the columns are X_Pixel, Y_Pixel, Angular_Error_Deg
        metric_column = df.columns[2]
        
        # Calculate the mean of the metric, ignoring NaN values (non-object pixels)
        # Note: If there were NaN values due to masking, .mean() handles them, but typically 
        # the CSV generation logic ensures only object pixels are saved.
        mean_value = df[metric_column].mean()
        
        return float(mean_value)
    except Exception as e:
        print(f"[ERROR] Failed to read/process {csv_path.name}: {e}")
        return None


def assemble_master_data(config: Dict) -> pd.DataFrame:
    """
    Loads geometric data and merges it with calculated mean error metrics.
    """
    paths = config['paths']
    
    # 1. Load Geometric/Loop Data from Optimization Report
    try:
        df_geo = pd.read_csv(paths['report_csv_path'])
        # We only need the aggregate rows which contain all geometric parameters
        df_geo = df_geo[df_geo['Source_ID'] == 'AGGREGATE'].copy()
        df_geo['Config_Key'] = df_geo['Configuration'] + '__' + df_geo['Light_Setup']
    except FileNotFoundError:
        print(f"[FATAL] Report CSV not found at {paths['report_csv_path']}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[FATAL] Error loading report CSV: {e}")
        return pd.DataFrame()

    print(f"Loaded {len(df_geo)} unique geometric configurations.")

    # 2. Iterate through all required plot specifications to calculate and merge error metrics
    master_plot_data = []

    # Get unique combinations of PS/EV methods needed from all plot specs
    unique_error_calculations = []
    for spec in config['plots']:
        calc_key = (spec['ps_method'], spec['ev_method'], spec['metric_file'])
        if calc_key not in unique_error_calculations:
            unique_error_calculations.append(calc_key)
            
    # Iterate through unique combinations needed
    for ps_method, ev_method, metric_file in unique_error_calculations:
        
        # Determine the name of the column that will hold the final error metric
        # E.g., 'Joint_Least_Squares_Aggregated_WMean_EV_Classical_Mean_Angular_Error_Deg'
        metric_suffix = metric_file.replace('_Pixel_Data.csv', '')
        error_column_name = f"{ps_method}_{ev_method}_Mean_{metric_suffix}"
        
        print(f"-> Calculating {error_column_name}...")
        
        # Merge error metric into a temporary DataFrame
        df_temp = df_geo[['Config_Key', 'Configuration', 'Light_Setup']].copy()
        df_temp['PS_Method'] = ps_method
        df_temp['EV_Method'] = ev_method
        df_temp['Metric_Name'] = error_column_name # The name of the resulting metric column
        df_temp['Mean_Error_Value'] = np.nan # Placeholder for the calculated mean error
        
        # We need to iterate over indices, not rows, for efficient modification using .loc
        indices = df_geo.index.tolist() 

        for index in indices:
            config_key = df_geo.loc[index, 'Config_Key']
            
            # Use a dummy plot_spec tailored for the current PS/EV iteration
            dummy_spec = {
                'ps_method': ps_method,
                'ev_method': ev_method,
                'metric_file': metric_file
            }
            
            # Calculate the mean error from the pixel CSV
            mean_error = calculate_mean_error(dummy_spec, config_key, Path(paths['analysis_base_dir']))
            
            # Update the temporary DataFrame using .loc for safety
            if mean_error is not None:
                df_temp.loc[index, 'Mean_Error_Value'] = mean_error
        
        # Filter out rows where the error could not be calculated (files not found)
        df_temp = df_temp.dropna(subset=['Mean_Error_Value']).rename(
            columns={'Mean_Error_Value': error_column_name}
        )
        
        # Merge the geometric data (excluding duplicate keys) with the calculated error
        # We merge df_geo with the new error column from df_temp
        df_final_row = df_geo.merge(
            df_temp[['Config_Key', error_column_name, 'PS_Method', 'EV_Method', 'Metric_Name']], 
            on='Config_Key', 
            how='inner'
        )
        
        master_plot_data.append(df_final_row)
    
    if not master_plot_data:
        return pd.DataFrame()

    df_master = pd.concat(master_plot_data, ignore_index=True)

    # Clean up and prepare columns for plotting (using .loc for setting data)
    df_master.loc[:, 'Light_Prop_Name'] = df_master['Light_Prop_Name'].apply(lambda x: re.sub(r'_(?=\d+\.\d+$)', '_Area', x) if isinstance(x, str) else x)
    df_master.loc[:, 'Psi_Angle_Deg'] = df_master['Psi_Angle_Deg'].astype(float)
    df_master.loc[:, 'Num_Lights'] = df_master['Num_Lights'].astype(int)
    
    return df_master


# --- Dynamic Plotting ---

def generate_plots(df_master: pd.DataFrame, config: Dict):
    """
    Generates plots based on the configuration, handling three grouping variables.
    """
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(config['plotting']['style'])
    
    # Define a set of markers and linestyles to cycle through
    markers = ['o', 's', '^', 'D', 'p', 'h']
    linestyles = ['-', '--', '-.', ':']

    for plot_spec in config['plots']:
        x_param = plot_spec['x_axis_param']
        color_group = plot_spec['color_group_by']
        linestyle_group = plot_spec['linestyle_group_by']
        ps_method = plot_spec['ps_method']
        ev_method = plot_spec['ev_method']
        
        # Filter master data for the specific PS and EV method
        plot_df = df_master[
            (df_master['PS_Method'] == ps_method) & 
            (df_master['EV_Method'] == ev_method)
        ].copy()
        
        if plot_df.empty:
            print(f"[SKIP] No data found for PS={ps_method} and EV={ev_method}. Plot skipped.")
            continue

        # The column holding the error metric for this plot iteration
        error_column = plot_df['Metric_Name'].iloc[0]
        
        # Convert grouping columns to string for consistent indexing/legend labels
        plot_df.loc[:, color_group] = plot_df[color_group].astype(str)
        plot_df.loc[:, linestyle_group] = plot_df[linestyle_group].astype(str)

        # Start Plotting
        fig, ax = plt.subplots(figsize=config['plotting']['figure_size'], dpi=config['plotting']['dpi'])
        
        color_groups = sorted(plot_df[color_group].unique())
        linestyle_groups = sorted(plot_df[linestyle_group].unique())
        
        # Cycle through 8 distinct colors
        color_map = {name: plt.cm.get_cmap('Dark2')(i % 8) for i, name in enumerate(color_groups)}

        # Iterate over both grouping variables simultaneously
        for i, color_val in enumerate(color_groups):
            for j, linestyle_val in enumerate(linestyle_groups):
                
                # Filter for the current combination
                subset = plot_df[
                    (plot_df[color_group] == color_val) & 
                    (plot_df[linestyle_group] == linestyle_val)
                ].sort_values(by=x_param)
                
                if subset.empty:
                    continue
                
                # Build the legend label
                label = f"{color_group.replace('_', ' ')}: {color_val}, {linestyle_group.replace('_', ' ')}: {linestyle_val}"

                # Plot the line
                ax.plot(
                    subset[x_param], 
                    subset[error_column], # Use the dynamically determined error column
                    label=label,
                    marker=markers[i % len(markers)], 
                    linestyle=linestyles[j % len(linestyles)], 
                    color=color_map[color_val], 
                    linewidth=2
                )

        # Final Plot Styling
        ax.set_title(
            f"{plot_spec['plot_name']}\nPS: {ps_method}, EV: {ev_method} ({plot_spec['title_suffix']})", 
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel(f"{x_param.replace('_', ' ')}", fontsize=14)
        ax.set_ylabel(plot_spec['y_axis_label'], fontsize=14)
        
        # Move legend outside the plot for clarity
        ax.legend(title=f"Grouping Variables", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
        
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        # Save Plot
        safe_filename = re.sub(r'[^\w\-_\. ]', '_', plot_spec['plot_name'])
        output_path = output_dir / f"{safe_filename}.png"
        fig.savefig(output_path, facecolor="white", dpi=config['plotting']['dpi'])
        plt.close(fig)
        
        print(f"[SUCCESS] Generated plot: {safe_filename}.png")
        
        # --- Save Plot Data to CSV ---
        # Select key columns and the mean error
        # Use a consistent set of columns for the plot data CSV
        columns_to_save = [x_param, color_group, linestyle_group, error_column, 'PS_Method', 'EV_Method', 'Configuration', 'Light_Setup']
        csv_output_path = output_dir / f"{safe_filename}_Plot_Data.csv"
        
        # Merge back to the main DataFrame structure for saving (if needed for all columns)
        # However, saving just the used columns from the subset is cleaner:
        plot_df[columns_to_save].to_csv(csv_output_path, index=False, float_format='%.6f')
        print(f"[SUCCESS] Saved plot data to: {csv_output_path.name}")


def main():
    # Path to the new configuration file (Adjust this path!)
    CONFIG_FILE_PATH = r"c:\Users\vishn\Desktop\avanthik\comparison_code\plot_analysis_config.json"
    
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load configuration from {CONFIG_FILE_PATH}: {e}")
        return
        
    print("\n" + "="*80)
    print("PHASE 6: PHOTOMETRIC STEREO ANALYSIS AND PLOTTING")
    print("="*80)

    # 1. Assemble master data from reports and calculated pixel CSVs
    df_master = assemble_master_data(config)
    
    if df_master.empty:
        print("[PIPELINE FAILED] Could not assemble master data for plotting. Check paths and input files.")
        return

    # 2. Generate all configured plots
    generate_plots(df_master, config)

    print("\n" + "="*80)
    print("[PIPELINE COMPLETE] Analysis and Plotting Finished.")
    print("="*80)


if __name__ == "__main__":
    main()