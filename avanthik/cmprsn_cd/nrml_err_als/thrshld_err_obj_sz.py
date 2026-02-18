import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- USER SETTING: DEFINE THRESHOLD HERE OR IN JSON ---
DEFAULT_THRESHOLD_DEG = 0.10  # Change this value as needed
# ------------------------------------------------------

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"CRITICAL ERROR: Config file not found at: {config_path}")
        return None
    with open(config_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None

def get_profile(df, sweep_col, fixed_col, fixed_val, tolerance):
    # Extract strip around the fixed value
    mask = (df[fixed_col] >= fixed_val - tolerance) & (df[fixed_col] <= fixed_val + tolerance)
    subset = df[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=[sweep_col, 'angular_error_deg'])
    return subset.sort_values(by=sweep_col)

def filter_to_zoom(df, col_name, limit):
    """Filters data to the zoom limit."""
    if df.empty: return df
    mask = (df[col_name] >= -limit) & (df[col_name] <= limit)
    return df[mask]

def find_crossing_distance(df, pos_col, error_col, threshold):
    """
    Finds the distance from center (0) where error FIRST exceeds threshold.
    Returns: (neg_limit, pos_limit)
    """
    if df.empty:
        return None, None

    # Split into Negative (Left) and Positive (Right) sides relative to center 0
    neg_side = df[df[pos_col] < 0].sort_values(by=pos_col, ascending=False) # 0 -> -inf
    pos_side = df[df[pos_col] >= 0].sort_values(by=pos_col, ascending=True) # 0 -> +inf

    # Find first failure point on Negative side (moving left from 0)
    neg_fail_dist = None
    for _, row in neg_side.iterrows():
        if row[error_col] > threshold:
            neg_fail_dist = row[pos_col]
            break # Stop at first violation
            
    # Find first failure point on Positive side (moving right from 0)
    pos_fail_dist = None
    for _, row in pos_side.iterrows():
        if row[error_col] > threshold:
            pos_fail_dist = row[pos_col]
            break # Stop at first violation

    return neg_fail_dist, pos_fail_dist

def main():
    # --- PATH TO JSON ---
    json_path = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\nrml_err_als\thrshld_err_obj_sz_cfg.json"
    # --------------------

    print(f"Loading config from: {json_path}")
    config = load_config(json_path)
    if config is None: return

    # Get Threshold from Config or use Default
    threshold_deg = config.get('analysis', {}).get('threshold_error_deg', DEFAULT_THRESHOLD_DEG)
    print(f"--- ANALYZING WITH THRESHOLD: {threshold_deg}° ---")

    # 1. Load Data
    try:
        df_errors = pd.read_csv(config['files']['errors'])
        df_mapping = pd.read_csv(config['files']['mapping'])
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    df = pd.merge(df_errors, df_mapping, on=['pixel_u', 'pixel_v'])
    if df.empty:
        print("Error: Merged dataframe is empty.")
        return

    # 2. UNIT CONVERSION (Meters -> Centimeters)
    x_range = df['x_world'].max() - df['x_world'].min()
    if x_range < 0.5: 
        print(f"Converting Meters -> Centimeters (Range: {x_range:.4f})")
        df['x_real'] = df['x_world'] * 100.0
        df['y_real'] = df['y_world'] * 100.0
    else:
        df['x_real'] = df['x_world']
        df['y_real'] = df['y_world']

    # 3. Center Data
    x_mid = (df['x_real'].max() + df['x_real'].min()) / 2
    y_mid = (df['y_real'].max() + df['y_real'].min()) / 2
    df['x_real'] = df['x_real'] - x_mid
    df['y_real'] = df['y_real'] - y_mid
    
    # 4. Parameters
    real_len = config['dimensions']['length_cm']
    real_width = config['dimensions']['width_cm']
    zoom_lim = config.get('plot_settings', {}).get('limit_view', 0.2)

    # Tolerance
    res_x = max(df['pixel_u'].nunique(), 100)
    res_y = max(df['pixel_v'].nunique(), 100)
    tol_x = (real_len / res_x) * 1.5
    tol_y = (real_width / res_y) * 1.5

    # 5. Extract Profiles & Calculate Limits
    
    # --- X-Sweep (Center Line) ---
    p1_c = filter_to_zoom(get_profile(df, 'x_real', 'y_real', 0, tol_y), 'x_real', zoom_lim)
    x_neg_lim, x_pos_lim = find_crossing_distance(p1_c, 'x_real', 'angular_error_deg', threshold_deg)

    # --- Y-Sweep (Center Line) ---
    p2_c = filter_to_zoom(get_profile(df, 'y_real', 'x_real', 0, tol_x), 'y_real', zoom_lim)
    y_neg_lim, y_pos_lim = find_crossing_distance(p2_c, 'y_real', 'angular_error_deg', threshold_deg)

    # 6. Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Helper to Plot Limit Lines ---
    def add_limit_markers(ax, neg_lim, pos_lim, threshold):
        # Draw Threshold Line
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold}°)')
        
        # Draw Vertical Limit Lines
        if neg_lim is not None:
            ax.axvline(x=neg_lim, color='green', linestyle='-', linewidth=2)
            ax.text(neg_lim, threshold * 1.1, f'{neg_lim:.3f}cm', color='green', ha='center', fontweight='bold')
        
        if pos_lim is not None:
            ax.axvline(x=pos_lim, color='green', linestyle='-', linewidth=2)
            ax.text(pos_lim, threshold * 1.1, f'{pos_lim:.3f}cm', color='green', ha='center', fontweight='bold')
            
        # Shade the "Safe Zone"
        safe_min = neg_lim if neg_lim is not None else ax.get_xlim()[0]
        safe_max = pos_lim if pos_lim is not None else ax.get_xlim()[1]
        ax.axvspan(safe_min, safe_max, color='green', alpha=0.1, label='Safe Zone')

    # Plot X-Sweep
    if not p1_c.empty:
        axs[0].plot(p1_c['x_real'], p1_c['angular_error_deg'], '-', color='black', label='Error Profile')
        add_limit_markers(axs[0], x_neg_lim, x_pos_lim, threshold_deg)
    
    axs[0].set_title(f'Length (X) Error vs Distance\nSafe Range: [{x_neg_lim if x_neg_lim else "Min"} to {x_pos_lim if x_pos_lim else "Max"}] cm')
    axs[0].set_xlabel('Position X (cm)')
    axs[0].set_ylabel('Angular Error (deg)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].set_xlim(-zoom_lim, zoom_lim) 

    # Plot Y-Sweep
    if not p2_c.empty:
        axs[1].plot(p2_c['y_real'], p2_c['angular_error_deg'], '-', color='black', label='Error Profile')
        add_limit_markers(axs[1], y_neg_lim, y_pos_lim, threshold_deg)

    axs[1].set_title(f'Width (Y) Error vs Distance\nSafe Range: [{y_neg_lim if y_neg_lim else "Min"} to {y_pos_lim if y_pos_lim else "Max"}] cm')
    axs[1].set_xlabel('Position Y (cm)')
    axs[1].set_ylabel('Angular Error (deg)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].set_xlim(-zoom_lim, zoom_lim) 

    plt.tight_layout()
    
    output_path = config['files'].get('output_plot', 'angular_error_threshold.png')
    plt.savefig(output_path)
    print(f"SUCCESS: Plot saved to: {output_path}")
    
    print("\n--- Threshold Analysis Results ---")
    print(f"Threshold: {threshold_deg} degrees")
    print(f"X-Axis Safe Range: {x_neg_lim} cm to {x_pos_lim} cm")
    print(f"Y-Axis Safe Range: {y_neg_lim} cm to {y_pos_lim} cm")

if __name__ == "__main__":
    main()