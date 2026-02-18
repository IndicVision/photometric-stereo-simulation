import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.kernel_ridge import KernelRidge

# --- USER SETTING: DEFINE THRESHOLD HERE OR IN JSON ---
DEFAULT_THRESHOLD_DEG = 0.10
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

def fit_curve(x, y, config):
    """
    Fits a curve to the data based on JSON configuration.
    Returns: Fitted Y values matching the input X.
    """
    fit_cfg = config.get('fitting', {})
    method = fit_cfg.get('method', 'polynomial').lower()
    
    # Reshape for sklearn if needed
    X_in = x.values.reshape(-1, 1)
    
    if method == 'gaussian':
        # Kernel Ridge Regression with RBF (Gaussian) kernel
        gamma = fit_cfg.get('gaussian', {}).get('gamma', 10.0)
        alpha = fit_cfg.get('gaussian', {}).get('alpha', 0.1) # Regularization
        
        try:
            krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
            krr.fit(X_in, y)
            y_fit = krr.predict(X_in)
            label = f'Gaussian Fit (gamma={gamma})'
        except Exception as e:
            print(f"Fitting Error (Gaussian): {e}")
            return y, "Fit Failed (Using Raw)"

    elif method == 'polynomial':
        degree = fit_cfg.get('polynomial', {}).get('degree', 2)
        try:
            coeffs = np.polyfit(x, y, degree)
            poly_func = np.poly1d(coeffs)
            y_fit = poly_func(x)
            label = f'Poly Fit (deg={degree})'
        except Exception as e:
            print(f"Fitting Error (Poly): {e}")
            return y, "Fit Failed (Using Raw)"
            
    else:
        # Fallback to raw if method unknown or explicitly None
        return y, "Raw Data"

    return y_fit, label

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
    json_path = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\nrml_err_als\upd_thrshld_err_obj_sz_cfg.json"
    # --------------------

    print(f"Loading config from: {json_path}")
    config = load_config(json_path)
    if config is None: return

    # Get Configuration
    threshold_deg = config.get('analysis', {}).get('threshold_error_deg', DEFAULT_THRESHOLD_DEG)
    fit_config = config.get('fitting', {})
    
    # DECISION: USE FIT OR RAW?
    use_fit = fit_config.get('use_fit_for_analysis', False)
    
    print(f"--- ANALYZING WITH THRESHOLD: {threshold_deg}° ---")
    print(f"--- FITTING METHOD: {fit_config.get('method', 'None').upper()} ---")
    print(f"--- CALCULATION SOURCE: {'BEST FIT CURVE' if use_fit else 'RAW DATA'} ---")

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

    # 5. Extract Profiles, Fit, & Calculate Limits
    
    # --- Helper to process a profile ---
    def process_profile(df_prof, x_col, err_col):
        if df_prof.empty: return df_prof, None, None, "No Data"
        
        # 1. Generate Fit
        y_fit, label = fit_curve(df_prof[x_col], df_prof[err_col], config)
        df_prof['fitted_error'] = y_fit
        
        # 2. CRITICAL: Switch Data Source for Limit Calculation
        if use_fit:
            target_col = 'fitted_error'
            calc_source = "Fit"
        else:
            target_col = err_col
            calc_source = "Raw"
            
        # 3. Find limits based on target column
        neg_lim, pos_lim = find_crossing_distance(df_prof, x_col, target_col, threshold_deg)
        
        return df_prof, neg_lim, pos_lim, label, calc_source

    # --- X-Sweep ---
    p1_c = filter_to_zoom(get_profile(df, 'x_real', 'y_real', 0, tol_y), 'x_real', zoom_lim)
    p1_c, x_neg_lim, x_pos_lim, x_lbl, x_src = process_profile(p1_c, 'x_real', 'angular_error_deg')

    # --- Y-Sweep ---
    p2_c = filter_to_zoom(get_profile(df, 'y_real', 'x_real', 0, tol_x), 'y_real', zoom_lim)
    p2_c, y_neg_lim, y_pos_lim, y_lbl, y_src = process_profile(p2_c, 'y_real', 'angular_error_deg')

    # 6. Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    def add_plot_elements(ax, data, x_col, neg_lim, pos_lim, fit_label, source_used):
        # Raw Data (Lighter)
        ax.plot(data[x_col], data['angular_error_deg'], '.', color='gray', alpha=0.3, label='Raw Data')
        
        # Fitted Data (Solid)
        if 'fitted_error' in data.columns:
            # Highlight fit if it's the active source
            style = '-' if source_used == "Fit" else '--'
            width = 2.5 if source_used == "Fit" else 1.5
            alpha = 1.0 if source_used == "Fit" else 0.6
            ax.plot(data[x_col], data['fitted_error'], style, color='blue', linewidth=width, alpha=alpha, label=fit_label)

        # Threshold Line
        ax.axhline(y=threshold_deg, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({threshold_deg}°)')
        
        # Vertical Limit Lines
        if neg_lim is not None:
            ax.axvline(x=neg_lim, color='green', linestyle='-', linewidth=2)
            ax.text(neg_lim, threshold_deg * 1.1, f'{neg_lim:.3f}cm', color='green', ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        if pos_lim is not None:
            ax.axvline(x=pos_lim, color='green', linestyle='-', linewidth=2)
            ax.text(pos_lim, threshold_deg * 1.1, f'{pos_lim:.3f}cm', color='green', ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
        # Shade Safe Zone
        safe_min = neg_lim if neg_lim is not None else ax.get_xlim()[0]
        safe_max = pos_lim if pos_lim is not None else ax.get_xlim()[1]
        ax.axvspan(safe_min, safe_max, color='green', alpha=0.1, label=f'Safe Zone ({source_used})')

    # Plot X-Sweep
    if not p1_c.empty:
        add_plot_elements(axs[0], p1_c, 'x_real', x_neg_lim, x_pos_lim, x_lbl, x_src)
    
    axs[0].set_title(f'Length (X) Error\nCalc Source: {x_src.upper()} | Limit: [{x_neg_lim if x_neg_lim else "Min"} : {x_pos_lim if x_pos_lim else "Max"}] cm')
    axs[0].set_xlabel('Position X (cm)')
    axs[0].set_ylabel('Angular Error (deg)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].set_xlim(-zoom_lim, zoom_lim) 

    # Plot Y-Sweep
    if not p2_c.empty:
        add_plot_elements(axs[1], p2_c, 'y_real', y_neg_lim, y_pos_lim, y_lbl, y_src)

    axs[1].set_title(f'Width (Y) Error\nCalc Source: {y_src.upper()} | Limit: [{y_neg_lim if y_neg_lim else "Min"} : {y_pos_lim if y_pos_lim else "Max"}] cm')
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
    print(f"Analysis Mode: {'FITTED CURVE' if use_fit else 'RAW DATA'}")
    print(f"Threshold: {threshold_deg} degrees")
    print(f"X-Axis Safe Range: {x_neg_lim} cm to {x_pos_lim} cm")
    print(f"Y-Axis Safe Range: {y_neg_lim} cm to {y_pos_lim} cm")

if __name__ == "__main__":
    main()