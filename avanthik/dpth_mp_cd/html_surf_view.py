import sys
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def generate_html_from_csv(config_path):
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    print(f"Reading CSV data from: {cfg['input_csv']}")
    df = pd.read_csv(cfg['input_csv'])

    # 1. Reconstruct the 2D grid from the 1D CSV columns
    u_idx = df['pixel_u'].values.astype(int)
    v_idx = df['pixel_v'].values.astype(int)

    max_u = np.max(u_idx) + 1
    max_v = np.max(v_idx) + 1

    Z_grid = np.full((max_v, max_u), np.nan)
    X_grid = np.full((max_v, max_u), np.nan)
    Y_grid = np.full((max_v, max_u), np.nan)

    # Use the normalized extrusion depth if available, otherwise raw world Z
    z_col = 'z_extrusion' if 'z_extrusion' in df.columns else 'z_world'
    
    Z_grid[v_idx, u_idx] = df[z_col].values
    X_grid[v_idx, u_idx] = df['x_world'].values
    Y_grid[v_idx, u_idx] = df['y_world'].values

    # 2. Crop to the valid masked area to remove wasted empty space
    valid_mask = ~np.isnan(Z_grid)
    r_idx, c_idx = np.where(valid_mask)
    r_min, r_max = np.min(r_idx), np.max(r_idx)
    c_min, c_max = np.min(c_idx), np.max(c_idx)

    Z_crop = Z_grid[r_min:r_max+1, c_min:c_max+1]
    X_crop = X_grid[r_min:r_max+1, c_min:c_max+1]
    Y_crop = Y_grid[r_min:r_max+1, c_min:c_max+1]

    # 3. Convert all measurements from meters to millimeters (mm)
    Z_mm = Z_crop * 1000.0
    X_mm = X_crop * 1000.0
    Y_mm = Y_crop * 1000.0

    # 4. Downsample to prevent WebGL browser crashes
    target_max = cfg['visualization'].get('downsample_target_max_pixels', 600)
    stride = max(1, max(Z_crop.shape) // target_max)
    print(f"Applying downsample stride of {stride} (Target Max: {target_max}px)")

    Z_render = Z_mm[::stride, ::stride].copy()
    X_render = X_mm[::stride, ::stride].copy()
    Y_render = Y_mm[::stride, ::stride].copy()

    # 5. Apply visual flips if requested
    if cfg['visualization'].get('flip_y', False):
        Z_render = np.flipud(Z_render)
    if cfg['visualization'].get('flip_x', False):
        Z_render = np.fliplr(Z_render)

    # 6. Build the Plotly 3D Figure
    print("Generating 3D mesh...")
    fig = go.Figure(data=[go.Surface(
        z=Z_render,
        x=X_render,
        y=Y_render,
        colorscale=cfg['visualization'].get('colorscale', 'RdYlBu'),
        lighting=cfg.get('lighting', {}),
        lightposition=cfg.get('light_position', {})
    )])

    # 7. Calculate Aspect Ratio and Apply Exaggeration
    range_x = float(np.nanmax(X_render) - np.nanmin(X_render))
    range_y = float(np.nanmax(Y_render) - np.nanmin(Y_render))
    range_z = float(np.nanmax(Z_render) - np.nanmin(Z_render))
    
    # Fallbacks in case of completely flat surfaces to prevent division by zero
    if range_x == 0: range_x = 1.0
    if range_y == 0: range_y = 1.0
    if range_z == 0: range_z = 1.0

    z_exag = float(cfg['visualization'].get('z_exaggeration', 1.0))

    fig.update_layout(
        title=f"3D Surface Reconstruction (Exaggeration: {z_exag}x)",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='manual',
            aspectratio=dict(
                x=1.0,
                y=range_y / range_x,
                z=(range_z / range_x) * z_exag
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 8. Save output
    output_path = cfg['output_html']
    fig.write_html(output_path)
    print(f"SUCCESS! Interactive HTML saved to: {output_path}")

if __name__ == "__main__":
    # You can pass the JSON file path as a command line argument, or it defaults to a local file
    config_file = r"C:\Users\vishn\Desktop\avanthik\dpth_mp_cd\html_surf_view_cfg.json"
    generate_html_from_csv(config_file)