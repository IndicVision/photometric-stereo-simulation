import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import time
import json
import os
import sys

def main():
    t_start_total = time.time()
    print('========================================')
    print('Surface Integration: Fixed Aspect & Interactive 3D')
    print('MATCHING MATLAB "TRUTH" LOGIC')
    print('========================================\n')

    # --- CONFIGURATION ---
    # 1. Try hardcoded path first, then fall back to local directory
    config_path_hardcoded = r"C:\Users\vishn\Desktop\avanthik\dpth_mp_cd\dpth_frm_csv_json.json"
    
    if os.path.exists(config_path_hardcoded):
        config_file = config_path_hardcoded
    elif len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "dpth_frm_csv_json.json"

    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        print("Please provide the full path or place the JSON in the script directory.")
        return

    with open(config_file, 'r') as f:
        cfg = json.load(f)

    FILENAME_CSV = cfg['io']['input_csv']
    OUTPUT_DIR   = cfg['io']['output_dir']
    TARGET_LEN_CM = cfg['geometry']['target_len_cm']
    TARGET_BRE_CM = cfg['geometry']['target_bre_cm']
    EPS_NZ = cfg['integration']['epsilon_nz']
    SHOW_PLOTS = cfg['visualization']['show_plots']
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. LOAD DATA ---
    print('[1/6] Loading data...')
    if not os.path.exists(FILENAME_CSV):
        print(f"Error: CSV not found at {FILENAME_CSV}")
        return
        
    df = pd.read_csv(FILENAME_CSV)
    u_idx = df['pixel_u'].values
    v_idx = df['pixel_v'].values
    
    # MATLAB: M_img = max(v_idx) (if 1-based) -> here max + 1 for size
    M_img = np.max(v_idx) + 1 
    N_img = np.max(u_idx) + 1
    
    nx = np.zeros((M_img, N_img))
    ny = np.zeros((M_img, N_img))
    nz = np.zeros((M_img, N_img))
    mask = np.zeros((M_img, N_img), dtype=bool)
    
    nx[v_idx, u_idx] = df['normal_x'].values
    # MATCH MATLAB: ny = -T.normal_y (Inverting Y-normal)
    ny[v_idx, u_idx] = -df['normal_y'].values 
    nz[v_idx, u_idx] = df['normal_z'].values
    mask[v_idx, u_idx] = True
    
    M, N = nx.shape
    print(f"  Grid size: {M} x {N}")
    print(f"  Valid pixels: {np.count_nonzero(mask)}")

    # --- 2. GEOMETRY ---
    print('[2/6] Computing gradients...')
    # MATCH MATLAB: linspace centered at 0
    x = np.linspace(-TARGET_LEN_CM/2, TARGET_LEN_CM/2, N)
    y = np.linspace(-TARGET_BRE_CM/2, TARGET_BRE_CM/2, M)
    X, Y = np.meshgrid(x, y)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # MATCH MATLAB: valid = mask & abs(nz) > eps_nz
    valid = mask & (np.abs(nz) > EPS_NZ)
    
    p = np.zeros((M, N))
    q = np.zeros((M, N))
    
    # MATCH MATLAB: p = -nx ./ nz * dx
    p[valid] = -nx[valid] / nz[valid] * dx
    q[valid] = -ny[valid] / nz[valid] * dy

    # --- 3. BUILD SYSTEM ---
    print('[3/6] Building sparse Poisson system...')
    node_ids = np.zeros((M, N), dtype=int) - 1
    num_unknowns = np.count_nonzero(mask)
    node_ids[mask] = np.arange(num_unknowns)
    
    # Horizontal Neighbors (Right)
    mask_H = mask[:, :-1] & mask[:, 1:]
    r_H, c_H = np.where(mask_H)
    id_self_H = node_ids[r_H, c_H]
    id_right_H = node_ids[r_H, c_H + 1]
    val_p = p[r_H, c_H] # Note: Includes 0s if pixel is valid mask but invalid normal (steep)
    
    # Vertical Neighbors (Down)
    mask_V = mask[:-1, :] & mask[1:, :]
    r_V, c_V = np.where(mask_V)
    id_self_V = node_ids[r_V, c_V]
    id_down_V = node_ids[r_V + 1, c_V]
    val_q = q[r_V, c_V]
    
    num_H = len(id_self_H)
    num_V = len(id_self_V)
    num_eq = num_H + num_V + 1
    
    row_idx = []
    col_idx = []
    vals = []
    b = np.zeros(num_eq)
    
    # Horizontal Equations: Z_right - Z_self = p
    row_idx.append(np.arange(num_H)); col_idx.append(id_right_H); vals.append(np.ones(num_H))
    row_idx.append(np.arange(num_H)); col_idx.append(id_self_H);  vals.append(np.full(num_H, -1.0))
    b[0:num_H] = val_p
    
    # Vertical Equations: Z_down - Z_self = q
    # Offset rows by num_H
    row_idx.append(np.arange(num_H, num_H+num_V)); col_idx.append(id_down_V); vals.append(np.ones(num_V))
    row_idx.append(np.arange(num_H, num_H+num_V)); col_idx.append(id_self_V); vals.append(np.full(num_V, -1.0))
    b[num_H:num_H+num_V] = val_q
    
    # Anchor Equation: Z_anchor = 0
    anchor_eq = num_eq - 1
    # Find first valid pixel index (like MATLAB 'find(mask, 1)')
    flat_idx = np.argmax(mask.flatten()) 
    anchor_r, anchor_c = np.unravel_index(flat_idx, (M, N))
    
    row_idx.append([anchor_eq]); col_idx.append([node_ids[anchor_r, anchor_c]]); vals.append([1.0])
    b[anchor_eq] = 0.0 # MATCH MATLAB: Anchor to 0
    
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(row_idx), np.concatenate(col_idx))), shape=(num_eq, num_unknowns))

    # --- 4. SOLVE ---
    print('[4/6] Solving linear system (Normal Equations)...')
    # MATLAB uses Cholesky on Normal Eq: (A'A)x = A'b
    # spla.spsolve is efficiently solving Ax=b or Normal Eq.
    # We explicitly form Normal Equations to match MATLAB's least-squares approach exactly
    AtA = A.T @ A
    Atb = A.T @ b
    z = spla.spsolve(AtA, Atb)

    # --- 5. RECONSTRUCT ---
    print('[5/6] Reconstructing depth map...')
    Z = np.full((M, N), np.nan)
    Z[mask] = z
    
    # Grounding (Global) - Used for Height Map
    edges = np.concatenate([Z[0, :], Z[-1, :], Z[:, 0], Z[:, -1]])
    edges = edges[~np.isnan(edges)]
    ground_level = np.median(edges) if edges.size > 0 else 0
    Z_grounded = Z - ground_level

    # --- 6. VISUALIZATION ---
    print('[6/6] Generating visualizations...')
    
    # Crop to valid area
    r_idx_list, c_idx_list = np.where(mask)
    r_min, r_max = r_idx_list.min(), r_idx_list.max()
    c_min, c_max = c_idx_list.min(), c_idx_list.max()
    
    X_crop = X[r_min:r_max+1, c_min:c_max+1]
    Y_crop = Y[r_min:r_max+1, c_min:c_max+1]
    Z_crop = Z_grounded[r_min:r_max+1, c_min:c_max+1]
    
    limit_val = np.nanmax(np.abs(Z_crop))
    if limit_val == 0: limit_val = 1e-5

    # Aspect Ratio Setup
    phys_w = X_crop.max() - X_crop.min()
    phys_h = Y_crop.max() - Y_crop.min()
    base_size = 6.0
    fig_w = base_size
    fig_h = base_size * (phys_h / phys_w) 
    
    # MATCH MATLAB: Red-Black-Green Colormap
    # MATLAB: reds=[1..0], greens=[0..1]. combined.
    colors_list = [(1, 0, 0), (0, 0, 0), (0, 1, 0)] # Red -> Black -> Green
    cmap_rbg = LinearSegmentedColormap.from_list('rbg', colors_list, N=64)

    # 1. Deviation Map
    fig_dev = plt.figure(figsize=(fig_w + 1.2, fig_h)) 
    ax_dev = fig_dev.add_subplot(111)
    im = ax_dev.imshow(Z_crop, extent=[X_crop.min(), X_crop.max(), Y_crop.max(), Y_crop.min()], 
               cmap=cmap_rbg, vmin=-limit_val, vmax=limit_val, origin='upper')
    ax_dev.set_title('Deviation Map')
    ax_dev.set_xlabel('X (cm)')
    ax_dev.set_ylabel('Y (cm)')
    plt.colorbar(im, ax=ax_dev)
    fig_dev.savefig(os.path.join(OUTPUT_DIR, 'deviation_map.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_dev)

    # 2. Height Map
    fig_hmap = plt.figure(figsize=(fig_w + 1.2, fig_h))
    ax_hmap = fig_hmap.add_subplot(111)
    im_h = ax_hmap.imshow(Z_crop, extent=[X_crop.min(), X_crop.max(), Y_crop.max(), Y_crop.min()], 
               cmap='gray', origin='upper')
    ax_hmap.set_title('Height Map')
    ax_hmap.set_xlabel('X (cm)')
    ax_hmap.set_ylabel('Y (cm)')
    plt.colorbar(im_h, ax=ax_hmap)
    fig_hmap.savefig(os.path.join(OUTPUT_DIR, 'height_map.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_hmap)

    # 3. Cross Sections with Linear Detrend (MATCHING MATLAB)
    # Find middle indices
    mid_r_rel = (r_max - r_min) // 2
    mid_c_rel = (c_max - c_min) // 2
    
    # X-Z Section (Horizontal Cut)
    z_slice_x = Z_crop[mid_r_rel, :]
    x_slice = X_crop[mid_r_rel, :]
    valid_x = ~np.isnan(z_slice_x)
    
    if np.any(valid_x):
        x_plot = x_slice[valid_x]
        z_plot = z_slice_x[valid_x]
        # MATCH MATLAB: Linear Detrend (Pin edges to 0)
        p_poly = np.polyfit([x_plot[0], x_plot[-1]], [z_plot[0], z_plot[-1]], 1)
        z_trend = np.polyval(p_poly, x_plot)
        z_flat_x = z_plot - z_trend

        fig_xc = plt.figure(figsize=(8, 4))
        plt.plot(x_plot, z_flat_x, 'b-', label='Estimated')
        plt.fill_between(x_plot, z_flat_x, color='b', alpha=0.1)
        plt.axhline(0, color='r', linestyle='-', label='Reference')
        plt.title(f'X-Z Cross-Section (Detrended)')
        plt.xlabel('X (cm)')
        plt.ylabel('Z (cm)')
        plt.legend()
        plt.grid(True)
        fig_xc.savefig(os.path.join(OUTPUT_DIR, 'cross_section_X.png'), dpi=150)
        plt.close(fig_xc)

    # Y-Z Section (Vertical Cut)
    z_slice_y = Z_crop[:, mid_c_rel]
    y_slice = Y_crop[:, mid_c_rel]
    valid_y = ~np.isnan(z_slice_y)

    if np.any(valid_y):
        y_plot = y_slice[valid_y]
        z_plot = z_slice_y[valid_y]
        # MATCH MATLAB: Linear Detrend
        p_poly = np.polyfit([y_plot[0], y_plot[-1]], [z_plot[0], z_plot[-1]], 1)
        z_trend = np.polyval(p_poly, y_plot)
        z_flat_y = z_plot - z_trend

        fig_yc = plt.figure(figsize=(8, 4))
        plt.plot(y_plot, z_flat_y, 'b-', label='Estimated')
        plt.fill_between(y_plot, z_flat_y, color='b', alpha=0.1)
        plt.axhline(0, color='r', linestyle='-', label='Reference')
        plt.title(f'Y-Z Cross-Section (Detrended)')
        plt.xlabel('Y (cm)')
        plt.ylabel('Z (cm)')
        plt.legend()
        plt.grid(True)
        fig_yc.savefig(os.path.join(OUTPUT_DIR, 'cross_section_Y.png'), dpi=150)
        plt.close(fig_yc)

    # --- 4. Interactive 3D (Keep Python Bonus) ---
    print("  -> Generating Interactive 3D Model...")
    skip = 1 
    if M > 500 or N > 500: skip = 2
        
    fig_plotly = go.Figure(data=[go.Surface(
        x=X_crop[::skip, ::skip], 
        y=Y_crop[::skip, ::skip], 
        z=Z_crop[::skip, ::skip],
        colorscale='RdBu', 
        cmin=-limit_val, cmax=limit_val
    )])

    fig_plotly.update_layout(
        title='Interactive 3D Reconstruction',
        autosize=False,
        width=800, height=800,
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data' 
        )
    )
    
    path_html = os.path.join(OUTPUT_DIR, 'reconstruction_3d.html')
    fig_plotly.write_html(path_html)
    print(f"  -> Saved Interactive 3D: {path_html}")

    if SHOW_PLOTS:
        # Since we closed plots, we reopen the images or just rely on saved files.
        # Here we just open the HTML.
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(path_html))

    print(f"Total Time: {time.time() - t_start_total:.4f} s")

if __name__ == "__main__":
    main()