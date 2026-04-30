import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

def process_from_config(config_file_path):
    # 1. Load Configuration
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return

    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            
        img1_path = config.get("input_image_1")
        img2_path = config.get("input_image_2")
        out_csv_path = config.get("output_diff_csv", "output.csv")
        out_heat_path = config.get("output_heatmap_image", "heatmap.png")
        K = float(config.get("K", 1.0))
        
    except Exception as e:
        print(f"Error parsing config: {e}")
        return

    print(f"--- Configuration Loaded ---")
    print(f"Input 1: {img1_path}")
    print(f"Input 2: {img2_path}")
    print(f"Multiplier K: {K}")
    print(f"----------------------------")

    # 2. Load Images (Unchanged depth)
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images.")
        return

    # Convert to grayscale if needed
    if len(img1.shape) > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Check Resolution
    if img1.shape != img2.shape:
        raise ValueError(f"Resolution Mismatch: {img1.shape} vs {img2.shape}")

    # 4. Calculate Difference
    # Convert to float64 for precision
    arr1 = img1.astype(np.float64)
    arr2 = img2.astype(np.float64)
    
    # Raw signed difference: (Input 1 - Input 2)
    diff_raw = arr1 - arr2 
    
    # Apply Multiplier K
    diff_scaled = diff_raw * K

    # ==========================================
    # OUTPUT 1: CSV File (pixel_u, pixel_v, error)
    # ==========================================
    
    print("Generating CSV data...")
    
    # Create coordinate grids
    # v = row indices (Y), u = column indices (X)
    rows, cols = diff_scaled.shape
    v_coords, u_coords = np.indices((rows, cols))
    
    # Flatten arrays
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    error_flat = diff_scaled.flatten()
    
    # Create DataFrame
    df = pd.DataFrame({
        'pixel_u': u_flat,
        'pixel_v': v_flat,
        'error': error_flat
    })
    
    # Save to CSV
    df.to_csv(out_csv_path, index=False)
    print(f"[Success] Saved CSV (u, v, error) to: {out_csv_path}")


    # ==========================================
    # OUTPUT 2: Heatmap with Statistics (Visual)
    # ==========================================
    
    stats = {
        "Max": np.max(diff_raw),
        "Min": np.min(diff_raw),
        "Mean": np.mean(diff_raw),
        "Std": np.std(diff_raw),
    }

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_txt = fig.add_subplot(gs[1])

    # Heatmap Plot (Using Raw Difference for visualization context)
    max_abs_val = max(abs(stats['Min']), abs(stats['Max']))
    if max_abs_val == 0: max_abs_val = 1 # Prevent warning if flat image
        
    cax = ax_map.imshow(diff_raw, cmap='seismic', vmin=-max_abs_val, vmax=max_abs_val)
    
    cbar = fig.colorbar(cax, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('Raw Intensity Difference (No K applied)')

    ax_map.set_title(f"Difference Heatmap\nResolution: {cols}x{rows}")
    ax_map.set_xlabel("Pixel U (Column)")
    ax_map.set_ylabel("Pixel V (Row)")

    # Statistics Text
    ax_txt.axis('off')
    stats_text = (
        f"STATISTICS REPORT\n"
        f"=================\n\n"
        f"Max Diff:\n {stats['Max']:.4f}\n\n"
        f"Min Diff:\n {stats['Min']:.4f}\n\n"
        f"Mean Diff:\n {stats['Mean']:.4f}\n\n"
        f"Std Dev:\n {stats['Std']:.4f}\n\n"
        f"-----------------\n"
        f"Config K:\n {K}\n"
    )
    
    ax_txt.text(0.05, 0.5, stats_text, transform=ax_txt.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", edgecolor="black", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_heat_path, dpi=150)
    print(f"[Success] Saved Heatmap to: {out_heat_path}")
    plt.close()

if __name__ == "__main__":
    config_path = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\inten_diff_heatmp_cfg.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    process_from_config(config_path)