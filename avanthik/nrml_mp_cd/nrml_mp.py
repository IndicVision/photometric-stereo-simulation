import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import math

# --- Core Helper Functions ---

def derive_normal_from_config(config_name: str, default_normal: List[float]) -> np.ndarray:
    """
    Parses the plane configuration name (Azimuth_Elevation_...) and computes 
    the ground truth normal vector (n) using spherical coordinates.
    """
    try:
        plane_parts = config_name.split('_')
        if len(plane_parts) < 2:
            raise ValueError("Configuration name is too short to contain Azimuth and Elevation.")
            
        azimuth_deg = float(plane_parts[0])
        elevation_deg = float(plane_parts[1])
        
        azimuth_rad = math.radians(azimuth_deg)
        elevation_rad = math.radians(elevation_deg)
        
        # Zenith angle (from +Z axis) is 90 - elevation
        zenith_rad = math.radians(90.0) - elevation_rad
        
        # Calculate Cartesian coordinates (Normal Vector)
        nx = math.sin(zenith_rad) * math.cos(azimuth_rad)
        ny = math.sin(zenith_rad) * math.sin(azimuth_rad)
        nz = math.cos(zenith_rad)
        
        normal = np.array([nx, ny, nz], dtype=np.float32)
        
        # Final normalization
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
             normal /= norm_val
        
        return normal
        
    except Exception as e:
        # Fallback to default
        return np.array(default_normal, dtype=np.float32)

def load_images(folder: Path, num_lights: int, ev_method_name: str) -> Optional[np.ndarray]:
    """Load and normalize images, filtering by the EV method name in the filename."""
    
    # Target file pattern: e.g., 001_Aggregated_WMean_EV_Classical.png
    image_files = sorted(folder.glob(f"*_{ev_method_name}.png"), key=lambda x: int(x.stem.split('_')[0]))
    
    if len(image_files) != num_lights:
        return None

    frames = []
    for path in image_files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    
    return np.stack(frames, axis=0)

def load_light_matrix(light_matrix_dir: Path, config_key: str) -> Optional[np.ndarray]:
    """Loads light matrix from the individual CSV file using the config_key."""
    matrix_file = light_matrix_dir / f"light_matrix_{config_key}.csv"
    
    if not matrix_file.exists():
        return None
        
    try:
        df = pd.read_csv(matrix_file)
        df_lights = df[['L_x', 'L_y', 'L_z']]
        return df_lights.values.astype(np.float32)
    except Exception as e:
        return None

def generate_object_mask(images: np.ndarray, use_mask: bool) -> np.ndarray:
    """Generate object mask from image stack, or full-image mask if disabled."""
    _, height, width, _ = images.shape
    if use_mask:
        max_intensity_per_pixel = np.max(images, axis=0)
        return np.sum(max_intensity_per_pixel, axis=2) > 0.01
    return np.ones((height, width), dtype=bool)

def get_S_pinv(S: np.ndarray) -> np.ndarray:
    """Returns the inverse or pseudo-inverse of the light matrix S."""
    if S.shape[0] == 3:
        return np.linalg.inv(S)
    return np.linalg.pinv(S)

def compute_angular_error(computed_normals: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Compute angular error in degrees between computed normals and ground truth."""
    if np.linalg.norm(ground_truth) < 1e-6:
        return np.full(computed_normals.shape[:2], np.nan)
        
    ground_truth = ground_truth / np.linalg.norm(ground_truth)
    dot_products = np.dot(computed_normals, ground_truth)
    computed_norms = np.linalg.norm(computed_normals, axis=2)
    
    angular_error = np.zeros_like(computed_norms)
    valid = computed_norms > 1e-6
    
    dot_products_clipped = np.clip(dot_products[valid], -1.0, 1.0)
    angular_error[valid] = np.arccos(dot_products_clipped) * 180.0 / np.pi
    
    return angular_error

def compute_cosine_similarity(computed_normals: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between computed normals and ground truth."""
    if np.linalg.norm(ground_truth) < 1e-6:
        return np.full(computed_normals.shape[:2], np.nan)
        
    ground_truth = ground_truth / np.linalg.norm(ground_truth)
    dot_products = np.dot(computed_normals, ground_truth)
    computed_norms = np.linalg.norm(computed_normals, axis=2)
    
    similarity = np.zeros_like(computed_norms)
    valid = computed_norms > 1e-6
    
    similarity[valid] = dot_products[valid]
    
    return np.clip(similarity, -1.0, 1.0)

# --- Normal Estimation Methods (Unchanged for logic) ---

def photometric_stereo_avg_intensity(images: np.ndarray, light_matrix: np.ndarray, object_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    num_images, height, width, _ = images.shape
    S_pinv = get_S_pinv(light_matrix)
    intensities_avg = np.mean(images, axis=3) 
    intensities_flat = intensities_avg.reshape(num_images, -1)
    G_avg_flat = S_pinv @ intensities_flat
    norms = np.linalg.norm(G_avg_flat, axis=0)
    valid = (norms > 1e-6) & object_mask.ravel()
    unit_normals_flat = np.zeros_like(G_avg_flat)
    unit_normals_flat[:, valid] = G_avg_flat[:, valid] / norms[valid]
    normals_final = unit_normals_flat.reshape(3, height, width).transpose(1, 2, 0)
    return normals_final, {} 

def photometric_stereo_avg_normals(images: np.ndarray, light_matrix: np.ndarray, object_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    num_images, height, width, _ = images.shape
    S_pinv = get_S_pinv(light_matrix)
    normals_per_channel = []
    channel_normal_maps = {}
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        intensities = images[:, :, :, channel_idx]
        intensities_flat = intensities.reshape(num_images, -1)
        normals_flat = S_pinv @ intensities_flat
        norms = np.linalg.norm(normals_flat, axis=0)
        valid = (norms > 1e-6) & object_mask.ravel()
        unit_normals_flat = np.zeros_like(normals_flat)
        unit_normals_flat[:, valid] = normals_flat[:, valid] / norms[valid]
        normals_channel = unit_normals_flat.reshape(3, height, width).transpose(1, 2, 0)
        normals_per_channel.append(normals_channel)
        channel_normal_maps[channel_name] = normals_channel
    normals_avg = np.mean(normals_per_channel, axis=0)
    norms_final = np.linalg.norm(normals_avg, axis=2, keepdims=True)
    valid_final = (norms_final > 1e-6) & object_mask[:, :, np.newaxis]
    normals_final = np.where(valid_final, normals_avg / norms_final, 0)
    return normals_final, channel_normal_maps

def photometric_stereo_joint_ls(images: np.ndarray, light_matrix: np.ndarray, object_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
    num_images, height, width, _ = images.shape
    A = np.kron(np.eye(3), light_matrix) 
    A_pinv = np.linalg.pinv(A)
    I_R = images[:, :, :, 0].reshape(num_images, -1)
    I_G = images[:, :, :, 1].reshape(num_images, -1)
    I_B = images[:, :, :, 2].reshape(num_images, -1)
    I_combined = np.concatenate([I_R, I_G, I_B], axis=0) 
    X_flat = A_pinv @ I_combined
    G_R_flat = X_flat[0:3, :]
    G_G_flat = X_flat[3:6, :]
    G_B_flat = X_flat[6:9, :]
    rho_R = np.linalg.norm(G_R_flat, axis=0)
    rho_G = np.linalg.norm(G_G_flat, axis=0)
    rho_B = np.linalg.norm(G_B_flat, axis=0)
    N_R_flat = np.zeros_like(G_R_flat)
    N_G_flat = np.zeros_like(G_G_flat)
    N_B_flat = np.zeros_like(G_B_flat)
    valid_pixels = (rho_R + rho_G + rho_B) > 1e-6
    N_R_flat[:, valid_pixels] = G_R_flat[:, valid_pixels] / (rho_R[valid_pixels] + 1e-6)
    N_G_flat[:, valid_pixels] = G_G_flat[:, valid_pixels] / (rho_G[valid_pixels] + 1e-6)
    N_B_flat[:, valid_pixels] = G_B_flat[:, valid_pixels] / (rho_B[valid_pixels] + 1e-6)
    N_avg_flat = (N_R_flat + N_G_flat + N_B_flat) / 3.0
    norms = np.linalg.norm(N_avg_flat, axis=0)
    final_valid = (norms > 1e-6) & object_mask.ravel()
    unit_normals_flat = np.zeros_like(N_avg_flat)
    unit_normals_flat[:, final_valid] = N_avg_flat[:, final_valid] / norms[final_valid]
    normals_final = unit_normals_flat.reshape(3, height, width).transpose(1, 2, 0)
    channel_normal_maps = {
        'R': N_R_flat.reshape(3, height, width).transpose(1, 2, 0),
        'G': N_G_flat.reshape(3, height, width).transpose(1, 2, 0),
        'B': N_B_flat.reshape(3, height, width).transpose(1, 2, 0)
    }
    return normals_final, channel_normal_maps

# --- NEW: CSV Saving Function ---

def save_pixel_data_csv(
    data_2d: np.ndarray, 
    data_3d: Optional[np.ndarray], 
    analysis_dir: Path, 
    method_name: str, 
    data_type: str, 
    object_mask: np.ndarray
):
    """
    Saves 2D or 3D pixel-wise data (Normals, Error, Similarity) to a CSV file.
    """
    # 1. Prepare coordinates
    y_coords, x_coords = np.where(object_mask)
    
    if data_3d is not None:
        # Saving Normals (N_x, N_y, N_z)
        data_to_save = data_3d[y_coords, x_coords]
        df_cols = ['N_X', 'N_Y', 'N_Z']
    else:
        # Saving Angular Error or Cosine Similarity (2D metrics)
        data_to_save = data_2d[y_coords, x_coords]
        # Reshape to (N, 1) for DataFrame construction
        data_to_save = data_to_save.reshape(-1, 1) 
        df_cols = [data_type]

    # 2. Construct DataFrame
    df = pd.DataFrame(data_to_save, columns=df_cols)
    df.insert(0, 'Y_Pixel', y_coords)
    df.insert(0, 'X_Pixel', x_coords)
    
    # 3. Save to CSV
    csv_filename = f"{method_name}_{data_type.replace(' ', '_')}_Pixel_Data.csv"
    output_path = analysis_dir / csv_filename
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"    Saved pixel data CSV for {data_type} to: {output_path.name}")


# --- MODIFIED: save_normal_component_map (Uses Configurable Pixel Range) ---

def save_normal_component_map(normals: np.ndarray, analysis_dir: Path, method_name: str, component_names: List[str], ps_cfg: dict):
    """
    Generates grayscale images for X, Y, Z components using configurable pixel range.
    """
    PIXEL_MIN = ps_cfg['grayscale_mapping']['pixel_min_value']
    PIXEL_MAX = ps_cfg['grayscale_mapping']['pixel_max_value']
    RANGE = PIXEL_MAX - PIXEL_MIN
    
    # 1. Map Normals (V in [-1, 1]) to Pixel Intensity (P in [PIXEL_MIN, PIXEL_MAX])
    normal_map_float = PIXEL_MIN + (RANGE / 2.0) * (normals + 1.0)
    
    # Identify foreground pixels
    object_mask_3d = np.linalg.norm(normals, axis=2, keepdims=True) > 1e-6
    
    # 2. Apply mask: foreground uses scaled value, background is set to 0.0
    normal_map_float_masked = np.where(object_mask_3d, normal_map_float, 0.0)
    
    # 3. Clip and convert to 8-bit image format (handles non-standard ranges)
    normal_map_8bit = np.clip(normal_map_float_masked, PIXEL_MIN, PIXEL_MAX).astype(np.uint8)

    # 4. Save individual components (X, Y, Z)
    for i, comp_name in enumerate(component_names):
        component_image = normal_map_8bit[:, :, i]
        output_path = analysis_dir / f"{method_name}_Normal_Map_Grayscale_{comp_name}.png"
        
        cv2.imwrite(str(output_path), component_image)
        print(f"    Saved Grayscale {comp_name} Component Map to: {output_path.name}")

# --- MODIFIED: save_normal_map_rgb (Uses Configurable Pixel Range) ---

def save_normal_map_rgb(normals: np.ndarray, output_path: Path, title: str, width: int, height: int, ps_cfg: dict):
    """
    Converts computed normals ([-1, 1]) to a standard color-coded normal map image 
    ([PIXEL_MIN, PIXEL_MAX]) using configurable pixel range. Uses R=X, G=Y, B=Z mapping.
    """
    PIXEL_MIN = ps_cfg['grayscale_mapping']['pixel_min_value']
    PIXEL_MAX = ps_cfg['grayscale_mapping']['pixel_max_value']
    RANGE = PIXEL_MAX - PIXEL_MIN
    
    # 1. Map Normals (V in [-1, 1]) to Pixel Intensity (P in [PIXEL_MIN, PIXEL_MAX])
    normal_map_float = PIXEL_MIN + (RANGE / 2.0) * (normals + 1.0)
    
    # 2. Identify foreground pixels
    object_mask_3d = np.linalg.norm(normals, axis=2, keepdims=True) > 1e-6
    
    # 3. Apply mask: foreground pixels use the calculated color, background is set to 0.0
    normal_map_float_masked = np.where(object_mask_3d, normal_map_float, 0.0)
    
    # 4. Clip and convert to 8-bit image format
    normal_map_8bit = np.clip(normal_map_float_masked, PIXEL_MIN, PIXEL_MAX).astype(np.uint8)

    # Save using OpenCV (converts RGB to BGR for file writing)
    bgr_image = cv2.cvtColor(normal_map_8bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_image)
    
    print(f"    Saved standard color Normal Map (RGB) to: {output_path.name} (Range: [{PIXEL_MIN}, {PIXEL_MAX}])")


def create_figure_with_cbar(fig_width: float, fig_height: float, include_colorbar: bool, fig_settings: dict) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_settings['dpi'])
    gs = fig.add_gridspec(1, 2, 
                          width_ratios=fig_settings['gridspec_width_ratios'], 
                          wspace=fig_settings['gridspec_wspace'])
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    if not include_colorbar:
        ax_cbar.axis('off')
    return fig, ax_main, ax_cbar

def save_heatmap_plot(data: np.ndarray, output_path: Path, title: str, label: str, 
                      vmin: float, vmax: float, cmap: str, show_axes_colorbar: bool, fig_settings: dict, 
                      width: int, height: int):
    
    # Adjust VMAX dynamically to stretch contrast if max error is small (e.g., < 5 degrees)
    data_max = np.nanmax(data) if data.size > 0 else 0.0
    final_vmax = vmax 
    
    # Apply dynamic scaling: If max error is small, set vmax to the actual max error.
    if data_max > 1e-4 and data_max < 5.0:
        final_vmax = data_max 
    
    if show_axes_colorbar:
        fig_settings['base_fig_height'] = 8
        fig_width = fig_settings['base_fig_height'] * (width / height)
        fig, ax, cbar_ax = create_figure_with_cbar(fig_width, fig_settings['base_fig_height'], True, fig_settings)
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=final_vmax,
                       origin='upper', aspect='equal', extent=[0, width, height, 0])
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'{label} (Max: {data_max:.4f}°)') 
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.tight_layout()
        fig.savefig(str(output_path), facecolor='white')
        plt.close(fig)
    else:
        h, w = data.shape
        dpi = fig_settings['dpi']
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        plt.imshow(data, cmap=cmap, origin='upper', vmin=vmin, vmax=final_vmax)
        plt.axis('off')
        plt.subplots_adjust(0, 0, 1, 1)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

# --- MODIFIED analyze_configuration with fixes ---
def analyze_configuration(plane_config_dir: Path, light_setup_dir: Path, ev_method_name: str, ps_cfg: dict, all_methods: dict):
    
    config_key = f"{plane_config_dir.name}__{light_setup_dir.name}"
    ground_truth = derive_normal_from_config(plane_config_dir.name, ps_cfg['general']['default_ground_truth_normal'])
    
    print(f"\n--- Analyzing Configuration: {config_key} (EV Method: {ev_method_name}) ---")
    
    # --- 1. Load Data and Setup ---
    light_matrix = load_light_matrix(
        Path(ps_cfg['general']['light_matrix_dir']), config_key
    )
    if light_matrix is None: 
        print(f"  [SKIP] Light matrix CSV not found for key {config_key}.")
        return
    
    num_lights = light_matrix.shape[0]
    
    images = load_images(light_setup_dir, num_lights, ev_method_name)
    if images is None: 
        print(f"  [SKIP] Required images not found for EV method {ev_method_name}.")
        return
    
    _, height, width, _ = images.shape
    
    # --- Output Folder Path (FIXED: Nested structure) ---
    analysis_dir = Path(ps_cfg['general']['base_output_dir']) / plane_config_dir.name / light_setup_dir.name / ev_method_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output path set to: {analysis_dir}")
    print(f"  Ground Truth Normal (Calculated): {ground_truth[0]:.4f}, {ground_truth[1]:.4f}, {ground_truth[2]:.4f}")
    
    # --- 2. Masking ---
    object_mask = generate_object_mask(images, ps_cfg['processing']['use_mask'])
    
    # --- 3. Core Analysis Loop (Methods) ---
    
    for method_name, func in all_methods.items():
        if ps_cfg['methods'].get(method_name.lower().replace('_', '_'), False):
            
            print(f"\n[RUN] Executing method: {method_name}")
            
            normals_final, channel_normal_maps = func(images, light_matrix, object_mask)

            normals_final[~object_mask] = 0.0
            
            # --- OUTPUT: Standard Normal Map (RGB Image) (MODIFIED) ---
            normal_map_output_path = analysis_dir / f"{method_name}_Normal_Map_RGB.png"
            save_normal_map_rgb(normals_final, normal_map_output_path, method_name, width, height, ps_cfg)
            
            # --- OUTPUT: Grayscale Component Maps (MODIFIED) ---
            save_normal_component_map(normals_final, analysis_dir, method_name, component_names=['X', 'Y', 'Z'], ps_cfg=ps_cfg)

            # --- OUTPUT: Normal Vector CSV (NEW) ---
            if ps_cfg['output']['save_pixel_csvs']:
                save_pixel_data_csv(
                    normals_final[:, :, 0], normals_final, analysis_dir, method_name, 'Normal_Vector', object_mask
                )
            
            # --- Angular Error ---
            if ps_cfg['output']['compute_angular_error']:
                angular_error = compute_angular_error(normals_final, ground_truth)
                angular_error_masked = angular_error.copy()
                angular_error_masked[~object_mask] = np.nan
                
                output_path = analysis_dir / f"{method_name}_Angular_Error.png"
                title = f"{method_name} Angular Error"
                
                valid_errors = angular_error[object_mask]
                mean_err = np.nanmean(valid_errors) if valid_errors.size > 0 else 0.0
                max_err = np.nanmax(valid_errors) if valid_errors.size > 0 else 0.0
                
                print(f"    Mean Error: {mean_err:.4f}°, Max Error: {max_err:.4f}°")
                
                save_heatmap_plot(
                    data=angular_error_masked,
                    output_path=output_path,
                    title=title,
                    label="Angular Error (degrees)",
                    vmin=0.0,
                    vmax=90.0,
                    cmap='RdYlGn_r',
                    show_axes_colorbar=ps_cfg['output']['show_axes_colorbar'],
                    fig_settings=ps_cfg['figure_settings'],
                    width=width,
                    height=height
                )
                
                # --- OUTPUT: Angular Error CSV (NEW) ---
                if ps_cfg['output']['save_pixel_csvs']:
                    save_pixel_data_csv(
                        angular_error, None, analysis_dir, method_name, 'Angular_Error_Deg', object_mask
                    )
            
            # --- Cosine Similarity ---
            if ps_cfg['output']['compute_cosine_similarity']:
                similarity_map = compute_cosine_similarity(normals_final, ground_truth)
                similarity_map_masked = similarity_map.copy()
                similarity_map_masked[~object_mask] = np.nan
                
                output_path = analysis_dir / f"{method_name}_Cosine_Similarity.png"
                title = f"{method_name} Cosine Similarity"
                
                save_heatmap_plot(
                    data=similarity_map_masked,
                    output_path=output_path,
                    title=title,
                    label="Cosine Similarity",
                    vmin=0.0,
                    vmax=1.0,
                    cmap='RdYlGn',
                    show_axes_colorbar=ps_cfg['output']['show_axes_colorbar'],
                    fig_settings=ps_cfg['figure_settings'],
                    width=width,
                    height=height
                )

                # --- OUTPUT: Cosine Similarity CSV (NEW) ---
                if ps_cfg['output']['save_pixel_csvs']:
                    save_pixel_data_csv(
                        similarity_map, None, analysis_dir, method_name, 'Cosine_Similarity', object_mask
                    )


# --- Main Function with Corrected Discovery Logic ---
def main():
    ALL_METHODS = {
        "Average_Intensity": photometric_stereo_avg_intensity,
        "Average_Normals": photometric_stereo_avg_normals,
        "Joint_Least_Squares": photometric_stereo_joint_ls
    }
    
    # --- Load Configuration ---
    CONFIG_FILE_PATH = r"C:\Users\vishn\Desktop\avanthik\normal_map_code\normal_map_config.json"
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            ps_cfg = json.load(f)
    except FileNotFoundError:
        print(f"[FATAL] Configuration file not found at: {CONFIG_FILE_PATH}")
        return
    except json.JSONDecodeError as e:
        print(f"[FATAL] Invalid JSON in config file: {e}")
        return

    BASE_INPUT = Path(ps_cfg['general']['base_input_dir'])
    
    # Target ONLY the EV method specified in the config
    TARGET_EV_METHOD = ps_cfg['general']['default_ev_method']
    
    if not BASE_INPUT.exists():
        print(f"[FATAL] Base input directory not found: {BASE_INPUT}")
        return

    print(f"Scanning base directory: {BASE_INPUT}")
    print(f"Targeting ONLY EV Method: {TARGET_EV_METHOD}")
    
    processed_count = 0
    
    # 1. Loop through plane configurations
    for plane_config_dir in BASE_INPUT.iterdir():
        if not plane_config_dir.is_dir(): continue
        
        # 2. Loop through light setups
        for light_setup_dir in plane_config_dir.iterdir():
            if not light_setup_dir.is_dir(): continue
            
            # 3. Process only if the target EV method files exist
            if list(light_setup_dir.glob(f"*_{TARGET_EV_METHOD}.png")):
                analyze_configuration(plane_config_dir, light_setup_dir, TARGET_EV_METHOD, ps_cfg, ALL_METHODS)
                processed_count += 1

    if processed_count == 0:
        print(f"[INFO] No complete configurations found using the target EV method: {TARGET_EV_METHOD}.")
        
    print("\n" + "="*70)
    print(f"[PIPELINE] Photometric Stereo Analysis Complete. Processed {processed_count} configuration(s).")
    print("="*70)

if __name__ == "__main__":
    main()