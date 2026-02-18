import os
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- OPTIMIZATION: Custom CUDA Kernel (Strategy 3) ---
# This C++ code runs directly on the GPU.
# It handles the loop over light samples INSIDE the thread,
# eliminating the need for massive memory allocations.

CUDA_KERNEL_SOURCE = r'''
extern "C" __global__
void integrate_area_light(
    const float* P_surf,       // Surface Points (N, 3)
    const float* light_samples, // Light Sample Points (S, 3)
    const float* light_norm,    // Light Normal (3)
    float cos_half_spread,      // Spread threshold
    int num_pixels,
    int num_samples,
    float* G_out               // Output Vector (N, 3)
) {
    // 1. Calculate Global Thread ID (Pixel Index)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= num_pixels) return;

    // 2. Load Surface Point for this thread
    float px = P_surf[idx * 3 + 0];
    float py = P_surf[idx * 3 + 1];
    float pz = P_surf[idx * 3 + 2];

    // 3. Load Light Normal
    float nx = light_norm[0];
    float ny = light_norm[1];
    float nz = light_norm[2];

    // 4. Initialize Accumulators
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;

    // 5. Loop over all light samples (The heavy lifting)
    for (int s = 0; s < num_samples; s++) {
        // Load Light Sample Position
        float lx = light_samples[s * 3 + 0];
        float ly = light_samples[s * 3 + 1];
        float lz = light_samples[s * 3 + 2];

        // Vector: Surface -> Light
        float vx = lx - px;
        float vy = ly - py;
        float vz = lz - pz;

        // Distance Squared
        float dist_sq = vx*vx + vy*vy + vz*vz;
        
        // Sqrt & Safe Check (Avoid div by zero)
        float dist = sqrtf(dist_sq);
        if (dist < 1e-8f) dist = 1e-8f;

        // Normalized Direction (Surface -> Light)
        float dx = vx / dist;
        float dy = vy / dist;
        float dz = vz / dist;

        // Emission Cosine
        // dot(LightNorm, -Direction) = -dot(LightNorm, Direction)
        float cos_emit = -(nx*dx + ny*dy + nz*dz);

        // Visibility & Spread Check
        if (cos_emit >= cos_half_spread) {
            if (cos_emit < 0.0f) cos_emit = 0.0f;
            
            // Weight = (Emission * Visibility) / Dist^2
            float weight = cos_emit / dist_sq;
            
            // Accumulate
            gx += dx * weight;
            gy += dy * weight;
            gz += dz * weight;
        }
    }

    // 6. Write Result (Normalized by sample count)
    G_out[idx * 3 + 0] = gx / num_samples;
    G_out[idx * 3 + 1] = gy / num_samples;
    G_out[idx * 3 + 2] = gz / num_samples;
}
'''

class GeneralizedNormalProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compile the CUDA Kernel once at initialization
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')
        
        # Binning Configuration
        binning_cfg = self.cfg['global_settings'].get('binning', {})
        self.binning_enabled = binning_cfg.get('enabled', False)
        self.bin_width, self.bin_height = binning_cfg.get('cell_size', [1, 1])
        
        if self.binning_enabled:
            print(f"[INFO] Binning enabled: {self.bin_width}x{self.bin_height} pixel cells")

    def apply_binning(self, img):
        """
        Bin the image by averaging cells of size [bin_width, bin_height].
        Returns binned image.
        """
        if not self.binning_enabled:
            return img
        
        h, w = img.shape[:2]
        
        # Calculate binned dimensions
        binned_h = h // self.bin_height
        binned_w = w // self.bin_width
        
        # Crop to make divisible
        crop_h = binned_h * self.bin_height
        crop_w = binned_w * self.bin_width
        
        img_cropped = img[:crop_h, :crop_w]
        
        # Reshape and average
        if len(img_cropped.shape) == 2:  # Grayscale
            binned = img_cropped.reshape(binned_h, self.bin_height, 
                                        binned_w, self.bin_width).mean(axis=(1, 3))
        else:  # Color
            binned = img_cropped.reshape(binned_h, self.bin_height, 
                                        binned_w, self.bin_width, -1).mean(axis=(1, 3))
        
        return binned

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: 
            raise FileNotFoundError(f"Missing image: {img_path}")
        
        if img.ndim == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0.0 - 1.0 (float32)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        
        # Apply Gamma Correction (Linearization)
        if gamma != 1.0:
            img = np.power(img, gamma)
        
        # Apply binning BEFORE transferring to GPU
        if self.binning_enabled:
            img = self.apply_binning(img)
            
        return cp.array(img)

    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        """
        Calculates the accumulated light vector G for area lights using Custom CUDA Kernel.
        Processing is done in a single pass (No chunking needed).
        """
        pos = np.array(light_cfg['pos_m'])
        dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir'])
        samples = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        # 1. Generate Light Samples (CPU)
        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)
        
        v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
        ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v
        
        # Move to GPU
        sample_pts_gpu = cp.array(sample_pts, dtype=cp.float32)
        norm_gpu = cp.array(norm, dtype=cp.float32)
        
        # Params
        cos_half_spread = np.cos(np.deg2rad(spread_deg / 2.0))
        num_pixels = P_surf_full.shape[0]
        num_samples = sample_pts_gpu.shape[0]
        
        # Output Buffer
        G_eff_full = cp.zeros((num_pixels, 3), dtype=cp.float32)

        # 2. Configure Kernel Launch
        threads_per_block = 128
        blocks_per_grid = (num_pixels + threads_per_block - 1) // threads_per_block

        # 3. Launch Kernel
        # Arguments: (P_surf, light_samples, light_norm, cos_spread, n_pix, n_samp, output)
        self.kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                P_surf_full,        # Pixel positions (Pointer)
                sample_pts_gpu,     # Light samples (Pointer)
                norm_gpu,           # Light normal (Pointer)
                cp.float32(cos_half_spread),
                cp.int32(num_pixels),
                cp.int32(num_samples),
                G_eff_full          # Output (Pointer)
            )
        )
        
        # Note: No memory cleanup needed here because we didn't allocate massive tensors!
        return G_eff_full

    def run(self):
        print("Executing Corrected Normal Reconstruction (Optimized: Custom CUDA Kernel + Binning)...")
        
        # --- TIMER START: TOTAL ---
        t_start_total = time.time()

        # 1. Load CSV Mapping
        t0 = time.time()
        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
            
        print(f"Loading Geometry from: {csv_path.name}")
        df = pd.read_csv(csv_path)
        
        # CSV contains one entry per cell (cell centers if binning enabled)
        u_idx_centers = df['pixel_u'].values
        v_idx_centers = df['pixel_v'].values
        
        # Load Surface Points (Global Frame) - one per cell
        P_surf = cp.ascontiguousarray(cp.array(df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32))
        
        # If binning enabled, convert center coordinates to binned indices for image lookup
        if self.binning_enabled:
            # Convert center pixel coordinates to binned cell indices
            u_idx = (u_idx_centers // self.bin_width).astype(int)
            v_idx = (v_idx_centers // self.bin_height).astype(int)
            print(f"[INFO] Processing {len(u_idx)} cells (one computation per {self.bin_width}x{self.bin_height} cell)")
        else:
            u_idx = u_idx_centers.astype(int)
            v_idx = v_idx_centers.astype(int)
        
        cp.cuda.Device(0).synchronize()
        print(f"  [Time] Geometry Loading: {time.time() - t0:.4f} sec")

        # 2. Extract Offsets
        cam_u, cam_v = self.cfg['camera']['manual_center_pixel']
        center_idx = ((df['x_world'])**2 + (df['y_world'])**2).idxmin()
        obj_u = int(df.loc[center_idx, 'pixel_u'])
        obj_v = int(df.loc[center_idx, 'pixel_v'])

        # 3. Solver Setup
        num_cells = len(df)  # Number of cells to compute (one per cell)
        num_lights = len(self.cfg['lights'])
        
        I = cp.zeros((num_cells, num_lights), dtype=cp.float32)
        G = cp.zeros((num_cells, num_lights, 3), dtype=cp.float32)

        # 4. Light Sampling Loop
        print(f"Processing {num_lights} Lights...")
        t_lights_start = time.time()

        for j, l_cfg in enumerate(self.cfg['lights']):

            t_light_single = time.time()

            print(f" Processing Light {l_cfg['id']}...")
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            img_gpu = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
            
            # Extract intensities using binned coordinates (u_idx, v_idx are already binned if binning enabled)
            I[:, j] = img_gpu[v_idx, u_idx]
            
            # Compute Geometry Matrix (G) - Now calls the CUDA Kernel
            G[:, j, :] = self.get_light_samples_gpu(l_cfg, P_surf)
            
            del img_gpu

            cp.cuda.Device(0).synchronize()
            print(f"  > Light {l_cfg['id']} processed in {time.time() - t_light_single:.4f} sec")
        
        cp.cuda.Device(0).synchronize()
        print(f"  [Time] Total Light Processing: {time.time() - t_lights_start:.4f} sec")

        # 5. Solve Photometric Stereo
        print(" Solving linear system with regularization...")
        t_solve_start = time.time()
        
        GT = G.transpose(0, 2, 1) 
        GTG = cp.matmul(GT, G)
        
        lambda_reg = 1e-4
        GTG = GTG + (cp.eye(3, dtype=cp.float32) * lambda_reg)
        
        GTI = cp.matmul(GT, I[:, :, None])
        inv_GTG = cp.linalg.inv(GTG)
        n_est = cp.matmul(inv_GTG, GTI).squeeze()
        
        albedo = cp.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / cp.where(albedo == 0, 1, albedo) 

        # 6. Calculate Angular Error and Expand to All Pixels
        gt_n = cp.array([0, 0, 1], dtype=cp.float32)
        dot_prod = cp.sum(normals * gt_n, axis=1)
        dot_prod = cp.clip(dot_prod, -1.0, 1.0)
        err_deg_gpu = cp.rad2deg(cp.arccos(dot_prod))
        
        err_cpu = cp.asnumpy(err_deg_gpu)
        normals_cpu = cp.asnumpy(normals)
        
        # Expand cells to all pixels for output
        if self.binning_enabled:
            print(f"[INFO] Expanding {len(normals_cpu)} cells to full pixel grid...")
            
            # Create arrays for all pixels
            all_u_pixels = []
            all_v_pixels = []
            all_normals = []
            all_errors = []
            
            for i in range(len(u_idx_centers)):
                # Get the cell center coordinates
                u_center = u_idx_centers[i]
                v_center = v_idx_centers[i]
                
                # Calculate cell boundaries
                u_start = int((u_center // self.bin_width) * self.bin_width)
                v_start = int((v_center // self.bin_height) * self.bin_height)
                
                # Generate all pixels in this cell
                for v_offset in range(self.bin_height):
                    for u_offset in range(self.bin_width):
                        u_pixel = u_start + u_offset
                        v_pixel = v_start + v_offset
                        
                        # Check bounds
                        if u_pixel < self.cfg['resolution']['width'] and v_pixel < self.cfg['resolution']['height']:
                            all_u_pixels.append(u_pixel)
                            all_v_pixels.append(v_pixel)
                            all_normals.append(normals_cpu[i])
                            all_errors.append(err_cpu[i])
            
            # Convert to arrays
            u_idx_full = np.array(all_u_pixels)
            v_idx_full = np.array(all_v_pixels)
            normals_cpu_full = np.array(all_normals)
            err_cpu_full = np.array(all_errors)
            
            print(f"[INFO] Expanded to {len(u_idx_full)} total pixels")
        else:
            u_idx_full = u_idx_centers.astype(int)
            v_idx_full = v_idx_centers.astype(int)
            normals_cpu_full = normals_cpu
            err_cpu_full = err_cpu
        
        min_err = float(np.min(err_cpu_full))
        max_err = float(np.max(err_cpu_full))
        mean_err = float(np.mean(err_cpu_full))
        med_err = float(np.median(err_cpu_full))

        cp.cuda.Device(0).synchronize()
        print(f"  [Time] Solver & Validation: {time.time() - t_solve_start:.4f} sec")
        
        print(f" Error Analysis -> Mean: {mean_err:.2f}°, Median: {med_err:.2f}°, Max: {max_err:.2f}°")

        # --- OUTPUT GENERATION ---

        print("Generating Outputs...")
        t_io_start = time.time()

        h, w = self.cfg['resolution']['width'], self.cfg['resolution']['height'] 
        h_res, w_res = self.cfg['resolution']['height'], self.cfg['resolution']['width']

        # A. Error Stats JSON
        stats = {
            "mean_error": mean_err,
            "median_error": med_err,
            "min_error": min_err,
            "max_error": max_err,
            "offsets": {
                "camera_center_pixel": [int(cam_u), int(cam_v)],
                "object_center_pixel": [obj_u, obj_v],
                "object_center_world": [
                    float(df.loc[center_idx, 'x_world']), 
                    float(df.loc[center_idx, 'y_world']), 
                    float(df.loc[center_idx, 'z_world'])
                ]
            },
            "scaling_info": {
                "error_map_min_val": min_err,
                "error_map_max_val": max_err,
                "note": "error_degree_map.png is scaled from [min, max] to [0, 65535]"
            }
        }
        
        if self.binning_enabled:
            stats["binning_info"] = {
                "enabled": True,
                "cell_size": [self.bin_width, self.bin_height],
                "note": "Processing was done on binned data for speed"
            }
        
        with open(self.output_dir / "error_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)

        # Use expanded pixel coordinates for output maps
        out_u_idx = u_idx_full
        out_v_idx = v_idx_full

        # B. Standard Normal Map
        n_map = np.zeros((h_res, w_res, 3), dtype=np.float32)
        n_map[out_v_idx, out_u_idx] = normals_cpu_full
        n_vis_uint16 = ((n_map + 1.0) / 2.0 * 65535).astype(np.uint16)
        cv2.imwrite(str(self.output_dir / "normal_map_linear.png"), cv2.cvtColor(n_vis_uint16, cv2.COLOR_RGB2BGR))

        # C. Strict Component Map
        n_strict_bgr = np.zeros((h_res, w_res, 3), dtype=np.uint16)
        norm_x_scaled = ((normals_cpu_full[:, 0] + 1.0) / 2.0 * 65535).astype(np.uint16)
        norm_y_scaled = ((normals_cpu_full[:, 1] + 1.0) / 2.0 * 65535).astype(np.uint16)
        norm_z_scaled = ((normals_cpu_full[:, 2] + 1.0) / 2.0 * 65535).astype(np.uint16)
        
        n_strict_bgr[out_v_idx, out_u_idx, 0] = norm_z_scaled 
        n_strict_bgr[out_v_idx, out_u_idx, 1] = norm_y_scaled 
        n_strict_bgr[out_v_idx, out_u_idx, 2] = norm_x_scaled 
        cv2.imwrite(str(self.output_dir / "normal_components_scaled.png"), n_strict_bgr)

        # D. Scaled Error Map
        err_img_map = np.zeros((h_res, w_res), dtype=np.uint16)
        if max_err > min_err:
            err_norm = (err_cpu_full - min_err) / (max_err - min_err)
            err_scaled = (err_norm * 65535).astype(np.uint16)
        else:
            err_scaled = np.zeros_like(err_cpu_full, dtype=np.uint16)
        err_img_map[out_v_idx, out_u_idx] = err_scaled
        cv2.imwrite(str(self.output_dir / "error_degree_map.png"), err_img_map)

        # E. CSV Export
        df_out = pd.DataFrame({
            'pixel_u': out_u_idx,
            'pixel_v': out_v_idx,
            'normal_x': normals_cpu_full[:, 0],
            'normal_y': normals_cpu_full[:, 1],
            'normal_z': normals_cpu_full[:, 2],
            'angular_error_deg': err_cpu_full
        })
        df_out.to_csv(self.output_dir / "pixelwise_normals_errors.csv", index=False)

        # F. Detailed Heatmap
        err_plt = np.full((h_res, w_res), np.nan, dtype=np.float32)
        err_plt[out_v_idx, out_u_idx] = err_cpu_full
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(err_plt, cmap='jet', interpolation='none')
        cbar = plt.colorbar(im)
        cbar.set_label('Angular Error (Degrees)', rotation=270, labelpad=15)
        
        plt.title(f"Reconstruction Error Distribution\nMean: {mean_err:.2f}° | Median: {med_err:.2f}°", fontsize=14)
        plt.xlabel("Pixel U (Width)", fontsize=12)
        plt.ylabel("Pixel V (Height)", fontsize=12)
        plt.savefig(self.output_dir / "error_heatmap_detailed.png", dpi=150, bbox_inches='tight')
        plt.close()

        # G. Geometry Verification
        plt.figure(figsize=(10, 8))
        subset = df.iloc[::100]
        plt.scatter(subset['pixel_u'], subset['pixel_v'], c=subset['z_world'], s=1, cmap='viridis', label='Object Surface Z')
        plt.scatter(cam_u, cam_v, color='red', s=100, marker='x', label='Camera Center')
        plt.scatter(obj_u, obj_v, color='blue', s=100, marker='+', label='Object Center')
        plt.title(f"Geometry Alignment Verification\nObj World Z Mean: {df['z_world'].mean():.4f}m")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal') 
        plt.gca().invert_yaxis() 
        plt.savefig(self.output_dir / "geometry_offset_verification.png")
        plt.close()

        print(f"  [Time] Output Saving: {time.time() - t_io_start:.4f} sec")
        print(f"Processing Complete. All outputs saved to: {self.output_dir}")
        print(f"Total Execution Time: {time.time() - t_start_total:.4f} sec")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        # REPLACE WITH YOUR CONFIG PATH
        cfg_path = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\big_px_nrml_cal\bat_proc_nrml_grl_fst2_BPC.json"
    
    if os.path.exists(cfg_path):
        processor = GeneralizedNormalProcessor(cfg_path)
        processor.run()
    else:
        print(f"Please provide a valid config path. File not found: {cfg_path}")
