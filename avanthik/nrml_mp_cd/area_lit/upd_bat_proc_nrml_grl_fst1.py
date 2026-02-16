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

# --- OPTIMIZATION START: Fused Kernels ---

@cp.fuse()
def fused_safe_sqrt(dist_sq):
    """
    Combines square root and safety clamping.
    """
    return cp.maximum(cp.sqrt(dist_sq), 1e-8)

@cp.fuse()
def fused_compute_weight(cos_emit, cos_half_spread, dist_sq):
    """
    Combines spread check, clamping, and inverse square law weighting.
    FIX: Removed 'type(cos_emit)(0)' which caused the crash. 
         Used simple scalar '0' which CuPy handles automatically.
    """
    # Simply using 0 works because CuPy broadcasts scalars in fused kernels correctly.
    return (cp.maximum(0, cos_emit) * (cos_emit >= cos_half_spread)) / dist_sq

# --- OPTIMIZATION END ---

class GeneralizedNormalProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: 
            raise FileNotFoundError(f"Missing image: {img_path}")
        
        if img.ndim == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 0.0 - 1.0 (float32)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        
        # Apply Gamma Correction
        if gamma != 1.0:
            img = np.power(img, gamma)
            
        return cp.array(img)

    def get_light_samples_gpu(self, light_cfg, P_surf_full, chunk_size=25000):
        """
        Calculates the accumulated light vector G for area lights.
        OPTIMIZED: Uses cp.fuse() kernels for element-wise math.
        """
        pos = np.array(light_cfg['pos_m'])
        dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir']) 
        samples = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        # 1. Create grid of points on the area light
        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)
        
        # Determine local axes for the light plane
        v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
        ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v
        sample_pts_gpu = cp.array(sample_pts, dtype=cp.float32)
        norm_gpu = cp.array(norm, dtype=cp.float32)
        
        # Pre-calculate spread threshold
        cos_half_spread = np.cos(np.deg2rad(spread_deg / 2.0))

        num_pixels = P_surf_full.shape[0]
        G_eff_full = cp.zeros((num_pixels, 3), dtype=cp.float32)

        # 2. Batch processing loop
        for i in range(0, num_pixels, chunk_size):
            end = min(i + chunk_size, num_pixels)
            P_chunk = P_surf_full[i:end] 

            # Vector from Surface Pixel -> Light Sample
            # Shape: (Num_Light_Samples, Batch_Size, 3)
            vec_surf_to_light = sample_pts_gpu[:, None, :] - P_chunk[None, :, :]
            
            # Distance Squared
            dist_sq = cp.sum(vec_surf_to_light**2, axis=2)
            
            # [OPTIMIZED] Fused Sqrt + Max
            dist = fused_safe_sqrt(dist_sq)
            
            # Unit Direction (Surface -> Light)
            dir_surf_to_light = vec_surf_to_light / dist[:, :, None]

            # 3. Emission Calculation
            cos_emit = -cp.sum(dir_surf_to_light * norm_gpu, axis=2)
            
            # [OPTIMIZED] Fused Logic
            weight = fused_compute_weight(cos_emit, cos_half_spread, dist_sq)
            
            # 6. Integrate
            G_chunk = cp.sum(dir_surf_to_light * weight[:, :, None], axis=0)
            
            # Normalize
            G_eff_full[i:end] = G_chunk / sample_pts_gpu.shape[0]
            
            cp.get_default_memory_pool().free_all_blocks()

        return G_eff_full

    def run(self):
        print("Executing Corrected Normal Reconstruction (Optimized with Fused Kernels)...")
        
        # --- TIMER START: TOTAL ---
        t_start_total = time.time()

        # 1. Load CSV Mapping
        t0 = time.time()
        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
            
        print(f"Loading Geometry from: {csv_path.name}")
        df = pd.read_csv(csv_path)
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        
        P_surf = cp.array(df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32)
        cp.cuda.Device(0).synchronize()
        print(f"  [Time] Geometry Loading: {time.time() - t0:.4f} sec")

        # 2. Extract Offsets
        cam_u, cam_v = self.cfg['camera']['manual_center_pixel']
        center_idx = ((df['x_world'])**2 + (df['y_world'])**2).idxmin()
        obj_u = int(df.loc[center_idx, 'pixel_u'])
        obj_v = int(df.loc[center_idx, 'pixel_v'])

        # 3. Solver Setup
        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])
        
        I = cp.zeros((num_pixels, num_lights), dtype=cp.float32)
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)

        # 4. Light Sampling Loop
        print(f"Processing {num_lights} Lights...")
        t_lights_start = time.time()

        for j, l_cfg in enumerate(self.cfg['lights']):

            t_light_single = time.time()

            print(f" Processing Light {l_cfg['id']}...")
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            img_gpu = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
            
            I[:, j] = img_gpu[v_idx, u_idx]
            
            G[:, j, :] = self.get_light_samples_gpu(l_cfg, P_surf)
            
            del img_gpu
            cp.get_default_memory_pool().free_all_blocks()

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

        # 6. Calculate Angular Error
        gt_n = cp.array([0, 0, 1], dtype=cp.float32)
        dot_prod = cp.sum(normals * gt_n, axis=1)
        dot_prod = cp.clip(dot_prod, -1.0, 1.0)
        err_deg_gpu = cp.rad2deg(cp.arccos(dot_prod))
        
        err_cpu = cp.asnumpy(err_deg_gpu)
        normals_cpu = cp.asnumpy(normals)
        
        min_err = float(np.min(err_cpu))
        max_err = float(np.max(err_cpu))
        mean_err = float(np.mean(err_cpu))
        med_err = float(np.median(err_cpu))

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
        with open(self.output_dir / "error_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)

        # B. Standard Normal Map
        n_map = np.zeros((h_res, w_res, 3), dtype=np.float32)
        n_map[v_idx, u_idx] = normals_cpu
        n_vis_uint16 = ((n_map + 1.0) / 2.0 * 65535).astype(np.uint16)
        cv2.imwrite(str(self.output_dir / "normal_map_linear.png"), cv2.cvtColor(n_vis_uint16, cv2.COLOR_RGB2BGR))

        # C. Strict Component Map
        n_strict_bgr = np.zeros((h_res, w_res, 3), dtype=np.uint16)
        norm_x_scaled = ((normals_cpu[:, 0] + 1.0) / 2.0 * 65535).astype(np.uint16)
        norm_y_scaled = ((normals_cpu[:, 1] + 1.0) / 2.0 * 65535).astype(np.uint16)
        norm_z_scaled = ((normals_cpu[:, 2] + 1.0) / 2.0 * 65535).astype(np.uint16)
        
        n_strict_bgr[v_idx, u_idx, 0] = norm_z_scaled 
        n_strict_bgr[v_idx, u_idx, 1] = norm_y_scaled 
        n_strict_bgr[v_idx, u_idx, 2] = norm_x_scaled 
        cv2.imwrite(str(self.output_dir / "normal_components_scaled.png"), n_strict_bgr)

        # D. Scaled Error Map
        err_img_map = np.zeros((h_res, w_res), dtype=np.uint16)
        if max_err > min_err:
            err_norm = (err_cpu - min_err) / (max_err - min_err)
            err_scaled = (err_norm * 65535).astype(np.uint16)
        else:
            err_scaled = np.zeros_like(err_cpu, dtype=np.uint16)
        err_img_map[v_idx, u_idx] = err_scaled
        cv2.imwrite(str(self.output_dir / "error_degree_map.png"), err_img_map)

        # E. CSV Export
        df_out = pd.DataFrame({
            'pixel_u': u_idx,
            'pixel_v': v_idx,
            'normal_x': normals_cpu[:, 0],
            'normal_y': normals_cpu[:, 1],
            'normal_z': normals_cpu[:, 2],
            'angular_error_deg': err_cpu
        })
        df_out.to_csv(self.output_dir / "pixelwise_normals_errors.csv", index=False)

        # F. Detailed Heatmap
        err_plt = np.full((h_res, w_res), np.nan, dtype=np.float32)
        err_plt[v_idx, u_idx] = err_cpu
        
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
        # REPLACE THIS WITH YOUR ACTUAL CONFIG PATH IF RUNNING DIRECTLY
        cfg_path = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\upd_bat_proc_nrml_grl_fst_cfg.json"
    
    if os.path.exists(cfg_path):
        processor = GeneralizedNormalProcessor(cfg_path)
        processor.run()
    else:
        print(f"Please provide a valid config path. File not found: {cfg_path}")