import os
import sys

# --- CRITICAL: Enable OpenEXR before importing OpenCV ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import time
import cv2
import pandas as pd
import numpy as np
import cupy as cp
from pathlib import Path

# --- CUDA KERNEL FOR AREA LIGHT INTEGRATION ---
CUDA_KERNEL_SOURCE = r'''
extern "C" __global__
void integrate_area_light(
    const float* P_surf,       
    const float* light_samples, 
    const float* light_norm,    
    float cos_half_spread,      
    int num_pixels,
    int num_samples,
    float* G_out               
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float px = P_surf[idx * 3 + 0];
    float py = P_surf[idx * 3 + 1];
    float pz = P_surf[idx * 3 + 2];

    float nx = light_norm[0];
    float ny = light_norm[1];
    float nz = light_norm[2];

    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;

    for (int s = 0; s < num_samples; s++) {
        float lx = light_samples[s * 3 + 0];
        float ly = light_samples[s * 3 + 1];
        float lz = light_samples[s * 3 + 2];

        float vx = lx - px;
        float vy = ly - py;
        float vz = lz - pz;

        float dist_sq = vx*vx + vy*vy + vz*vz;
        float dist = sqrtf(dist_sq);
        if (dist < 1e-8f) dist = 1e-8f;

        float dx = vx / dist;
        float dy = vy / dist;
        float dz = vz / dist;

        float cos_emit = -(nx*dx + ny*dy + nz*dz);

        if (cos_emit >= cos_half_spread) {
            if (cos_emit < 0.0f) cos_emit = 0.0f;
            float weight = cos_emit / dist_sq;
            gx += dx * weight;
            gy += dy * weight;
            gz += dz * weight;
        }
    }

    G_out[idx * 3 + 0] = gx / num_samples;
    G_out[idx * 3 + 1] = gy / num_samples;
    G_out[idx * 3 + 2] = gz / num_samples;
}
'''

class IntensitySimulator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')

    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        pos = np.array(light_cfg['pos_m'])
        dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir'])
        samples = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        # Generate Samples (CPU)
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
        
        cos_half_spread = np.cos(np.deg2rad(spread_deg / 2.0))
        num_pixels = P_surf_full.shape[0]
        num_samples = sample_pts_gpu.shape[0]
        
        G_out = cp.zeros((num_pixels, 3), dtype=cp.float32)

        threads_per_block = 128
        blocks_per_grid = (num_pixels + threads_per_block - 1) // threads_per_block

        self.kernel(
            (blocks_per_grid,), (threads_per_block,),
            (P_surf_full, sample_pts_gpu, norm_gpu, cp.float32(cos_half_spread),
             cp.int32(num_pixels), cp.int32(num_samples), G_out)
        )
        return G_out

    def run(self):
        # Load Settings
        settings = self.cfg['simulation_settings']
        K = settings['proportionality_constant_k']
        
        res = settings.get('resolution', {'width': 2100, 'height': 1400})
        W_res = res['width']
        H_res = res['height']

        n_fixed_cpu = np.array(settings['fixed_normal'], dtype=np.float32)
        n_fixed = cp.array(n_fixed_cpu)
        
        save_exr = settings.get('save_exr_linear', False)
        
        # --- MODIFIED SECTION START ---
        # Check for new 'save_png' flag, fallback to old 'save_png_16bit' if missing
        save_png = settings.get('save_png', settings.get('save_png_16bit', False))
        png_depth = settings.get('png_bit_depth', 16) # Default to 16 if not specified
        
        if png_depth not in [8, 16]:
            print(f"Warning: Invalid png_bit_depth ({png_depth}). Defaulting to 16-bit.")
            png_depth = 16
            
        print(f"--- Starting Intensity Simulation ---")
        print(f"K Factor: {K}")
        print(f"Normal: {n_fixed_cpu}")
        print(f"Outputs: EXR={save_exr}, PNG={save_png} (Depth: {png_depth}-bit)")
        # --- MODIFIED SECTION END ---

        # 1. Load CSV
        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        print(f"Loading CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)

        csv_source = settings.get('csv_from', 'MskHom')
        if csv_source == 'Blender':
            print("  [Info] CSV Source is 'Blender'. Flipping World X and Y coordinates.")
            df['x_world'] = -df['x_world']
            df['y_world'] = -df['y_world']
        elif csv_source == 'MskHom':
            print("  [Info] CSV Source is 'MskHom'. Using coordinates as-is.")
        else:
            print(f"  [Warning] Unknown csv_from value: '{csv_source}'. Defaulting to no flip.")

        # 2. Filter Valid Pixels (Alpha == 1)
        if 'alpha' not in df.columns:
            raise ValueError("CSV missing 'alpha' column.")
        
        valid_df = df[df['alpha'] == 1].copy()
        if len(valid_df) == 0:
            raise ValueError("No pixels found with alpha == 1")

        print(f"Processing {len(valid_df)} valid pixels.")

        # 3. Prepare Data
        u_idx = valid_df['pixel_u'].values.astype(int)
        v_idx = valid_df['pixel_v'].values.astype(int)
        P_surf = cp.ascontiguousarray(cp.array(valid_df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32))
        
        # 4. Process Each Light
        for l_cfg in self.cfg['lights']:
            t0 = time.time()
            l_id = l_cfg['id']
            print(f"\nProcessing Light {l_id}...")
            
            # A. Calculate G
            G_gpu = self.get_light_samples_gpu(l_cfg, P_surf)
            
            # B. Calculate I = K * (n . G)
            dot_prod = cp.dot(G_gpu, n_fixed)
            I_raw = cp.maximum(dot_prod * K, 0.0) # Clip negatives
            
            # --- SAVE EXR (Linear) ---
            if save_exr:
                exr_img_gpu = cp.zeros((H_res, W_res), dtype=cp.float32)
                exr_img_gpu[v_idx, u_idx] = I_raw
                
                exr_path = self.output_dir / f"light_{l_id}.exr"
                cv2.imwrite(str(exr_path), cp.asnumpy(exr_img_gpu))
                print(f"  > Saved EXR: {exr_path.name}")

            # --- SAVE PNG (8-bit or 16-bit) ---
            if save_png:
                if png_depth == 16:
                    # Clip to 16-bit max and cast
                    max_val = 65535
                    I_out = cp.clip(I_raw, 0, max_val).astype(cp.uint16)
                    dtype_np = np.uint16
                else: # 8-bit
                    # Clip to 8-bit max and cast
                    # NOTE: If your K is calibrated for 16-bit (high values), 
                    # you must lower K in JSON or this will just appear white (clipped).
                    max_val = 255
                    I_out = cp.clip(I_raw, 0, max_val).astype(cp.uint8)
                    dtype_np = np.uint8

                # Place into image buffer
                png_img_gpu = cp.zeros((H_res, W_res), dtype=dtype_np)
                
                # Check bounds to ensure u,v don't exceed resolution
                # (Optional safety, assuming CSV is correct)
                png_img_gpu[v_idx, u_idx] = I_out
                
                png_path = self.output_dir / f"light_{l_id}.png"
                cv2.imwrite(str(png_path), cp.asnumpy(png_img_gpu))
                print(f"  > Saved PNG ({png_depth}-bit): {png_path.name}")
            
            print(f"  > Light processed in {time.time() - t0:.3f}s")

        print("\nDone.")

if __name__ == "__main__":
    # UPDATE THIS PATH TO YOUR NEW JSON FILE
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\nrml_err_als\theor_img_rndr_cfg.json"
    
    if os.path.exists(CONFIG_PATH):
        IntensitySimulator(CONFIG_PATH).run()
    else:
        print(f"Config file not found: {CONFIG_PATH}")