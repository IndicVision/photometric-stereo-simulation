import os
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path

# ==============================
# CUDA KERNEL
# ==============================

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

# ==============================
# MAIN CLASS
# ==============================

class GeneralizedNormalProcessor:

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

        # We strictly read paths but do NOT create output dir since we aren't saving files.
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')

    def load_image(self, img_path, gamma, bit_depth_json):
        """
        Loads image and returns RAW intensity values.
        STRICTLY respects gamma settings. If gamma=1.0, NO processing is done.
        """
        img_path_str = str(img_path)
        img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
        
        if img is None: 
            raise FileNotFoundError(f"Missing image: {img_path}")
        
        if img.ndim == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        
        # Determine if it's integer based (8/16 bit) or Float (EXR)
        # We only apply gamma logic if the user EXPLICITLY requested it (gamma != 1.0)
        if gamma != 1.0:
            is_float_data = (img.dtype == np.float32 or img.dtype == np.float16)
            
            if is_float_data:
                 # EXR: Direct gamma
                 img = np.power(img, gamma)
            else:
                # Integer: Normalize -> Gamma -> Scale Back
                max_val = float(2**bit_depth_json - 1)
                img = img / max_val             
                img = np.power(img, gamma)      
                img = img * max_val             

        return cp.array(img)

    def get_light_samples_gpu(self, light_cfg, P_surf):
        pos = np.array(light_cfg['pos_m'])
        dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir'])
        samples = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        step_u = dims[0] / samples[0]
        step_v = dims[1] / samples[1]

        u = np.linspace(-dims[0]/2 + step_u/2, dims[0]/2 - step_u/2, samples[0])
        v = np.linspace(-dims[1]/2 + step_v/2, dims[1]/2 - step_v/2, samples[1])
        uu, vv = np.meshgrid(u, v)

        v_dummy = np.array([0,1,0]) if abs(norm[1]) < 0.9 else np.array([1,0,0])
        ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:,None]*ax_u + vv.flatten()[:,None]*ax_v

        sample_pts_gpu = cp.array(sample_pts, dtype=cp.float32)
        norm_gpu = cp.array(norm, dtype=cp.float32)

        cos_half_spread = np.cos(np.deg2rad(spread_deg/2.0))

        num_pixels = P_surf.shape[0]
        num_samples = sample_pts_gpu.shape[0]

        G_out = cp.zeros((num_pixels,3), dtype=cp.float32)

        threads = 128
        blocks = (num_pixels + threads - 1) // threads

        self.kernel(
            (blocks,), (threads,),
            (
                P_surf,
                sample_pts_gpu,
                norm_gpu,
                cp.float32(cos_half_spread),
                cp.int32(num_pixels),
                cp.int32(num_samples),
                G_out
            )
        )

        return G_out

    def run(self):

        print("Executing Intensity Ratio Check (Console Only)...")
        t_start = time.time()

        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        csv_source = self.cfg.get('global_settings', {}).get('csv_from', 'MskHom')
        if csv_source == 'Blender':
            print("  [Info] CSV Source is 'Blender'. Flipping World X and Y coordinates.")
            df['x_world'] = -df['x_world']
            df['y_world'] = -df['y_world']
        
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)

        P_surf = cp.ascontiguousarray(
            cp.array(df[['x_world','y_world','z_world']].values, dtype=cp.float32)
        )

        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])

        I = cp.zeros((num_pixels, num_lights), dtype=cp.float32)
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)

        for j, l_cfg in enumerate(self.cfg['lights']):
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            
            # Use JSON bit_depth to check for max values, but do not normalize
            img_gpu = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])

            I[:,j] = img_gpu[v_idx, u_idx]
            G[:,j,:] = self.get_light_samples_gpu(l_cfg, P_surf)

            del img_gpu

        cp.cuda.Device(0).synchronize()

        # ==========================================================
        # INTENSITY & RATIO ANALYSIS (RAW VALUES)
        # ==========================================================
        print("\n" + "="*60)
        print("  INTENSITY & RATIO ANALYSIS (RAW DN vs PREDICTED)")
        print("="*60)
        
        # ==========================================================
        # LOAD NORMAL FROM CONFIG
        # ==========================================================
        # Fetch from JSON, default to [0, 0, 1] if missing
        norm_list = self.cfg.get('global_settings', {}).get('surface_normal', [0.0, 0.0, 1.0])
        
        # Normalize on CPU first to ensure unit length (crucial for n^T * G)
        n_cpu = np.array(norm_list, dtype=np.float32)
        norm_len = np.linalg.norm(n_cpu)
        
        if norm_len > 1e-6:
            n_cpu /= norm_len
        else:
            print("[Warning] Invalid normal vector in JSON. Defaulting to [0,0,1].")
            n_cpu = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        gt_n = cp.array(n_cpu, dtype=cp.float32)
        print(f"  [Info] Using Surface Normal: {n_cpu}")

        
        for j in range(num_lights):
            # 1. Observed Intensity (I) - RAW Values
            I_obs = I[:, j]
            mean_obs = cp.mean(I_obs)
            
            # 2. Predicted Intensity (n^T * G)
            I_pred = cp.sum(G[:, j, :] * gt_n, axis=1)
            mean_pred = cp.mean(I_pred)
            
            # Safe predicted intensity
            I_pred_safe = I_pred + 1e-8
            
            # 3. Ratio of Averages
            ratio_of_averages = mean_obs / (mean_pred + 1e-8)
            
            # 4. Average of Ratios
            pixelwise_ratio = I_obs / I_pred_safe
            average_of_ratios = cp.mean(pixelwise_ratio)
            
            print(f"Light {j+1} ({self.cfg['lights'][j]['id']}):")
            print(f"  Avg Observed (Raw DN):  {float(mean_obs):.4f}")
            print(f"  Avg Predicted (Geom):   {float(mean_pred):.4f}")
            print(f"  Ratio (Mean/Mean):      {float(ratio_of_averages):.6f}")
            print(f"  Ratio (Mean of Ratios): {float(average_of_ratios):.6f}")
            print("-" * 40)
            
        print("\nAnalysis Complete.")
        print(f"Total Time: {time.time()-t_start:.3f} sec")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        # REPLACE WITH YOUR CONFIG PATH
        cfg_path = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\nrml_err_als\inten_ratio_cfg.json"

    if os.path.exists(cfg_path):
        processor = GeneralizedNormalProcessor(cfg_path)
        processor.run()
    else:
        print(f"Please provide a valid config path. File not found: {cfg_path}")