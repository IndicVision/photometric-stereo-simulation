import os
import json
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CUDA Kernel (Unchanged) ---
CUDA_KERNEL_SOURCE = r'''
extern "C" __global__
void integrate_area_light(
    const float* P_surf, const float* light_samples, const float* light_norm,    
    float cos_half_spread, int num_pixels, int num_samples, float* G_out               
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float px = P_surf[idx * 3 + 0];
    float py = P_surf[idx * 3 + 1];
    float pz = P_surf[idx * 3 + 2];

    float nx = light_norm[0];
    float ny = light_norm[1];
    float nz = light_norm[2];

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;

    for (int s = 0; s < num_samples; s++) {
        float lx = light_samples[s * 3 + 0];
        float ly = light_samples[s * 3 + 1];
        float lz = light_samples[s * 3 + 2];

        float vx = lx - px, vy = ly - py, vz = lz - pz;
        float dist_sq = vx*vx + vy*vy + vz*vz;
        float dist = sqrtf(dist_sq);
        if (dist < 1e-8f) dist = 1e-8f;

        float dx = vx / dist, dy = vy / dist, dz = vz / dist;
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

class ConditionNumberMapGenerator:
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

        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)
        
        v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
        ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v
        
        sample_pts_gpu = cp.array(sample_pts, dtype=cp.float32)
        norm_gpu = cp.array(norm, dtype=cp.float32)
        
        cos_half_spread = np.cos(np.deg2rad(spread_deg / 2.0))
        num_pixels = P_surf_full.shape[0]
        num_samples = sample_pts_gpu.shape[0]
        
        G_eff_full = cp.zeros((num_pixels, 3), dtype=cp.float32)
        threads_per_block = 128
        blocks_per_grid = (num_pixels + threads_per_block - 1) // threads_per_block

        self.kernel(
            (blocks_per_grid,), (threads_per_block,),
            (P_surf_full, sample_pts_gpu, norm_gpu, cp.float32(cos_half_spread),
             cp.int32(num_pixels), cp.int32(num_samples), G_eff_full)
        )
        return G_eff_full

    def run(self):
        print("Calculating Condition Number Heatmap...")
        
        # 1. Load Geometry
        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        print(f"Loading Geometry from: {csv_path.name}")
        df = pd.read_csv(csv_path)

        if self.cfg.get('global_settings', {}).get('csv_from') == 'Blender':
            df['x_world'] = -df['x_world']
            df['y_world'] = -df['y_world']

        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        P_surf = cp.ascontiguousarray(cp.array(df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32))

        # 2. Build Light Matrix G
        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)

        print(f"Computing Light Vectors for {num_lights} lights...")
        for j, l_cfg in enumerate(self.cfg['lights']):
            G[:, j, :] = self.get_light_samples_gpu(l_cfg, P_surf)
        
        cp.cuda.Device(0).synchronize()

        # 3. Calculate Condition Number (SVD)
        print("Computing Singular Value Decomposition (SVD)...")
        S = cp.linalg.svd(G, compute_uv=False) 

        # Condition Number = Max Sigma / Min Sigma
        sigma_max = S[:, 0]
        sigma_min = S[:, 2]
        cond_gpu = sigma_max / (sigma_min + 1e-10)
        cond_cpu = cp.asnumpy(cond_gpu)

        # 4. Generate Visualization
        print("Generating Heatmap...")
        h, w = self.cfg['resolution']['height'], self.cfg['resolution']['width']
        cond_map = np.full((h, w), np.nan, dtype=np.float32)
        cond_map[v_idx, u_idx] = cond_cpu

        # Stats
        c_min, c_max = np.nanmin(cond_cpu), np.nanmax(cond_cpu)
        c_mean, c_med = np.nanmean(cond_cpu), np.nanmedian(cond_cpu)
        c_p99 = np.nanpercentile(cond_cpu, 99) 

        # Plot
        plt.figure(figsize=(14, 10))
        plt.imshow(cond_map, cmap='jet', interpolation='none', vmin=1.0, vmax=c_p99)
        cbar = plt.colorbar()
        cbar.set_label('Condition Number (Lower is Better)', rotation=270, labelpad=20)
        
        plt.title(f"Condition Number Heatmap\nClipped Visual Max: {c_p99:.2f}", fontsize=15)
        
        stats_text = (
            f"Stats:\nMin: {c_min:.2f}\nMed: {c_med:.2f}\nMean: {c_mean:.2f}\nMax: {c_max:.2f}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props, family='monospace')

        save_path = self.output_dir / "condition_number_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save Raw CSV
        pd.DataFrame({
            'pixel_u': u_idx, 'pixel_v': v_idx, 'condition_number': cond_cpu
        }).to_csv(self.output_dir / "condition_number_data.csv", index=False)

        print(f"Done. Saved to: {save_path}")

if __name__ == "__main__":
    import sys
    cfg_path = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\nrml_err_als\cdn_nbr_cal_cfg.json"
    if os.path.exists(cfg_path):
        ConditionNumberMapGenerator(cfg_path).run()
    else:
        print(f"Config not found: {cfg_path}")