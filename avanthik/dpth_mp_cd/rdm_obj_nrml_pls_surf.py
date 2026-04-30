import os
import time
import json
import cupy as cp
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

CUDA_KERNEL_SOURCE = r'''
extern "C" __global__
void integrate_area_light(
    const float* __restrict__ P_surf,       
    const float* __restrict__ light_samples, 
    const float* __restrict__ light_norm,    
    float cos_half_spread,      
    int num_pixels,
    int num_samples,
    float* __restrict__ G_out               
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sh_lx[128];
    __shared__ float sh_ly[128];
    __shared__ float sh_lz[128];

    float nx = light_norm[0];
    float ny = light_norm[1];
    float nz = light_norm[2];

    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (idx < num_pixels) {
        px = P_surf[idx * 3 + 0];
        py = P_surf[idx * 3 + 1];
        pz = P_surf[idx * 3 + 2];
    }

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    int num_tiles = (num_samples + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < num_tiles; ++t) {
        int sample_idx = t * blockDim.x + threadIdx.x;
        if (sample_idx < num_samples) {
            sh_lx[threadIdx.x] = light_samples[sample_idx * 3 + 0];
            sh_ly[threadIdx.x] = light_samples[sample_idx * 3 + 1];
            sh_lz[threadIdx.x] = light_samples[sample_idx * 3 + 2];
        }
        __syncthreads();

        int num_samples_in_tile = min(blockDim.x, num_samples - t * blockDim.x);
        
        if (idx < num_pixels) {
            for (int s = 0; s < num_samples_in_tile; ++s) {
                float vx = sh_lx[s] - px;
                float vy = sh_ly[s] - py;
                float vz = sh_lz[s] - pz;
                float dist_sq = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                if (dist_sq < 1e-16f) dist_sq = 1e-16f;

                float inv_dist = rsqrtf(dist_sq);
                float dx = vx * inv_dist;
                float dy = vy * inv_dist;
                float dz = vz * inv_dist;
                float cos_emit = -(nx*dx + ny*dy + nz*dz);

                if (cos_emit >= cos_half_spread) {
                    cos_emit = fmaxf(cos_emit, 0.0f);
                    float weight = cos_emit * (inv_dist * inv_dist);
                    gx = fmaf(dx, weight, gx);
                    gy = fmaf(dy, weight, gy);
                    gz = fmaf(dz, weight, gz);
                }
            }
        }
        __syncthreads();
    }

    if (idx < num_pixels) {
        float inv_num_samples = 1.0f / (float)num_samples;
        G_out[idx * 3 + 0] = gx * inv_num_samples;
        G_out[idx * 3 + 1] = gy * inv_num_samples;
        G_out[idx * 3 + 2] = gz * inv_num_samples;
    }
}
'''

class AutoIterativePipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')
        
        self.max_iterations = self.cfg.get('global_settings', {}).get('max_iterations', 15)
        self.convergence_threshold = self.cfg.get('global_settings', {}).get('convergence_threshold',  1e-5)

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(f"Missing image: {img_path}")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        if gamma != 1.0: img = np.power(img, gamma)
        return cp.array(img)

    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        pos = np.array(light_cfg['pos_m'])
        dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir'])
        samples = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)
        
        if 'ax_u' in light_cfg and 'ax_v' in light_cfg:
            ax_u = np.array(light_cfg['ax_u'], dtype=np.float32)
            ax_v = np.array(light_cfg['ax_v'], dtype=np.float32)
        else:
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
        
        chunk_size = 50000 
        threads_per_block = 128
        
        for start_idx in range(0, num_pixels, chunk_size):
            end_idx = min(start_idx + chunk_size, num_pixels)
            current_chunk = end_idx - start_idx
            
            P_surf_chunk = P_surf_full[start_idx:end_idx]
            G_out_chunk = G_eff_full[start_idx:end_idx]
            blocks_per_grid = (current_chunk + threads_per_block - 1) // threads_per_block
            
            self.kernel((blocks_per_grid,), (threads_per_block,),
                (P_surf_chunk, sample_pts_gpu, norm_gpu, cp.float32(cos_half_spread),
                 cp.int32(current_chunk), cp.int32(num_samples), G_out_chunk))
            
            cp.cuda.Device(0).synchronize()
            
        return G_eff_full

    def estimate_normals(self, df):
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])

        if not hasattr(self, 'I_w_cache'):
            print("    [DEBUG] Caching image intensities to VRAM (One-time cost)...")
            I = cp.zeros((num_pixels, num_lights), dtype=cp.float32)

            for j, l_cfg in enumerate(self.cfg['lights']):
                img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
                img_gpu = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
                I[:, j] = img_gpu[v_idx, u_idx]
                del img_gpu

            global_cfg = self.cfg.get('global_settings', {})
            apply_thresh = global_cfg.get('apply_dark_threshold', True)
            thresh_val = global_cfg.get('dark_threshold_value', 0.025)
            self.min_lights = global_cfg.get('min_valid_lights', 3)
            
            if apply_thresh:
                W = (I >= thresh_val).astype(cp.float32)
                self.valid_light_count = cp.sum(W, axis=1)
                self.I_w_cache = I * W
                self.W_cache = W
            else:
                self.I_w_cache = I
                self.W_cache = cp.ones_like(I)
                self.valid_light_count = cp.full((num_pixels,), num_lights, dtype=cp.float32)

        P_surf = cp.ascontiguousarray(cp.array(df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32))
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)

        for j, l_cfg in enumerate(self.cfg['lights']):
            G[:, j, :] = self.get_light_samples_gpu(l_cfg, P_surf)

        G_w = G * self.W_cache[:, :, None]

        GT_w = G_w.transpose(0, 2, 1) 
        GTG = cp.matmul(GT_w, G_w) + (cp.eye(3, dtype=cp.float32) * 1e-4)
        GTI = cp.matmul(GT_w, self.I_w_cache[:, :, None])
        
        n_est = cp.linalg.solve(GTG, GTI).squeeze()
        
        invalid_mask = self.valid_light_count < self.min_lights
        if cp.sum(invalid_mask) > 0:
            n_est[invalid_mask] = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
        
        albedo = cp.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / cp.where(albedo == 0, 1, albedo)
        
        normals_cpu = cp.asnumpy(normals)
        
        del G, G_w, GT_w, GTG, GTI, n_est, P_surf
        cp.get_default_memory_pool().free_all_blocks()
        
        return normals_cpu

    def reconstruct_surface(self, df, normals):
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        
        M_img = np.max(v_idx) + 1
        N_img = np.max(u_idx) + 1
        mask = np.zeros((M_img, N_img), dtype=bool)
        mask[v_idx, u_idx] = True

        neg_z_count = np.sum(normals[:, 2] < 0)
        total_norms = len(normals)
        print(f"    [DEBUG] Negative Z-normals before correction: {neg_z_count} / {total_norms} ({(neg_z_count/total_norms)*100:.2f}%)")

        n_corr = normals.copy()
        
        if (neg_z_count / total_norms) > 0.5:
            print("    [DEBUG] Coordinate system inversion detected! Flipping all normal vectors.")
            n_corr = -n_corr
        else:
            inverted_mask = n_corr[:, 2] < 0
            n_corr[inverted_mask] = -n_corr[inverted_mask]
        
        nx_grid = np.zeros((M_img, N_img))
        ny_grid = np.zeros((M_img, N_img))
        nz_grid = np.zeros((M_img, N_img))
        
        nx_grid[v_idx, u_idx] = n_corr[:, 0]
        ny_grid[v_idx, u_idx] = n_corr[:, 1] 
        nz_grid[v_idx, u_idx] = n_corr[:, 2]

        X_grid = np.full((M_img, N_img), np.nan)
        Y_grid = np.full((M_img, N_img), np.nan)

        X_grid[v_idx, u_idx] = df['x_world'].values
        Y_grid[v_idx, u_idx] = df['y_world'].values

        r_idx, c_idx = np.where(mask)
        center_r, center_c = int(np.mean(r_idx)), int(np.mean(c_idx))

        try: dx = np.abs(X_grid[center_r, center_c+1] - X_grid[center_r, center_c])
        except IndexError: dx = 0.0001
        try: dy = np.abs(Y_grid[center_r+1, center_c] - Y_grid[center_r, center_c])
        except IndexError: dy = 0.0001
        
        if np.isnan(dx) or dx == 0: dx = 0.0001
        if np.isnan(dy) or dy == 0: dy = 0.0001

        eps_nz = 0.15
        valid = mask & (np.abs(nz_grid) > eps_nz)

        p = np.zeros((M_img, N_img))
        q = np.zeros((M_img, N_img))
        
        nz_safe = nz_grid[valid]  
        
        p[valid] = (nx_grid[valid] / nz_safe) * dx   
        q[valid] = -(ny_grid[valid] / nz_safe) * dy   

        if not hasattr(self, 'poisson_solver'):
            print("    [DEBUG] Factoring Poisson Matrix (One-time CPU cost)...")
            node_ids = np.zeros((M_img, N_img), dtype=int)
            num_unknowns = np.sum(mask)
            node_ids[mask] = np.arange(num_unknowns)

            mask_H = mask[:, :-1] & mask[:, 1:]
            self.r_H, self.c_H = np.where(mask_H)
            id_self_H = node_ids[self.r_H, self.c_H]
            id_right_H = node_ids[self.r_H, self.c_H + 1]

            mask_V = mask[:-1, :] & mask[1:, :]
            self.r_V, self.c_V = np.where(mask_V)
            id_self_V = node_ids[self.r_V, self.c_V]
            id_down_V = node_ids[self.r_V + 1, self.c_V]

            num_H = len(id_self_H)
            num_V = len(id_self_V)
            num_eq = num_H + num_V + 1

            I_list = np.concatenate([
                np.arange(num_H), np.arange(num_H),
                np.arange(num_H, num_H + num_V), np.arange(num_H, num_H + num_V),
                [num_eq - 1]
            ])
            
            J_list = np.concatenate([
                id_right_H, id_self_H,
                id_down_V, id_self_V,
                [node_ids[center_r, center_c]] 
            ])
            
            V_list = np.concatenate([
                np.ones(num_H), -np.ones(num_H),
                np.ones(num_V), -np.ones(num_V),
                [1.0]
            ])
            
            A = coo_matrix((V_list, (I_list, J_list)), shape=(num_eq, num_unknowns)).tocsr()
            self.A_T = A.T
            C = self.A_T @ A
            self.poisson_solver = factorized(C) 

        val_p = p[self.r_H, self.c_H]
        val_q = q[self.r_V, self.c_V]
        b = np.concatenate([val_p, val_q, [0.0]])

        d = self.A_T @ b
        z = self.poisson_solver(d) 

        Z = np.full((M_img, N_img), np.nan)
        Z[mask] = z

        print(f"    [DEBUG] Pre-Detrend Z | Min: {np.nanmin(Z):.6f}, Max: {np.nanmax(Z):.6f}, Mean: {np.nanmean(Z):.6f}")

        detrend_mode = self.cfg.get('global_settings', {}).get('detrending_mode', 'linear').lower()
        
        valid_X = X_grid[mask]
        valid_Y = Y_grid[mask]
        valid_Z = Z[mask]
        
        if detrend_mode == 'quadratic':
            print("    [DEBUG] Detrend Mode | QUADRATIC (Flattening macro-bowls for flat objects)")
            A_quad = np.c_[valid_X**2, valid_Y**2, valid_X*valid_Y, valid_X, valid_Y, np.ones_like(valid_X)]
            C_quad, _, _, _ = np.linalg.lstsq(A_quad, valid_Z, rcond=None)
            Detrend_Z = (C_quad[0] * X_grid**2 + C_quad[1] * Y_grid**2 + 
                         C_quad[2] * X_grid * Y_grid + C_quad[3] * X_grid + 
                         C_quad[4] * Y_grid + C_quad[5])
        else:
            print("    [DEBUG] Detrend Mode | LINEAR (Preserving natural curves for 3D objects)")
            A_plane = np.c_[valid_X, valid_Y, np.ones_like(valid_X)]
            C_plane, _, _, _ = np.linalg.lstsq(A_plane, valid_Z, rcond=None)
            Detrend_Z = (C_plane[0] * X_grid) + (C_plane[1] * Y_grid) + C_plane[2]
            
        Z[mask] = Z[mask] - Detrend_Z[mask]

        surface_floor = np.nanpercentile(Z[mask], 2)
        Z[mask] = Z[mask] - surface_floor
        
        if 'z_world' in df.columns and not (df['z_world'] == 0).all():
            original_floor = np.percentile(df['z_world'].values, 2)
            Z[mask] = Z[mask] + original_floor

        print(f"    [DEBUG] Final Output Z| Min: {np.nanmin(Z):.6f}, Max: {np.nanmax(Z):.6f}, Mean: {np.nanmean(Z):.6f}")

        return Z, dx, dy, mask

    def save_visualizations(self, iter_num, normals, df, Z, dx, dy, mask):
        iter_dir = self.output_dir / f"iteration_{iter_num:02d}"
        iter_dir.mkdir(exist_ok=True)

        FLIP_X_VISUALLY = False 
        FLIP_Y_VISUALLY = False

        Z_EXAGGERATION = float(self.cfg.get('global_settings', {}).get('z_exaggeration', 1.0))

        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        h_res, w_res = self.cfg['resolution']['height'], self.cfg['resolution']['width']
        
        n_map = np.zeros((h_res, w_res, 3), dtype=np.float32)
        n_map[v_idx, u_idx] = normals
        n_vis_uint16 = ((n_map + 1.0) / 2.0 * 65535).astype(np.uint16)
        cv2.imwrite(str(iter_dir / "normal_map.png"), cv2.cvtColor(n_vis_uint16, cv2.COLOR_RGB2BGR))

        r_idx, c_idx = np.where(mask)
        r_min, r_max = np.min(r_idx), np.max(r_idx)
        c_min, c_max = np.min(c_idx), np.max(c_idx)
        
        Z_crop = Z[r_min:r_max+1, c_min:c_max+1]
        base_z_crop = np.nanmedian(Z_crop)

        Z_vis_mm = (Z_crop - base_z_crop) * 1000.0
        x_plot_mm = (np.arange(Z_crop.shape[1]) * dx) * 1000.0
        y_plot_mm = (np.arange(Z_crop.shape[0]) * dy) * 1000.0

        x_extent = [x_plot_mm[-1], x_plot_mm[0]] if FLIP_X_VISUALLY else [x_plot_mm[0], x_plot_mm[-1]]
        y_extent = [y_plot_mm[0], y_plot_mm[-1]] if FLIP_Y_VISUALLY else [y_plot_mm[-1], y_plot_mm[0]]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(Z_vis_mm, extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]], cmap='viridis')
        ax.set_title(f"2D Depth Map (Extrusions) - Iteration {iter_num}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        fig.colorbar(im, ax=ax, label='Extrusion Depth (mm)', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(iter_dir / "2D_depth_map.png", dpi=200)
        plt.close(fig)

        print("    [DEBUG] Generating 3D Plotly file (this may take a few seconds)...")
        Z_render = Z_vis_mm.copy()
        if FLIP_Y_VISUALLY: Z_render = np.flipud(Z_render)
        if FLIP_X_VISUALLY: Z_render = np.fliplr(Z_render)

        fig3d = go.Figure(data=[go.Surface(
            z=Z_render,  
            x=x_plot_mm, 
            y=y_plot_mm, 
            colorscale='RdYlBu',
            lighting=dict(ambient=0.2, diffuse=1.0, roughness=0.8, specular=0.1, fresnel=0.0),
            lightposition=dict(x=-1000, y=0, z=1) 
        )])


        range_x = float(np.ptp(x_plot_mm)) if len(x_plot_mm) > 1 else 1.0
        range_y = float(np.ptp(y_plot_mm)) if len(y_plot_mm) > 1 else 1.0
        z_valid = Z_vis_mm[~np.isnan(Z_vis_mm)]
        range_z = float(np.ptp(z_valid)) if len(z_valid) > 0 else 5.0
        
        if np.isnan(range_z) or range_z < 1e-4:
            range_z = 5.0

        fig3d.update_layout(
            title=f"3D Surface (Extrusions) - Iteration {iter_num}",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='manual',
                aspectratio=dict(
                    x=1.0, 
                    y=range_y / range_x, 
                    z=(range_z / range_x) * Z_EXAGGERATION 
                ) 
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig3d.write_html(str(iter_dir / f"3D_surface_interactive.html"))


        df_out = df.copy()
        df_out['z_world'] = Z[v_idx, u_idx] 
        base_z_full = np.nanmedian(Z[mask])
        df_out['z_extrusion'] = Z[v_idx, u_idx] - base_z_full 
        
        out_csv = iter_dir / f"mapping_iter{iter_num}.csv"
        df_out.to_csv(out_csv, index=False)
        
        return out_csv, Z[v_idx, u_idx]

    def run(self):
        print("\n" + "="*50)
        print("Starting Automated Photometric Stereo Pipeline (ULTRA FAST)")
        print("="*50)

        current_csv = Path(self.cfg['paths']['world_coordinate_csv'])
        previous_z_vals = None

        for i in range(self.max_iterations):
            print(f"\n--- [ Iteration {i} ] ---")
            t_start = time.time()

            df = pd.read_csv(current_csv)
            csv_source = self.cfg.get('global_settings', {}).get('csv_from', 'MskHom')
            if csv_source == 'Blender' and i == 0: 
                df['x_world'] = -df['x_world']
                df['y_world'] = -df['y_world']

            if i == 0:
                print("  > Initializing Physical Camera Grid...")
                cam_cfg = self.cfg.get('camera', {})
                
                sensor_width_mm = cam_cfg.get('sensor_width_mm', 35.9)
                focal_length_mm = cam_cfg.get('focal_length_mm', 50.0)
                object_dist_m   = cam_cfg.get('object_distance_m', 0.5)
                res_w           = self.cfg['resolution']['width']
                
                sensor_pixel_size_mm = sensor_width_mm / res_w
                dx_meters = sensor_pixel_size_mm * (object_dist_m / focal_length_mm)
                dy_meters = dx_meters 
                
                if cam_cfg.get('use_auto_center', True):
                    cam_u = res_w / 2.0
                    cam_v = self.cfg['resolution']['height'] / 2.0
                else:
                    cam_u, cam_v = cam_cfg.get('manual_center_pixel', [0, 0])

                df['x_world'] = (df['pixel_u'] - cam_u) * dx_meters
                df['y_world'] = -(df['pixel_v'] - cam_v) * dy_meters  
                
                if 'z_world' not in df.columns or (df['z_world'] == 0).all():
                    df['z_world'] = 0.0
                    print("    [DEBUG] Init Grid | No Z-data found, defaulting to Z=0.0")
                else:
                    print(f"    [DEBUG] Init Grid | Preserved starting Z-height: {df['z_world'].mean():.6f}m")

                print(f"    [DEBUG] Init Grid | dx: {dx_meters:.6f}m, dy: {dy_meters:.6f}m")
                print(f"    [DEBUG] Init Grid | X Bounds: {df['x_world'].min():.6f} to {df['x_world'].max():.6f}")
                print(f"    [DEBUG] Init Grid | Y Bounds: {df['y_world'].min():.6f} to {df['y_world'].max():.6f}")

            print("  > Estimating Surface Normals...")
            normals = self.estimate_normals(df)

            print("  > Reconstructing 3D Surface...")
            Z_map, dx, dy, mask = self.reconstruct_surface(df, normals)

            raw_new_z = Z_map[df['pixel_v'].values.astype(int), df['pixel_u'].values.astype(int)]
            df['z_world'] = raw_new_z

            print("  > Saving Visualizations and CSV...")
            current_csv, current_z_vals = self.save_visualizations(i, normals, df, Z_map, dx, dy, mask)

            t_end = time.time()
            print(f"  [Time] Iteration {i} completed in {t_end - t_start:.2f} seconds.")

            if previous_z_vals is not None:
                mad = np.mean(np.abs(current_z_vals - previous_z_vals))
                print(f"  [Convergence] Mean Absolute Depth Change: {mad:.6f} meters")
                if mad < self.convergence_threshold:
                    print(f"\nSUCCESS: Geometry converged after {i} iterations! (Change < {self.convergence_threshold}m)")
                    break
            else:
                print("  [Convergence] Base iteration complete. Tracking changes next run.")

            previous_z_vals = current_z_vals

        print("\n" + "="*50)
        print(f"Pipeline finished! All outputs saved in: {self.output_dir}")
        print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\vishn\Desktop\avanthik\oth_wrk\hemant\surface reconstruction\upd_basis_copy (1).json"
    if os.path.exists(cfg_path):
        pipeline = AutoIterativePipeline(cfg_path)
        pipeline.run()
    else:
        print(f"File not found: {cfg_path}")