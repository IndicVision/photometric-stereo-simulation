import cupy as cp  # GPU Acceleration
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class GPUNormalProcessor:
    def __init__(self, config_path):
        """Initializes paths and reconstruction settings from JSON."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.input_base = Path(self.cfg['paths']['renders_dir'])
        self.output_base = Path(self.cfg['paths']['reconstruction_output_dir'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        self.bg_threshold = self.cfg['reconstruction_settings'].get('background_threshold', 0.05)
        self.erosion_iter = self.cfg['reconstruction_settings'].get('mask_erosion_iter', 2)
        
        # USER INPUT: Base constant (L_A * rho / pi)
        self.base_albedo_const = self.cfg['reconstruction_settings'].get('base_albedo_constant', 1.0)

    def parse_folder_params(self, p_folder_name, l_folder_name):
        """Extracts configuration parameters from directory names."""
        p_parts = p_folder_name.split('_')
        l_parts = l_folder_name.split('_')
        return {
            'elev': float(p_parts[1]), 
            'azim': float(p_parts[0]), 
            'area': float(p_parts[5]),
            'num_l': int(l_parts[0]), 
            'dist': float(l_parts[1]),
            'spread': float(l_parts[4]), 
            'psi': float(l_parts[5]), 
            'energy': float(l_parts[6])
        }

    def process(self):
        sample_folders = sorted([d for d in self.input_base.iterdir() if d.is_dir() and d.name.startswith("samples_")])
        print(f"Found {len(sample_folders)} sample folders: {[f.name for f in sample_folders]}")
        
        for s_folder in sample_folders:
            samples_val = int(s_folder.name.split('_')[1])
            for p_folder in [d for d in s_folder.iterdir() if d.is_dir()]:
                for l_folder in [d for d in p_folder.iterdir() if d.is_dir()]:
                    params = self.parse_folder_params(p_folder.name, l_folder.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_folder.name}__{l_folder.name}.csv"
                    
                    if not matrix_file.exists():
                        continue
                    
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values.astype(np.float32)
                    method = self.cfg['analysis_settings']['exposure_method']
                    
                    images = []
                    for i in range(1, params['num_l'] + 1):
                        img_p = l_folder / f"{i:03d}_{method}.png"
                        if img_p.exists():
                            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                            images.append(img.astype(np.float32)/255.0)

                    if len(images) != len(l_vecs):
                        continue
                        
                    print(f"GPU Processing: {s_folder.name} | {p_folder.name} | {l_folder.name}")
                    
                    mask = self.create_mask(images)
                    p_surf, true_n = self.get_geometry(params, images[0].shape[1], images[0].shape[0])
                    
                    # Solve on GPU
                    n_map, albedo, errs = self.solve_normals_gpu(images, l_vecs, p_surf, true_n, params)
                    
                    # DYNAMIC CALCULATION: True Albedo based on spread
                    half_spread_rad = np.deg2rad(params['spread'] / 2.0)
                    folder_true_albedo = self.base_albedo_const / (np.sin(half_spread_rad)**2 + 1e-9)
                    
                    # Calculate Albedo Error Map
                    albedo_err_map = cp.abs(albedo - folder_true_albedo)
                    
                    # Compute stats
                    valid_errs = errs[mask]
                    valid_alb_errs = albedo_err_map[mask]
                    
                    stats = {
                        "elev": params['elev'],
                        "azim": params['azim'],
                        "area": params['area'],
                        "num_l": params['num_l'],
                        "dist": params['dist'],
                        "spread": params['spread'],
                        "psi": params['psi'],
                        "energy": params['energy'],
                        "samples": samples_val,
                        "true_albedo_target": float(folder_true_albedo),
                        "mean_error": float(cp.mean(valid_errs)) if valid_errs.size > 0 else 0,
                        "max_error": float(cp.max(valid_errs)) if valid_errs.size > 0 else 0,
                        "min_error": float(cp.min(valid_errs)) if valid_errs.size > 0 else 0,
                        "mean_albedo_error": float(cp.mean(valid_alb_errs)) if valid_alb_errs.size > 0 else 0,
                        "max_albedo_error": float(cp.max(valid_alb_errs)) if valid_alb_errs.size > 0 else 0,
                        "min_albedo_error": float(cp.min(valid_alb_errs)) if valid_alb_errs.size > 0 else 0,
                        "valid_pixels": int(valid_errs.size)
                    }
                    
                    self.save_and_report(s_folder.name, p_folder.name, l_folder.name, n_map, albedo, errs, albedo_err_map, mask, stats)

    def create_mask(self, images):
        avg_img = np.mean(images, axis=0)
        raw_mask = (avg_img > self.bg_threshold).astype(np.uint8)
        return cv2.erode(raw_mask, np.ones((3,3), np.uint8), iterations=self.erosion_iter).astype(bool)

    def get_geometry(self, params, w, h):
        zenith_rad = np.deg2rad(90.0 - params['elev'])
        az_rad = np.deg2rad(params['azim'])
        nx, ny, nz = np.sin(zenith_rad)*np.cos(az_rad), np.sin(zenith_rad)*np.sin(az_rad), np.cos(zenith_rad)
        side = params['area']**0.5
        x, y = np.linspace(-side/2, side/2, w), np.linspace(side/2, -side/2, h)
        xv, yv = np.meshgrid(x, y)
        zv = -(nx * xv + ny * yv) / (nz + 1e-9)
        return cp.array(np.stack((xv, yv, zv), axis=-1)), cp.array([nx, ny, nz], dtype=cp.float32)

    def solve_normals_gpu(self, images, light_vecs, p_surf, true_n, params):
        I_gpu = cp.array(images, dtype=cp.float32)
        L_vecs_gpu = cp.array(light_vecs, dtype=cp.float32)
        
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['reconstruction_settings']['samples_per_axis']
        half_spread = np.deg2rad(params['spread'] / 2.0)
        
        energy_scale = 1.0
        if self.cfg['reconstruction_settings'].get('use_blender_energy_scaling', True) and params['spread'] < 180.0:
            energy_scale = 1.0 / (np.sin(half_spread)**2 + 1e-9)

        G_list = []
        for l_vec in L_vecs_gpu:
            pts = self.get_light_samples_gpu(l_vec, params['dist'], l_dims, samp_ax)
            n_A = -l_vec
            G_acc = cp.zeros((p_surf.shape[0], p_surf.shape[1], 3), dtype=cp.float32)
            
            for pt in pts:
                v = pt - p_surf
                dist_sq = cp.sum(v**2, axis=2) + 1e-9
                l_k = v / cp.sqrt(dist_sq)[:,:,None]
                cos_emitter = cp.sum(-n_A * l_k, axis=2)
                term = (cp.maximum(0, cos_emitter) / dist_sq)[:,:,None] * l_k
                
                if self.cfg['reconstruction_settings'].get('discard_irrelevant_l_vectors', True):
                    G_acc += cp.where(cos_emitter[:,:,None] >= np.cos(half_spread), term, 0)
                else:
                    G_acc += term

            G_list.append((G_acc * energy_scale) / len(pts))

        I_flat = I_gpu.reshape(len(light_vecs), -1).T
        G_flat = cp.stack(G_list, axis=2).reshape(-1, len(light_vecs), 3)
        
        GT = G_flat.transpose(0, 2, 1)
        GTG_inv = cp.linalg.inv(cp.matmul(GT, G_flat) + 1e-6 * cp.eye(3))
        Norms = cp.einsum('ijk,ik->ij', cp.matmul(GTG_inv, GT), I_flat)
        
        Albedo = cp.linalg.norm(Norms, axis=1)
        N_Map = (Norms / (Albedo[:, None] + 1e-9)).reshape(p_surf.shape[0], p_surf.shape[1], 3)
        Albedo_Map = Albedo.reshape(p_surf.shape[0], p_surf.shape[1])
        
        errs = cp.degrees(cp.arccos(cp.clip(cp.sum(N_Map * true_n, axis=2), -1, 1)))
        return N_Map, Albedo_Map, errs

    def get_light_samples_gpu(self, l_vec, dist, dims, n_samp):
        center = l_vec * dist
        normal = -l_vec
        up = cp.array([0, 0, 1], dtype=cp.float32)
        right = cp.cross(up, normal)
        if cp.linalg.norm(right) < 1e-3: right = cp.array([1, 0, 0], dtype=cp.float32)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(normal, right)
        
        w, h = dims
        steps = cp.linspace(-0.5, 0.5, n_samp)
        ii, jj = cp.meshgrid(steps, steps)
        samples = center + (right[None, None, :] * ii[:, :, None] * w) + (up_loc[None, None, :] * jj[:, :, None] * h)
        return samples.reshape(-1, 3)

    def save_and_report(self, s_n, p_n, l_n, n_map, albedo, errs, alb_errs, mask, stats):
        out_path = self.output_base / s_n / p_n / l_n
        out_path.mkdir(parents=True, exist_ok=True)
        
        errs_cpu = cp.asnumpy(errs)
        alb_errs_cpu = cp.asnumpy(alb_errs)
        n_map_cpu = cp.asnumpy(n_map)
        albedo_cpu = cp.asnumpy(albedo)
        
        self.generate_heatmap(errs_cpu, mask, out_path / "error_map.png", f"Normal Error Heatmap: {p_n}", "Angular Error (Degrees)", stats['mean_error'])
        self.generate_heatmap(alb_errs_cpu, mask, out_path / "albedo_error_map.png", f"Albedo Error Heatmap: {p_n}", "Absolute Albedo Error", stats['mean_albedo_error'])

        if self.cfg['outputs'].get('save_stats_report', True):
            pd.DataFrame([stats]).to_csv(out_path / "recon_report.csv", index=False)
            
        if self.cfg['outputs'].get('save_per_pixel_csv', True):
            H, W = mask.shape
            yy, xx = np.mgrid[0:H, 0:W]
            df_pixels = pd.DataFrame({
                'x': xx[mask], 'y': yy[mask],
                'nx': n_map_cpu[mask][:,0], 'ny': n_map_cpu[mask][:,1], 'nz': n_map_cpu[mask][:,2],
                'albedo': albedo_cpu[mask],
                'err_deg': errs_cpu[mask],
                'albedo_err': alb_errs_cpu[mask]
            })
            df_pixels.to_csv(out_path / "pixel_data.csv", index=False)

    def generate_heatmap(self, data_cpu, mask, save_path, title, label, mean_val):
        plt.figure(figsize=(10, 8))
        masked_data = np.where(mask, data_cpu, np.nan)
        coords = np.argwhere(mask)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            plot_data = masked_data[y_min:y_max+1, x_min:x_max+1]
            vmin, vmax = float(np.min(data_cpu[mask])), float(np.max(data_cpu[mask]))
        else:
            plot_data = masked_data
            vmin, vmax = 0, 1
        
        img_plot = plt.imshow(plot_data, cmap='inferno', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img_plot)
        cbar.set_label(label, rotation=270, labelpad=15)
        plt.title(f"{title}\nMin: {vmin:.4f} | Max: {vmax:.4f} | Mean: {mean_val:.4f}")
        plt.axis('off')
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\normal_map_code\batch_process_normals_fast_config.json"
    GPUNormalProcessor(CONFIG_PATH).process()