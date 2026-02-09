import numpy as np
import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import cupy as cp

class IntensityGAnalyzerGPU:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.render_base = Path(self.cfg['paths']['rendered_output_base'])
        self.geo_csv_base = Path(self.cfg['paths']['world_coordinate_csv_base']) 
        self.output_csv_base = Path(self.cfg['paths']['csv_output_base'])
        self.plot_base = Path(self.cfg['paths']['plot_output_base'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        # New Resolution & Threshold Settings
        self.res_x = self.cfg['resolution']['width']
        self.res_y = self.cfg['resolution']['height']
        self.alpha_thresh = self.cfg['processing_settings'].get('alpha_threshold', 0.5)

        self.method = self.cfg['processing_settings']['exposure_method']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']
        self.num_g_bins = self.cfg['processing_settings'].get('num_g_plot_samples', 15)
        self.plot_channel = self.cfg['plot_settings'].get('color_channel', 'Gray')

    def calculate_nT_G_gpu(self, p_surf_gpu, n_surf_gpu, light_vec, dist, params):
        """Calculates nT.G using explicit world coordinates from CSV."""
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['processing_settings']['samples_per_axis']
        total_area = l_dims[0] * l_dims[1]
        half_spread = cp.deg2rad(params['spread'] / 2.0)
        
        # 1. Blender Energy Scaling
        energy_scale = 1.0
        if self.cfg['processing_settings'].get('use_blender_energy_scaling', True) and params['spread'] < 180.0:
            energy_scale = 1.0 / (cp.sin(half_spread)**2 + 1e-9)

        # 2. Setup Light Geometry
        light_center = cp.array(light_vec * dist, dtype=cp.float32)
        n_A = -cp.array(light_vec, dtype=cp.float32) 
        
        up = cp.array([0, 0, 1], dtype=cp.float32)
        right = cp.cross(up, n_A)
        if cp.linalg.norm(right) < 1e-3: right = cp.array([1, 0, 0], dtype=cp.float32)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(n_A, right)
        
        steps = cp.linspace(-0.5, 0.5, samp_ax)
        ii, jj = cp.meshgrid(steps, steps)
        led_pts = light_center + (right[None, None, :] * ii[:, :, None] * l_dims[0]) + \
                                 (up_loc[None, None, :] * jj[:, :, None] * l_dims[1])
        led_pts = led_pts.reshape(-1, 3)

        # 3. Integration Loop
        G_acc = cp.zeros(p_surf_gpu.shape[0], dtype=cp.float32)
        
        for pt in led_pts:
            v = pt - p_surf_gpu 
            dist_sq = cp.sum(v**2, axis=1) + 1e-9
            l_k = v / cp.sqrt(dist_sq)[:, None]
            
            # Emitter cosine
            cos_emitter = cp.sum(-n_A * l_k, axis=1)
            
            # Receiver cosine (nT)
            cos_receiver = cp.sum(n_surf_gpu * l_k, axis=1)
            
            # Combined Term with Backface Culling
            term = (cp.maximum(0, cos_emitter) * cp.maximum(0, cos_receiver)) / dist_sq
            
            # Cone Clipping
            if self.cfg['processing_settings'].get('discard_irrelevant_l_vectors', True):
                G_acc += cp.where(cos_emitter >= cp.cos(half_spread), term, 0)
            else:
                G_acc += term
        
        final_g = (G_acc * total_area * energy_scale) / len(led_pts)
        return cp.asnumpy(final_g)

    def extract_params(self, p_n, l_n):
        p, l = p_n.split('_'), l_n.split('_')
        return {'elev': float(p[1]), 'azim': float(p[0]), 'area': float(p[5]), 'dist': float(l[1]), 'spread': float(l[4])}

    def process(self):
        print("Starting nT.G Analysis (Using World Coordinate CSVs)...")
        
        # Walk through Render Directory Structure
        for s_fold in [d for d in self.render_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                
                # 1. Locate World Coordinate CSV for this geometry
                # New Logic: Base / Resolution_Folder / Config_Name / world_position.csv
                resolution_folder = f"{self.res_x}_{self.res_y}"
                geo_csv_path = self.geo_csv_base / resolution_folder / p_fold.name / "world_position.csv"
                
                if not geo_csv_path.exists():
                    print(f"Skipping {p_fold.name}: No world_position.csv found at {geo_csv_path}")
                    continue
                
                # 2. Load Geometry Data (CPU)
                print(f"  Loading Geometry: {p_fold.name}")
                df_geo = pd.read_csv(geo_csv_path)
                
                # Filter using Configurable Alpha Threshold
                if 'alpha' in df_geo.columns:
                    df_geo = df_geo[df_geo['alpha'] > self.alpha_thresh]

                u_coords = df_geo['pixel_u'].values.astype(int)
                v_coords = df_geo['pixel_v'].values.astype(int)
                
                # Prepare Surface positions for GPU
                p_surf_gpu = cp.array(df_geo[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32)
                
                # Calculate Normal (Constant per plane)
                params_proto = self.extract_params(p_fold.name, "0_0_AREA_0_0_0") 
                zenith = cp.deg2rad(90.0 - params_proto['elev'])
                az = cp.deg2rad(params_proto['azim'])
                n_vec = cp.array([cp.sin(zenith)*cp.cos(az), cp.sin(zenith)*cp.sin(az), cp.cos(zenith)], dtype=cp.float32)
                n_surf_gpu = n_vec.reshape(1, 3) 

                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    l_params = self.extract_params(p_fold.name, l_fold.name)
                    
                    # Load Light Matrix
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    if not matrix_file.exists(): continue
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values

                    for l_id in self.target_lights:
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        if not img_path.exists(): continue
                        
                        # Load Image
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is None: continue
                        
                        # --- PIXEL MAPPING ---
                        h, w = img_bgr.shape[:2]
                        valid_idx = (u_coords < w) & (v_coords < h) & (u_coords >= 0) & (v_coords >= 0)
                        
                        if not np.any(valid_idx): continue
                        
                        curr_u = u_coords[valid_idx]
                        curr_v = v_coords[valid_idx]
                        curr_p_surf = p_surf_gpu[valid_idx]
                        
                        # Extract Intensities
                        pixel_colors = img_bgr[curr_v, curr_u] 
                        pixel_gray = cv2.cvtColor(pixel_colors.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
                        
                        # --- CALCULATE G ---
                        ntg_vals = self.calculate_nT_G_gpu(curr_p_surf, n_surf_gpu, l_vecs[l_id-1], l_params['dist'], l_params)
                        
                        # --- EXPORT DATA ---
                        df_out = pd.DataFrame({
                            'pixel_u': curr_u,
                            'pixel_v': curr_v,
                            'x': cp.asnumpy(curr_p_surf[:, 0]),
                            'y': cp.asnumpy(curr_p_surf[:, 1]),
                            'Gray': pixel_gray,
                            'R': pixel_colors[:, 2],
                            'G': pixel_colors[:, 1],
                            'B': pixel_colors[:, 0],
                            'G_Value': ntg_vals
                        })
                        
                        # Save CSV
                        csv_dir = self.output_csv_base / s_fold.name / p_fold.name / l_fold.name
                        csv_dir.mkdir(parents=True, exist_ok=True)
                        df_out.to_csv(csv_dir / f"light_{l_id:03d}_pixel_data.csv", index=False)
                        
                        # Generate Plots
                        plot_dir = self.plot_base / s_fold.name / p_fold.name / l_fold.name
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        self.generate_plot_and_stats(df_out, plot_dir, l_id, l_fold.name)
                
                del p_surf_gpu
                cp.get_default_memory_pool().free_all_blocks()

    def generate_plot_and_stats(self, df, plot_dir, l_id, l_name):
        g_min, g_max = df['G_Value'].min(), df['G_Value'].max()
        target_g_values = np.linspace(g_min, g_max, self.num_g_bins) if self.num_g_bins > 1 else [g_min]

        plot_df_list = []
        stats_list = []
        
        for target in target_g_values:
            closest_g = df.iloc[(df['G_Value'] - target).abs().argsort()[:1]]['G_Value'].values[0]
            subset = df[np.isclose(df['G_Value'], closest_g, atol=1e-8)].copy()
            data = subset[self.plot_channel]
            
            if len(data) > 0:
                subset['G_Label'] = f"{closest_g:.5f}"
                plot_df_list.append(subset)
                
                stats_list.append({
                    'G_Value': closest_g,
                    'Min': data.min(),
                    'Q1': data.quantile(0.25),
                    'Median': data.median(),
                    'Q2_Mean': data.mean(),
                    'Q3': data.quantile(0.75),
                    'Max': data.max(),
                    'Range': data.max() - data.min(),
                    'IQR': data.quantile(0.75) - data.quantile(0.25),
                    'Pixel_Count': len(data)
                })
        
        pd.DataFrame(stats_list).to_csv(plot_dir / f"light_{l_id:03d}_{self.plot_channel}_stats.csv", index=False)

        if plot_df_list:
            plot_df = pd.concat(plot_df_list)
            fig, ax = plt.subplots(figsize=(16, 9))
            plot_df.boxplot(column=self.plot_channel, by='G_Label', grid=False, 
                            showfliers=self.cfg['plot_settings']['show_outliers'], 
                            ax=ax, showmeans=True)
            
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.title(f"Intensity vs nT.G | Light {l_id}\n{l_name}")
            plt.suptitle("") 
            plt.xlabel("nT.G (Geometric Vector Dot Product)")
            plt.ylabel(f"Pixel Intensity ({self.plot_channel})")
            plt.xticks(rotation=45, fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"light_{l_id:03d}_{self.plot_channel}_boxplot.png", dpi=self.cfg['plot_settings']['dpi'])
            plt.close()

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\nrfld_inv_sqr_plt_cfg.json"
    IntensityGAnalyzerGPU(CONFIG_PATH).process()