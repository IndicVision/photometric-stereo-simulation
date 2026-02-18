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
        self.input_base = Path(self.cfg['paths']['rendered_output_base'])
        self.csv_base = Path(self.cfg['paths']['csv_output_base'])
        self.plot_base = Path(self.cfg['paths']['plot_output_base'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        self.method = self.cfg['processing_settings']['exposure_method']
        self.bg_thresh = self.cfg['processing_settings']['background_threshold']
        self.erosion = self.cfg['processing_settings']['mask_erosion_iter']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']
        self.num_g_bins = self.cfg['processing_settings'].get('num_g_plot_samples', 15)
        self.plot_channel = self.cfg['plot_settings'].get('color_channel', 'Gray')
    def get_geometry_and_normal_gpu(self, params, w, h):
        """Constructs 3D surface positions and the unit normal n from path parameters."""
        zenith = cp.deg2rad(90.0 - params['elev'])
        az = cp.deg2rad(params['azim'])
        # Unit normal n based on Elevation and Azimuth from folder path
        nx = cp.sin(zenith) * cp.cos(az)
        ny = cp.sin(zenith) * cp.sin(az)
        nz = cp.cos(zenith)
        n = cp.array([nx, ny, nz], dtype=cp.float32)
        side = params['area']**0.5
        x = cp.linspace(-side/2, side/2, w)
        y = cp.linspace(side/2, -side/2, h)
        xv, yv = cp.meshgrid(x, y)
        zv = -(n[0]*xv + n[1]*yv) / (n[2] + 1e-9)
        p_surf = cp.stack((xv, yv, zv), axis=-1)
        return p_surf, n
    def calculate_nT_G_gpu(self, p_surf, n_surf, light_vec, dist, params):
        """Calculates nT.G with visibility clipping and energy scaling."""
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
        G_acc = cp.zeros((p_surf.shape[0], p_surf.shape[1], 3), dtype=cp.float32)
        for pt in led_pts:
            v = pt - p_surf
            dist_sq = cp.sum(v**2, axis=2) + 1e-9
            l_k = v / cp.sqrt(dist_sq)[:, :, None]
            # Emitter cosine visibility check
            cos_emitter = cp.sum(-n_A * l_k, axis=2)
            term = (cp.maximum(0, cos_emitter) / dist_sq)[:, :, None] * l_k
            # 3. Cone Clipping (discard_irrelevant_l_vectors)
            if self.cfg['processing_settings'].get('discard_irrelevant_l_vectors', True):
                G_acc += cp.where(cos_emitter[:, :, None] >= cp.cos(half_spread), term, 0)
            else:
                G_acc += term
        # Proportionality: nT * G_vector
        nT_G = cp.sum(G_acc * n_surf, axis=2)
        final_g = (nT_G * total_area * energy_scale) / len(led_pts)
        return cp.asnumpy(final_g)
    def process(self):
        print("Starting nT.G Analysis and Pixel Data Export...")
        for s_fold in [d for d in self.input_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    params = self.extract_params(p_fold.name, l_fold.name)
                    # Correctly locate matrix file using the new folder naming convention
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    if not matrix_file.exists():
                        print(f"Matrix file not found: {matrix_file}")
                        continue
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values
                    first_img_path = next(l_fold.glob(f"*.png"))
                    img_sample = cv2.imread(str(first_img_path))
                    h, w = img_sample.shape[:2]
                    p_surf_gpu, n_surf_gpu = self.get_geometry_and_normal_gpu(params, w, h)
                    for l_id in self.target_lights:
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        if not img_path.exists(): continue
                        img_bgr = cv2.imread(str(img_path))
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                        mask = cv2.erode((img_gray > self.bg_thresh).astype(np.uint8), np.ones((3,3)), iterations=self.erosion).astype(bool)
                        ntg_map = self.calculate_nT_G_gpu(p_surf_gpu, n_surf_gpu, l_vecs[l_id-1], params['dist'], params)
                        y, x = np.indices(img_gray.shape)
                        # 1. Generate Full Pixel Data CSV
                        df_pixel_all = pd.DataFrame({
                            'x': x[mask], 'y': y[mask],
                            'Gray': (img_gray[mask] * 255).astype(np.uint8),
                            'R': img_bgr[mask][:, 2], 'G': img_bgr[mask][:, 1], 'B': img_bgr[mask][:, 0],
                            'G_Value': ntg_map[mask]
                        })
                        csv_dir = self.csv_base / s_fold.name / p_fold.name / l_fold.name
                        csv_dir.mkdir(parents=True, exist_ok=True)
                        df_pixel_all.to_csv(csv_dir / f"light_{l_id:03d}_pixel_data.csv", index=False)
                        # 2. Generate Plots and Comprehensive Summary CSV
                        plot_dir = self.plot_base / s_fold.name / p_fold.name / l_fold.name
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        self.generate_plot_and_stats(df_pixel_all, plot_dir, l_id, l_fold.name)
                    del p_surf_gpu, n_surf_gpu
                    cp.get_default_memory_pool().free_all_blocks()
    def extract_params(self, p_n, l_n):
        """
        Modified to handle:
        p_n: azimuth_elev_x_y_z_area (e.g., 0.00_60.00_0.00_0.00_0.00_9.57)
        l_n: num_dist_type_shape_spread_Man_energy (e.g., 4_30.00_area_rectangle_180.0_Man_1.17)
        """
        p, l = p_n.split('_'), l_n.split('_')
        return {
            'elev': float(p[1]),
            'azim': float(p[0]),
            'area': float(p[5]), # Corrected index for Plane Area
            'dist': float(l[1]),
            'spread': float(l[4])
        }
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
                # Extended Statistics for Summary CSV
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
        # Save the full summary CSV
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
    # Ensure this points to your specific configuration file
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\non_uni_pos_lit_rndr\upd_nrfld_inv_sqr_plt_cfg.json"
    IntensityGAnalyzerGPU(CONFIG_PATH).process()