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
        """
        Initializes the analyzer.
        Loads configuration and establishes file paths.
        """
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.render_base = Path(self.cfg['paths']['rendered_output_base'])
        self.geo_csv_base = Path(self.cfg['paths']['world_coordinate_csv_base']) 
        self.output_csv_base = Path(self.cfg['paths']['csv_output_base'])
        self.plot_base = Path(self.cfg['paths']['plot_output_base'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        # Dimensions & Thresholds
        self.res_x = self.cfg['resolution']['width']
        self.res_y = self.cfg['resolution']['height']
        self.alpha_thresh = self.cfg['processing_settings'].get('alpha_threshold', 0.5)
        
        # Gamma (Safety Feature - Default 1.0 for Raw renders)
        self.gamma = self.cfg['processing_settings'].get('gamma_correction', 1.0)

        # Analysis Settings
        self.method = self.cfg['processing_settings']['exposure_method']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']
        self.plot_channel = self.cfg['plot_settings'].get('color_channel', 'Gray')
        
        # Fitting Settings
        self.poly_degree = self.cfg['curve_fitting']['polynomial_degree']
        self.fit_strategy = self.cfg['curve_fitting']['fit_strategy'] 
        self.fit_bins = self.cfg['curve_fitting'].get('fitting_bins', 50) 
        self.heatmap_bins = self.cfg['plot_settings'].get('heatmap_bins', 100)

    def calculate_nT_G_gpu(self, p_surf_gpu, n_surf_gpu, light_vec, dist_cm, params):
        """
        Calculates the Geometric Factor (G) for every pixel using GPU acceleration.
        
        MATH LOGIC:
        G = Integral( (cos_emit * cos_recv) / r^2 ) dA
        Approximated as: Sum( ... ) * (Total_Area / Num_Points)
        """
        
        # --- 1. UNIT CONVERSION (Centimeters -> Meters) ---
        # World CSV is Meters. Config is CM. We standardize to Meters.
        scale_factor = 0.01 
        
        dim_a_m = self.cfg['area_light_props']['dim_a_cm'] * scale_factor
        dim_b_m = self.cfg['area_light_props']['dim_b_cm'] * scale_factor
        l_dims = (dim_a_m, dim_b_m)
        dist_m = dist_cm * scale_factor
        
        # Area in m^2 (Required for Irradiance Calculation)
        total_area = dim_a_m * dim_b_m 
        # --------------------------------------------------

        samp_ax = self.cfg['processing_settings']['samples_per_axis']
        half_spread = cp.deg2rad(params['spread'] / 2.0)
        
        # Energy Scaling: Disable if comparing to Theoretical Equation with sin^2 term
        energy_scale = 1.0
        if self.cfg['processing_settings'].get('use_blender_energy_scaling', False) and params['spread'] < 180.0:
            energy_scale = 1.0 / (cp.sin(half_spread)**2 + 1e-9)

        # Light Frame Construction
        light_center = cp.array(light_vec * dist_m, dtype=cp.float32)
        n_A = -cp.array(light_vec, dtype=cp.float32) 
        
        up = cp.array([0, 0, 1], dtype=cp.float32)
        right = cp.cross(up, n_A)
        # Robust handling for lights pointing straight up/down
        if cp.linalg.norm(right) < 1e-3: right = cp.array([1, 0, 0], dtype=cp.float32)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(n_A, right)
        
        # --- 2. SAMPLING CORRECTION (Midpoint vs Edge) ---
        # We sample the CENTER of each grid cell to improve accuracy.
        # offset = (Range) / (2 * Count) = 1.0 / (2 * samp_ax)
        offset = 1.0 / (2.0 * samp_ax)
        steps = cp.linspace(-0.5 + offset, 0.5 - offset, samp_ax)
        ii, jj = cp.meshgrid(steps, steps)
        
        # Generate Grid Points in 3D Space (Meters)
        led_pts = light_center + (right[None, None, :] * ii[:, :, None] * l_dims[0]) + \
                                 (up_loc[None, None, :] * jj[:, :, None] * l_dims[1])
        led_pts = led_pts.reshape(-1, 3)

        G_acc = cp.zeros(p_surf_gpu.shape[0], dtype=cp.float32)
        
        # --- 3. INTEGRATION LOOP ---
        for pt in led_pts:
            v = pt - p_surf_gpu 
            dist_sq = cp.sum(v**2, axis=1) + 1e-9
            l_k = v / cp.sqrt(dist_sq)[:, None]
            
            cos_emitter = cp.sum(-n_A * l_k, axis=1)
            cos_receiver = cp.sum(n_surf_gpu * l_k, axis=1)
            
            # The Geometric Term (1/m^2)
            term = (cp.maximum(0, cos_emitter) * cp.maximum(0, cos_receiver)) / dist_sq
            
            # Cone Clipping (Barn Door Effect)
            if self.cfg['processing_settings'].get('discard_irrelevant_l_vectors', True):
                mask = cos_emitter >= cp.cos(half_spread)
                G_acc += term * mask
            else:
                G_acc += term
        
        # --- 4. FINAL SCALING ---
        # Convert Sum to Irradiance: Sum * (dA)
        # dA = Total_Area / Number_of_Points
        final_g = (G_acc * total_area * energy_scale) / len(led_pts)
        
        return cp.asnumpy(final_g)

    def extract_params(self, p_n, l_n):
        """Parses folder names to extract physical parameters."""
        p, l = p_n.split('_'), l_n.split('_')
        return {
            'elev': float(p[1]), 'azim': float(p[0]), 'area': float(p[5]),
            'dist': float(l[1]), 'spread': float(l[4])
        }

    def process(self):
        print(f"Starting Analysis | Gamma: {self.gamma} | Scale: 0.01 (Meters)")
        
        for s_fold in [d for d in self.render_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                
                # Load Geometry (Meters)
                resolution_folder = f"{self.res_x}_{self.res_y}"
                geo_csv_path = self.geo_csv_base / resolution_folder / p_fold.name / "world_position.csv"
                
                if not geo_csv_path.exists(): continue
                print(f"  Processing: {p_fold.name}")
                
                df_geo = pd.read_csv(geo_csv_path)
                if 'alpha' in df_geo.columns:
                    df_geo = df_geo[df_geo['alpha'] > self.alpha_thresh]

                u_coords = df_geo['pixel_u'].values.astype(int)
                v_coords = df_geo['pixel_v'].values.astype(int)
                p_surf_gpu = cp.array(df_geo[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32)
                
                # Analytical Surface Normal
                params_proto = self.extract_params(p_fold.name, "0_0_AREA_0_0_0") 
                zenith = cp.deg2rad(90.0 - params_proto['elev'])
                az = cp.deg2rad(params_proto['azim'])
                n_vec = cp.array([cp.sin(zenith)*cp.cos(az), cp.sin(zenith)*cp.sin(az), cp.cos(zenith)], dtype=cp.float32)
                n_surf_gpu = n_vec.reshape(1, 3) 

                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    l_params = self.extract_params(p_fold.name, l_fold.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    
                    if not matrix_file.exists(): continue
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values
                    
                    slopes_data = []

                    for l_id in self.target_lights:
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        if not img_path.exists(): continue
                        
                        # Load Image (0-255 sRGB/Raw)
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is None: continue
                        
                        h, w = img_bgr.shape[:2]
                        valid_idx = (u_coords < w) & (v_coords < h) & (u_coords >= 0) & (v_coords >= 0)
                        
                        if not np.any(valid_idx): continue
                        
                        curr_u = u_coords[valid_idx]
                        curr_v = v_coords[valid_idx]
                        curr_p_surf = p_surf_gpu[valid_idx]
                        
                        pixel_colors = img_bgr[curr_v, curr_u] 
                        pixel_gray = cv2.cvtColor(pixel_colors.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
                        
                        # --- GAMMA LINEARIZATION ---
                        # If gamma != 1.0, convert to Linear. 
                        # I_lin = (I_srgb / 255)^gamma * 255
                        if self.gamma != 1.0:
                            pixel_gray = 255.0 * ((pixel_gray.astype(np.float32) / 255.0) ** self.gamma)
                        
                        # Calculate G
                        ntg_vals = self.calculate_nT_G_gpu(curr_p_surf, n_surf_gpu, l_vecs[l_id-1], l_params['dist'], l_params)
                        
                        # Save Per-Light CSV
                        df_out = pd.DataFrame({
                            'pixel_u': curr_u, 'pixel_v': curr_v,
                            'x': cp.asnumpy(curr_p_surf[:, 0]), 'y': cp.asnumpy(curr_p_surf[:, 1]),
                            'Gray': pixel_gray, 'G_Value': ntg_vals
                        })
                        
                        csv_dir = self.output_csv_base / s_fold.name / p_fold.name / l_fold.name
                        csv_dir.mkdir(parents=True, exist_ok=True)
                        df_out.to_csv(csv_dir / f"light_{l_id:03d}_pixel_data.csv", index=False)
                        
                        # Generate Plot
                        plot_dir = self.plot_base / s_fold.name / p_fold.name / l_fold.name
                        plot_dir.mkdir(parents=True, exist_ok=True)
                        metrics = self.generate_density_plot_with_fit(df_out, plot_dir, l_id, l_fold.name)
                        
                        if metrics:
                            slopes_data.append({
                                'Light_ID': l_id,
                                'Slope': metrics['slope'],
                                'R_Squared': metrics['r2'],
                                'Intercept': metrics['intercept']
                            })
                    
                    if slopes_data:
                        summary_df = pd.DataFrame(slopes_data)
                        summary_filename = f"slopes_summary_{self.plot_channel}_{self.fit_strategy}_Poly{self.poly_degree}.csv"
                        save_path = self.plot_base / s_fold.name / p_fold.name / l_fold.name / summary_filename
                        summary_df.to_csv(save_path, index=False)

                del p_surf_gpu
                cp.get_default_memory_pool().free_all_blocks()

    def generate_density_plot_with_fit(self, df, plot_dir, l_id, l_name):
        x = df['G_Value'].values
        y = df['Gray'].values # Using Gray channel
        
        fig, ax = plt.subplots(figsize=(12, 8))
        h = ax.hist2d(x, y, bins=self.heatmap_bins, cmap='inferno', density=True, zorder=1)
        plt.colorbar(h[3], ax=ax, label='Pixel Density')
        
        fit_x, fit_y = np.array([]), np.array([])
        if self.fit_strategy == "Global_OLS":
            fit_x, fit_y = x, y
        else:
            bins = np.linspace(x.min(), x.max(), self.fit_bins)
            digitized = np.digitize(x, bins)
            bin_x, bin_y = [], []
            for i in range(1, len(bins)):
                mask = digitized == i
                if np.any(mask):
                    bin_x.append(x[mask].mean())
                    bin_y.append(np.median(y[mask]) if self.fit_strategy == "Binned_Median" else y[mask].mean())
            fit_x, fit_y = np.array(bin_x), np.array(bin_y)

        metrics = None
        if len(fit_x) > self.poly_degree:
            coeffs = np.polyfit(fit_x, fit_y, self.poly_degree)
            poly_func = np.poly1d(coeffs)
            slope = coeffs[self.poly_degree - 1] if self.poly_degree >= 1 else 0.0
            intercept = coeffs[-1]
            
            # --- MODIFICATION: Create Equation String ---
            # Example: y = 2.34e+02x + 1.23e+00
            eq_str = f"$y = $"
            for i, c in enumerate(coeffs):
                power = self.poly_degree - i
                sign = "+" if c >= 0 and i > 0 else ""
                if power > 1: eq_str += f"{sign}{c:.2e}$x^{power}$ "
                elif power == 1: eq_str += f"{sign}{c:.2e}$x$ "
                else: eq_str += f"{sign}{c:.2e}"
            # --------------------------------------------
            
            x_line = np.linspace(x.min(), x.max(), 500)
            y_pred = poly_func(x)
            
            # Calculate R2 based on fit method (binned vs global)
            ss_res = np.sum((fit_y - poly_func(fit_x)) ** 2)
            ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-9))
            
            metrics = {'slope': slope, 'intercept': intercept, 'r2': r2}
            
            ax.plot(x_line, poly_func(x_line), color='cyan', linewidth=2.5, 
                    label=f"Fit: {self.fit_strategy}\n$R^2 = {r2:.4f}$")
            
            # --- MODIFICATION: Add Text Box to Plot ---
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, eq_str, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            # ------------------------------------------
            
        plt.title(f"Intensity vs nT.G | Light {l_id}\n{l_name}")
        plt.xlabel("nT.G (Irradiance W/m2)")
        plt.ylabel("Pixel Intensity (0-255)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(plot_dir / f"light_{l_id:03d}_{self.plot_channel}_{self.fit_strategy}_Poly{self.poly_degree}.png", dpi=300)
        plt.close()
        return metrics

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\upd_nrfld_inv_sqr_plt_cfg.json"
    IntensityGAnalyzerGPU(CONFIG_PATH).process()