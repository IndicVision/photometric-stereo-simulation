import cupy as cp  # GPU Acceleration
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

class GPUNormalProcessor:
    def __init__(self, config_path):
        """Initializes paths and reconstruction settings from JSON."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_base = Path(self.cfg['paths']['renders_dir'])
        self.output_base = Path(self.cfg['paths']['reconstruction_output_dir'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        self.geo_csv_base = Path(self.cfg['paths']['world_coordinate_csv_base'])
        self.ev_csv_path = Path(self.cfg['paths']['exposure_report_csv_path'])
        
        # Settings
        self.res_x = self.cfg['resolution']['width']
        self.res_y = self.cfg['resolution']['height']
        self.alpha_thresh = self.cfg['reconstruction_settings'].get('alpha_threshold', 0.5)
        
        # Gamma Safety (Default 1.0 for Raw)
        self.gamma = self.cfg['reconstruction_settings'].get('gamma_correction', 1.0)
        
        # Physics Constants (For Error Calculation Only)
        self.target_rho_constant = self.cfg['reconstruction_settings'].get('base_albedo_constant', 0.8)
        self.bit_depth = self.cfg['reconstruction_settings'].get('bit_depth', 8)

        # Inside __init__
        self.force_z = self.cfg['reconstruction_settings'].get('force_positive_z', False)

        # Load EV Report Lookup Table
        self.ev_df = None
        if self.ev_csv_path.exists():
            print("Loading Exposure Value Report...")
            self.ev_df = pd.read_csv(self.ev_csv_path)
        else:
            print(f"[WARNING] EV Report not found at {self.ev_csv_path}. E will be 1.0")

    def parse_folder_params(self, p_folder_name, l_folder_name):
        """Extracts configuration parameters from directory names."""
        p_parts = p_folder_name.split('_')
        l_parts = l_folder_name.split('_')
        return {
            'elev': float(p_parts[1]), 
            'azim': float(p_parts[0]), 
            'area': float(p_parts[5]),
            'num_l': int(l_parts[0]), 
            'dist': float(l_parts[1]), # CM
            'spread': float(l_parts[4]), 
            'psi': float(l_parts[5]), 
            'energy': float(l_parts[6])
        }

    def get_exposure_value(self, samples, config_name, light_setup, light_id):
        """Looks up the specific EV used for a specific light from the report."""
        if self.ev_df is None: return 1.0
        
        method = self.cfg['analysis_settings']['exposure_method']
        
        subset = self.ev_df[
            (self.ev_df['Samples'] == samples) & 
            (self.ev_df['Configuration'] == config_name) & 
            (self.ev_df['Light_Setup'] == light_setup)
        ]
        
        if subset.empty: return 1.0
        
        if 'Source_ID' in subset.columns:
            row = subset[subset['Source_ID'] == light_id]
            if not row.empty:
                return float(row.iloc[0].get(method, 1.0))
            
            row = subset[subset['Source_ID'] == 'AGGREGATE']
            if not row.empty:
                return float(row.iloc[0].get(method, 1.0))

        return float(subset.iloc[0].get(method, 1.0))

    def process(self):
        sample_folders = sorted([d for d in self.input_base.iterdir() if d.is_dir() and d.name.startswith("samples_")])
        print(f"Found {len(sample_folders)} sample folders.")
        print(f"Note: Applying CM -> Meter scaling. Gamma: {self.gamma}")
        
        for s_folder in sample_folders:
            try:
                samples_val = int(s_folder.name.split('_')[1])
            except: continue

            for p_folder in [d for d in s_folder.iterdir() if d.is_dir()]:
                
                # 1. Load Geometry from CSV
                resolution_folder = f"{self.res_x}_{self.res_y}"
                geo_csv_path = self.geo_csv_base / resolution_folder / p_folder.name / "world_position.csv"
                
                if not geo_csv_path.exists():
                    continue

                df_geo = pd.read_csv(geo_csv_path)
                if 'alpha' in df_geo.columns:
                    df_geo = df_geo[df_geo['alpha'] > self.alpha_thresh]
                
                # Extract valid pixel indices and coordinates
                u_coords = df_geo['pixel_u'].values.astype(int)
                v_coords = df_geo['pixel_v'].values.astype(int)
                p_surf_gpu = cp.array(df_geo[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32)

                # Calculate Ground Truth Normal (Only for Error Checking)
                params_proto = self.parse_folder_params(p_folder.name, "0_0_AREA_0_0_0_0")
                zenith = np.deg2rad(90.0 - params_proto['elev'])
                az = np.deg2rad(params_proto['azim'])
                nx, ny, nz = np.sin(zenith)*np.cos(az), np.sin(zenith)*np.sin(az), np.cos(zenith)
                true_n_gpu = cp.array([nx, ny, nz], dtype=cp.float32)

                for l_folder in [d for d in p_folder.iterdir() if d.is_dir()]:
                    params = self.parse_folder_params(p_folder.name, l_folder.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_folder.name}__{l_folder.name}.csv"
                    
                    if not matrix_file.exists(): continue
                    
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values.astype(np.float32)
                    method = self.cfg['analysis_settings']['exposure_method']
                    
                    # 2. Collect Images and EVs
                    intensities_list = []
                    valid_light_vecs = []
                    ev_list = []

                    for i in range(1, params['num_l'] + 1):
                        img_p = l_folder / f"{i:03d}_{method}.png"
                        if img_p.exists():
                            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                            valid_pixels = img[v_coords, u_coords].astype(np.float32)
                            
                            # Gamma Linearization (Safety)
                            if self.gamma != 1.0:
                                valid_pixels = 255.0 * ((valid_pixels / 255.0) ** self.gamma)
                            
                            intensities_list.append(valid_pixels)
                            valid_light_vecs.append(l_vecs[i-1])
                            
                            ev = self.get_exposure_value(samples_val, p_folder.name, l_folder.name, i)
                            ev_list.append(ev)

                    if not intensities_list: continue

                    print(f"Processing: {l_folder.name}")

                    # Upload to GPU
                    I_gpu = cp.stack([cp.array(x) for x in intensities_list], axis=1)
                    L_vecs_gpu = cp.array(valid_light_vecs, dtype=cp.float32)
                    
                    # 3. Solve Normals (Returns Physical Albedo via inversion)
                    avg_ev = np.mean(ev_list)
                    n_map_flat, albedo_physical_flat, errs_flat = self.solve_normals_data_driven(
                        I_gpu, L_vecs_gpu, p_surf_gpu, true_n_gpu, params, avg_ev
                    )
                    
                    # Error vs Constant Target (0.8)
                    alb_err_flat = cp.abs(albedo_physical_flat - self.target_rho_constant)

                    # 4. Reconstruct 2D Maps for Output
                    H, W = self.res_y, self.res_x
                    
                    mask_2d = np.zeros((H, W), dtype=bool)
                    mask_2d[v_coords, u_coords] = True
                    
                    def to_2d(flat_arr, channels=1):
                        cpu_arr = cp.asnumpy(flat_arr)
                        if channels == 1:
                            out = np.zeros((H, W), dtype=np.float32)
                            out[v_coords, u_coords] = cpu_arr
                            return out
                        else:
                            out = np.zeros((H, W, channels), dtype=np.float32)
                            out[v_coords, u_coords] = cpu_arr
                            return out

                    n_map_2d = to_2d(n_map_flat, 3)
                    albedo_2d = to_2d(albedo_physical_flat)
                    err_2d = to_2d(errs_flat)
                    alb_err_2d = to_2d(alb_err_flat)

                    stats = {
                        "elev": params['elev'], "azim": params['azim'], "area": params['area'],
                        "num_l": params['num_l'], "dist": params['dist'], "spread": params['spread'],
                        "psi": params['psi'], "energy": params['energy'], "samples": samples_val,
                        "mean_error_deg": float(np.mean(cp.asnumpy(errs_flat))),
                        "mean_albedo": float(np.mean(cp.asnumpy(albedo_physical_flat))),
                        "mean_albedo_error": float(np.mean(cp.asnumpy(alb_err_flat))),
                        "valid_pixels": int(len(errs_flat))
                    }

                    self.save_outputs(s_folder.name, p_folder.name, l_folder.name, 
                                      n_map_2d, albedo_2d, err_2d, alb_err_2d, mask_2d, 
                                      stats, u_coords, v_coords)
                    
                    del I_gpu, L_vecs_gpu, n_map_flat
                    cp.get_default_memory_pool().free_all_blocks()

    def solve_normals_data_driven(self, I, L_vecs, P_surf, true_n, params, avg_ev):
        num_pixels = P_surf.shape[0]
        num_lights = L_vecs.shape[0]
        
        # --- A. G-Matrix Calculation (Physics Model) ---
        
        # Unit Conversion (cm -> m)
        scale_factor = 0.01
        
        dim_a_m = self.cfg['area_light_props']['dim_a_cm'] * scale_factor
        dim_b_m = self.cfg['area_light_props']['dim_b_cm'] * scale_factor
        l_dims = (dim_a_m, dim_b_m)
        dist_m = params['dist'] * scale_factor
        
        samp_ax = self.cfg['reconstruction_settings']['samples_per_axis']
        half_spread = np.deg2rad(params['spread'] / 2.0)
        
        G_stack = []

        for i in range(num_lights):
            l_vec = L_vecs[i]
            # Get Samples (Midpoint Logic)
            light_pts = self.get_light_samples_gpu(l_vec, dist_m, l_dims, samp_ax)
            
            G_accum = cp.zeros((num_pixels, 3), dtype=cp.float32)
            n_A = -l_vec 
            
            for k in range(light_pts.shape[0]):
                pt = light_pts[k]
                v = pt - P_surf 
                dist_sq = cp.sum(v**2, axis=1) + 1e-9
                dist = cp.sqrt(dist_sq)
                l_k = v / dist[:, None]
                
                cos_emitter = cp.sum(-n_A * l_k, axis=1)
                
                # Raw Geometric Sum (1/r^2)
                # Note: No Area Multiplication here. G is Raw.
                contrib = (cp.maximum(0, cos_emitter) / dist_sq)[:, None] * l_k
                
                if self.cfg['reconstruction_settings'].get('discard_irrelevant_l_vectors', True):
                    mask = cos_emitter >= np.cos(half_spread)
                    G_accum += contrib * mask[:, None]
                else:
                    G_accum += contrib
            
            G_stack.append(G_accum)
            
        G = cp.stack(G_stack, axis=1)
        
        # --- B. Solver (Pseudo-Inverse) ---
        # N = (G.T * G)^-1 * G.T * I
        GT = G.transpose(0, 2, 1)
        GTG = cp.matmul(GT, G)
        GTG += cp.eye(3, dtype=cp.float32) * 1e-6 # Regularization
        GTG_inv = cp.linalg.inv(GTG)
        
        I_col = I[:, :, None]
        GTI = cp.matmul(GT, I_col)
        N_raw = cp.matmul(GTG_inv, GTI).squeeze(2)


        # --- NEW: Hemisphere Constraint Logic ---
        if self.force_z:
            # If Nz is negative, flip the entire (Nx, Ny, Nz) vector
            mask = N_raw[:, 2] < 0
            N_raw[mask] *= -1.0
        
        
        # --- C. Physical Albedo Extraction ---
        # Magnitude M = I / G_raw
        Magnitude = cp.linalg.norm(N_raw, axis=1)
        
        # Equation Reversion:
        # Rho_Phys = M * [ (pi^2 * K * sin^2) / (Phi * E * 255) ]
        K_points = samp_ax[0] * samp_ax[1]
        sin_sq_sigma = np.sin(half_spread)**2 + 1e-9
        quant_scale = (2 ** self.bit_depth) - 1
        
        numerator = (np.pi**2) * K_points * sin_sq_sigma
        denominator = params['energy'] * avg_ev * quant_scale
        
        correction_factor = numerator / denominator
        Physical_Albedo = Magnitude * correction_factor
        
        # Normalize to get Unit Normal
        Unit_Normal = N_raw / (Magnitude[:, None] + 1e-9)
        
        # --- D. Error Check (Ground Truth Comparison) ---
        dot = cp.sum(Unit_Normal * true_n, axis=1)
        dot = cp.clip(dot, -1.0, 1.0)
        errs = cp.degrees(cp.arccos(dot))
        
        return Unit_Normal, Physical_Albedo, errs

    def get_light_samples_gpu(self, l_vec, dist, dims, n_samp):
        center = l_vec * dist 
        normal = -l_vec
        up = cp.array([0, 0, 1], dtype=cp.float32)
        right = cp.cross(up, normal)
        if cp.linalg.norm(right) < 1e-3: 
            right = cp.array([1, 0, 0], dtype=cp.float32)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(normal, right)
        
        w, h = dims
        
        # --- FIX STARTS HERE ---
        nx = int(n_samp[0]) # Width samples
        ny = int(n_samp[1]) # Height samples

        # Calculate offsets separately
        off_x = 1.0 / (2.0 * nx)
        off_y = 1.0 / (2.0 * ny)

        # Generate separate linspaces for Width (X) and Height (Y)
        steps_x = cp.linspace(-0.5 + off_x, 0.5 - off_x, nx)
        steps_y = cp.linspace(-0.5 + off_y, 0.5 - off_y, ny)
        
        # Create the grid using the separate axes
        # Note: meshgrid indexing='xy' (Cartesian) is standard, 
        # meaning ii corresponds to steps_x (width), jj to steps_y (height)
        ii, jj = cp.meshgrid(steps_x, steps_y) 
        # --- FIX ENDS HERE ---

        # The rest of your projection logic remains valid
        samples = center + (right[None, None, :] * ii[:, :, None] * w) + \
                           (up_loc[None, None, :] * jj[:, :, None] * h)
        return samples.reshape(-1, 3)

    def save_outputs(self, s_n, p_n, l_n, n_map, albedo, errs, alb_errs, mask, stats, u_c, v_c):
        out_path = self.output_base / s_n / p_n / l_n
        out_path.mkdir(parents=True, exist_ok=True)
        
        self.generate_heatmap(errs, mask, out_path / "error_map.png", 
                              f"Normal Error: {p_n}", "Deg", stats['mean_error_deg'])
        self.generate_heatmap(alb_errs, mask, out_path / "albedo_error_map.png", 
                              f"Albedo Error: {p_n}", "Diff", stats['mean_albedo_error'])
        
        if self.cfg['outputs'].get('save_stats_report', True):
            cols = ["elev","azim","area","num_l","dist","spread","psi","energy","samples",
                    "mean_albedo", "mean_error_deg", "mean_albedo_error", "valid_pixels"]
            pd.DataFrame([stats], columns=cols).to_csv(out_path / "recon_report.csv", index=False)
        
        if self.cfg['outputs'].get('save_per_pixel_csv', True):
            df_pixels = pd.DataFrame({
                'pixel_u': u_c,
                'pixel_v': v_c,
                'nx': n_map[mask][:,0], 
                'ny': n_map[mask][:,1], 
                'nz': n_map[mask][:,2],
                'albedo': albedo[mask], # This is now PHYSICAL ALBEDO (approx 0.8)
                'err_deg': errs[mask],
                'albedo_err': alb_errs[mask]
            })
            df_pixels.to_csv(out_path / "pixel_data.csv", index=False)

    def generate_heatmap(self, data, mask, save_path, title, label, mean_val):
        plt.figure(figsize=(10, 8))
        coords = np.argwhere(mask)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            plot_data = data[y_min:y_max+1, x_min:x_max+1]
            crop_mask = mask[y_min:y_max+1, x_min:x_max+1]
            plot_data = np.where(crop_mask, plot_data, np.nan)
            valid_vals = data[mask]
            vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))
        else:
            plot_data = data
            vmin, vmax = 0, 1
            
        img_plot = plt.imshow(plot_data, cmap='inferno', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img_plot)
        cbar.set_label(label, rotation=270, labelpad=15)
        plt.title(f"{title}\nMin: {vmin:.4f} | Max: {vmax:.4f} | Mean: {mean_val:.4f}")
        plt.axis('off')
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\bat_proc_nrml_fst_cfg.json"
    GPUNormalProcessor(CONFIG_PATH).process()