import numpy as np
import cv2
import os
import json
import pandas as pd
from pathlib import Path

class DirectoryNormalProcessor:
    def __init__(self, config_path):
        """Initializes paths and reconstruction settings from JSON."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.input_base = Path(self.cfg['paths']['renders_dir'])
        self.output_base = Path(self.cfg['paths']['reconstruction_output_dir'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        self.bg_threshold = self.cfg['reconstruction_settings'].get('background_threshold', 0.05)
        self.erosion_iter = self.cfg['reconstruction_settings'].get('mask_erosion_iter', 2)

    def parse_folder_params(self, plane_folder, light_folder):
        """Extracts configuration parameters from directory names."""
        p_parts = plane_folder.split('_')
        l_parts = light_folder.split('_')
        
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
        all_stats = []
        sample_folders = [d for d in self.input_base.iterdir() if d.is_dir() and d.name.startswith("samples_")]
        
        for s_folder in sample_folders:
            samples_val = int(s_folder.name.split('_')[1])
            for p_folder in [d for d in s_folder.iterdir() if d.is_dir()]:
                for l_folder in [d for d in p_folder.iterdir() if d.is_dir()]:
                    
                    params = self.parse_folder_params(p_folder.name, l_folder.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_folder.name}__{l_folder.name}.csv"
                    
                    if not matrix_file.exists(): continue
                    light_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values.astype(np.float32)
                    method = self.cfg['analysis_settings']['exposure_method']
                    
                    images = []
                    for i in range(1, params['num_l'] + 1):
                        img_p = l_folder / f"{i:03d}_{method}.png"
                        if img_p.exists():
                            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                            images.append(img.astype(np.float32)/255.0)

                    if len(images) != len(light_vecs): continue
                    
                    print(f"Processing: {s_folder.name} | {p_folder.name} | {l_folder.name}")
                    
                    H, W = images[0].shape
                    avg_img = np.mean(images, axis=0)
                    raw_mask = (avg_img > self.bg_threshold).astype(np.uint8)
                    mask = cv2.erode(raw_mask, np.ones((3,3), np.uint8), iterations=self.erosion_iter).astype(bool)
                    
                    # Compute geometry and solve normals
                    P_surf, TRUE_N = self.get_geometry(params, W, H)
                    N_Map, Albedo, errs = self.solve_normals(images, light_vecs, P_surf, TRUE_N, params)
                    
                    valid_errs = errs[mask]
                    
                    # Store statistics for the final report
                    stats = {
                        "Samples": samples_val,
                        "Configuration": p_folder.name,
                        "Light_Setup": l_folder.name,
                        "Mean_Error_Deg": np.mean(valid_errs) if valid_errs.size > 0 else 0,
                        "Valid_Pixels": int(valid_errs.size),
                        "Spread": params['spread'],
                        "Distance": params['dist'],
                        "Energy": params['energy']
                    }
                    all_stats.append(stats)
                    
                    out_path = self.output_base / s_folder.name / p_folder.name / l_folder.name
                    out_path.mkdir(parents=True, exist_ok=True)
                    self.save_results(out_path, N_Map, errs, mask, W, H, params, stats)

        # Save aggregated report after all folders are processed
        if self.cfg['outputs'].get('save_stats_report', False) and all_stats:
            report_df = pd.DataFrame(all_stats)
            report_path = self.output_base / "recon_report.csv"
            report_df.to_csv(report_path, index=False)
            print(f"\n[SUCCESS] Aggregated report saved to: {report_path}")

    def get_geometry(self, params, w_pix, h_pix):
        """Constructs 3D surface positions and true normal for error calculation[cite: 150, 157]."""
        zenith_rad = np.deg2rad(90.0 - params['elev'])
        az_rad = np.deg2rad(params['azim'])
        nx, ny, nz = np.sin(zenith_rad)*np.cos(az_rad), np.sin(zenith_rad)*np.sin(az_rad), np.cos(zenith_rad)
        
        side = params['area']**0.5
        x_vals = np.linspace(-side/2, side/2, w_pix)
        y_vals = np.linspace(side/2, -side/2, h_pix) 
        xv, yv = np.meshgrid(x_vals, y_vals)
        zv = -(nx * xv + ny * yv) / (nz + 1e-9)
        return np.stack((xv, yv, zv), axis=-1), np.array([nx, ny, nz], dtype=np.float32)

    def solve_normals(self, images, light_vecs, P_surf, TRUE_N, params):
        """Photometric Stereo solver including Emitter Cosine and Spread constraints[cite: 131, 138, 195]."""
        H, W = images[0].shape
        G_list = []
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['reconstruction_settings']['samples_per_axis']
        
        discard_irrelevant = self.cfg['reconstruction_settings'].get('discard_irrelevant_l_vectors', True)
        use_blender_scale = self.cfg['reconstruction_settings'].get('use_blender_energy_scaling', True)
        
        # Energy scaling logic (inverse sine squared) for Blender area lights
        half_spread = np.deg2rad(params['spread'] / 2.0)
        energy_scale = 1.0
        if use_blender_scale and params['spread'] < 180.0:
            energy_scale = 1.0 / (np.sin(half_spread)**2 + 1e-9)

        for l_vec in light_vecs:
            pts = self.get_light_samples(l_vec, params['dist'], l_dims, samp_ax)
            n_A = -l_vec  # Normal of area light points toward origin [cite: 122]
            
            G_acc = np.zeros((H, W, 3), dtype=np.float32)
            
            for pt in pts:
                v = pt - P_surf # Direction from surface to light sample [cite: 134]
                dist_sq = np.sum(v**2, axis=2) + 1e-9
                l_k = v / np.sqrt(dist_sq)[:,:,np.newaxis] 
                
                # Paper Cosine Term: -n_A dot l_k [cite: 131, 138]
                cos_emitter = np.sum(-n_A * l_k, axis=2) 
                
                # Apply spread angle constraint (cone clipping)
                if discard_irrelevant:
                    mask_spread = cos_emitter >= np.cos(half_spread)
                else:
                    mask_spread = np.ones((H, W), dtype=bool)
                
                # (cos_emitter / dist_sq) * l_k
                term = (np.maximum(0, cos_emitter) / dist_sq)[:,:,np.newaxis] * l_k
                G_acc[mask_spread] += term[mask_spread]
                
            # Compute final G-vector per light [cite: 195]
            G_list.append((G_acc * energy_scale) / len(pts))
            
        I_flat = np.stack(images, axis=-1).reshape(-1, len(light_vecs))
        G_flat = np.stack(G_list, axis=2).reshape(-1, len(light_vecs), 3)
        
        # Solve over-determined linear system for scaled normal [cite: 142]
        GTG = np.matmul(G_flat.transpose(0,2,1), G_flat) + 1e-6 * np.eye(3)
        G_pinv = np.matmul(np.linalg.inv(GTG), G_flat.transpose(0,2,1))
        Norms = np.einsum('ijk,ik->ij', G_pinv, I_flat)
        
        Albedo = np.linalg.norm(Norms, axis=1)
        N_Map = (Norms / (Albedo[:, np.newaxis] + 1e-9)).reshape(H, W, 3)
        
        dots = np.sum(N_Map * TRUE_N, axis=2)
        errs = np.degrees(np.arccos(np.clip(dots, -1, 1)))
        return N_Map, Albedo, errs

    def get_light_samples(self, light_vec, distance, dims, num_samples_axis):
        """Discretizes area light into K sampling points[cite: 131]."""
        light_center = light_vec * distance
        light_normal = -light_vec 
        up_world = np.array([0, 0, 1])
        right_local = np.cross(up_world, light_normal) if not np.allclose(np.abs(light_normal), up_world, atol=1e-3) else np.array([1, 0, 0])
        right_local /= (np.linalg.norm(right_local) + 1e-9)
        up_local = np.cross(light_normal, right_local)
        up_local /= (np.linalg.norm(up_local) + 1e-9)
        
        w, h = dims
        step_x, step_y = w / num_samples_axis, h / num_samples_axis
        samples = []
        for i in range(num_samples_axis):
            for j in range(num_samples_axis):
                pos = light_center + (right_local * (-w/2 + step_x*(i+0.5))) + (up_local * (-h/2 + step_y*(j+0.5)))
                samples.append(pos)
        return np.array(samples)

    def save_results(self, out_path, N_Map, errs, mask, W, H, params, stats):
        """Saves individual configuration results."""
        if self.cfg['outputs']['save_per_pixel_csv']:
            y, x = np.indices((H, W))
            df = pd.DataFrame({
                'x': x[mask], 'y': y[mask], 
                'nx': N_Map[mask][:,0], 'ny': N_Map[mask][:,1], 'nz': N_Map[mask][:,2], 
                'err_deg': errs[mask]
            })
            df.to_csv(out_path / "pixel_data.csv", index=False)
        
        if self.cfg['outputs']['save_error_heatmap']:
            heatmap_data = np.zeros((H, W), dtype=np.float32)
            heatmap_data[mask] = errs[mask]
            # Normalize for visualization (0 to 10 degrees)
            vis = (np.clip(heatmap_data, 0, 10)/10*255).astype(np.uint8)
            cv2.imwrite(str(out_path / "error_map.png"), cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO))

if __name__ == "__main__":
    # Ensure this path is correct for your system
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\bat_proc_nrml_cfg.json"
    processor = DirectoryNormalProcessor(config_file)
    processor.process()