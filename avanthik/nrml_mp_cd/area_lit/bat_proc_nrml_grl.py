import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

class GeneralizedNormalProcessor:
    def __init__(self, config_path):
        """Initializes processing environment from JSON configuration."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_image(self, img_path, gamma, bit_depth):
        """Generalized loader: OpenCV for standard/EXR, RawPy for CR2."""
        path_str = str(img_path)
        ext = img_path.suffix.lower()
        
        if not img_path.exists():
            raise FileNotFoundError(f"[Error] Image not found: {path_str}")

        # --- CASE 1: CR2 / Camera Raw ---
        if ext in ['.cr2', '.nef', '.dng']:
            import rawpy
            with rawpy.imread(path_str) as raw:
                # Postprocess: linear=True ensures we get physical light intensity (no gamma applied yet)
                # no_auto_bright=True prevents the library from scaling up the image arbitrarily
                rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
                # Convert RGB to Grayscale (standard weights)
                img = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
                # Normalize 16-bit int to 0.0-1.0 float
                img = img.astype(np.float32) / 65535.0

        # --- CASE 2: EXR / JPG / PNG (OpenCV) ---
        else:
            img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"[Error] Failed to decode image: {path_str}")

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img.dtype == np.float32:
                pass # EXR is already float
            else:
                img = img.astype(np.float32) / (2**bit_depth - 1)
            
            # Gamma Linearization (Only applies to non-RAW formats that have gamma)
            if gamma != 1.0:
                img = np.power(img, gamma)
            
        # Optional Signal-to-Noise Dark Thresholding
        if self.cfg['global_settings'].get('apply_dark_threshold', False):
            thresh = self.cfg['global_settings']['dark_threshold_value']
            img[img < thresh] = 0.0
            
        return img

    def get_light_samples_gpu(self, l_cfg):
        """Generates 3D sample points based on shape and Centroid Rule."""
        center = cp.array(l_cfg['pos_m'], dtype=cp.float32)
        normal = cp.array(l_cfg['norm_dir'], dtype=cp.float32)
        normal /= cp.linalg.norm(normal)
        
        # Define local coordinate frame (Basis vectors)
        up = cp.array([0, 0, 1], dtype=cp.float32)
        if cp.abs(cp.dot(up, normal)) > 0.99:
            up = cp.array([1, 0, 0], dtype=cp.float32)
        right = cp.cross(up, normal)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(normal, right)

        if l_cfg['shape'] == "RECT":
            w, h = l_cfg['dims_m']
            nx, ny = l_cfg['sampling']
            off_x, off_y = 1.0/(2*nx), 1.0/(2*ny)
            sx = cp.linspace(-0.5 + off_x, 0.5 - off_x, nx) * w
            sy = cp.linspace(-0.5 + off_y, 0.5 - off_y, ny) * h
            ii, jj = cp.meshgrid(sx, sy)
            samples = center + (right[None,None,:] * ii[:,:,None]) + (up_loc[None,None,:] * jj[:,:,None])
            
        elif l_cfg['shape'] == "DISK":
            r_max = l_cfg['radius_m']
            n_r, n_theta = l_cfg['sampling']
            r_steps = cp.linspace(0, r_max, n_r + 1)
            theta_steps = cp.linspace(0, 2*cp.pi, n_theta + 1)
            
            sample_list = []
            for i in range(n_r):
                for j in range(n_theta):
                    r1, r2 = r_steps[i], r_steps[i+1]
                    t1, t2 = theta_steps[j], theta_steps[j+1]
                    # Polar Area Centroid Formulas
                    rc = (2/3) * (r2**3 - r1**3) / (r2**2 - r1**2 + 1e-9)
                    tc = (t1 + t2) / 2.0
                    x_loc, y_loc = rc * cp.cos(tc), rc * cp.sin(tc)
                    sample_list.append(center + (x_loc * right) + (y_loc * up_loc))
            samples = cp.stack(sample_list)

        return samples.reshape(-1, 3)

    def compute_ground_truth(self):
        """Calculates Ground Truth Normal from Elevation and Azimuth."""
        elev = self.cfg['global_settings']['ground_truth_elev_deg']
        azim = self.cfg['global_settings']['ground_truth_azim_deg']
        zenith = np.deg2rad(90.0 - elev)
        az = np.deg2rad(azim)
        nx = np.sin(zenith) * np.cos(az)
        ny = np.sin(zenith) * np.sin(az)
        nz = np.cos(zenith)
        return cp.array([nx, ny, nz], dtype=cp.float32)

    def process(self):
        # 1. Load Valid Pixel Geometry
        df = pd.read_csv(self.cfg['paths']['world_coordinate_csv'])
        df = df[df['alpha'] > self.cfg['global_settings']['alpha_threshold']]
        
        u_coords, v_coords = df['pixel_u'].values.astype(int), df['pixel_v'].values.astype(int)
        P_surf = cp.array(df[['x_world', 'y_world', 'z_world']].values, dtype=cp.float32)

        intensities, G_stack = [], []
        
        # 2. Construct Intensity Matrix and Geometric Matrix
        for l_cfg in self.cfg['lights']:
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            img = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
            intensities.append(cp.array(img[v_coords, u_coords]))
            
            l_samples = self.get_light_samples_gpu(l_cfg)
            G_accum = cp.zeros((P_surf.shape[0], 3), dtype=cp.float32)
            n_A = -cp.array(l_cfg['norm_dir'], dtype=cp.float32)
            cos_half_spread = np.cos(np.deg2rad(l_cfg['spread_deg']/2))

            for pt in l_samples:
                v = pt - P_surf
                dist_sq = cp.sum(v**2, axis=1) + 1e-9
                l_k = v / cp.sqrt(dist_sq)[:, None]
                cos_emit = cp.sum(n_A * l_k, axis=1)
                mask = cos_emit >= cos_half_spread
                # Accumulate geometric contribution (1/r^2 falloff)
                G_accum += (cp.maximum(0, cos_emit) / dist_sq)[:, None] * l_k * mask[:, None]
            
            G_stack.append(G_accum / l_samples.shape[0])

        # 3. Solver: Compute Unit Normals
        I = cp.stack(intensities, axis=1)[:, :, None]
        G = cp.stack(G_stack, axis=1)
        GT = G.transpose(0, 2, 1)
        GTG_inv = cp.linalg.inv(cp.matmul(GT, G) + cp.eye(3)*1e-6)
        N_raw = cp.matmul(GTG_inv, cp.matmul(GT, I)).squeeze(2)
        Normals = N_raw / (cp.linalg.norm(N_raw, axis=1)[:, None] + 1e-9)
        
        # 4. Error Calculation
        true_n = self.compute_ground_truth()
        dot = cp.sum(Normals * true_n, axis=1)
        dot = cp.clip(dot, -1.0, 1.0)
        angular_errors = cp.degrees(cp.arccos(dot))
        
        # 5. Export Normal Maps and Error Heatmap
        self.generate_outputs(Normals, angular_errors, u_coords, v_coords)

    def generate_outputs(self, normals, errors, u, v):
        H, W = self.cfg['resolution']['height'], self.cfg['resolution']['width']
        
        # Normal Map (Float TIFF)
        n_map = np.zeros((H, W, 3), dtype=np.float32)
        n_map[v, u] = cp.asnumpy(normals)
        imageio.imwrite(self.output_dir / "normal_map.tif", n_map)
        
        # Angular Error Heatmap
        err_map = np.zeros((H, W), dtype=np.float32)
        err_map[v, u] = cp.asnumpy(errors)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(err_map, cmap='inferno')
        plt.colorbar(label='Degrees Error')
        plt.title(f"Mean Angular Error: {np.mean(cp.asnumpy(errors)):.4f}°")
        plt.axis('off')
        plt.savefig(self.output_dir / "error_heatmap.png", bbox_inches='tight')
        plt.close()

        # Visual Normal Map (RGB shifted)
        vis_map = (n_map + 1.0) / 2.0
        plt.imsave(self.output_dir / "normal_vis.png", np.clip(vis_map, 0, 1))
        
        print(f"Normal Reconstruction Complete. Mean Error: {np.mean(cp.asnumpy(errors)):.4f} deg")

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\bat_proc_nrml_grl_cfg.json"
    GeneralizedNormalProcessor(config_file).process()