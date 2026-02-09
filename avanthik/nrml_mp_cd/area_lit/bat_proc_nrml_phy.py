import cupy as cp  # GPU Acceleration
import numpy as np
import cv2
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

class PhysicalGPUNormalProcessor:
    def __init__(self, config_path):
        """Initializes paths and physics settings from JSON."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_dir = Path(self.cfg['paths']['input_images_dir'])
        self.output_base = Path(self.cfg['paths']['output_dir'])
        self.light_yaml_path = Path(self.cfg['paths']['light_yaml_path'])
        
        # Camera & Resolution
        self.res_w = self.cfg['camera']['resolution'][0]
        self.res_h = self.cfg['camera']['resolution'][1]
        
        # Load Light Data
        with open(self.light_yaml_path, 'r') as f:
            self.light_data = yaml.safe_load(f)
            
        print(f"Loaded configuration for {len(self.light_data)} lights.")
        self.output_base.mkdir(parents=True, exist_ok=True)

    def process(self):
        print(f"="*60)
        print(f"Starting Physical Normal Reconstruction")
        print(f"Resolution: {self.res_w}x{self.res_h}")
        print(f"="*60)

        # 1. Compute Geometry (P_surf) on GPU
        print("Computing Ray-Plane Intersection on GPU...")
        P_surf_gpu, valid_mask_gpu, true_n_gpu = self.compute_geometry_gpu()
        
        # 2. Load and Preprocess Images
        I_gpu = self.load_images_aligned_gpu()
        
        # Validation
        if I_gpu.shape[2] != len(self.light_data):
            raise ValueError(f"Error: Loaded {I_gpu.shape[2]} image channels but YAML has {len(self.light_data)} lights.")

        # Filter invalid pixels
        print("Creating active pixel mask...")
        img_intensity = cp.mean(I_gpu, axis=2) 
        
        thresh_val = self.cfg['image_processing']['mask_threshold'] / 255.0
        active_mask = (img_intensity > thresh_val) & valid_mask_gpu
        
        active_indices = cp.where(active_mask.flatten())[0]
        
        if len(active_indices) == 0:
            raise ValueError("No valid pixels found! Check mask_threshold or camera/plane geometry.")

        P_surf_flat = P_surf_gpu.reshape(-1, 3)[active_indices]
        I_flat = I_gpu.reshape(-1, I_gpu.shape[2])[active_indices] 
        
        print(f"Solving for {len(active_indices)} valid pixels...")

        # 3. Solve Normals
        n_map_flat, errs_flat = self.solve_normals_gpu(
            I_flat, P_surf_flat, true_n_gpu, active_indices
        )

        # 4. Reconstruct and Save
        print("Saving Outputs...")
        self.save_outputs(n_map_flat, errs_flat, active_mask, true_n_gpu)
        
        # Cleanup
        del I_gpu, P_surf_gpu, P_surf_flat, I_flat, n_map_flat
        cp.get_default_memory_pool().free_all_blocks()
        print("Done.")

    def compute_geometry_gpu(self):
        """Generates 3D surface points (x,y,z) for every pixel on GPU."""
        cam_cfg = self.cfg['camera']
        f_mm = cam_cfg['focal_length_mm']
        sw_mm, sh_mm = cam_cfg['sensor_size_mm']
        
        u = cp.linspace(0, self.res_w - 1, self.res_w)
        v = cp.linspace(0, self.res_h - 1, self.res_h)
        uu, vv = cp.meshgrid(u, v)
        
        cx = self.res_w / 2.0
        cy = self.res_h / 2.0
        
        fx = f_mm * (self.res_w / sw_mm)
        fy = f_mm * (self.res_h / sh_mm)
        
        x_norm = (uu - cx) / fx
        y_norm = -(vv - cy) / fy 
        
        rays_cam = cp.stack([x_norm, y_norm, -cp.ones_like(x_norm)], axis=-1)
        rays_norm = cp.linalg.norm(rays_cam, axis=2, keepdims=True)
        rays_cam /= rays_norm
        
        cam_rot = cp.array(self._euler_to_matrix(cam_cfg['rotation_euler_deg']), dtype=cp.float32)
        cam_pos = cp.array(cam_cfg['position_cm'], dtype=cp.float32)
        
        rays_world = cp.tensordot(rays_cam, cam_rot.T, axes=1)
        
        plane_cfg = self.cfg['plane']
        plane_center = cp.array(plane_cfg['center_cm'], dtype=cp.float32)
        plane_rot = cp.array(self._euler_to_matrix(plane_cfg['rotation_euler_deg']), dtype=cp.float32)
        
        plane_n = plane_rot @ cp.array([0, 0, 1], dtype=cp.float32)
        
        numer = cp.dot(plane_center - cam_pos, plane_n)
        denom = cp.sum(rays_world * plane_n, axis=2)
        
        valid_mask = (cp.abs(denom) > 1e-6)
        
        t = numer / (denom + 1e-9)
        valid_mask = valid_mask & (t > 0)
        
        P_surf = cam_pos + rays_world * t[:, :, None]
        
        return P_surf, valid_mask, plane_n

    def load_images_aligned_gpu(self):
        """Loads images corresponding strictly to the order in YAML."""
        ext = self.cfg['image_processing']['file_extension']
        images = []
        gamma = self.cfg['image_processing']['gamma_value']
        do_gamma = self.cfg['image_processing']['apply_gamma_correction']
        
        print(f"Loading images matched to YAML...")
        
        for idx, light_conf in enumerate(self.light_data):
            lid = light_conf['light_id']
            fname = f"light_{int(lid):03d}{ext}"
            fpath = self.input_dir / fname
            
            if not fpath.exists():
                raise FileNotFoundError(f"Missing image for Light ID {lid}: {fpath}")
            
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None: 
                raise ValueError(f"Failed to read image: {fpath}")
                
            if img.shape[0] != self.res_h or img.shape[1] != self.res_w:
                print(f"[WARN] Resizing {fname} from {img.shape} to ({self.res_h}, {self.res_w})")
                img = cv2.resize(img, (self.res_w, self.res_h))

            img_float = img.astype(np.float32) / 255.0
            
            if do_gamma:
                img_float = np.power(img_float, gamma)
                
            images.append(img_float)
            print(f"  [{idx+1}/{len(self.light_data)}] Loaded {fname}")
            
        I_stack = np.stack(images, axis=-1)
        return cp.array(I_stack, dtype=cp.float32)

    def solve_normals_gpu(self, I_flat, P_surf_flat, true_n, indices):
        """Builds G matrix and solves for Normals."""
        num_pixels = P_surf_flat.shape[0]
        
        G_stack = []
        
        l_shape = self.cfg['light_settings']['shape']
        l_dims = self.cfg['light_settings']['dimensions_cm']
        n_samp = self.cfg['light_settings']['samples_per_axis']
        
        spread_deg = self.cfg['light_settings'].get('spread_angle_deg', 180.0)
        half_spread_rad = np.deg2rad(spread_deg / 2.0)
        min_cos = np.cos(half_spread_rad)
        do_clip = self.cfg['reconstruction_settings']['discard_irrelevant_l_vectors']
        
        for i, l_conf in enumerate(self.light_data):
            pos = cp.array(l_conf['position_cm'], dtype=cp.float32)
            rot = l_conf.get('angles_deg', {'roll':0, 'pitch':0, 'yaw':0})
            if isinstance(rot, dict):
                rot_list = [rot.get('roll',0), rot.get('pitch',0), rot.get('yaw',0)]
            else:
                rot_list = rot 
            
            rot_mat = cp.array(self._euler_to_matrix(rot_list), dtype=cp.float32)
            direction = rot_mat @ cp.array([0, 0, -1], dtype=cp.float32)
            
            power = float(l_conf.get('radiant_power_w', 1.0))
            
            samples = self.get_light_samples_gpu(pos, rot_mat, l_shape, l_dims, n_samp)
            
            G_light = cp.zeros((num_pixels, 3), dtype=cp.float32)
            
            # Optimization: If samples count is low, loop is fine. 
            for k in range(samples.shape[0]):
                s = samples[k]
                v = s - P_surf_flat 
                dist_sq = cp.sum(v**2, axis=1) + 1e-9
                l_k = v / cp.sqrt(dist_sq)[:, None] 
                
                cos_emitter = cp.sum(direction * (-l_k), axis=1)
                
                geo_term = cp.maximum(0, cos_emitter) / dist_sq
                contrib = geo_term[:, None] * l_k
                
                if do_clip:
                    valid_cone = (cos_emitter >= min_cos)
                    contrib *= valid_cone[:, None]
                
                G_light += contrib
            
            # Weighting: Total Power distributed among active samples
            # This ensures energy conservation even if some samples are masked
            if samples.shape[0] > 0:
                area = self._get_area(l_shape, l_dims)
                weight = (area / samples.shape[0]) * power
                G_stack.append(G_light * weight)
            else:
                # Should not happen with centroid logic, but safe fallback
                G_stack.append(cp.zeros_like(G_light))
            
        G = cp.stack(G_stack, axis=1)
        
        I_col = I_flat[:, :, None]
        GT = G.transpose(0, 2, 1)
        GTG = cp.matmul(GT, G)
        
        reg = self.cfg['reconstruction_settings']['regularization']
        GTG += cp.eye(3, dtype=cp.float32) * reg
        GTG_inv = cp.linalg.inv(GTG)
        GTI = cp.matmul(GT, I_col)
        N_raw = cp.matmul(GTG_inv, GTI).squeeze(2)
        
        norm_mag = cp.linalg.norm(N_raw, axis=1, keepdims=True)
        N_final = N_raw / (norm_mag + 1e-9)
        
        dot = cp.sum(N_final * true_n, axis=1)
        dot = cp.clip(dot, -1.0, 1.0)
        errs = cp.degrees(cp.arccos(dot))
        
        return N_final, errs

    def get_light_samples_gpu(self, pos, rot_mat, shape, dims, n_samp):
        """
        Generates discretized light samples using Centroid/Midpoint Rule.
        Instead of sampling edges (-w/2, w/2), we sample the center of each grid cell.
        """
        right = rot_mat[:, 0]
        up = rot_mat[:, 1]
        
        w, h = dims[0], dims[1]
        
        # Calculate step size (size of one grid cell)
        step_x = w / n_samp
        step_y = h / n_samp
        
        # Generate points at the center of each cell
        # Start: -w/2 + step/2
        # End:    w/2 - step/2
        x = cp.linspace(-w/2 + step_x/2, w/2 - step_x/2, n_samp)
        y = cp.linspace(-h/2 + step_y/2, h/2 - step_y/2, n_samp)
        
        xx, yy = cp.meshgrid(x, y)
        xx, yy = xx.flatten(), yy.flatten()

        if shape == 'circle':
            # Diameter is stored in dims[0]
            r = w / 2.0
            mask = (xx**2 + yy**2) <= (r**2)
            xx, yy = xx[mask], yy[mask]
            
            # Safety: If grid is too coarse relative to circle, center should survive.
            # But with centroid logic, n=1 gives (0,0) which is valid.
            if xx.size == 0:
                 xx = cp.array([0.0], dtype=cp.float32)
                 yy = cp.array([0.0], dtype=cp.float32)
            
        offsets = (right[None, :] * xx[:, None]) + (up[None, :] * yy[:, None])
        samples = pos + offsets
        return samples

    def _get_area(self, shape, dims):
        if shape == 'rectangle': return dims[0] * dims[1]
        elif shape == 'circle': return np.pi * (dims[0]/2.0)**2
        return 1.0

    def _euler_to_matrix(self, angles):
        rx, ry, rz = np.radians(angles)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        return Rz @ Ry @ Rx

    def save_outputs(self, n_flat, err_flat, mask, true_n):
        H, W = self.res_h, self.res_w
        cpu_mask = cp.asnumpy(mask)
        coords = np.where(cpu_mask)
        
        n_map = np.zeros((H, W, 3), dtype=np.float32)
        n_cpu = cp.asnumpy(n_flat)
        n_map[coords] = n_cpu
        
        e_map = np.zeros((H, W), dtype=np.float32)
        e_cpu = cp.asnumpy(err_flat)
        e_map[coords] = e_cpu
        
        if self.cfg['outputs']['save_visualizations']:
            vis_n = (n_map + 1.0) / 2.0
            vis_n = np.clip(vis_n, 0, 1)
            vis_n[~cpu_mask] = 0
            plt.imsave(self.output_base / "normal_viz.png", vis_n)
            
            if self.cfg['outputs']['save_error_heatmap']:
                plt.figure(figsize=(10,8))
                plt.imshow(e_map, cmap='inferno', vmin=0, vmax=np.percentile(e_cpu, 98))
                plt.colorbar(label="Angular Error (Deg)")
                plt.title(f"Mean Error: {np.mean(e_cpu):.2f} Deg")
                plt.savefig(self.output_base / "error_heatmap.png")
                plt.close()

        if self.cfg['outputs']['save_per_pixel_csv']:
            df = pd.DataFrame({
                'u': coords[1], 'v': coords[0],
                'nx': n_cpu[:, 0], 'ny': n_cpu[:, 1], 'nz': n_cpu[:, 2],
                'err_deg': e_cpu
            })
            df.to_csv(self.output_base / "pixel_normals.csv", index=False)
            
        print(f"Results saved to {self.output_base}")

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\bat_proc_nrml_phy_cfg.json"
    PhysicalGPUNormalProcessor(CONFIG_PATH).process()