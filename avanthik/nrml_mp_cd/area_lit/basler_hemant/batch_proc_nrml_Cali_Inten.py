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
        self.verify_dir = self.output_dir / "verification"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verify_dir.mkdir(parents=True, exist_ok=True)

        self.mask_path = Path(self.cfg['paths']['input_mask_path'])
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask not found at: {self.mask_path}")
        
    def load_image(self, img_path, gamma, bit_depth):
        path_str = str(img_path)
        ext = img_path.suffix.lower()
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {path_str}")

        # --- CASE 1: Headerless RAW (.raw) ---
        if ext == '.raw':
            w = self.cfg['resolution']['width']
            h = self.cfg['resolution']['height']
            file_size = os.path.getsize(path_str)
            size_rgb_8bit = w * h * 3 
            size_gray_16bit = w * h * 2
            size_gray_8bit = w * h

            if file_size == size_rgb_8bit:
                raw_data = np.fromfile(path_str, dtype=np.uint8)
                img_rgb = raw_data.reshape((h, w, 3)).astype(np.float32) / 255.0
                img = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
            elif file_size == size_gray_16bit:
                img = np.fromfile(path_str, dtype=np.uint16).reshape((h, w))
                img = img.astype(np.float32) / 65535.0
            elif file_size == size_gray_8bit:
                img = np.fromfile(path_str, dtype=np.uint8).reshape((h, w))
                img = img.astype(np.float32) / 255.0
            else:
                raise ValueError(f"File {path_str} size {file_size} does not match resolution {w}x{h}")

        # --- CASE 2: Camera RAW Formats ---
        elif ext in ['.cr2', '.nef', '.dng']:
            import rawpy
            with rawpy.imread(path_str) as raw:
                # user_flip=0 ensures we ignore rotation tags and get raw sensor data
                rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True, user_flip=0)
                img = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
                img = img.astype(np.float32) / 65535.0

        # --- CASE 3: Standard Image Formats ---
        else:
            img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if img is None: raise ValueError(f"Failed to decode: {path_str}")
            if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Auto-detect depth for PNG/TIFF
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            
            if gamma != 1.0: img = np.power(img, gamma)
            
        if self.cfg['global_settings'].get('apply_dark_threshold', False):
            img[img < self.cfg['global_settings']['dark_threshold_value']] = 0.0
            
        return img
    
    def get_light_samples_gpu(self, l_cfg):
        center = cp.array(l_cfg['pos_m'], dtype=cp.float32)
        normal = cp.array(l_cfg['norm_dir'], dtype=cp.float32)
        normal /= cp.linalg.norm(normal)
        
        up = cp.array([0, 0, 1], dtype=cp.float32)
        if cp.abs(cp.dot(up, normal)) > 0.99: up = cp.array([1, 0, 0], dtype=cp.float32)
        right = cp.cross(up, normal); right /= cp.linalg.norm(right)
        up_loc = cp.cross(normal, right)

        if l_cfg['shape'] == "RECT":
            w, h = l_cfg['dims_m']
            nx, ny = l_cfg['sampling']
            sx = cp.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx) * w 
            sy = cp.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny) * h
            ii, jj = cp.meshgrid(sx, sy)
            samples = center + (right[None,None,:] * ii[:,:,None]) + (up_loc[None,None,:] * jj[:,:,None])
        elif l_cfg['shape'] == "DISK":
            r_max = l_cfg['radius_m']; n_r, n_theta = l_cfg['sampling']
            r_steps = cp.linspace(0, r_max, n_r + 1); theta_steps = cp.linspace(0, 2*cp.pi, n_theta + 1)
            sl = []
            for i in range(n_r):
                for j in range(n_theta):
                    r1, r2 = r_steps[i], r_steps[i+1]; t1, t2 = theta_steps[j], theta_steps[j+1]
                    rc = (2/3) * (r2**3 - r1**3) / (r2**2 - r1**2 + 1e-9); tc = (t1 + t2) / 2.0
                    sl.append(center + (rc * cp.cos(tc) * right) + (rc * cp.sin(tc) * up_loc))
            samples = cp.stack(sl)
        return samples.reshape(-1, 3)

    def detect_plane_corners(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]      # TL
        ordered[2] = pts[np.argmax(s)]      # BR
        ordered[1] = pts[np.argmin(diff)]   # TR
        ordered[3] = pts[np.argmax(diff)]   # BL
        return ordered

    def compute_geometry_with_offset(self, corners, mask, h, w):
        plane_cfg = self.cfg['plane']
        cam_cfg = self.cfg['camera']
        
        # --- FIXED: Use explicit Center Position (Z=2.7cm) ---
        center_pos = np.array(plane_cfg.get('center_position_cm', [0.0, 0.0, 0.0]))
        if self.cfg['global_settings']['system_units'] == 'm':
            center_pos = center_pos * 0.01

        center_px = np.array([w/2, h/2]) if cam_cfg.get('use_auto_center', True) else np.array(cam_cfg['manual_center_pixel'])
        scale = 0.01 if self.cfg['global_settings']['system_units'] == 'm' else 1.0
        w_real, h_real = plane_cfg['dimensions_cm'][0] * scale, plane_cfg['dimensions_cm'][1] * scale
        
        # Define Object Corners in 2D Local Space (centered at 0,0)
        obj_corners_2d = np.array([[-w_real/2, -h_real/2], [w_real/2, -h_real/2], [w_real/2, h_real/2], [-w_real/2, h_real/2]], dtype=np.float32)
        
        # Homography: Maps Pixel -> Local 2D World
        H, _ = cv2.findHomography(corners, obj_corners_2d)
        
        # Calculate Offset correction to align Camera Center
        center_h = (H @ np.array([center_px[0], center_px[1], 1.0]))
        offset_2d = center_h[:2] / center_h[2]

        # Generate Full Pixel Grid
        y_grid, x_grid = np.indices((h, w))
        ones = np.ones_like(x_grid.flatten())
        coords_homo = np.stack([x_grid.flatten(), y_grid.flatten(), ones]) 
        mapped_homo = H @ coords_homo
        p_world_2d = (mapped_homo[:2] / mapped_homo[2]).T
        
        # Align Grid so Camera Axis hits (0,0) in 2D space
        p_world_2d -= offset_2d 
        p_world_3d = np.column_stack([p_world_2d, np.zeros(len(p_world_2d))])

        # Apply Rotations (Elevation/Azimuth)
        z_rad = np.deg2rad(90 - plane_cfg['elevation_deg'])
        a_rad = np.deg2rad(plane_cfg['azimuth_deg'])
        Rz = np.array([[np.cos(a_rad), -np.sin(a_rad), 0], [np.sin(a_rad), np.cos(a_rad), 0], [0,0,1]])
        Ry = np.array([[np.cos(z_rad), 0, np.sin(z_rad)], [0, 1, 0], [-np.sin(z_rad), 0, np.cos(z_rad)]])
        R = Rz @ Ry
        
        p_world_final = p_world_3d @ R.T
        
        # --- SHIFT: Move Surface up to Z=2.7cm ---
        p_world_final += center_pos 

        P_map = p_world_final.reshape(h, w, 3).astype(np.float32)
        true_n = cp.array(np.array([0, 0, 1.0]) @ R.T, dtype=cp.float32)

        vis_data = {'corners': corners, 'cam_origin': center_px, 'plane_center_proj': center_px} # Simplified vis
        return cp.array(P_map), true_n, vis_data

    def process(self):
        print("Starting Processing with Auto-Calibration...")
        
        mask_cpu = cv2.imread(str(self.mask_path), 0)
        h_ref, w_ref = mask_cpu.shape
        print(f"Reference: {w_ref}x{h_ref}")

        corners = self.detect_plane_corners(mask_cpu)
        P_full_gpu, true_n, vis_data = self.compute_geometry_with_offset(corners, mask_cpu, h_ref, w_ref)
        
        df_coords = pd.read_csv(self.cfg['paths']['world_coordinate_csv'])
        df_coords = df_coords[df_coords['alpha'] > self.cfg['global_settings']['alpha_threshold']]
        
        u_coords = df_coords['pixel_u'].values.astype(int)
        v_coords = df_coords['pixel_v'].values.astype(int)
        u_coords = np.clip(u_coords, 0, w_ref - 1)
        v_coords = np.clip(v_coords, 0, h_ref - 1)

        P_surf = P_full_gpu[v_coords, u_coords]

        # --- COORDINATE FIX: Image Y vs World Y ---
        # Image Y is DOWN. World Y is UP. 
        # If Light coords in JSON are standard (Y+ is "Up/Left" on table), we flip P_surf Y.
        P_surf[:, 0] *= 1.0
        P_surf[:, 1] *= -1.0 

        intensities, G_stack = [], []
        
        # 3. Load Images & Calculate Physics
        print("Loading images and calculating light physics...")
        for light_idx, l_cfg in enumerate(self.cfg['lights']):
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            img = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
            
            # Check Orientation/Size
            h_img, w_img = img.shape
            if (h_img != h_ref) or (w_img != w_ref):
                if (h_img == w_ref) and (w_img == h_ref):
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                else:
                    img = cv2.resize(img, (w_ref, h_ref))

            intensities.append(cp.array(img[v_coords, u_coords]))
            
            # Physics Calculation
            l_samples = self.get_light_samples_gpu(l_cfg)
            G_accum = cp.zeros((P_surf.shape[0], 3), dtype=cp.float32)
            n_A = cp.array(l_cfg['norm_dir'], dtype=cp.float32)
            cos_half_spread = np.cos(np.deg2rad(l_cfg['spread_deg']/2))

            for pt in l_samples:
                v = pt - P_surf; dist_sq = cp.sum(v**2, axis=1) + 1e-9; l_k = v / cp.sqrt(dist_sq)[:, None] 
                cos_emit = -cp.sum(n_A * l_k, axis=1) # Dot product for emission
                
                # Check if emission is negative (back of light)
                # valid_emit = (cos_emit > 0)
                
                G_accum += (cp.maximum(0, cos_emit) / dist_sq)[:, None] * l_k
                
            G_stack.append(G_accum / l_samples.shape[0])
        
        # --- 4. AUTO-CALIBRATION STEP ---
        # We compute a scalar to bridge the gap between Theoretical G and Real I
        print("\nPer-Light Auto-Calibration:")
        scaling_factors = []
        
        for i in range(len(G_stack)):
            I_real = intensities[i]
            # Since board is flat, Normal is [0,0,1]. Expected Brightness = G_z
            I_theo = G_stack[i][:, 2] 
            
            # Robust Mean: Ignore 0s and very bright spots
            valid_mask = (I_real > 0.05) & (I_real < 0.95)
            if cp.sum(valid_mask) > 100:
                mean_real = cp.mean(I_real[valid_mask])
                mean_theo = cp.mean(I_theo[valid_mask])
                scale = mean_real / mean_theo
            else:
                scale = 1.0 # Fallback
            
            scaling_factors.append(scale)
            print(f"  Light {i+1} ({self.cfg['lights'][i]['file_name']}): Scaling G by {scale:.6f}")
            
            # Apply Scale
            G_stack[i] *= scale

        # --- 5. Verify Calibration (Light 1) ---
        if len(G_stack) > 0:
            idx = 0
            I_r = intensities[idx]
            I_p = G_stack[idx][:, 2] # Predicted intensity on flat surface
            
            ratio = I_r / (I_p + 1e-6)
            ratio_img = np.zeros((h_ref, w_ref), dtype=np.float32)
            ratio_img[v_coords, u_coords] = cp.asnumpy(ratio)
            
            # Plot Gradient Profile
            plt.figure(figsize=(10, 5))
            plt.subplot(1,2,1)
            plt.imshow(ratio_img, cmap='RdBu', vmin=0.5, vmax=1.5)
            plt.colorbar(label='Ratio (Real / Pred)')
            plt.title(f'Calibrated Ratio Map - Light {idx+1}')
            
            plt.subplot(1,2,2)
            mid_row = ratio_img[h_ref//2, :]
            plt.plot(mid_row)
            plt.title('Horizontal Profile (Center Row)')
            plt.ylim(0, 2)
            plt.savefig(self.output_dir / "calibration_check.png")
            plt.close()

        # 6. Solve Normals
        I = cp.stack(intensities, axis=1)[:, :, None]
        G = cp.stack(G_stack, axis=1)
        GT = G.transpose(0, 2, 1)
        
        GTG_inv = cp.linalg.inv(cp.matmul(GT, G) + cp.eye(3)*1e-2) # Slightly higher reg for stability
        
        N_raw = cp.matmul(GTG_inv, cp.matmul(GT, I)).squeeze(2)
        Normals = N_raw / (cp.linalg.norm(N_raw, axis=1)[:, None] + 1e-9)
        
        dot = cp.sum(Normals * true_n, axis=1)
        angular_errors = cp.degrees(cp.arccos(cp.clip(dot, -1.0, 1.0)))
        
        self.export_to_csv(u_coords, v_coords, Normals, angular_errors)
        self.generate_outputs(Normals, angular_errors, u_coords, v_coords, h_ref, w_ref)

    def export_to_csv(self, u, v, normals, errors):
        print("Exporting results...")
        n_cpu = cp.asnumpy(normals)
        err_cpu = cp.asnumpy(errors)
        results_df = pd.DataFrame({
            'pixel_u': u, 'pixel_v': v,
            'normal_x': n_cpu[:, 0], 'normal_y': n_cpu[:, 1], 'normal_z': n_cpu[:, 2],
            'angular_error_deg': err_cpu
        })
        results_df.to_csv(self.output_dir / "pixelwise_normals.csv", index=False)

    def generate_outputs(self, normals, errors, u, v, h, w):
        err_cpu = cp.asnumpy(errors)
        stats = {k: float(v) for k, v in {"Mean": np.mean(err_cpu), "Median": np.median(err_cpu), "RMSE": np.sqrt(np.mean(err_cpu**2))}.items()}

        print("\n" + "="*40 + "\n CALIBRATED STATISTICS\n" + "="*40)
        for k, val in stats.items(): print(f" {k:<10}: {val:.4f} degrees")
        
        n_map = np.zeros((h, w, 3), dtype=np.float32); n_map[v, u] = cp.asnumpy(normals)
        vis_map = (n_map + 1.0) / 2.0; vis_map[vis_map == 0.5] = 0
        plt.imsave(self.output_dir / "normal_vis.png", np.clip(vis_map, 0, 1))
        
        err_map = np.full((h, w), np.nan, dtype=np.float32); err_map[v, u] = err_cpu
        plt.figure(figsize=(10, 8)); plt.cm.inferno.set_bad(color='black')
        plt.imshow(err_map, cmap='inferno', vmin=0, vmax=20)
        plt.colorbar(label='Error (deg)'); plt.title(f"Calibrated Error (Mean: {stats['Mean']:.2f})")
        plt.savefig(self.output_dir / "error_heatmap.png")
        print(f"Done. Outputs in {self.output_dir}")

if __name__ == "__main__":
    import sys
    default_cfg = r"C:\Users\chand\OneDrive\Desktop\cropped_png_jpg_5th_feb\upd_bat_proc_nrml_grl_cfg.json"
    config_file = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    if os.path.exists(config_file): GeneralizedNormalProcessor(config_file).process()
    else: print(f"Config file not found: {config_file}")