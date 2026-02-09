import os
# CRITICAL: Enable OpenEXR support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

class PhotometricStereoFullReconstructor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.input_dir = Path(self.cfg['paths']['input_images_dir'])
        self.mask_path = Path(self.cfg['paths']['input_mask_path'])
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.light_yaml_path = Path(self.cfg['paths']['light_yaml_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.light_yaml_path, 'r') as f:
            self.light_data = yaml.safe_load(f)

    def load_images_gpu(self, target_h, target_w):
        images = []
        ext = self.cfg['image_processing']['file_extension'].upper()
        for l_conf in self.light_data:
            fname = f"light_{int(l_conf['light_id']):03d}{ext}"
            fpath = self.input_dir / fname
            if not fpath.exists(): 
                fpath = self.input_dir / fname.replace(ext, ext.lower())

            if ext in ['.CR2', '.NEF']:
                if not RAW_SUPPORT: raise ImportError("rawpy not installed")
                with rawpy.imread(str(fpath)) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=True).astype(np.float32)/65535.0
                    img = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
                    # Check for 90 degree sensor tilt
                    if (img.shape[0] > img.shape[1] and target_w > target_h) or \
                       (img.shape[1] > img.shape[0] and target_h > target_w):
                        img = np.rot90(img, k=-1)
            elif ext == '.EXR':
                # OpenEXR is now enabled via the environment flag above
                img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED).astype(np.float32)
                if len(img.shape) == 3: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.imread(str(fpath)).astype(np.float32)/255.0
                if self.cfg['image_processing']['apply_gamma_correction_if_jpg']:
                    img = np.power(img, self.cfg['image_processing']['gamma_value'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, (target_w, target_h))
            images.append(cp.array(img))
        return cp.stack(images, axis=-1)

    def process(self):
        mask_cpu = cv2.imread(str(self.mask_path), 0)
        if mask_cpu is None: raise FileNotFoundError(f"Mask not found at {self.mask_path}")
        h, w = mask_cpu.shape
        
        # Calculate origin (Boresight)
        if self.cfg['camera']['use_auto_center']:
            center_px = [w/2, h/2]
        else:
            center_px = self.cfg['camera']['manual_center_pixel']

        I_gpu = self.load_images_gpu(h, w)
        corners = self.detect_plane_corners(mask_cpu)
        
        # Geometry with enhanced coordinate tracking
        P_surf_gpu, valid_geo_gpu, true_n_gpu, global_corners, plane_center_global = \
            self.compute_geometry_with_offset(corners, mask_cpu, h, w, center_px)
        
        active = cp.where(valid_geo_gpu.flatten())[0]
        I_flat = I_gpu.reshape(-1, len(self.light_data))[active]
        P_flat = P_surf_gpu.reshape(-1, 3)[active]
        
        means = cp.mean(I_flat, axis=0)
        I_calib = I_flat * (cp.mean(means) / (means + 1e-9))
        
        N_flat, errs_flat = self.solve_normals_gpu(I_calib, P_flat, true_n_gpu)
        
        # Output Visualizations and CSV
        self.visualize_coordinate_offset(global_corners, plane_center_global)
        self.save_results(N_flat, errs_flat, valid_geo_gpu, h, w)

    def compute_geometry_with_offset(self, corners, mask, h, w, center_px):
        plane_cfg = self.cfg['plane']
        w_cm, h_cm = plane_cfg['dimensions_cm']
        # Object frame centered at (0,0) locally
        obj_corners_2d = np.array([[-w_cm/2, -h_cm/2], [w_cm/2, -h_cm/2], [w_cm/2, h_cm/2], [-w_cm/2, h_cm/2]], dtype=np.float32)
        H, _ = cv2.findHomography(corners, obj_corners_2d)
        
        # Offset: Camera Z-axis intersection with plane
        offset_h = (H @ np.array([center_px[0], center_px[1], 1.0]))
        offset_2d = offset_h[:2] / offset_h[2]

        # Use only object pixels defined by mask
        y, x = np.where(mask > 127)
        mapped_h = (H @ np.stack([x, y, np.ones_like(x)], axis=0))
        p_local_2d = (mapped_h[:2]/mapped_h[2]).T - offset_2d
        p_local_3d = np.column_stack([p_local_2d, np.zeros(len(x))])
        
        z_rad, a_rad = np.radians(90-plane_cfg['elevation_deg']), np.radians(plane_cfg['azimuth_deg'])
        R = self._create_rotation_matrix(z_rad, a_rad)
            
        P_gpu = cp.zeros((h, w, 3), dtype=cp.float32)
        P_gpu[y, x] = cp.array(p_local_3d @ R.T, dtype=cp.float32)
        true_n = cp.array([np.sin(z_rad)*np.cos(a_rad), np.sin(z_rad)*np.sin(a_rad), np.cos(z_rad)], dtype=cp.float32)
        
        # 6 global points: origin (Camera Z), plane center, 4 corners
        obj_corners_3d = np.column_stack([obj_corners_2d - offset_2d, np.zeros(4)])
        global_corners = obj_corners_3d @ R.T
        plane_center_global = np.array([-offset_2d[0], -offset_2d[1], 0.0]) @ R.T
        
        return P_gpu, cp.array(mask > 127), true_n, global_corners, plane_center_global

    def _create_rotation_matrix(self, z_rad, a_rad):
        Rz = np.array([[np.cos(a_rad), -np.sin(a_rad), 0], [np.sin(a_rad), np.cos(a_rad), 0], [0,0,1]])
        Ry = np.array([[np.cos(z_rad), 0, np.sin(z_rad)], [0, 1, 0], [-np.sin(z_rad), 0, np.cos(z_rad)]])
        return Rz @ Ry

    def solve_normals_gpu(self, I, P, true_n):
        G_cols = []
        l_set = self.cfg['light_settings']
        for l_conf in self.light_data:
            l_dir = cp.array(l_conf['direction_vector_local'], dtype=cp.float32)
            l_dir /= cp.linalg.norm(l_dir)
            samples, area = self._get_light_samples_gpu(l_conf)
            G_accum = cp.zeros((P.shape[0], 3), dtype=cp.float32)
            spread = l_conf.get('spread_angle_deg', l_set['defaults']['spread_angle_deg'])
            min_cos = np.cos(np.radians(spread / 2.0))
            for s_pt in samples:
                V = s_pt - P
                d2 = cp.sum(V**2, axis=1) + 1e-9
                l_k = V / cp.sqrt(d2)[:, None]
                cos_emit = cp.dot(l_k, -l_dir)
                geo = cp.maximum(0, cos_emit) / d2
                if l_set['clip_if_spread_less_than_180'] and spread < 180:
                    geo *= (cos_emit >= min_cos)
                G_accum += geo[:, None] * l_k
            G_cols.append((G_accum / len(samples)) * area)
        G = cp.stack(G_cols, axis=1)
        GT = G.transpose(0, 2, 1)
        GTG = cp.matmul(GT, G) + cp.eye(3, dtype=cp.float32) * self.cfg['reconstruction_settings']['regularization']
        N = cp.matmul(cp.linalg.inv(GTG), cp.matmul(GT, I[:, :, None])).squeeze(2)
        N /= (cp.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
        errs = cp.degrees(cp.arccos(cp.clip(cp.sum(N * true_n, axis=1), -1.0, 1.0)))
        return N, errs

    def _get_light_samples_gpu(self, l_conf):
        center = np.array(l_conf['coordinates_local_cm'])
        shape = l_conf.get('shape', self.cfg['light_settings']['defaults']['shape'])
        dims = l_conf.get('dimensions_cm', self.cfg['light_settings']['defaults']['dimensions_cm'])
        angles = l_conf.get('orientation_local_deg', {"roll": 0, "pitch": 0, "yaw": 0})
        rx, ry, rz = np.radians([angles['roll'], angles['pitch'], angles['yaw']])
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        R_light = Rz @ Ry @ Rx
        right, up = R_light[:, 0], R_light[:, 1]
        samples = []
        if shape == 'circle':
            radius = dims[0] / 2.0
            n_ang = self.cfg['light_settings']['discretization']['circle_angular_steps']
            n_rad = self.cfg['light_settings']['discretization']['circle_radial_steps']
            for i in range(n_rad):
                r = (i + 0.5) * (radius / n_rad)
                for j in range(n_ang):
                    theta = j * (2 * np.pi / n_ang)
                    samples.append(center + r*np.cos(theta)*right + r*np.sin(theta)*up)
            area = np.pi * radius**2
        else:
            w, h = dims
            nx = self.cfg['light_settings']['discretization']['rect_x_steps']
            ny = self.cfg['light_settings']['discretization']['rect_y_steps']
            for x_v in np.linspace(-w/2, w/2, nx):
                for y_v in np.linspace(-h/2, h/2, ny):
                    samples.append(center + x_v*right + y_v*up)
            area = w * h
        return cp.array(samples, dtype=cp.float32), area

    def detect_plane_corners(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        # Restored detected corners image output
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for p in ordered: 
            cv2.circle(vis, tuple(p.astype(int)), 20, (0, 0, 255), -1)
        cv2.imwrite(str(self.output_dir / "detected_corners_vis.png"), vis)
        return ordered

    def visualize_coordinate_offset(self, corners, center):
        plt.figure(figsize=(10,10))
        # 1. Camera Z-axis Origin
        plt.scatter(0, 0, color='red', marker='x', s=150, label='Camera Z-Axis Origin (0,0)')
        plt.annotate("Origin (0,0)", (0, 0.1), ha='center', color='red', weight='bold')
        
        # 2. Plane Center
        plt.scatter(center[0], center[1], color='green', marker='+', s=150, label='Plane Center')
        plt.annotate(f"Center\n({center[0]:.2f}, {center[1]:.2f})", (center[0], center[1]), 
                     xytext=(5,5), textcoords="offset points", color='green', weight='bold')
        
        # 3. Plane corners and Boundary
        pts = np.vstack([corners, corners[0]])
        plt.plot(pts[:, 0], pts[:, 1], 'b-o', label='Plane Boundary', linewidth=2)
        for i, p in enumerate(corners):
            plt.annotate(f"Corner {i+1}\n({p[0]:.2f}, {p[1]:.2f})", (p[0], p[1]), 
                         xytext=(5,5), textcoords="offset points", fontsize=9)
        
        plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
        plt.grid(True); plt.legend(loc='upper right'); plt.axis('equal')
        plt.title('Coordinate Verification (Origin = Camera Boresight)')
        plt.savefig(self.output_dir / "coordinate_verification_enhanced.png", dpi=300)
        plt.close()

    def save_results(self, N, errs, mask_gpu, h, w):
        mask_cpu = cp.asnumpy(mask_gpu)
        y, x = np.where(mask_cpu)
        N_cpu = cp.asnumpy(N)
        errs_cpu = cp.asnumpy(errs)
        
        # CSV output for data analysis
        pd.DataFrame({'u':x, 'v':y, 'nx':N_cpu[:,0], 'ny':N_cpu[:,1], 'nz':N_cpu[:,2], 'err_deg':errs_cpu}).to_csv(self.output_dir / "pixel_normals.csv", index=False)
        
        # Error Heatmap output
        e_map = np.full((h, w), np.nan)
        e_map[y, x] = errs_cpu
        plt.figure(figsize=(10,8))
        plt.imshow(e_map, cmap='inferno')
        plt.colorbar(label='Angular Error (Degrees)')
        plt.title(f"Mean Error: {np.mean(errs_cpu):.4f}°")
        plt.savefig(self.output_dir / "error_heatmap.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    # Ensure correct JSON path
    PhotometricStereoFullReconstructor(r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\protype_2_nrml_cfg.json").process()