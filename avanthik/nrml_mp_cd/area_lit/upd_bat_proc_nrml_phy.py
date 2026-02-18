import cupy as cp
import numpy as np
import cv2
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

class UltimateNormalProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.input_dir = Path(self.cfg['paths']['input_images_dir'])
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg['paths']['light_yaml_path'], 'r') as f:
            self.light_data = yaml.safe_load(f)

    def _get_lens_params(self):
        """Computes v (image dist) and u (object dist) from lens math."""
        f_mm = self.cfg['camera']['focal_length_mm']
        m = self.cfg['camera']['magnification']
        v_mm = f_mm * (1 + m)
        u_mm = v_mm / m
        return v_mm, u_mm / 10.0  # Return u in cm for world space

    def _get_cam_extrinsics(self, u_cm):
        """Builds camera position and rotation based on computed u_cm."""
        dir_vec = np.array(self.cfg['camera']['camera_direction_vector'])
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-9)
        origin = np.array(self.cfg['geometry_settings']['plane_origin_cm'])
        cam_pos = origin + (dir_vec * u_cm)
        
        mapping = self.cfg['camera']['axis_mapping']
        R = np.zeros((3, 3))
        axes = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
        for i, cam_ax in enumerate(['cam_x', 'cam_y', 'cam_z']):
            target = mapping[cam_ax].lower()
            sign = -1 if target.startswith('-') else 1
            R[:, i] = sign * np.array(axes[target.strip('+-')])
        return cp.array(cam_pos, dtype=cp.float32), cp.array(R, dtype=cp.float32)

    def load_images_and_mask(self):
        """Standardized Image Acquisition with Shape Safety."""
        ext = self.cfg['image_processing']['file_extension']
        res_w, res_h = self.cfg['camera']['resolution']
        thresh = self.cfg['image_processing']['mask_threshold']
        if not self.cfg['image_processing']['is_threshold_normalized']: thresh /= 255.0

        images, union_mask = [], cp.zeros((res_h, res_w), dtype=bool)
        coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        for l_conf in self.light_data:
            pattern = str(self.input_dir / f"light_{int(l_conf['light_id']):03d}*")
            files = glob.glob(pattern)
            fpath = [f for f in files if f.lower().endswith(ext.lower())][0]

            if ext.upper() in ['.CR2', '.DNG', '.NEF']:
                with rawpy.imread(fpath) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
                    img = (rgb.astype(np.float32) / 65535.0) @ coeffs
            else:
                bgr = cv2.imread(fpath).astype(np.float32) / 255.0
                img = bgr[:,:,::-1] @ coeffs
                if self.cfg['image_processing']['apply_gamma_if_ldr']:
                    img = np.power(img, self.cfg['image_processing']['gamma_value'])
            
            # --- SHAPE SAFETY: Handle Tilted/Transposed images ---
            if img.shape == (res_w, res_h):
                img = img.T
            if img.shape != (res_h, res_w):
                img = cv2.resize(img, (res_w, res_h), interpolation=cv2.INTER_AREA)
            
            img_gpu = cp.array(img, dtype=cp.float32)
            union_mask |= (img_gpu > thresh)
            images.append(img_gpu)
        return cp.stack(images, axis=-1), union_mask

    def compute_geometry(self, mask_gpu):
        """Generalized Perspective or Independent Homography."""
        mode = self.cfg['geometry_settings']['mode']
        res_w, res_h = self.cfg['camera']['resolution']
        v_mm, u_cm = self._get_lens_params()
        
        if mode == "PERSPECTIVE":
            sw, sh = self.cfg['camera']['sensor_size_mm']
            u_g, v_g = cp.meshgrid(cp.linspace(0, res_w-1, res_w), cp.linspace(0, res_h-1, res_h))
            rays = cp.stack([(u_g - res_w/2)/(v_mm*res_w/sw), (v_g - res_h/2)/(v_mm*res_h/sh), cp.ones_like(u_g)], axis=-1)
            rays /= (cp.linalg.norm(rays, axis=2, keepdims=True) + 1e-9)
            
            cam_pos, R_cam = self._get_cam_extrinsics(u_cm)
            rays_world = cp.tensordot(rays, R_cam.T, axes=1)
            
            p_pos, p_norm = cp.array(self.cfg['geometry_settings']['plane_origin_cm']), \
                             cp.array(self.cfg['geometry_settings']['plane_normal'])
            denom = cp.sum(rays_world * p_norm, axis=2)
            t = cp.dot(p_pos - cam_pos, p_norm) / (denom + 1e-9)
            return cam_pos + rays_world * t[:,:,None], (t > 0), p_norm

        else: # HOMOGRAPHY
            mask_cpu = (cp.asnumpy(mask_gpu) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
            d = np.diff(pts, axis=1); rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
            
            L, B = self.cfg['geometry_settings']['plane_dimensions_cm']
            pts_world = np.array([[-L/2, B/2], [L/2, B/2], [L/2, -B/2], [-L/2, -B/2]], dtype="float32")
            H, _ = cv2.findHomography(rect, pts_world)
            
            v_idx, u_idx = np.indices((res_h, res_w))
            px_coords = np.stack([u_idx.ravel(), v_idx.ravel(), np.ones(res_h*res_w)], axis=0)
            world = (H @ px_coords); world /= (world[2,:] + 1e-9)
            P_3d = cp.zeros((res_h, res_w, 3), dtype=cp.float32)
            P_3d[:,:,:2] = cp.array(world[:2,:].reshape(2, res_h, res_w).transpose(1,2,0))
            return P_3d, cp.ones((res_h, res_w), dtype=bool), cp.array([0,0,1], dtype=cp.float32)

    def _get_samples(self, l_conf):
        """Discretization using exact grid centroids and YAML rotations."""
        center = np.array(l_conf['coordinates_local_cm'])
        r, p, y = np.radians([l_conf['orientation_local_deg'].get(k, 0) for k in ['roll', 'pitch', 'yaw']])
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        rot = Rz @ Ry @ Rx
        right, up = rot[:, 0], rot[:, 1]
        
        nx, ny = self.cfg['light_settings']['discretization']['rect_steps']
        dims = l_conf.get('dimensions_cm', self.cfg['light_settings']['defaults']['dimensions_cm'])
        dx, dy = dims[0], (dims[1] if len(dims) > 1 else dims[0])
        
        xs = np.linspace(-dx/2, dx/2, nx+1); xc = (xs[:-1] + xs[1:]) / 2
        ys = np.linspace(-dy/2, dy/2, ny+1); yc = (ys[:-1] + ys[1:]) / 2
        return cp.array([center + x*right + y*up for x in xc for y in yc], dtype=cp.float32)

    def process(self):
        I, a_mask = self.load_images_and_mask()
        P, v_mask, true_n = self.compute_geometry(a_mask)
        final_mask = v_mask & a_mask; active_idx = cp.where(final_mask.flatten())[0]
        I_flat, P_flat = I.reshape(-1, len(self.light_data))[active_idx], P.reshape(-1, 3)[active_idx]
        
        G_cols = []
        for l_conf in self.light_data:
            l_dir, pwr = cp.array(l_conf['direction_vector_local']), l_conf.get('radiant_power_w', 1.0)
            spread = l_conf.get('spread_angle_deg', self.cfg['light_settings']['defaults']['spread_angle_deg'])
            min_cos = np.cos(np.radians(spread / 2.0))
            samples = self._get_samples(l_conf); G_acc = cp.zeros((len(active_idx), 3), dtype=cp.float32)
            for s in samples:
                V = s - P_flat; d2 = cp.sum(V**2, axis=1) + 1e-9; l_k = V / cp.sqrt(d2)[:, None]
                cos_emit = cp.dot(l_k, -l_dir)
                geo = (pwr * cp.maximum(0, cos_emit) / d2)
                if self.cfg['light_settings']['clip_by_spread']: geo *= (cos_emit >= min_cos)
                G_acc += geo[:, None] * l_k
            G_cols.append(G_acc / len(samples))

        G = cp.stack(G_cols, axis=1); GT = G.transpose(0, 2, 1)
        reg = self.cfg['reconstruction']['regularization']
        N_raw = cp.linalg.inv(cp.matmul(GT, G) + cp.eye(3)*reg) @ cp.matmul(GT, I_flat[:, :, None])
        N_est = N_raw.squeeze(2); N_est /= (cp.linalg.norm(N_est, axis=1, keepdims=True) + 1e-9)
        
        errs = cp.degrees(cp.arccos(cp.clip(cp.sum(N_est * true_n, axis=1), -1, 1)))
        self.save_results(final_mask, N_est, P_flat, errs)

    def save_results(self, mask, N_est, P_flat, errs):
        w, h = self.cfg['camera']['resolution']
        v_idx, u_idx = np.where(cp.asnumpy(mask))
        df = pd.DataFrame({'u': u_idx, 'v': v_idx, 'world_x': cp.asnumpy(P_flat[:,0]), 'world_y': cp.asnumpy(P_flat[:,1]),
                           'nx': cp.asnumpy(N_est[:,0]), 'ny': cp.asnumpy(N_est[:,1]), 'nz': cp.asnumpy(N_est[:,2]), 'err': cp.asnumpy(errs)})
        df.to_csv(self.output_dir / f"results_{self.cfg['geometry_settings']['mode'].lower()}.csv", index=False)
        
        err_map = np.zeros((h, w)); err_map[v_idx, u_idx] = cp.asnumpy(errs)
        plt.figure(figsize=(12, 10))
        plt.imshow(np.ma.masked_where(err_map == 0, err_map), cmap='inferno', vmin=0, vmax=np.percentile(cp.asnumpy(errs), 98))
        plt.colorbar(label="Error (Deg)"); plt.title(f"PS: {self.cfg['geometry_settings']['mode']} | m={self.cfg['camera']['magnification']}\nMean Error: {cp.mean(errs):.4f}°")
        plt.axis('off'); plt.savefig(self.output_dir / f"heatmap_{self.cfg['geometry_settings']['mode'].lower()}.png", dpi=300)
        print(f"Complete. Output saved to: {self.output_dir}")

if __name__ == "__main__":
    UltimateNormalProcessor(r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\upd_bat_proc_nrml_phy_cfg.json").process()