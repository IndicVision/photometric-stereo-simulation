import numpy as np
import cv2
import os
import json
import pandas as pd
from pathlib import Path
import cupy as cp # GPU Acceleration
import math

CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\upd_nrfld_inv_sqr_ctr_cfg.json"

class GContourGeneratorGPU:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_base = Path(self.cfg['paths']['rendered_output_base'])
        self.csv_base = Path(self.cfg['paths']['csv_input_base'])
        self.output_base = Path(self.cfg['paths']['contour_output_base'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        self.method = self.cfg['processing_settings']['exposure_method']
        self.bg_thresh = self.cfg['processing_settings']['background_threshold']
        self.erosion = self.cfg['processing_settings']['mask_erosion_iter']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']
        self.num_n = self.cfg['processing_settings'].get('num_g_plot_samples', 20)
        self.thresh_mode = self.cfg['processing_settings'].get('threshold_mode', 'Range')
        self.thresh_val = self.cfg['processing_settings'].get('intensity_threshold_value', 5.0)

    def get_geometry_gpu(self, params, w, h):
        """Constructs surface positions on GPU."""
        zenith = cp.deg2rad(90.0 - params['elev'])
        az = cp.deg2rad(params['azim'])
        n = cp.array([cp.sin(zenith)*cp.cos(az), cp.sin(zenith)*cp.sin(az), cp.cos(zenith)], dtype=cp.float32)
        side = params['area']**0.5
        x = cp.linspace(-side/2, side/2, w)
        y = cp.linspace(side/2, -side/2, h)
        xv, yv = cp.meshgrid(x, y)
        zv = -(n[0]*xv + n[1]*yv) / (n[2] + 1e-9)
        return cp.stack((xv, yv, zv), axis=-1)

    def calculate_g_map_gpu(self, p_surf, light_vec, dist, params):
        """Vectorized G-map calculation using CuPy."""
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['processing_settings']['samples_per_axis']
        total_area = l_dims[0] * l_dims[1]
        
        light_center = cp.array(light_vec * dist, dtype=cp.float32)
        n_A = -cp.array(light_vec, dtype=cp.float32) 
        
        up = cp.array([0, 0, 1], dtype=cp.float32)
        right = cp.cross(up, n_A)
        if cp.linalg.norm(right) < 1e-3: right = cp.array([1, 0, 0], dtype=cp.float32)
        right /= cp.linalg.norm(right)
        up_loc = cp.cross(n_A, right)
        
        steps = cp.linspace(-0.5, 0.5, samp_ax)
        si, sj = cp.meshgrid(steps, steps)
        pts = light_center + (right[None, None, :] * si[:, :, None] * l_dims[0]) + \
                             (up_loc[None, None, :] * sj[:, :, None] * l_dims[1])
        pts = pts.reshape(-1, 3)

        G_acc = cp.zeros(p_surf.shape[:2], dtype=cp.float32)
        for pt in pts:
            v = pt - p_surf
            r2 = cp.sum(v**2, axis=2) + 1e-9
            l_k = v / cp.sqrt(r2)[:, :, None]
            cos_emitter = cp.sum(-n_A * l_k, axis=2)
            G_acc += cp.maximum(0, cos_emitter) / r2
            
        return cp.asnumpy(G_acc * total_area / len(pts))

    def process(self):
        print(f"Starting Fast Contour Generation (Direct Curve Mode)...")
        for s_fold in [d for d in self.input_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    p_parts = p_fold.name.split('_')
                    l_parts = l_fold.name.split('_')
                    params = {'elev': float(p_parts[1]), 'azim': float(p_parts[0]), 'area': float(p_parts[5]), 'dist': float(l_parts[1]), 'spread': float(l_parts[4])}
                    
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    if not matrix_file.exists(): continue
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values
                    
                    for l_id in self.target_lights:
                        csv_path = self.csv_base / s_fold.name / p_fold.name / l_fold.name / f"light_{l_id:03d}_pixel_data.csv"
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        if not (csv_path.exists() and img_path.exists()): continue
                        
                        df = pd.read_csv(csv_path)
                        img_bgr = cv2.imread(str(img_path))
                        
                        p_surf_gpu = self.get_geometry_gpu(params, img_bgr.shape[1], img_bgr.shape[0])
                        g_map = self.calculate_g_map_gpu(p_surf_gpu, l_vecs[l_id-1], params['dist'], params)
                        
                        mask = cv2.erode((cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) > (self.bg_thresh*255)).astype(np.uint8), np.ones((3,3)), iterations=self.erosion).astype(bool)
                        
                        coords = np.argwhere(mask)
                        roi = (coords.min(axis=0)[0], coords.max(axis=0)[0], coords.min(axis=0)[1], coords.max(axis=0)[1]) if coords.size > 0 else None

                        target_levels = np.linspace(df['G_Value'].min(), df['G_Value'].max(), self.num_n)
                        self.draw_iso_g_contours(img_bgr, g_map, mask, target_levels, df, s_fold.name, p_fold.name, l_fold.name, l_id, roi)
                        
                        del p_surf_gpu
                        cp.get_default_memory_pool().free_all_blocks()

    def draw_iso_g_contours(self, img, g_map, mask, levels, df_pixels, s_n, p_n, l_n, l_id, roi):
        contour_img = img.copy()
        vis = self.cfg['contour_visuals']
        h_img, w_img = img.shape[:2]
        
        for i, val in enumerate(levels):
            closest_g = df_pixels.iloc[(df_pixels['G_Value'] - val).abs().argsort()[:1]]['G_Value'].values[0]
            subset = df_pixels[np.isclose(df_pixels['G_Value'], closest_g, atol=1e-7)]
            
            metric = (subset['Gray'].quantile(0.75) - subset['Gray'].quantile(0.25)) if self.thresh_mode == 'IQR' else (subset['Gray'].max() - subset['Gray'].min())
            color = vis['good_color_bgr'] if metric <= self.thresh_val else vis['bad_color_bgr']

            level_mask = (g_map >= val).astype(np.uint8) * 255
            level_mask[~mask] = 0
            contours, _ = cv2.findContours(level_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw labels on every 3rd contour for clarity
            label_this = (i % 3 == 0) and vis['show_text_labels'] and len(contours) > 0
            
            for cnt in contours:
                if label_this and cv2.contourArea(cnt) > 100:
                    label_text = f"{val:.4e}"
                    # Find a stable point away from image boundaries
                    points = cnt.reshape(-1, 2)
                    mid = len(points) // 2
                    pt = points[mid]
                    
                    # Calculate Tangent
                    p1 = points[max(0, mid-15)]
                    p2 = points[min(len(points)-1, mid+15)]
                    angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
                    
                    # Flip angle if upside down
                    if angle > 90: angle -= 180
                    if angle < -90: angle += 180
                    
                    # 1. Clear a small space for the text (prevent strike-through)
                    self.clear_text_path(contour_img, label_text, pt, angle, 0.25)
                    
                    # 2. Draw the contour (will be broken where text is)
                    cv2.drawContours(contour_img, [cnt], -1, color, vis['line_thickness'])
                    
                    # 3. Draw the actual text over the cleared path
                    self.draw_rotated_text_clean(contour_img, label_text, pt, angle, 0.25, (255, 255, 255))
                    label_this = False # Only label once per level
                else:
                    cv2.drawContours(contour_img, [cnt], -1, color, vis['line_thickness'])

        if vis.get('crop_to_object', True) and roi:
            y1, y2, x1, x2 = roi
            pad = 50
            contour_img = contour_img[max(0, y1-pad):min(h_img, y2+pad), max(0, x1-pad):min(w_img, x2+pad)]

        out_dir = self.output_base / s_n / p_n / l_n
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"light_{l_id:03d}_G_contours.png"), contour_img)

    def clear_text_path(self, img, text, center, angle, scale):
        """Creates a small gap in the contour line for the label."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, scale, 1)[0]
        # Slightly larger box to provide buffer
        rect = ((center[0], center[1]), (size[0] + 4, size[1] + 2), angle)
        box = np.int0(cv2.boxPoints(rect))
        cv2.fillPoly(img, [box], (0, 0, 0)) # Wipe the contour under the text

    def draw_rotated_text_clean(self, img, text, center, angle, scale, color):
        """Draws rotated text centered on the point without any background box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, scale, 1)[0]
        
        # Create canvas for text
        text_canvas = np.zeros((size[1]*2, size[0]*2, 3), dtype=np.uint8)
        cv2.putText(text_canvas, text, (size[0]//2, size[1] + size[1]//2), font, scale, color, 1, cv2.LINE_AA)
        
        # Rotate
        M = cv2.getRotationMatrix2D((size[0], size[1]), angle, 1.0)
        rotated = cv2.warpAffine(text_canvas, M, (size[0]*2, size[1]*2))
        
        # Place on image
        h, w = rotated.shape[:2]
        sy, sx = int(center[1] - h/2), int(center[0] - w/2)
        
        for i in range(h):
            for j in range(w):
                if rotated[i, j].any():
                    ty, tx = sy + i, sx + j
                    if 0 <= ty < img.shape[0] and 0 <= tx < img.shape[1]:
                        img[ty, tx] = rotated[i, j]

if __name__ == "__main__":
    GContourGeneratorGPU(CONFIG_PATH).process()