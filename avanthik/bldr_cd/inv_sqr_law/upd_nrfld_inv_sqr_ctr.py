import numpy as np
import cv2
import os
import json
import pandas as pd
from pathlib import Path
import math

class GContourGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.render_base = Path(self.cfg['paths']['rendered_output_base'])
        self.csv_input_base = Path(self.cfg['paths']['csv_input_base']) # From previous script
        self.output_base = Path(self.cfg['paths']['contour_output_base'])
        
        self.method = self.cfg['processing_settings']['exposure_method']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']
        self.num_n = self.cfg['processing_settings'].get('num_g_plot_samples', 20)
        self.thresh_mode = self.cfg['processing_settings'].get('threshold_mode', 'Range')
        self.thresh_val = self.cfg['processing_settings'].get('intensity_threshold_value', 5.0)

    def reconstruct_images_from_csv(self, df, h, w):
        """Reconstructs 2D G-Map and Mask from sparse CSV data."""
        g_map = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        u = df['pixel_u'].values.astype(int)
        v = df['pixel_v'].values.astype(int)
        g = df['G_Value'].values.astype(np.float32)
        
        # Valid bounds check
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, g = u[valid], v[valid], g[valid]
        
        # Fill maps
        g_map[v, u] = g
        mask[v, u] = 255 # Valid object pixels
        
        return g_map, mask.astype(bool)

    def process(self):
        print(f"Starting Contour Generation (Pixel-Perfect Mode)...")
        
        for s_fold in [d for d in self.render_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    
                    for l_id in self.target_lights:
                        # 1. Inputs
                        csv_path = self.csv_input_base / s_fold.name / p_fold.name / l_fold.name / f"light_{l_id:03d}_pixel_data.csv"
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        
                        if not (csv_path.exists() and img_path.exists()): continue
                        
                        # 2. Load Data
                        df = pd.read_csv(csv_path)
                        img_bgr = cv2.imread(str(img_path))
                        h, w = img_bgr.shape[:2]
                        
                        # 3. Reconstruct 2D Maps
                        g_map, mask = self.reconstruct_images_from_csv(df, h, w)
                        
                        # 4. Determine ROI for cropping
                        coords = np.argwhere(mask)
                        roi = None
                        if coords.size > 0:
                            roi = (coords.min(axis=0)[0], coords.max(axis=0)[0], 
                                   coords.min(axis=0)[1], coords.max(axis=0)[1])

                        # 5. Draw Contours
                        target_levels = np.linspace(df['G_Value'].min(), df['G_Value'].max(), self.num_n)
                        self.draw_iso_g_contours(img_bgr, g_map, mask, target_levels, df, s_fold.name, p_fold.name, l_fold.name, l_id, roi)

    def draw_iso_g_contours(self, img, g_map, mask, levels, df_pixels, s_n, p_n, l_n, l_id, roi):
        contour_img = img.copy()
        vis = self.cfg['contour_visuals']
        h_img, w_img = img.shape[:2]
        
        for i, val in enumerate(levels):
            # Find the actual G value in the data closest to the target level
            closest_g = df_pixels.iloc[(df_pixels['G_Value'] - val).abs().argsort()[:1]]['G_Value'].values[0]
            subset = df_pixels[np.isclose(df_pixels['G_Value'], closest_g, atol=1e-7)]
            
            # Metric Check
            metric = (subset['Gray'].quantile(0.75) - subset['Gray'].quantile(0.25)) if self.thresh_mode == 'IQR' else (subset['Gray'].max() - subset['Gray'].min())
            color = vis['good_color_bgr'] if metric <= self.thresh_val else vis['bad_color_bgr']

            # Create binary mask for this level (G >= val) for finding contour boundary
            level_mask = (g_map >= val).astype(np.uint8) * 255
            level_mask[~mask] = 0
            
            # Find Contours
            contours, _ = cv2.findContours(level_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            label_this = (i % 3 == 0) and vis['show_text_labels'] and len(contours) > 0
            
            for cnt in contours:
                if label_this and cv2.contourArea(cnt) > 100:
                    label_text = f"{val:.4e}"
                    points = cnt.reshape(-1, 2)
                    mid = len(points) // 2
                    pt = points[mid]
                    
                    # Calculate Angle
                    p1 = points[max(0, mid-15)]
                    p2 = points[min(len(points)-1, mid+15)]
                    angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
                    
                    if angle > 90: angle -= 180
                    if angle < -90: angle += 180
                    
                    self.clear_text_path(contour_img, label_text, pt, angle, 0.25)
                    cv2.drawContours(contour_img, [cnt], -1, color, vis['line_thickness'])
                    self.draw_rotated_text_clean(contour_img, label_text, pt, angle, 0.25, (255, 255, 255))
                    label_this = False
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, scale, 1)[0]
        rect = ((center[0], center[1]), (size[0] + 4, size[1] + 2), angle)
        box = np.int0(cv2.boxPoints(rect))
        cv2.fillPoly(img, [box], (0, 0, 0))

    def draw_rotated_text_clean(self, img, text, center, angle, scale, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = cv2.getTextSize(text, font, scale, 1)[0]
        text_canvas = np.zeros((size[1]*2, size[0]*2, 3), dtype=np.uint8)
        cv2.putText(text_canvas, text, (size[0]//2, size[1] + size[1]//2), font, scale, color, 1, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((size[0], size[1]), angle, 1.0)
        rotated = cv2.warpAffine(text_canvas, M, (size[0]*2, size[1]*2))
        
        h, w = rotated.shape[:2]
        sy, sx = int(center[1] - h/2), int(center[0] - w/2)
        
        # Minimal bounds check manual blend
        for i in range(h):
            for j in range(w):
                if rotated[i, j].any():
                    ty, tx = sy + i, sx + j
                    if 0 <= ty < img.shape[0] and 0 <= tx < img.shape[1]:
                        img[ty, tx] = rotated[i, j]

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\inv_sqr_law\upd_nrfld_inv_sqr_ctr_cfg.json"
    GContourGenerator(CONFIG_PATH).process()