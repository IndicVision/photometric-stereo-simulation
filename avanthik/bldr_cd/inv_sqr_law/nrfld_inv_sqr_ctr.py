import numpy as np
import cv2
import os
import json
import pandas as pd
from pathlib import Path

class GContourGenerator:
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

    def parse_params(self, p_folder, l_folder):
        p_parts = p_folder.split('_')
        l_parts = l_folder.split('_')
        return {
            'elev': float(p_parts[1]), 
            'azim': float(p_parts[0]), 
            'area': float(p_parts[5]),
            'dist': float(l_parts[1])
        }

    def get_geometry(self, params, w, h):
        """Constructs 3D surface positions."""
        zenith = np.deg2rad(90.0 - params['elev'])
        az = np.deg2rad(params['azim'])
        # Calculate normal vector [cite: 117, 142]
        n = [np.sin(zenith)*np.cos(az), np.sin(zenith)*np.sin(az), np.cos(zenith)]
        
        side = params['area']**0.5
        x = np.linspace(-side/2, side/2, w)
        y = np.linspace(side/2, -side/2, h)
        xv, yv = np.meshgrid(x, y)
        zv = -(n[0]*xv + n[1]*yv) / (n[2] + 1e-9)
        return np.stack((xv, yv, zv), axis=-1)

    def calculate_g_map(self, p_surf, light_vec, dist):
        """Calculates the discretized G-field including the emitter cosine term[cite: 125, 131, 138]."""
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['processing_settings']['samples_per_axis']
        total_area = l_dims[0] * l_dims[1]
        K = samp_ax ** 2 
        
        light_center = light_vec * dist
        n_A = -light_vec # Light normal [cite: 122]
        
        up = np.array([0, 0, 1])
        right = np.cross(up, n_A)
        if np.linalg.norm(right) < 1e-3: right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up_loc = np.cross(n_A, right)
        
        G_acc = np.zeros(p_surf.shape[:2], dtype=np.float32)
        steps = np.linspace(-0.5, 0.5, samp_ax)
        
        for i in steps:
            for j in steps:
                pt = light_center + (right * i * l_dims[0]) + (up_loc * j * l_dims[1])
                v = pt - p_surf 
                r2 = np.sum(v**2, axis=2) + 1e-9
                l_k = v / np.sqrt(r2)[:, :, np.newaxis] 
                
                # Emitter cosine term [cite: 125, 131]
                cos_emitter = np.sum(-n_A * l_k, axis=2)
                cos_emitter = np.maximum(0, cos_emitter) 
                G_acc += (cos_emitter / r2)
        
        return (total_area / K) * G_acc

    def process(self):
        # Crawl through folders matching the standard naming convention
        sample_folds = [d for d in self.input_base.iterdir() if d.is_dir()]
        for s_fold in sample_folds:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    
                    params = self.parse_params(p_fold.name, l_fold.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    if not matrix_file.exists(): continue
                    
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values
                    
                    for l_id in self.target_lights:
                        # Find corresponding CSV to get min/max G values
                        csv_path = self.csv_base / s_fold.name / p_fold.name / l_fold.name / f"light_{l_id:03d}_pixel_data.csv"
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        
                        if not (csv_path.exists() and img_path.exists()): continue
                        
                        # Load data
                        df = pd.read_csv(csv_path)
                        img_bgr = cv2.imread(str(img_path))
                        
                        # Calculate discretized G-levels: (max-min)/(N-1)
                        g_min, g_max = df['G_Value'].min(), df['G_Value'].max()
                        step = (g_max - g_min) / (self.num_n - 1)
                        target_levels = [g_min + i * step for i in range(self.num_n)]
                        
                        # Re-calculate G-map for the whole image to find contour lines
                        p_surf = self.get_geometry(params, img_bgr.shape[1], img_bgr.shape[0])
                        g_map = self.calculate_g_map(p_surf, l_vecs[l_id-1], params['dist'])
                        
                        # Apply background mask
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        mask = cv2.erode((gray > (self.bg_thresh*255)).astype(np.uint8), np.ones((3,3)), iterations=self.erosion)
                        g_map_masked = g_map * mask.astype(bool)

                        # Draw Contours
                        self.draw_iso_g_contours(img_bgr, g_map_masked, target_levels, s_fold.name, p_fold.name, l_fold.name, l_id)

    def draw_iso_g_contours(self, img, g_map, levels, s_n, p_n, l_n, l_id):
        contour_img = img.copy()
        vis = self.cfg['contour_visuals']
        
        for val in levels:
            # Create binary mask for this level
            level_mask = (g_map >= val).astype(np.uint8) * 255
            contours, _ = cv2.findContours(level_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the line
            cv2.drawContours(contour_img, contours, -1, vis['line_color_bgr'], vis['line_thickness'])
            
            # Optionally add text label at the first point of the contour
            if vis['show_text_labels'] and len(contours) > 0:
                p = contours[0][0][0]
                cv2.putText(contour_img, f"{val:.4f}", (p[0], p[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, vis['font_scale'], (255, 255, 255), 1)

        # Save output
        out_dir = self.output_base / s_n / p_n / l_n
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"light_{l_id:03d}_G_contours.png"), contour_img)
        print(f"Saved contours for Light {l_id} in {l_n}")

if __name__ == "__main__":
    generator = GContourGenerator(r"C:\Users\vishn\Desktop\avanthik\blender_code\inverse_square_law_analysis\nearfield_inverse_square_law_contour_config.json")
    generator.process()