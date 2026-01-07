import numpy as np
import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class IntensityGAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_base = Path(self.cfg['paths']['rendered_output_base'])
        self.csv_base = Path(self.cfg['paths']['csv_output_base'])
        self.plot_base = Path(self.cfg['paths']['plot_output_base'])
        self.matrix_base = Path(self.cfg['paths']['light_matrix_dir'])
        
        self.method = self.cfg['processing_settings']['exposure_method']
        self.bg_thresh = self.cfg['processing_settings']['background_threshold']
        self.erosion = self.cfg['processing_settings']['mask_erosion_iter']
        self.target_lights = self.cfg['processing_settings']['target_light_ids']

    def parse_params(self, p_folder, l_folder):
        p_parts = p_folder.split('_')
        l_parts = l_folder.split('_')
        return {
            'elev': float(p_parts[1]), 
            'azim': float(p_parts[0]), 
            'area': float(p_parts[5]),
            'dist': float(l_parts[1]), 
            'num_l': int(l_parts[0])
        }

    def get_geometry(self, params, w, h):
        zenith = np.deg2rad(90.0 - params['elev'])
        az = np.deg2rad(params['azim'])
        n = [np.sin(zenith)*np.cos(az), np.sin(zenith)*np.sin(az), np.cos(zenith)]
        
        side = params['area']**0.5
        x = np.linspace(-side/2, side/2, w)
        y = np.linspace(side/2, -side/2, h)
        xv, yv = np.meshgrid(x, y)
        zv = -(n[0]*xv + n[1]*yv) / (n[2] + 1e-9)
        return np.stack((xv, yv, zv), axis=-1)

    def calculate_g_val(self, p_surf, light_vec, dist, params):
        """
        Refined G-value calculation including the emitter cosine term 
        and Area scaling as per Equation (5) of the paper.
        """
        l_dims = (self.cfg['area_light_props']['dim_a_cm'], self.cfg['area_light_props']['dim_b_cm'])
        samp_ax = self.cfg['processing_settings']['samples_per_axis']
        total_area = l_dims[0] * l_dims[1] # A in the paper [cite: 133]
        K = samp_ax ** 2 # Number of sampling points [cite: 133]
        
        light_center = light_vec * dist
        # Area light normal n_A points back toward the origin [cite: 122]
        n_A = -light_vec 
        
        # Orient the area light coordinate system
        up = np.array([0, 0, 1])
        right = np.cross(up, n_A)
        if np.linalg.norm(right) < 1e-3: 
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up_loc = np.cross(n_A, right)
        
        G_acc = np.zeros(p_surf.shape[:2], dtype=np.float32)
        steps = np.linspace(-0.5, 0.5, samp_ax)
        
        for i in steps:
            for j in steps:
                # Calculate sample point position on light face
                pt = light_center + (right * i * l_dims[0]) + (up_loc * j * l_dims[1])
                
                # Vector from surface point to light sample
                v = pt - p_surf 
                r2 = np.sum(v**2, axis=2) + 1e-9
                l_k = v / np.sqrt(r2)[:, :, np.newaxis] # Unit direction vector l_k [cite: 134]
                
                # Emitter Cosine Term: (-n_A dot l_k) 
                # This accounts for the Lambertian emission of the area light
                cos_emitter = np.sum(-n_A * l_k, axis=2)
                cos_emitter = np.maximum(0, cos_emitter) # Ensure light only comes from the front face
                
                # Accumulate term: cos_emitter / r^2
                G_acc += (cos_emitter / r2)
        
        # Multiply by (A/K) to complete Equation (5) 
        return (total_area / K) * G_acc

    def process(self):
        for s_fold in [d for d in self.input_base.iterdir() if d.is_dir()]:
            for p_fold in [d for d in s_fold.iterdir() if d.is_dir()]:
                for l_fold in [d for d in p_fold.iterdir() if d.is_dir()]:
                    params = self.parse_params(p_fold.name, l_fold.name)
                    matrix_file = self.matrix_base / f"light_matrix_{p_fold.name}__{l_fold.name}.csv"
                    if not matrix_file.exists(): continue
                    
                    l_vecs = pd.read_csv(matrix_file)[['L_x', 'L_y', 'L_z']].values
                    
                    for l_id in self.target_lights:
                        img_path = l_fold / f"{l_id:03d}_{self.method}.png"
                        if not img_path.exists(): continue
                        
                        # 1. Load and Mask
                        img_bgr = cv2.imread(str(img_path))
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                        
                        raw_mask = (img_gray > self.bg_thresh).astype(np.uint8)
                        kernel = np.ones((3,3), np.uint8)
                        mask = cv2.erode(raw_mask, kernel, iterations=self.erosion).astype(bool)
                        
                        # 2. Geometry and Refined G-value
                        p_surf = self.get_geometry(params, img_gray.shape[1], img_gray.shape[0])
                        g_map = self.calculate_g_val(p_surf, l_vecs[l_id-1], params['dist'], params)
                        
                        # 3. Create CSV with Intensity and G_Value
                        y, x = np.indices(img_gray.shape)
                        df = pd.DataFrame({
                            'x': x[mask], 
                            'y': y[mask],
                            'R': img_bgr[mask][:, 2], 
                            'G': img_bgr[mask][:, 1], 
                            'B': img_bgr[mask][:, 0],
                            'Gray': (img_gray[mask] * 255).astype(np.uint8),
                            'G_Value': g_map[mask]
                        })
                        
                        out_dir = self.csv_base / s_fold.name / p_fold.name / l_fold.name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = out_dir / f"light_{l_id:03d}_pixel_data.csv"
                        df.to_csv(csv_path, index=False)
                        
                        # 4. Generate Plot
                        self.generate_plot(df, s_fold.name, p_fold.name, l_fold.name, l_id)

    def generate_plot(self, df, s_n, p_n, l_n, l_id):
        plt.figure(figsize=(14, 8))
        # Round G values to create discrete bins for the boxplot
        df['G_rounded'] = df['G_Value'].round(6)
        
        df.boxplot(column='Gray', by='G_rounded', grid=False, 
                   showfliers=self.cfg['plot_settings']['show_outliers'])
        
        plt.title(f"Intensity vs G-Value (Lambertian Emitter) | Light {l_id}\n{l_n}")
        plt.suptitle("") 
        plt.xlabel("G Value (Paper Eq. 5 with Cosine Emitter Term)")
        plt.ylabel("Intensity (8-bit Gray)")
        plt.xticks(rotation=90, fontsize=8)
        
        plot_dir = self.plot_base / s_n / p_n / l_n
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"light_{l_id:03d}_boxplot.png", dpi=self.cfg['plot_settings']['dpi'])
        plt.close()

if __name__ == "__main__":
    # Ensure this path matches your environment
    config_file = r"C:\Users\vishn\Desktop\avanthik\blender_code\inverse_square_law_analysis\nearfield_inverse_square_law_plot_config.json"
    analyzer = IntensityGAnalyzer(config_file)
    analyzer.process()