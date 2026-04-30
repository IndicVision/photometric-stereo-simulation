import os
# Force OpenEXR support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path

# Try importing rawpy for CR2 support
try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

class IntegratedPhotometricMapper:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_dir = Path(self.cfg['paths']['input_images_dir'])
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.res_w, self.res_h = self.cfg['camera']['resolution']
        self.plane_dim = np.array(self.cfg['plane']['dimensions_cm'])
        self.center_pos = np.array(self.cfg['plane']['center_position_cm'])

    def load_and_composite(self):
        ext = self.cfg['image_processing']['file_extension']
        running_max = None

        print(f"Scanning for light_XXX{ext} files...")
        for i in range(1, 1000): 
            fname = f"light_{i:03d}{ext}"
            fpath = self.input_dir / fname
            if not fpath.exists(): break

            if ext.upper() in ['.CR2', '.NEF']:
                with rawpy.imread(str(fpath)) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
                    img = (rgb.astype(np.float32) / 65535.0)
            else:
                img = cv2.imread(str(fpath)).astype(np.float32) / 255.0
            
            gray = 0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]
            if running_max is None:
                running_max = gray
            else:
                running_max = np.maximum(running_max, gray)
        
        return running_max

    def generate_mask(self, composite):
        method = self.cfg['image_processing']['masking_method']
        comp_8bit = (composite * 255).astype(np.uint8)
        
        if method == "otsu":
            _, mask = cv2.threshold(comp_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "manual":
            thresh = self.cfg['image_processing']['manual_threshold']
            _, mask = cv2.threshold(comp_8bit, thresh, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(str(self.output_dir / "final_mask.png"), mask)
        return mask > 0

    def detect_corners(self, mask, composite):
        """Finds 4 corners and prints which method succeeded."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        
        corners = None
        used_method = "None"

        # Method 1: Polygon Approximation (Squinting)
        for eps_factor in [0.01, 0.015, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
            if len(approx) == 4:
                corners = self.order_points(approx.reshape(4, 2))
                used_method = f"Polygon Approximation (Epsilon: {eps_factor*100}%)"
                break
        
        # Method 2: Minimum Area Rectangle (Fallback)
        if corners is None:
            rect = cv2.minAreaRect(largest)
            corners = self.order_points(cv2.boxPoints(rect))
            used_method = "Minimum Area Bounding Box (Fallback)"

        print(f"\n[DEBUG] Corner Detection Method: {used_method}")

        # Save Visualized Corners
        vis = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, pt in enumerate(corners):
            cv2.circle(vis, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(vis, f"C{i}", tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(self.output_dir / "detected_corners.png"), vis)
        
        return corners

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    def map_to_world(self, points_uv, H):
        points_uv = np.array(points_uv, dtype="float32")
        if points_uv.ndim == 1: points_uv = points_uv.reshape(1, 2)
        
        ones = np.ones((len(points_uv), 1))
        pixels_homo = np.hstack([points_uv, ones])
        world_2d_homo = (H @ pixels_homo.T).T
        world_2d = world_2d_homo[:, :2] / world_2d_homo[:, 2:3]
        
        elev = np.deg2rad(90 - self.cfg['plane']['elevation_deg'])
        world_3d = np.zeros((len(world_2d), 3))
        world_3d[:, 0] = world_2d[:, 0] + self.center_pos[0]
        world_3d[:, 1] = world_2d[:, 1] + self.center_pos[1]
        if elev != 0:
             world_3d[:, 2] = world_2d[:, 1] * np.sin(elev)
        
        return world_3d

    def process(self):
        comp = self.load_and_composite()
        mask = self.generate_mask(comp)
        pixel_corners = self.detect_corners(mask, comp)
        
        w, h = self.plane_dim
        plane_corners = np.array([[-w/2, h/2], [w/2, h/2], [w/2, -h/2], [-w/2, -h/2]], dtype="float32")
        H, _ = cv2.findHomography(pixel_corners, plane_corners)

        world_corners = self.map_to_world(pixel_corners, H)
        pixel_center = np.mean(pixel_corners, axis=0)
        world_center = self.map_to_world(pixel_center, H)[0]

        v_idx, u_idx = np.where(mask)
        object_pixels_uv = np.stack([u_idx, v_idx], axis=1)
        world_coords = self.map_to_world(object_pixels_uv, H)

        obj_w = world_coords[:, 0].max() - world_coords[:, 0].min()
        obj_h = world_coords[:, 1].max() - world_coords[:, 1].min()

        print("="*50)
        print("GEOMETRY DEBUG INFORMATION")
        print("="*50)
        labels = ["Top-Left ", "Top-Right", "Bot-Right", "Bot-Left "]
        for i in range(4):
            print(f"{labels[i]} | Pixel: ({pixel_corners[i][0]:.1f}, {pixel_corners[i][1]:.1f}) "
                  f"-> World: ({world_corners[i][0]:.3f}, {world_corners[i][1]:.3f}, {world_corners[i][2]:.3f}) cm")
        
        print(f"Center      | Pixel: ({pixel_center[0]:.1f}, {pixel_center[1]:.1f}) "
              f"-> World: ({world_center[0]:.3f}, {world_center[1]:.3f}, {world_center[2]:.3f}) cm")
        
        print("\n" + "="*50)
        print("DIMENSION ANALYSIS")
        print("="*50)
        print(f"Actual Plane Dimensions: {self.plane_dim[0]} x {self.plane_dim[1]} cm")
        print(f"Detected Object B-Box:   {obj_w:.3f} x {obj_h:.3f} cm")
        print("="*50 + "\n")

        df = pd.DataFrame({
            'pixel_u': u_idx, 'pixel_v': v_idx,
            'x_world': world_coords[:, 0]*0.01, 'y_world': world_coords[:, 1]*0.01,
            'z_world': world_coords[:, 2]*0.01, 'alpha': 1
        })
        df.to_csv(self.output_dir / "object_mapping.csv", index=False)
        print(f"Done. Files saved to {self.output_dir}")

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\msk_pls_homgrph_cfg.json"
    if os.path.exists(CONFIG_PATH):
        IntegratedPhotometricMapper(CONFIG_PATH).process()