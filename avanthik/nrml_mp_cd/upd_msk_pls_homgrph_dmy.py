import os
# Force OpenEXR support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy

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
        
        # Plane Configuration
        self.plane_dim = np.array(self.cfg['plane']['dimensions_cm'])
        self.center_pos = np.array(self.cfg['plane']['center_position_cm'])
        self.elevation = self.cfg['plane']['elevation_deg']
        self.azimuth = self.cfg['plane']['azimuth_deg']

    def load_and_composite(self):
        ext = self.cfg['image_processing']['file_extension']
        running_max = None
        
        # Pre-calculate expected file sizes
        num_pixels = self.res_w * self.res_h
        size_8bit = num_pixels       # 1 byte/pixel (Grayscale)
        size_16bit = num_pixels * 2  # 2 bytes/pixel (16-bit Grayscale)
        size_24bit = num_pixels * 3  # 3 bytes/pixel (8-bit RGB)

        print(f"Scanning for light_XXX{ext} files...")
        for i in range(1, 1000): 
            fname = f"light_{i:03d}{ext}"
            fpath = self.input_dir / fname
            if not fpath.exists(): break

            # --- CASE 1: Headerless RAW (.raw) ---
            if ext.upper() == '.RAW':
                file_size = os.path.getsize(str(fpath))
                
                if file_size == size_24bit:
                    # NEW: Load 8-bit RGB (3 channels)
                    raw_data = np.fromfile(str(fpath), dtype=np.uint8)
                    img_rgb = raw_data.reshape((self.res_h, self.res_w, 3))
                    
                    # Convert RGB to Grayscale (float 0-1)
                    img_rgb = img_rgb.astype(np.float32) / 255.0
                    # Standard luminance: 0.299*R + 0.587*G + 0.114*B
                    img = (0.299 * img_rgb[:,:,0] + 
                           0.587 * img_rgb[:,:,1] + 
                           0.114 * img_rgb[:,:,2])
                           
                elif file_size == size_16bit:
                    img = np.fromfile(str(fpath), dtype=np.uint16).reshape((self.res_h, self.res_w))
                    img = img.astype(np.float32) / 65535.0
                elif file_size == size_8bit:
                    img = np.fromfile(str(fpath), dtype=np.uint8).reshape((self.res_h, self.res_w))
                    img = img.astype(np.float32) / 255.0
                else:
                    raise ValueError(f"File {fname} size {file_size} does not match resolution {self.res_w}x{self.res_h} (Expected {size_8bit}, {size_16bit}, or {size_24bit})")
            
            # --- CASE 2: Camera RAW Formats (.CR2, .NEF, etc) ---
            elif ext.upper() in ['.CR2', '.NEF', '.DNG']:
                if not RAW_SUPPORT: raise ImportError("Rawpy needed for RAW files.")
                with rawpy.imread(str(fpath)) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
                    img = (rgb.astype(np.float32) / 65535.0)
                    if len(img.shape) == 3:
                        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

            # --- CASE 3: Standard Image Formats ---
            else:
                img = cv2.imread(str(fpath))
                if img is None: raise ValueError(f"Failed to load {fname}")
                img = img.astype(np.float32) / 255.0
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if running_max is None: running_max = img
            else: running_max = np.maximum(running_max, img)
        
        if running_max is None: raise FileNotFoundError(f"No images found in {self.input_dir}")
        return running_max

    def generate_mask(self, composite):
        method = self.cfg['image_processing']['masking_method']
        comp_8bit = (composite * 255).astype(np.uint8)
        
        if method == "otsu":
            _, mask = cv2.threshold(comp_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "manual":
            thresh = self.cfg['image_processing']['manual_threshold']
            _, mask = cv2.threshold(comp_8bit, thresh, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown masking method: {method}")
            
        # Clean up noise
        erosion_iter = self.cfg['image_processing'].get('erosion_iterations', 0)
        dilation_iter = self.cfg['image_processing'].get('dilation_iterations', 0)
        
        if erosion_iter > 0: mask = cv2.erode(mask, None, iterations=erosion_iter)
        if dilation_iter > 0: mask = cv2.dilate(mask, None, iterations=dilation_iter)
        
        cv2.imwrite(str(self.output_dir / "final_mask.png"), mask)
        return mask > 0

    def detect_corners(self, mask, composite):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: raise ValueError("No contours found.")
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        
        corners = None
        used_method = "None"
        for eps_factor in [0.01, 0.015, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
            if len(approx) == 4:
                corners = self.order_points(approx.reshape(4, 2))
                used_method = f"Polygon Approximation (Eps: {eps_factor*100}%)"
                break
        
        if corners is None:
            rect = cv2.minAreaRect(largest)
            corners = self.order_points(cv2.boxPoints(rect))
            used_method = "Min Area Rect (Fallback)"

        print(f"\n[DEBUG] Corner Detection Method: {used_method}")

        vis = cv2.cvtColor((composite * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, pt in enumerate(corners):
            cv2.circle(vis, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(vis, f"C{i}", tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(self.output_dir / "detected_corners.png"), vis)
        
        return corners

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def get_rigid_transform_matrix(self):
        """
        Calculates the Rotation Matrix R based on Azimuth and Elevation.
        FIX: Changed rotation order to 'zy' to ensure Tilt applies to the X-Axis (Width).
        """
        tilt_angle = 90.0 - self.elevation
        
        # 'zy' = Rotate Z (Azimuth) -> Rotate Y (Tilt)
        # Rotating around Y moves the X coordinates into Z.
        # This ensures Z is proportional to the Width (X) dimension.
        r = R_scipy.from_euler('zy', [self.azimuth, tilt_angle], degrees=True)
        return r.as_matrix()

    def map_to_local_plane(self, points_uv, H):
        points_uv = np.array(points_uv, dtype="float32")
        if points_uv.ndim == 1: points_uv = points_uv.reshape(1, 2)
        ones = np.ones((len(points_uv), 1))
        pixels_homo = np.hstack([points_uv, ones])
        local_2d_homo = (H @ pixels_homo.T).T
        local_2d = local_2d_homo[:, :2] / local_2d_homo[:, 2:3]
        return np.hstack([local_2d, np.zeros((len(local_2d), 1))])

    def apply_rigid_transform(self, local_points_3d):
        R = self.get_rigid_transform_matrix()
        T = self.center_pos
        return (R @ local_points_3d.T).T + T

    def process(self):
        # 1. Load and Basic Mask
        comp = self.load_and_composite()
        otsu_mask = self.generate_mask(comp)
        
        # 2. Corner Detection
        pixel_corners = self.detect_corners(otsu_mask, comp)
        
        # 3. MASK CROPPING (Fix for Convex Bulge)
        h, w = otsu_mask.shape
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        corners_int = pixel_corners.astype(np.int32)
        cv2.fillConvexPoly(poly_mask, corners_int, 255)
        
        # Only keep pixels inside the corner polygon
        final_cropped_mask = cv2.bitwise_and(otsu_mask.astype(np.uint8)*255, poly_mask)
        
        # Save verified mask
        save_path = self.output_dir / "mask_cropped_to_corners.png"
        cv2.imwrite(str(save_path), final_cropped_mask)
        print(f"[INFO] Saved corner-constrained mask to: {save_path}")

        # 4. Geometry Setup
        # Dims[0] (3.3) is Local X. Dims[1] (2.9) is Local Y.
        w_dim, h_dim = self.plane_dim
        
        local_corners_2d = np.array([
            [-w_dim/2,  h_dim/2], # TL
            [ w_dim/2,  h_dim/2], # TR
            [ w_dim/2, -h_dim/2], # BR
            [-w_dim/2, -h_dim/2]  # BL
        ], dtype="float32")
        
        # 5. Compute Homography & Transforms
        H, _ = cv2.findHomography(pixel_corners, local_corners_2d)
        
        # Debug Corners
        local_corners_3d = self.map_to_local_plane(pixel_corners, H)
        world_corners = self.apply_rigid_transform(local_corners_3d)
        
        # 6. Map Pixels
        v_idx, u_idx = np.where(final_cropped_mask > 0)
        object_pixels_uv = np.stack([u_idx, v_idx], axis=1)
        
        object_local_3d = self.map_to_local_plane(object_pixels_uv, H)
        object_global_3d = self.apply_rigid_transform(object_local_3d)

        # 7. Reporting
        print("="*50 + "\nGEOMETRY DEBUG INFORMATION\n" + "="*50)
        labels = ["TL", "TR", "BR", "BL"]
        for i in range(4):
            print(f"{labels[i]} | Pixel: ({pixel_corners[i][0]:.0f}, {pixel_corners[i][1]:.0f}) "
                  f"-> World: ({world_corners[i][0]:.3f}, {world_corners[i][1]:.3f}, {world_corners[i][2]:.3f}) cm")
        
        obj_w = np.linalg.norm(world_corners[0] - world_corners[1])
        obj_h = np.linalg.norm(world_corners[1] - world_corners[2])
        print(f"\nReconstructed Size: {obj_w:.3f} x {obj_h:.3f} cm (Target: {self.plane_dim[0]} x {self.plane_dim[1]})")

        # 8. Save CSV (Meters)
        df = pd.DataFrame({
            'pixel_u': u_idx, 'pixel_v': v_idx,
            'x_world': object_global_3d[:, 0] * 0.01,
            'y_world': object_global_3d[:, 1] * 0.01,
            'z_world': object_global_3d[:, 2] * 0.01,
            'alpha': 1
        })
        df.to_csv(self.output_dir / "object_mapping.csv", index=False)
        print(f"Done. Files saved to {self.output_dir}")

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\upd_msk_pls_homgrph_dmy_cfg.json"
    if os.path.exists(CONFIG_PATH):
        IntegratedPhotometricMapper(CONFIG_PATH).process()
    else:
        print(f"Configuration file not found: {CONFIG_PATH}")