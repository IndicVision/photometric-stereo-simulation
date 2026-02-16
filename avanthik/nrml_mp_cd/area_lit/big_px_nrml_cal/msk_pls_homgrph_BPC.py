import os
# Force OpenEXR support
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy

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
        
        # Binning Configuration
        binning_cfg = self.cfg['image_processing'].get('binning', {})
        self.binning_enabled = binning_cfg.get('enabled', False)
        self.bin_width, self.bin_height = binning_cfg.get('cell_size', [1, 1])
        
        if self.binning_enabled:
            print(f"[INFO] Binning enabled: {self.bin_width}x{self.bin_height} pixel cells")

    def apply_binning(self, img):
        """
        Bin the image by averaging cells of size [bin_width, bin_height].
        Returns binned image and binning parameters for coordinate conversion.
        """
        if not self.binning_enabled:
            return img, 1, 1, 0, 0
        
        h, w = img.shape[:2]
        
        # Calculate binned dimensions
        binned_h = h // self.bin_height
        binned_w = w // self.bin_width
        
        # Calculate crops to make dimensions divisible
        crop_h = h - (binned_h * self.bin_height)
        crop_w = w - (binned_w * self.bin_width)
        
        # Crop image to make it divisible by bin size
        if crop_h > 0 or crop_w > 0:
            img = img[:h-crop_h, :w-crop_w]
            h, w = img.shape[:2]
        
        # Reshape and average
        if len(img.shape) == 2:  # Grayscale
            binned = img.reshape(binned_h, self.bin_height, 
                                binned_w, self.bin_width).mean(axis=(1, 3))
        else:  # Color
            binned = img.reshape(binned_h, self.bin_height, 
                                binned_w, self.bin_width, -1).mean(axis=(1, 3))
        
        return binned, self.bin_width, self.bin_height, crop_w, crop_h

    def binned_to_original_coords(self, binned_coords, bin_w, bin_h, crop_w, crop_h):
        """
        Convert binned pixel coordinates back to original image coordinates.
        Maps to the center of each bin cell.
        """
        original_coords = binned_coords.copy()
        original_coords[:, 0] = binned_coords[:, 0] * bin_w + bin_w / 2.0
        original_coords[:, 1] = binned_coords[:, 1] * bin_h + bin_h / 2.0
        return original_coords

    def load_and_composite(self):
        ext = self.cfg['image_processing']['file_extension']
        running_max = None
        
        print(f"Scanning for light_XXX{ext} files...")
        for i in range(1, 1000): 
            fname = f"light_{i:03d}{ext}"
            fpath = self.input_dir / fname
            if not fpath.exists(): break

            # Load with UNCHANGED to keep float32 or 16-bit data intact
            img = cv2.imread(str(fpath), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

            if img is None: 
                raise ValueError(f"Failed to load {fname}")

            # --- SMART DATA TYPE HANDLING ---
            if img.dtype == np.float32:
                img_float = img
            elif img.dtype == np.uint16:
                img_float = img.astype(np.float32) / 65535.0
            else:
                img_float = img.astype(np.float32) / 255.0

            # --- COLOR/CHANNEL HANDLING ---
            if len(img_float.shape) == 3: # BGR
                img_gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
            elif len(img_float.shape) == 4: # BGRA
                b, g, r, a = cv2.split(img_float)
                gray = 0.299*r + 0.587*g + 0.114*b
                img_gray = gray * a 
            else:
                img_gray = img_float

            if running_max is None: 
                running_max = img_gray
            else: 
                running_max = np.maximum(running_max, img_gray)
        
        if running_max is None: 
            raise FileNotFoundError(f"No images found in {self.input_dir}")
        
        # Apply binning to composite
        binned_comp, self.bin_w, self.bin_h, self.crop_w, self.crop_h = self.apply_binning(running_max)
        
        if self.binning_enabled:
            print(f"[INFO] Original size: {running_max.shape}, Binned size: {binned_comp.shape}")
        
        return binned_comp

    def generate_mask(self, composite): 
        # EXR/Float handling
        v_min, v_max = np.min(composite), np.percentile(composite, 99.9)
        
        if v_max == v_min:
            comp_8bit = (composite * 255).astype(np.uint8)
        else:
            comp_8bit = (np.clip((composite - v_min) / (v_max - v_min), 0, 1) * 255).astype(np.uint8)
        
        method = self.cfg['image_processing']['masking_method']
        
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
        
        # Scale mask back to original resolution for saving
        if self.binning_enabled:
            original_h = self.res_h - self.crop_h
            original_w = self.res_w - self.crop_w
            mask_fullres = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_fullres = mask
        
        cv2.imwrite(str(self.output_dir / "final_mask.png"), mask_fullres)
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
        
        # Convert binned corners to original coordinates for visualization
        if self.binning_enabled:
            corners_original = self.binned_to_original_coords(corners, self.bin_w, self.bin_h, 
                                                             self.crop_w, self.crop_h)
        else:
            corners_original = corners

        # Visualize on full resolution composite
        vis_8bit = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        
        if self.binning_enabled:
            original_h = self.res_h - self.crop_h
            original_w = self.res_w - self.crop_w
            vis_8bit = cv2.resize(vis_8bit, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        vis = cv2.cvtColor(vis_8bit, cv2.COLOR_GRAY2BGR)

        for i, pt in enumerate(corners_original):
            center = tuple(pt.astype(int))
            cv2.circle(vis, center, 10, (0, 0, 255), -1)
            cv2.putText(vis, f"C{i}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(str(self.output_dir / "detected_corners.png"), vis)
        
        return corners

    def order_points(self, pts):
        """
        Main dispatcher that selects the ordering method based on JSON config.
        """
        method = self.cfg['image_processing'].get('corner_method', 'sum_diff')
        
        if method == 'radial':
            return self._order_points_radial(pts)
        else:
            return self._order_points_sum_diff(pts)

    def _order_points_sum_diff(self, pts):
        """
        METHOD 2: Standard Sum/Difference (The original method).
        Fast and reliable for upright rectangles.
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        
        return rect

    def _order_points_radial(self, pts):
        """
        METHOD 1: Radial/Angular Sorting.
        Robust against 45-degree rotations (Diamond shapes).
        """
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        s = sorted_pts.sum(axis=1)
        tl_idx = np.argmin(s)
        ordered_ccw = np.roll(sorted_pts, -tl_idx, axis=0)
        
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = ordered_ccw[0] # TL
        rect[1] = ordered_ccw[3] # TR
        rect[2] = ordered_ccw[2] # BR
        rect[3] = ordered_ccw[1] # BL
        
        return rect

    def get_rigid_transform_matrix(self):
        """
        Calculates the Rotation Matrix R based on Azimuth and Elevation.
        """
        tilt_angle = 90.0 - self.elevation
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
        # 1. Load and Basic Mask (binned)
        comp = self.load_and_composite()
        otsu_mask = self.generate_mask(comp)
        
        # 2. Corner Detection (in binned space)
        pixel_corners = self.detect_corners(otsu_mask, comp)
        
        # 3. MASK CROPPING (Fix for Convex Bulge) - in binned space
        h, w = otsu_mask.shape
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        corners_int = pixel_corners.astype(np.int32)
        cv2.fillConvexPoly(poly_mask, corners_int, 255)
        
        final_cropped_mask = cv2.bitwise_and(otsu_mask.astype(np.uint8)*255, poly_mask)
        
        # Scale mask back to original resolution for saving
        if self.binning_enabled:
            original_h = self.res_h - self.crop_h
            original_w = self.res_w - self.crop_w
            mask_fullres = cv2.resize(final_cropped_mask, (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            mask_fullres = final_cropped_mask
        
        save_path = self.output_dir / "mask_cropped_to_corners.png"
        cv2.imwrite(str(save_path), mask_fullres)
        print(f"[INFO] Saved corner-constrained mask to: {save_path}")

        # 4. Geometry Setup
        w_dim, h_dim = self.plane_dim
        
        local_corners_2d = np.array([
            [-w_dim/2,  h_dim/2], # TL
            [ w_dim/2,  h_dim/2], # TR
            [ w_dim/2, -h_dim/2], # BR
            [-w_dim/2, -h_dim/2]  # BL
        ], dtype="float32")
        
        # 5. Compute Homography & Transforms (using binned corners)
        H, _ = cv2.findHomography(pixel_corners, local_corners_2d)
        
        # Debug Corners - convert to original coordinates for reporting
        if self.binning_enabled:
            pixel_corners_original = self.binned_to_original_coords(pixel_corners, 
                                                                    self.bin_w, self.bin_h,
                                                                    self.crop_w, self.crop_h)
        else:
            pixel_corners_original = pixel_corners
        
        local_corners_3d = self.map_to_local_plane(pixel_corners, H)
        local_corners_3d = local_corners_3d * -1.0
        world_corners = self.apply_rigid_transform(local_corners_3d)
        
        # 6. Map Pixels (in binned space)
        v_idx_binned, u_idx_binned = np.where(final_cropped_mask > 0)
        object_pixels_uv_binned = np.stack([u_idx_binned, v_idx_binned], axis=1)
        
        # Compute 3D coordinates for binned cells
        object_local_3d = self.map_to_local_plane(object_pixels_uv_binned, H)
        object_local_3d = object_local_3d * -1.0
        object_global_3d = self.apply_rigid_transform(object_local_3d)
        
        # For CSV output: convert binned coordinates to original image space
        # We'll store the CENTER pixel of each cell as the representative
        if self.binning_enabled:
            # Convert binned cell indices to the center pixel of each cell in original image
            u_idx_original = (u_idx_binned * self.bin_w + self.bin_w // 2).astype(np.float32)
            v_idx_original = (v_idx_binned * self.bin_h + self.bin_h // 2).astype(np.float32)
            
            print(f"[INFO] Storing {len(u_idx_binned)} cell centers in CSV (one per {self.bin_w}x{self.bin_h} cell)")
        else:
            u_idx_original = u_idx_binned.astype(np.float32)
            v_idx_original = v_idx_binned.astype(np.float32)

        # 7. Reporting
        print("="*50 + "\nGEOMETRY DEBUG INFORMATION\n" + "="*50)
        labels = ["TL", "TR", "BR", "BL"]
        for i in range(4):
            print(f"{labels[i]} | Pixel: ({pixel_corners_original[i][0]:.0f}, {pixel_corners_original[i][1]:.0f}) "
                  f"-> World: ({world_corners[i][0]:.3f}, {world_corners[i][1]:.3f}, {world_corners[i][2]:.3f}) cm")
        
        obj_w = np.linalg.norm(world_corners[0] - world_corners[1])
        obj_h = np.linalg.norm(world_corners[1] - world_corners[2])
        print(f"\nReconstructed Size: {obj_w:.3f} x {obj_h:.3f} cm (Target: {self.plane_dim[0]} x {self.plane_dim[1]})")

        # 8. Save CSV (Meters) - using original pixel coordinates
        df = pd.DataFrame({
            'pixel_u': u_idx_original, 
            'pixel_v': v_idx_original,
            'x_world': object_global_3d[:, 0] * 0.01,
            'y_world': object_global_3d[:, 1] * 0.01,
            'z_world': object_global_3d[:, 2] * 0.01,
            'alpha': 1
        })
        df.to_csv(self.output_dir / "object_mapping.csv", index=False)
        print(f"Done. Files saved to {self.output_dir}")

if __name__ == "__main__":
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\area_lit\big_px_nrml_cal\msk_pls_homgrph_BPC_cfg.json"
    if os.path.exists(CONFIG_PATH):
        IntegratedPhotometricMapper(CONFIG_PATH).process()
    else:
        print(f"Configuration file not found: {CONFIG_PATH}")
