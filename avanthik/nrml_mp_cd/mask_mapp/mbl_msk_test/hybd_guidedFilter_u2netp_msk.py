import cv2
import numpy as np
import csv
import os
from pathlib import Path
import json

# Offline fix for the AI model
os.environ["U2NET_HOME"] = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test"
from rembg import remove, new_session

class HybridGuidedFilterMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading AI Brain: u2netp (Pocket Mobile Model)...")
        self.session = new_session("u2netp")
        
    def process(self):
        print("Step 1: Creating Max-Composite & Gamma Stretch...")
        combined_img = None
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # Drop the Alpha channel if the image has 4 channels
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            img_float = img.astype(np.float32)
            if combined_img is None:
                combined_img = img_float
            else:
                combined_img = np.maximum(combined_img, img_float)
                
        max_val = 65535.0 if combined_img.max() > 255 else 255.0
        combined_gamma = np.power(combined_img / max_val, 1.0 / 2.2) 
        combined_8bit = (combined_gamma * 255.0).astype(np.uint8)
        
        # We use the grayscale version as our "Guide" image
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        
        print("Step 2: AI Brain - Generating Rough Context Mask...")
        output_rgba = remove(combined_8bit, session=self.session)
        ai_mask_rough = output_rgba[:, :, 3]

        print("Step 3: Edge-Aware Guided Filtering...")
        # We slightly blur the AI mask first so the Guided Filter has a gradient to push around
        ai_mask_blurred = cv2.GaussianBlur(ai_mask_rough, (21, 21), 0)
        
        # Apply the Guided Filter
        # radius: How far to look for edges (20-30 is good for high-res images)
        # eps: Regularization. Smaller means it snaps harder to the guide's edges.
        try:
            refined_mask = cv2.ximgproc.guidedFilter(guide=gray, src=ai_mask_blurred, radius=30, eps=0.01)
        except AttributeError:
            print("ERROR: cv2.ximgproc not found! Please run: pip install opencv-contrib-python")
            return
            
        print("Step 4: Binarizing and Morphological Cleanup...")
        # The filter outputs a soft mask. We threshold it exactly at the middle to get a crisp edge.
        _, binary_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Keep only the largest contour to ensure no isolated noise remains
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(binary_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        print("Step 5: Saving Outputs...")
        ys, xs = np.where(final_mask > 0)
        
        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation is disabled in the config. Skipping CSV output.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            
            csv_path = self.output_dir / "valid_pixels_guided_filter.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        # Save a visual comparison of the raw AI vs the Guided Filter output
        cv2.imwrite(str(self.output_dir / "1_raw_ai_mask.jpg"), ai_mask_rough)
        cv2.imwrite(str(self.output_dir / "2_guided_filter_mask.png"), final_mask)
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your existing u2netp config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\hybd_guidedFilter_u2netp_msk_cfg.json"
    if os.path.exists(config_file):
        processor = HybridGuidedFilterMasker(config_file)
        processor.process()
    else:
        print("Config not found!")