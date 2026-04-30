import cv2
import numpy as np
import csv
import os
from pathlib import Path
import json

# Offline fix for the AI model
os.environ["U2NET_HOME"] = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test"
from rembg import remove, new_session

class HybridMaskerISNet:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading AI Brain: isnet-general-use (Heavy High-Accuracy Model)...")
        self.session = new_session("isnet-general-use")
        
    def process(self):
        print("Step 1: Creating Max-Composite & Gamma Stretch...")
        combined_img = None
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

            # --- NEW FIX: Drop the Alpha channel if the image has 4 channels ---
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
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        
        print("Step 2: AI Brain - Generating Rough Context Mask...")
        output_rgba = remove(combined_8bit, session=self.session)
        ai_mask_rough = output_rgba[:, :, 3]

        print("Step 3: Creating the 'Safe Zone' (Dilating AI Mask)...")
        # Expand the AI mask by ~40 pixels. 
        kernel_dilate = np.ones((41, 41), np.uint8)
        safe_zone = cv2.dilate(ai_mask_rough, kernel_dilate, iterations=1)
        
        print("Step 4: Traditional Scalpel - Calculating Otsu Threshold...")
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        otsu_thresh_val, trad_cv_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        print("Step 5: The Hybrid Merge...")
        hybrid_mask = cv2.bitwise_and(trad_cv_mask, safe_zone)

        print("Step 6: Morphological Cleanup...")
        kernel_close = np.ones((9, 9), np.uint8)
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((5, 5), np.uint8)
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(hybrid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(hybrid_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        print("Step 7: Saving Outputs...")
        ys, xs = np.where(final_mask > 0)
        
        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation is disabled in the config. Skipping CSV output.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            
            csv_path = self.output_dir / "valid_pixels_hybrid_isnet.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        cv2.imwrite(str(self.output_dir / "1_hybrid_safe_zone.jpg"), safe_zone)
        cv2.imwrite(str(self.output_dir / "2_hybrid_final_mask.png"), final_mask)
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your existing config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\hybd_trad_INGU_msk_cfg.json"
    if os.path.exists(config_file):
        processor = HybridMaskerISNet(config_file)
        processor.process()
    else:
        print("Config not found!")