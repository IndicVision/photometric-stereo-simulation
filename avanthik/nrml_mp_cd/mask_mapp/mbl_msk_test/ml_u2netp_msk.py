import cv2
import numpy as np
import csv
from pathlib import Path
import json
from rembg import remove, new_session

class MLMobileMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the lightweight mobile-friendly model (u2netp)
        print("Loading lightweight ML model (u2netp)...")
        self.session = new_session("u2netp")
        
    def process(self):
        print("Step 1: Creating Max-Composite...")
        combined_img = None
        
        for path in self.image_paths:
            # We can use standard 8-bit loading here because the ML model 
            # is smart enough to understand shape without extreme gamma stretching.
            img = cv2.imread(path)
            if img is None: continue
                
            # --- NEW FIX: Drop the Alpha channel if the image has 4 channels ---
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
                
            if combined_img is None:
                combined_img = img.astype(np.float32)
            else:
                combined_img = np.maximum(combined_img, img.astype(np.float32))
                
        combined_8bit = np.clip(combined_img, 0, 255).astype(np.uint8)
        
        print("Step 2: Running AI Segmentation...")
        # rembg automatically finds the salient object and removes the background
        # We pass it the BGR image and it returns an RGBA image where the background is transparent
        output_rgba = remove(combined_8bit, session=self.session)
        
        # Extract the alpha channel as our mask
        mask = output_rgba[:, :, 3]

        print("Step 3: Morphological Cleanup & Extraction...")
        # Just in case the ML leaves tiny holes, we fill them
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Keep only the largest object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        print("Step 4: Saving Outputs & Generating CSV...")
        ys, xs = np.where(final_mask > 0)

        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation is disabled in the config. Skipping CSV output.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            
            csv_path = self.output_dir / "valid_pixels_ml.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        cv2.imwrite(str(self.output_dir / "ml_combined_input.jpg"), combined_8bit)
        cv2.imwrite(str(self.output_dir / "ml_final_mask.png"), final_mask)
        
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your existing ml_u2netp_msk_cfg.json
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\ml_u2netp_msk_cfg.json"
    
    import os
    if os.path.exists(config_file):
        processor = MLMobileMasker(config_file)
        processor.process()
    else:
        print("Config not found!")