import cv2
import numpy as np
import csv
import os
from pathlib import Path
import json

# Offline fix for the AI model
os.environ["U2NET_HOME"] = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test"
from rembg import remove, new_session

class HybridTripleMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading AI Brain: u2netp...")
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
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        
        print("Step 2: AI Brain - Finding Definite Background...")
        output_rgba = remove(combined_8bit, session=self.session)
        ai_mask_rough = output_rgba[:, :, 3]

        # Dilate the AI mask. Everything OUTSIDE this is Definite Background.
        kernel_dilate = np.ones((41, 41), np.uint8)
        safe_zone = cv2.dilate(ai_mask_rough, kernel_dilate, iterations=1)
        definite_bgd = cv2.bitwise_not(safe_zone)
        
        print("Step 3: Otsu - Finding Definite Foreground Core...")
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Mask Otsu with the AI safe zone to remove the scale/edge lights
        otsu_clean = cv2.bitwise_and(otsu_mask, safe_zone)
        
        # Erode (shrink) the Otsu mask. Everything INSIDE this is Definite Foreground.
        # This keeps Otsu away from the delicate edges.
        kernel_erode = np.ones((25, 25), np.uint8)
        definite_fgd = cv2.erode(otsu_clean, kernel_erode, iterations=1)
        
        print("Step 4: GrabCut - Snapping the Final Curves...")
        # Initialize the GrabCut mask with "Probable Background" (cv2.GC_PR_BGD = 2)
        gc_mask = np.full(combined_8bit.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        
        # Add "Probable Foreground" where Otsu guessed there was an object (cv2.GC_PR_FGD = 3)
        gc_mask[otsu_clean == 255] = cv2.GC_PR_FGD
        
        # Lock in the Definite Background from AI (cv2.GC_BGD = 0)
        gc_mask[definite_bgd == 255] = cv2.GC_BGD
        
        # Lock in the Definite Foreground from Shrunk Otsu (cv2.GC_FGD = 1)
        gc_mask[definite_fgd == 255] = cv2.GC_FGD
        
        # Run GrabCut (Only 3 iterations needed because our initial guess is so good)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(combined_8bit, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        
        # Extract the final mask (Pixels marked as Definite FGD or Probable FGD)
        final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        print("Step 5: Morphological Cleanup...")
        kernel_close = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(final_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        print("Step 6: Saving Outputs...")
        ys, xs = np.where(clean_mask > 0)
        
        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation disabled.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            csv_path = self.output_dir / "valid_pixels_triple_hybrid.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        # Optional: Save the trimap visually so you can see how it works
        trimap_vis = (gc_mask * 85).astype(np.uint8) # Scale 0-3 to 0-255 for viewing
        cv2.imwrite(str(self.output_dir / "trimap_visualization.png"), trimap_vis)
        cv2.imwrite(str(self.output_dir / "triple_hybrid_mask.png"), clean_mask)
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your u2netp config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\trad_grabcut_u2netp_msk_cfg.json"
    if os.path.exists(config_file):
        processor = HybridTripleMasker(config_file)
        processor.process()
    else:
        print("Config not found!")