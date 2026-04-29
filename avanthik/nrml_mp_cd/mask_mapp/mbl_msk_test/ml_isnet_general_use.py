import cv2
import numpy as np
import csv
import os
from pathlib import Path
import json

# --- OFFLINE FIX ---
# Set the environment variable BEFORE importing rembg.
# Point this to the folder where you placed the downloaded .onnx file.
os.environ["U2NET_HOME"] = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test"

from rembg import remove, new_session

class MLMobileMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Using a much smarter model for general objects.
        # Because we set U2NET_HOME, it will look for it locally and NOT download.
        print("Loading local ML model (isnet-general-use)...")
        self.session = new_session("isnet-general-use")
        
    def process(self):
        print("Step 1: Creating Max-Composite...")
        combined_img = None
        
        for path in self.image_paths:
            # Load in 16-bit to preserve data
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
                
        print("Step 1.5: Applying Gamma 2.2 for AI Compatibility...")
        # AI models expect sRGB (Gamma 2.2) images, not Linear. 
        # We must stretch it so it looks like a normal photo to the AI.
        max_val = 65535.0 if combined_img.max() > 255 else 255.0
        combined_norm = combined_img / max_val
        combined_gamma = np.power(combined_norm, 1.0 / 2.2) 
        
        # Convert to 8-bit for the AI
        combined_8bit = (combined_gamma * 255.0).astype(np.uint8)
        
        print("Step 2: Running AI Segmentation...")
        # rembg automatically finds the salient object
        output_rgba = remove(combined_8bit, session=self.session)
        
        # Extract the alpha channel as our mask
        mask = output_rgba[:, :, 3]

        print("Step 3: Morphological Cleanup & Extraction...")
        # Close tiny holes
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
                
        # Save visualizations
        cv2.imwrite(str(self.output_dir / "ml_combined_input_gamma.jpg"), combined_8bit)
        cv2.imwrite(str(self.output_dir / "ml_final_mask.png"), final_mask)
        
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your existing JSON config
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\ml_isnet_general_use_cfg.json"
    
    if os.path.exists(config_file):
        processor = MLMobileMasker(config_file)
        processor.process()
    else:
        print("Config not found!")