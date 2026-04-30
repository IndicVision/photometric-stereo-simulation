import cv2
import numpy as np
import csv
from pathlib import Path
import json

class FastMobileMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self):
        print("Step 1: Simulating Mobile Max-Composite (16-bit Linear Aware)...")
        combined_img = None
        
        for path in self.image_paths:
            # FIX 1: Read the image in its raw 16-bit format, do not let OpenCV crush it
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not read {path}. Skipping.")
                continue
            
            # --- NEW FIX: Drop the Alpha channel if the image has 4 channels ---
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
                
            img_float = img.astype(np.float32)
            
            if combined_img is None:
                combined_img = img_float
            else:
                combined_img = np.maximum(combined_img, img_float)
                
        print("Step 2: Applying Temporary Gamma Stretch for Masking...")
        # FIX 2: Normalize the 16-bit data and stretch the shadows so Otsu can see them
        max_val = 65535.0 if combined_img.max() > 255 else 255.0
        combined_norm = combined_img / max_val
        
        # Apply Gamma 2.2 to brighten shadows visually
        combined_gamma = np.power(combined_norm, 1.0 / 2.2) 
        combined_8bit = (combined_gamma * 255.0).astype(np.uint8)
        
        if len(combined_8bit.shape) == 3:
            gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        else:
            gray = combined_8bit
        
        print("Step 3: Applying Noise Reduction (Anti-Cloth Texture)...")
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        

        print("Step 4: Applying Spatial Vignette (Darkening Edges)...")
        # Create a coordinate grid from -1 to 1
        h, w = blurred.shape
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        x, y = np.meshgrid(X, Y)
        
        # Calculate distance from the center
        radius = np.sqrt(x**2 + y**2)
        
        # Create a gradient that is 1.0 in the center and drops off towards the edges.
        # The '1.2' controls how wide the central safe zone is. 
        # (Increase to 1.5 if it cuts off your object, decrease to 1.0 if it still catches cloth)
        vignette = np.clip(1.5 - radius, 0.0, 1.0)
        print(f" -> Vignette gradient created with shape: {vignette.shape} and max value: {vignette.max():.2f}")
        print(" -> This will darken the edges of the image, helping Otsu focus on the center object and ignore cloth textures near the borders.")
        print(" -> Adjust the parameter in the code if you find that it's cutting off parts of your object or still catching too much cloth texture.")
        
        # Multiply the blurred image by this gradient
        blurred = (blurred * vignette).astype(np.uint8)
        
        # Save this out just so you can visually see what the math is doing
        cv2.imwrite(str(self.output_dir / "vignette_view.jpg"), blurred)


        print("Step 5: Calculating Otsu Threshold...")
        otsu_thresh_val, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f" -> Optimal Threshold found at: {otsu_thresh_val}")

        print("Step 6: Morphological Cleanup...")
        # CLOSE: Fills small black holes inside the white object
        kernel_close = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # OPEN: Removes small white noise (like lint or dust on the black cloth)
        kernel_open = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        print("Step 7: Isolating Largest Contour (The Object)...")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        
        if contours:
            # Assume the object is the largest bright thing in the frame
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            print("Warning: No objects found!")

        print("Step 8: Saving Outputs & Generating CSV...")
        # Generate the CSV mapping just like your original code
        ys, xs = np.where(final_mask > 0)

        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation is disabled in the config. Skipping CSV output.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            
            csv_path = self.output_dir / "valid_pixels.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        # Save visualizations
        cv2.imwrite(str(self.output_dir / "combined_max_image.jpg"), combined_8bit)
        cv2.imwrite(str(self.output_dir / "final_object_mask.png"), final_mask)
        
        print(f"Done! Outputs saved to {self.output_dir}")
        print(f"Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your existing JSON config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\trad_cv_msk_cfg.json"
    
    import os
    if os.path.exists(config_file):
        processor = FastMobileMasker(config_file)
        processor.process()
    else:
        print(f"Please update the config_file path. Could not find: {config_file}")