import json
import cv2
import numpy as np
import csv
from pathlib import Path

class ImageProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self):
        print("Processing images...")
        combined_img = None
        
        # 1. Generate Combined Image
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read {path}. Skipping.")
                continue
            
            if combined_img is None:
                combined_img = img.astype(np.float32)
            else:
                if img.shape[:2] != combined_img.shape[:2]:
                    img = cv2.resize(img, (combined_img.shape[1], combined_img.shape[0]))
                combined_img = np.maximum(combined_img, img.astype(np.float32))
        
        if combined_img is None:
            print("Error: No valid images found. Check your JSON paths.")
            return
            
        combined_8bit = np.clip(combined_img, 0, 255).astype(np.uint8)
        
        # 2. Generate Object Mask based on JSON method
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        method = self.cfg.get('method', 'otsu').lower()
        
        if method == 'manual':
            percent = self.cfg.get('manual_threshold_percent', 10.0)
            thresh_val = (percent / 100.0) * 255.0
            _, raw_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            print(f"Using MANUAL threshold: {thresh_val:.2f} ({percent}%)")
        else:
            # Default to Otsu
            otsu_thresh_val, raw_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_percent = (otsu_thresh_val / 255.0) * 100.0
            print(f"Using OTSU threshold: {otsu_thresh_val:.2f} ({otsu_percent:.2f}%)")

        # 3. Conditionally remove outliers (isolated pixels/small clusters)
        remove_outliers = self.cfg.get('remove_outliers', True)
        if remove_outliers:
            k_size = int(self.cfg.get('outlier_kernel_size', 5))
            kernel = np.ones((k_size, k_size), np.uint8)
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
            print(f"Outlier removal: ENABLED (Kernel size: {k_size}x{k_size})")
        else:
            print("Outlier removal: DISABLED")

        # 4. Conditionally fill holes based on JSON
        fill_holes = self.cfg.get('fill_holes', True)
        if fill_holes:
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(raw_mask)
            cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)
            print("Hole filling: ENABLED")
        else:
            final_mask = raw_mask
            print("Hole filling: DISABLED")
        
        # 5. Generate CSV with specific coordinate formatting
        # Grab the y (v) and x (u) coordinates of all white pixels
        ys, xs = np.where(final_mask > 0)
        valid_pixel_count = len(ys)
        
        csv_path = self.output_dir / "valid_pixels.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha']) # Header
            
            # Write the formatted row for every valid pixel found
            for u, v in zip(xs, ys):
                writer.writerow([u, v, 0.0, 0.0, 0.0, 1.0])
                
        # Save the image outputs
        cv2.imwrite(str(self.output_dir / "combined_image.jpg"), combined_8bit)
        cv2.imwrite(str(self.output_dir / "object_mask.png"), final_mask)
        
        print(f"Done! Outputs saved to {self.output_dir}")
        print(f"Total valid pixels found and written to CSV: {valid_pixel_count}")

if __name__ == "__main__":
    processor = ImageProcessor(r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\grl_msk_init_csv_cfg.json")
    processor.process()