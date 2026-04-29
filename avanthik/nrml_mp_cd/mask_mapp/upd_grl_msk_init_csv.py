import json
import cv2
import numpy as np
import csv
from pathlib import Path

# AI Support Imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_SUPPORT = True
except ImportError:
    SAM_SUPPORT = False

class ImageProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.method = self.cfg.get('method', 'otsu').lower()
        
        # Initialize AI Model if requested
        self.predictor = None
        if self.method == 'ai':
            if not SAM_SUPPORT:
                raise ImportError("SAM dependencies (segment_anything, torch) are not installed. Cannot use 'ai' method.")
            print("Initializing SAM Model...")
            model_path = self.cfg.get('sam_model_path', '')
            model_type = self.cfg.get('sam_model_type', 'vit_b')
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = SamPredictor(sam)
        
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
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)
        
        # 2. Generate Object Mask based on JSON method
        if self.method == 'manual':
            percent = self.cfg.get('manual_threshold_percent', 10.0)
            thresh_val = (percent / 100.0) * 255.0
            _, raw_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            print(f"Using MANUAL threshold: {thresh_val:.2f} ({percent}%)")
            
        elif self.method == 'ai':
            print("Using AI (SAM) for masking...")
            image_rgb = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            
            h, w = combined_8bit.shape[:2]
            
            # Fetch padding percent from config, default to 10%
            pad_percent = self.cfg.get('sam_box_padding_percent', 10.0) / 100.0
            pad_x, pad_y = int(w * pad_percent), int(h * pad_percent)
            
            input_box = np.array([pad_x, pad_y, w - pad_x, h - pad_y])
            print(f"SAM Prompt: Using bounding box inset by {pad_percent*100}% -> {input_box}")
            
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            raw_mask = (masks[0] * 255).astype(np.uint8)
            print(f"SAM prediction complete. Confidence Score: {scores[0]:.3f}")
            
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
            final_mask = raw_mask.copy()
            print("Hole filling: DISABLED")

        # 4.5 Conditionally keep ONLY the largest contour
        keep_largest = self.cfg.get('keep_largest_contour_only', False)
        if keep_largest:
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_mask = np.zeros_like(final_mask)
                cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                final_mask = cv2.bitwise_and(final_mask, largest_mask)
                print("Keep largest contour only: ENABLED")
            else:
                print("Keep largest contour only: No contours found to filter.")
        else:
            print("Keep largest contour only: DISABLED")
        
        # 5. Generate CSV with specific coordinate formatting
        ys, xs = np.where(final_mask > 0)
        valid_pixel_count = len(ys)
        
        # Fetch the target world coordinates from the JSON config
        world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
        x_w, y_w, z_w = world_coords
        
        csv_path = self.output_dir / "valid_pixels.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
            for u, v in zip(xs, ys):
                # Write the dynamic coordinates fetched from JSON
                writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        # Save outputs
        cv2.imwrite(str(self.output_dir / "combined_image.jpg"), combined_8bit)
        cv2.imwrite(str(self.output_dir / "object_mask.png"), final_mask)
        
        print(f"Done! Outputs saved to {self.output_dir}")
        print(f"Total valid pixels found and written to CSV: {valid_pixel_count}")
        print(f"Pixels assigned world coordinates: X={x_w}, Y={y_w}, Z={z_w}")

if __name__ == "__main__":
    processor = ImageProcessor(r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\upd_grl_msk_init_csv_cfg.json")
    processor.process()