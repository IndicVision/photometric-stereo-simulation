import cv2
import numpy as np
import csv
import os
import torch
from pathlib import Path
import json

# Offline fix for the u2netp model
os.environ["U2NET_HOME"] = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test"
from rembg import remove, new_session

# MobileSAM import
from mobile_sam import sam_model_registry, SamPredictor

class HybridMobileSAM:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.image_paths = self.cfg['image_paths']
        self.output_dir = Path(self.cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading AI Scout: u2netp (Pocket Mobile Model)...")
        self.u2netp_session = new_session("u2netp")
        
        print("Loading AI Sniper: MobileSAM...")
        model_type = "vit_t"
        sam_checkpoint = self.cfg.get('mobile_sam_weights', 'mobile_sam.pt')
        
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"MobileSAM weights not found at {sam_checkpoint}. Please download them.")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()
        self.sam_predictor = SamPredictor(self.mobile_sam)
        
    def process(self):
        print("Step 1: Creating Max-Composite & Gamma Stretch...")
        combined_img = None
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: continue

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
        
        print("Step 2: AI Scout - Finding the Object's Location...")
        output_rgba = remove(combined_8bit, session=self.u2netp_session)
        ai_mask_rough = output_rgba[:, :, 3]

        print("Step 3: Calculating Dynamic Bounding Box...")
        contours, _ = cv2.findContours(ai_mask_rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("ERROR: u2netp could not find any object.")
            return
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add 5% padding so MobileSAM can clearly see the background transition
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        
        x_min = max(0, x - pad_x)
        y_min = max(0, y - pad_y)
        x_max = min(combined_8bit.shape[1], x + w + pad_x)
        y_max = min(combined_8bit.shape[0], y + h + pad_y)
        
        input_box = np.array([x_min, y_min, x_max, y_max])
        
        # Save the box for visual debugging
        debug_img = combined_8bit.copy()
        cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imwrite(str(self.output_dir / "1_scout_bounding_box.jpg"), debug_img)

        print("Step 4: AI Sniper - MobileSAM Pixel-Perfect Masking...")
        image_rgb = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        
        # masks is a boolean array. Convert to 0-255 image.
        sam_mask = (masks[0] * 255).astype(np.uint8)

        print("Step 5: Morphological Cleanup...")
        # SAM is extremely clean, but a tiny 5x5 close kernel patches any micro-holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_CLOSE, kernel_close)

        print("Step 6: Saving Outputs...")
        ys, xs = np.where(final_mask > 0)
        
        generate_csv = self.cfg.get('generate_csv', True)
        if not generate_csv:
            print("CSV generation is disabled in the config. Skipping CSV output.")     
        else:
            world_coords = self.cfg.get('world_coordinates', [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            csv_path = self.output_dir / "valid_pixels_mobile_sam.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['pixel_u', 'pixel_v', 'x_world', 'y_world', 'z_world', 'alpha'])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
                
        cv2.imwrite(str(self.output_dir / "2_mobile_sam_mask.png"), final_mask)
        print(f"Done! Total valid pixels found: {len(ys)}")

if __name__ == "__main__":
    # Point this to your new JSON config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\nrml_mp_cd\mask_mapp\mbl_msk_test\hybd_mbSAM_u2netp_msk_cfg.json"
    if os.path.exists(config_file):
        processor = HybridMobileSAM(config_file)
        processor.process()
    else:
        print("Config not found!")