import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_SUPPORT = True
except ImportError:
    SAM_SUPPORT = False

class UnifiedMasker:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.input_dir = Path(self.cfg['paths']['input_dir'])
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_res = None 
        
        self.predictor = None
        if self.cfg['settings']['method'] == 'ai' and SAM_SUPPORT:
            self.init_ai()

    def init_ai(self):
        model_path = self.cfg['settings']['sam_model_path']
        sam = sam_model_registry["vit_b"](checkpoint=model_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(sam)

    def load_image(self, path):
        ext = path.suffix.lower()
        if ext in ['.cr2', '.nef', '.dng']:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(gamma=(1,1), use_camera_wb=True, no_auto_bright=True)
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        elif ext == '.exr':
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        else:
            img = cv2.imread(str(path)).astype(np.float32)

        if self.current_res is None:
            self.current_res = (img.shape[1], img.shape[0])
            print(f"Native Resolution set to: {self.current_res}")
        
        if (img.shape[1], img.shape[0]) != self.current_res:
            img = cv2.resize(img, self.current_res)
        return img

    def process(self):
        image_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in {".jpg", ".cr2", ".png", ".exr", ".nef", ".dng"}]
        
        # Add this safeguard:
        if not image_files:
            print(f"No valid images found in {self.input_dir}. Exiting.")
            return

        running_max_img = None

        print("Generating Composite Image...")
        for img_file in tqdm(image_files):
            img = self.load_image(img_file)
            if running_max_img is None: running_max_img = img
            else: running_max_img = np.maximum(running_max_img, img)

        composite_8bit = self.normalize_for_vis(running_max_img)
        
        method = self.cfg['settings']['method']
        if method == 'ai' and self.predictor:
            final_mask = self.run_ai_mask(composite_8bit)
        elif method == 'manual':
            final_mask = self.run_manual_mask(composite_8bit)
        else:
            final_mask = self.run_otsu(composite_8bit)

        # Restored outputs
        b, g, r = cv2.split(composite_8bit)
        object_transparent = cv2.merge([b, g, r, final_mask])

        cv2.imwrite(str(self.output_dir / "1_debug_composite.jpg"), composite_8bit)
        cv2.imwrite(str(self.output_dir / "2_final_mask.png"), final_mask)
        cv2.imwrite(str(self.output_dir / "3_object_only.png"), object_transparent)
        print(f"Masking complete. Outputs saved to {self.output_dir}")

    def run_ai_mask(self, img_8bit):
        self.predictor.set_image(cv2.cvtColor(img_8bit, cv2.COLOR_BGR2RGB))
        h, w = img_8bit.shape[:2]
        
        # Create a bounding box slightly inset from the edges (10% padding)
        # Format: [x_min, y_min, x_max, y_max]
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        input_box = np.array([pad_x, pad_y, w - pad_x, h - pad_y])
        
        # Pass the box to the predictor instead of points
        masks, _, _ = self.predictor.predict(
            point_coords=None, 
            point_labels=None, 
            box=input_box[None, :], # SAM expects a batch dimension, hence [None, :]
            multimask_output=False
        )
        return (masks[0] * 255).astype(np.uint8)

    def run_manual_mask(self, img_8bit):
        bg_path = self.cfg['paths']['background_image']
        bg = cv2.imread(bg_path, 0)
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, cv2.resize(bg, (gray.shape[1], gray.shape[0])))
        _, mask = cv2.threshold(diff, self.cfg['settings']['manual_threshold'], 255, cv2.THRESH_BINARY)
        return mask

    def run_otsu(self, img_8bit):
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY) if len(img_8bit.shape)==3 else img_8bit
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def normalize_for_vis(self, img_float):
        img = np.nan_to_num(img_float)
        return (img / np.max(img) * 255).astype(np.uint8) if np.max(img) > 0 else img.astype(np.uint8)

if __name__ == "__main__":
    UnifiedMasker(r"C:\Users\chand\OneDrive\Desktop\4th_march_code\unified_masker_cfg.json").process()