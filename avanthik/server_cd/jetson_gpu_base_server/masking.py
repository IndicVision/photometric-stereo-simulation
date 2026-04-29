import cv2
import numpy as np
import csv
import os
from pathlib import Path

os.environ["U2NET_HOME"] = os.path.expanduser("~/.u2net")
from rembg import remove, new_session


class HybridMaskerU2NetP:
    """
    Hybrid masker: AI (U2NetP) + Traditional CV (Otsu + Guided Filter).
    Accepts a config dict directly — no config file needed.

    Required keys in cfg:
        image_paths      : list of absolute paths to the 4 PNG images
        output_dir       : directory where mask + CSV will be saved
        world_coordinates: [x_w, y_w, z_w]  (e.g. [0.0, 0.0, 0.017])
        generate_csv     : bool
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.image_paths = cfg["image_paths"]
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading AI Brain: u2netp ...")
        self.session = new_session("u2netp")

    def process(self):
        print("Step 1: Creating Max-Composite & Gamma Stretch...")
        combined_img = None
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  WARNING: could not read {path}, skipping.")
                continue
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            img_float = img.astype(np.float32)
            combined_img = img_float if combined_img is None else np.maximum(combined_img, img_float)

        if combined_img is None:
            raise RuntimeError("No valid images found for masking.")

        max_val = 65535.0 if combined_img.max() > 255 else 255.0
        combined_gamma = np.power(combined_img / max_val, 1.0 / 2.2)
        combined_8bit = (combined_gamma * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(combined_8bit, cv2.COLOR_BGR2GRAY)

        print("Step 2: AI Brain - Generating Rough Context Mask...")
        output_rgba = remove(combined_8bit, session=self.session)
        ai_mask_rough = output_rgba[:, :, 3]

        print("Step 3: Creating the Safe Zone (Dilating AI Mask)...")
        kernel_dilate = np.ones((41, 41), np.uint8)
        safe_zone = cv2.dilate(ai_mask_rough, kernel_dilate, iterations=1)

        print("Step 4: Traditional Scalpel - Calculating Otsu Threshold...")
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, trad_cv_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print("Step 5: Hybrid Merge...")
        hybrid_mask = cv2.bitwise_and(trad_cv_mask, safe_zone)

        print("Step 6: Morphological Cleanup & Edge Smoothing...")
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_OPEN, kernel_open)

        mask_for_guide = cv2.GaussianBlur(hybrid_mask, (11, 11), 0)
        try:
            refined_soft_mask = cv2.ximgproc.guidedFilter(
                guide=gray, src=mask_for_guide, radius=15, eps=0.01
            )
            _, hybrid_mask = cv2.threshold(refined_soft_mask, 127, 255, cv2.THRESH_BINARY)
            print("  -> Guided Filter applied successfully.")
        except AttributeError:
            print("  -> WARNING: ximgproc not found, falling back to Gaussian.")
            smoothed_mask = cv2.GaussianBlur(hybrid_mask, (15, 15), 0)
            _, hybrid_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(hybrid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(hybrid_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        print("Step 7: Saving Outputs...")
        ys, xs = np.where(final_mask > 0)

        generate_csv = self.cfg.get("generate_csv", True)
        if not generate_csv:
            print("  CSV generation disabled.")
        else:
            world_coords = self.cfg.get("world_coordinates", [0.0, 0.0, 0.0])
            x_w, y_w, z_w = world_coords
            csv_path = self.output_dir / "valid_pixels_hybrid_u2netp.csv"
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["pixel_u", "pixel_v", "x_world", "y_world", "z_world", "alpha"])
                for u, v in zip(xs, ys):
                    writer.writerow([u, v, x_w, y_w, z_w, 1.0])
            print(f"  CSV saved: {csv_path}  ({len(ys):,} valid pixels)")

        cv2.imwrite(str(self.output_dir / "1_hybrid_safe_zone.jpg"), safe_zone)
        cv2.imwrite(str(self.output_dir / "2_hybrid_final_mask.png"), final_mask)
        print(f"Done! Total valid pixels: {len(ys):,}")
