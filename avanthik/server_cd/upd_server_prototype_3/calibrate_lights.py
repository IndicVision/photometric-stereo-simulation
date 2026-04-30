#!/usr/bin/env python3
"""
PhotoStereo — Light Calibration Script
Reads the four white-paper PNG images from photostereo_sessions,
computes intensity correction factors from the green channel centre crop,
and auto-updates process_pipeline.py with the new values.

Called by the server's /upload_calibration endpoint.
Can also be run standalone from the desktop shortcut.
"""

import cv2
import numpy as np
import re
import os
import sys
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR         = Path(os.path.expanduser("~/photostereo_sessions"))
CALIB_DIR        = SAVE_DIR / "calibration"
PIPELINE_SCRIPT  = Path(__file__).parent / "process_pipeline.py"
CROP_HALF        = 250   # half-width of centre crop → 500×500 box
FILES            = ["light_001.png", "light_002.png",
                    "light_003.png", "light_004.png"]


def compute_factors(image_dir: Path) -> list[float]:
    """
    Compute intensity correction factors from 4 white-paper PNG images.
    Uses green channel mean of a centre crop.
    Returns [f1, f2, f3, f4] where the brightest light has factor 1.0000.
    Raises FileNotFoundError if any image is missing.
    Raises ValueError if any image mean is zero.
    """
    means = []
    for fname in FILES:
        path = image_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Calibration image not found: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        if img.ndim < 3:
            raise ValueError(f"Expected colour image but got single channel: {path}")

        # Green channel only — no perceptual weighting skew from BGR2GRAY
        green = img[:, :, 1].astype(np.float64)
        h, w  = green.shape
        cy, cx = h // 2, w // 2

        # Guard against images smaller than the crop box
        half = min(CROP_HALF, cy - 1, cx - 1)
        crop = green[cy - half : cy + half, cx - half : cx + half]
        m    = crop.mean()
        if m <= 0:
            raise ValueError(f"Centre crop mean is zero for {fname}. Is the image blank?")
        means.append(m)
        print(f"  {fname}  green centre mean = {m:.2f}")

    max_mean = max(means)
    factors  = [round(max_mean / m, 4) for m in means]
    return factors


def update_pipeline_script(factors: list[float]) -> bool:
    """
    Replaces the intensity_correction list in process_pipeline.py in-place.
    Matches the pattern:  "intensity_correction": [...]
    Returns True if the replacement was made, False if pattern not found.
    """
    if not PIPELINE_SCRIPT.exists():
        print(f"WARNING: {PIPELINE_SCRIPT} not found — cannot auto-update.")
        return False

    text = PIPELINE_SCRIPT.read_text(encoding="utf-8")
    new_val = "[" + ", ".join(f"{f:.4f}" for f in factors) + "]"
    pattern = r'("intensity_correction"\s*:\s*)\[.*?\]'
    new_text, count = re.subn(pattern, rf'\g<1>{new_val}', text)

    if count == 0:
        print("WARNING: Could not find intensity_correction in process_pipeline.py")
        return False

    PIPELINE_SCRIPT.write_text(new_text, encoding="utf-8")
    print(f"  process_pipeline.py updated: intensity_correction = {new_val}")
    return True


def save_factors_json(factors: list[float]):
    """Save factors to a JSON file for the phone app to read."""
    out = {
        "intensity_correction": factors,
        "light_means": None,   # populated by caller if needed
        "status": "ok"
    }
    out_path = CALIB_DIR / "calibration_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out_path


def run(image_dir: Path = None) -> dict:
    """
    Main entry point. image_dir defaults to CALIB_DIR.
    Returns a dict with keys: status, factors, message.
    """
    if image_dir is None:
        image_dir = CALIB_DIR

    print(f"\n--- PhotoStereo Light Calibration ---")
    print(f"  Reading images from: {image_dir}")

    try:
        factors = compute_factors(image_dir)
    except (FileNotFoundError, IOError, ValueError) as e:
        msg = str(e)
        print(f"ERROR: {msg}")
        return {"status": "error", "factors": None, "message": msg}

    print(f"\n  Computed factors:")
    for i, f in enumerate(factors):
        print(f"    Light {i+1}: {f:.4f}")

    update_pipeline_script(factors)
    save_factors_json(factors)

    msg = (f"Calibration complete. "
           f"Factors: [{', '.join(f'{f:.4f}' for f in factors)}]")
    print(f"\n  {msg}")
    return {"status": "ok", "factors": factors, "message": msg}


if __name__ == "__main__":
    # Standalone use: optionally pass image directory as first argument
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run(d)
    sys.exit(0 if result["status"] == "ok" else 1)