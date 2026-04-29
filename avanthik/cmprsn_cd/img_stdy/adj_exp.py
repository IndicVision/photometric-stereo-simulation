import json
import os
import cv2
import numpy as np

# Enable OpenEXR support in OpenCV before anything else
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def apply_exposure(img, mode, value):
    dtype = img.dtype
    
    # Convert to float64 to prevent overflow during mathematical operations
    img_float = img.astype(np.float64)
    
    if mode == 'linear':
        # Linear multiplier: I_out = I_in * value
        img_float = img_float * value
        
    elif mode == 'gamma':
        # Gamma adjustment: I_out = max_val * (I_in / max_val)^(1 / gamma)
        # Determine the maximum value based on the container datatype
        if dtype == np.uint8:
            max_val = 255.0
        elif dtype == np.uint16:
            max_val = 65535.0
        else:
            # For float16/float32 (like EXR), assume 1.0 is the normalized white point.
            # EXRs can exceed 1.0 (HDR), and the math still scales correctly.
            max_val = 1.0 
            
        # Prevent math errors with negative numbers or zero
        img_float = np.clip(img_float, 0, None)
        
        if dtype in [np.uint8, np.uint16]:
            img_normalized = img_float / max_val
            img_float = np.power(img_normalized, 1.0 / value) * max_val
        else:
            # Direct power application for floating point images
            img_float = np.power(img_float, 1.0 / value)
    else:
        raise ValueError(f"Unsupported exposure_type: {mode}")

    # Re-cast back to the original datatype safely
    if dtype == np.uint8:
        return np.clip(img_float, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        return np.clip(img_float, 0, 65535).astype(np.uint16)
    elif dtype == np.float16:
        return img_float.astype(np.float16)
    elif dtype == np.float32:
        return img_float.astype(np.float32)
    else:
        return img_float.astype(dtype)

def process_images(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    inputs = config.get("inputs", [])
    output_dir = config.get("output_dir", "./output")
    exp_mode = config.get("exposure_type", "linear").lower()
    exp_val = float(config.get("exposure_value", 1.0))
    use_same_name = config.get("use_same_name", True)
    append_str = config.get("append_string", "")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for img_path in inputs:
        if not os.path.exists(img_path):
            print(f"Skipping: File not found -> {img_path}")
            continue
            
        # Read image exactly as it is (preserves float32, uint16, EXR, etc.)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Failed to read (might be unsupported RAW): {img_path}")
            continue
            
        print(f"Processing {img_path} | Type: {img.dtype} | Shape: {img.shape}")
        
        # Apply exposure
        adjusted_img = apply_exposure(img, exp_mode, exp_val)
        
        # Determine output filename
        filename = os.path.basename(img_path)
        base, ext = os.path.splitext(filename)
        
        # Handle RAW extensions (Fallback to EXR to preserve high bit depth)
        if ext.lower() in ['.cr2', '.dng', '.arw', '.nef']:
            print(f"Warning: Cannot natively write to {ext}. Saving as .exr to preserve bit depth.")
            ext = '.exr'
            
        if use_same_name:
            out_filename = f"{base}{ext}"
        else:
            out_filename = f"{base}{append_str}{ext}"
            
        out_path = os.path.join(output_dir, out_filename)
        
        # Save image
        cv2.imwrite(out_path, adjusted_img)
        print(f"Saved -> {out_path}\n")

if __name__ == "__main__":
    process_images(r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\adj_exp_cfg.json")