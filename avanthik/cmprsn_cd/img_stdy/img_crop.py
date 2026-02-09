import rawpy
import os
import cv2
import numpy as np

# --- CONFIGURATION -----------------------------------------------------------
INPUT_PATH = r"C:\Users\vishn\Desktop\avanthik\cmr_op\basler\simulation_003\cr2_cmr_op\cr2_TO_png_linear\cropped\nrml_op\otsu\error_degree_map.png"  # Change to any file
OUTPUT_PATH = r"C:\Users\vishn\Desktop\avanthik\cmr_op\basler\simulation_003\cr2_cmr_op\cr2_TO_png_linear\cropped\nrml_op\otsu\error_degree_map_cropped.png"

# Crop Settings (Coordinates based on the "Native/Sensor" orientation)
CROP_X = 501
CROP_Y = 431
CROP_W = 1119
CROP_H = 737

# --- INTENSITY CONTROL SETTINGS ---
# RAW files only: How should we "develop" the sensor data?
RAW_SETTINGS = {
    'gamma': (1, 1),        # (1,1) = Linear (Scientific/Dark). Set to (2.222, 4.5) for standard look.
    'no_auto_bright': True, # True = Exact intensity. False = Auto-scale to fill histogram.
    'use_camera_wb': True,  # True = Use camera settings. False = Auto WB.
    'user_flip': 0          # 0 = Force Sensor Orientation (Ignore rotation tags)
}

# -----------------------------------------------------------------------------

def read_image_universal(filepath):
    """
    Reads image data into a standard format (Linear Float or Integer), 
    handling specific logic for RAW vs Standard files.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    # 1. RAW HANDLING (CR2, NEF, ARW, DNG)
    if ext in ['.cr2', '.nef', '.arw', '.dng']:
        print(f"[Info] Detected RAW file: {ext}")
        with rawpy.imread(filepath) as raw:
            # Determine output bit depth based on intended file extension (not strictly necessary here but good practice)
            # We default to 16-bit for RAW processing to keep precision
            rgb = raw.postprocess(
                gamma=RAW_SETTINGS['gamma'],
                no_auto_bright=RAW_SETTINGS['no_auto_bright'],
                use_camera_wb=RAW_SETTINGS['use_camera_wb'],
                user_flip=RAW_SETTINGS['user_flip'],
                output_bps=16
            )
            # Rawpy gives RGB, OpenCV needs BGR
            image_data = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return image_data, 16  # Return data and bit-depth info

    # 2. STANDARD & HDR HANDLING (JPG, PNG, TIFF, EXR)
    else:
        print(f"[Info] Detected Standard/HDR file: {ext}")
        # IMREAD_UNCHANGED is critical:
        # - Loads 16-bit PNGs as 16-bit (not downscaled to 8-bit)
        # - Loads EXRs as float32
        # - Does NOT auto-rotate (perfect for your "ignore rotation" requirement)
        image_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if image_data is None:
            raise FileNotFoundError(f"Could not read file: {filepath}")

        # Check what we actually loaded
        dtype = image_data.dtype
        print(f"      Loaded Input Depth: {dtype}")
        
        return image_data, dtype

def save_image_smart(image, output_path, input_dtype):
    """
    Saves the image, handling the tricky conversion between Float (EXR) and Int (PNG)
    if necessary.
    """
    out_ext = os.path.splitext(output_path)[1].lower()
    
    # A. EXR Input -> PNG/TIFF/JPG Output (Needs conversion)
    # Check if input was float (EXR) but output is integer format
    if (image.dtype == np.float32) and (out_ext not in ['.exr', '.hdr']):
        print("[Warn] Converting Float (EXR) to Integer (PNG/JPG). Clipping values > 1.0.")
        
        # 1. Clip "super white" values to 1.0
        image = np.clip(image, 0.0, 1.0)
        
        # 2. Scale to 16-bit range (0-65535)
        image = (image * 65535).astype(np.uint16)

    # B. Integer Input -> EXR Output (Uncommon, but needs scaling)
    elif (image.dtype != np.float32) and (out_ext == '.exr'):
        print("[Info] Converting Integer to Float32 for EXR.")
        if image.dtype == np.uint16:
            image = image / 65535.0
        elif image.dtype == np.uint8:
            image = image / 255.0
        image = image.astype(np.float32)

    # C. Save
    cv2.imwrite(output_path, image)
    print(f"[Success] Saved to: {output_path}")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found: {INPUT_PATH}")
        return

    # 1. READ
    try:
        img, original_depth = read_image_universal(INPUT_PATH)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    # 2. CROP
    # Safety: Ensure crop doesn't go out of bounds
    img_h, img_w = img.shape[:2]
    
    if (CROP_Y + CROP_H > img_h) or (CROP_X + CROP_W > img_w):
        print(f"Error: Crop window is outside image dimensions!")
        print(f"Image: {img_w}x{img_h} | Crop End: {CROP_X+CROP_W}x{CROP_Y+CROP_H}")
        return

    # Slicing: [Rows, Cols]
    cropped_img = img[CROP_Y : CROP_Y+CROP_H, CROP_X : CROP_X+CROP_W]
    
    print(f"Original Size: {img.shape}")
    print(f"Cropped Size:  {cropped_img.shape}")

    # 3. SAVE
    save_image_smart(cropped_img, OUTPUT_PATH, original_depth)

if __name__ == "__main__":
    main()