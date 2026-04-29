import json
import cv2
import numpy as np
import sys
import os

def average_images_from_json(json_path):
    # Load JSON configuration
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

    input_paths = config.get("input_images", [])
    output_path = config.get("output_image", "output.png")

    if not input_paths:
        print("Error: No input images provided in JSON.")
        sys.exit(1)

    # Read the first image to establish the baseline shape and datatype
    first_img_path = input_paths[0]
    if not os.path.exists(first_img_path):
        print(f"Error: Image not found - {first_img_path}")
        sys.exit(1)

    # IMREAD_UNCHANGED ensures bit-depth (e.g., 16-bit) and alpha channels are conserved
    baseline_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    if baseline_img is None:
        print(f"Error: Could not decode image - {first_img_path}")
        sys.exit(1)

    baseline_shape = baseline_img.shape
    original_dtype = baseline_img.dtype

    # Initialize a float64 accumulator to prevent overflow during addition
    accumulator = np.zeros(baseline_shape, dtype=np.float64)
    
    # Process all images
    for path in input_paths:
        if not os.path.exists(path):
            print(f"Error: Image not found - {path}")
            sys.exit(1)
            
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not decode image - {path}")
            sys.exit(1)

        # 1. Check if resolution/shape matches the first image exactly
        if img.shape != baseline_shape:
            print(f"Error: Resolution mismatch! \n"
                  f"{first_img_path} is {baseline_shape} \n"
                  f"{path} is {img.shape}. Aborting.")
            sys.exit(1)

        # Add to accumulator
        accumulator += img

    # 2. Calculate the average
    num_images = len(input_paths)
    averaged_img_float = accumulator / num_images

    # 3. Conserve original datatype
    # If the original image was an integer type, round it before casting to avoid truncation bias
    if np.issubdtype(original_dtype, np.integer):
        averaged_img_float = np.round(averaged_img_float)

    # Cast back to the exact original bit-depth/datatype
    final_img = averaged_img_float.astype(original_dtype)

    # Save the output image
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success = cv2.imwrite(output_path, final_img)
    if success:
        print(f"Successfully saved denoised image to: {output_path}")
        print(f"Preserved Data Type: {original_dtype}")
        print(f"Resolution: {baseline_shape}")
    else:
        print("Error: Failed to save the output image. Check directory permissions or file extension.")

if __name__ == "__main__":
    # You can change this to accept a command line argument if you prefer
    json_config_file = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\img_avg_denoise_cfg.json"
    average_images_from_json(json_config_file)