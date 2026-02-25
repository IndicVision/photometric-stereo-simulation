import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ExifTags

CONFIG_FILE = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram.json"

def load_config(filename):
    with open(filename, "r") as f:
        return json.load(f)

from fractions import Fraction
from PIL.TiffImagePlugin import IFDRational

def read_exif_basic(img_path):
    try:
        exif = Image.open(img_path)._getexif()
        if exif is None:
            return "Unknown", "Unknown", "Unknown"

        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

        shutter = exif.get("ExposureTime", "Unknown")

        if isinstance(shutter, IFDRational):
            shutter = f"{shutter.numerator}/{shutter.denominator}"
        elif isinstance(shutter, float):
            shutter = str(Fraction(shutter).limit_denominator())
        elif isinstance(shutter, tuple):
            shutter = f"{shutter[0]}/{shutter[1]}"
        else:
            shutter = str(shutter)

        return shutter, str(exif.get("FNumber", "Unknown")), str(exif.get("ISOSpeedRatings", "Unknown"))
    except:
        return "Unknown", "Unknown", "Unknown"

def compute_histogram(img):
    low = 40
    mask = (img > low).astype(np.uint8) * 255
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
    return hist, mask

def compute_peak_support(hist):
    hist = hist.flatten()
    hist_smooth = gaussian_filter1d(hist, sigma=2)
    peak_idx = np.argmax(hist_smooth)
    peak_value = hist_smooth[peak_idx]
    background_level = np.percentile(hist_smooth, 5)
    support_threshold = background_level + 0.02 * (peak_value - background_level)

    left_support = peak_idx
    for i in range(peak_idx, 0, -1):
        if hist_smooth[i] <= support_threshold:
            left_support = i
            break

    right_support = peak_idx
    for i in range(peak_idx, len(hist_smooth)):
        if hist_smooth[i] <= support_threshold:
            right_support = i
            break

    return {
        "peak_idx": peak_idx,
        "left_support": left_support,
        "right_support": right_support
    }

def process_image(img_path, output_folder, global_y_max):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist, mask = compute_histogram(img)
    peak_info = compute_peak_support(hist)

    shutter, aperture, iso = read_exif_basic(img_path)

    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Plot and save histogram with peak start and peak end marked
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(hist, color='blue', linewidth=1.5, label='Smoothed Histogram')
    
    # Mark peak start (left support)
    ax.axvline(peak_info["left_support"], color='purple', linestyle='--', linewidth=1.5, 
               label='Peak Start')
    
    # Mark peak center
    ax.axvline(peak_info["peak_idx"], color='red', linestyle=':', linewidth=1.5, 
               label='Peak Center')
    
    # Mark peak end (right support)
    ax.axvline(peak_info["right_support"], color='orange', linestyle='--', linewidth=1.5, 
               label='Peak End')
    
    # Add smaller yellow boxes with lighter font at the top
    y_position = global_y_max * 0.97
    ax.text(peak_info["left_support"], y_position, str(peak_info["left_support"]), 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=0.8),
            ha='center', va='top', fontsize=8, fontweight='normal')
    
    ax.text(peak_info["peak_idx"], y_position, str(peak_info["peak_idx"]), 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=0.8),
            ha='center', va='top', fontsize=8, fontweight='normal')
    
    ax.text(peak_info["right_support"], y_position, str(peak_info["right_support"]), 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=0.8),
            ha='center', va='top', fontsize=8, fontweight='normal')
    
    ax.set_ylim(0, global_y_max)
    ax.set_xlabel("Intensity", fontsize=11)
    ax.set_ylabel("Pixel Count", fontsize=11)
    ax.set_title(f"Grayscale Histogram with Peak Support", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_folder, f"{img_name}_histogram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "image_name": img_name,
        "peak_start": peak_info["left_support"],
        "peak": peak_info["peak_idx"],
        "peak_end": peak_info["right_support"],
        "shutter_speed": shutter,
        "aperture": aperture,
        "iso": iso
    }

def process_orientation(sample_folder, set_num, orientation, output_base, sample_name, results_list, sample_id, angle_name, global_y_max):
    # Construct input path: sample_folder/set_num/orientation
    angle_folder = os.path.join(sample_folder, f"set_{set_num}", orientation)
    
    # Construct output path
    output_folder = os.path.join(output_base, sample_name, f"set_{set_num}", orientation)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images in this folder
    image_paths = sorted(glob.glob(os.path.join(angle_folder, "*.JPG")))
    
    if not image_paths:
        print(f"  set_{set_num}/{orientation}: No images found")
        return
    
    print(f"  set_{set_num}/{orientation}: {len(image_paths)} images")
    
    # Process each image
    for img_path in image_paths:
        result = process_image(img_path, output_folder, global_y_max)
        result["sample_number"] = sample_id
        result["set_number"] = set_num
        result["orientation_angle_deg"] = angle_name
        # result["brightness_level"] = level  # Commented out as requested
        results_list.append(result)

def process_sample_head_on(sample_id, base_path, output_base, sample_df, results_list):
    """Process head-on images for a sample"""
    row = sample_df[sample_df["Sample Number"] == sample_id]
    if row.empty:
        print(f"Sample {sample_id} not found in CSV")
        return
    
    length = row["Length"].values[0]
    breadth = row["Breadth"].values[0]
    sample_name = f"sample_{sample_id}_{length}x{breadth}"
    sample_folder = os.path.join(base_path, sample_name)
    head_on_folder = os.path.join(sample_folder, "head_on")
    
    if not os.path.exists(head_on_folder):
        print(f"Head-on folder not found: {head_on_folder}")
        return
    
    print(f"\nProcessing head-on images for {sample_name}")
    
    # Output path: output_base/headon_closeup/sample_name
    output_folder = os.path.join(output_base, "headon_closeup", sample_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images in head_on folder
    image_paths = sorted(glob.glob(os.path.join(head_on_folder, "*.JPG")))
    
    if not image_paths:
        print(f"  No images found in head_on folder")
        return
    
    print(f"  Found {len(image_paths)} head-on images")
    
    # Compute global y max for head-on images
    global_y_max = 0
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist, _ = compute_histogram(img)
        global_y_max = max(global_y_max, hist.max())
    
    print(f"  Global Y-max: {global_y_max}")
    
    # Process each image
    for img_path in image_paths:
        result = process_image(img_path, output_folder, global_y_max)
        result["sample_number"] = sample_id
        results_list.append(result)

def process_sample_orientation(sample_id, base_path, output_base, sample_df, results_list, config):
    row = sample_df[sample_df["Sample Number"] == sample_id]
    if row.empty:
        print(f"Sample {sample_id} not found in CSV")
        return
    
    length = row["Length"].values[0]
    breadth = row["Breadth"].values[0]
    sample_name = f"sample_{sample_id}_{length}x{breadth}"
    sample_folder = os.path.join(base_path, sample_name)
    
    print(f"\nProcessing {sample_name}")
    
    set_num = config["orientation"]["set_num"]
    
    # Compute global y max across ALL angles for consistent scaling
    global_y_max = 0
    for angle in config["orientation"]["angle_range"]:
        orientation = f"{angle}_deg"
        angle_folder = os.path.join(sample_folder, f"set_{set_num}", orientation)
        image_paths = sorted(glob.glob(os.path.join(angle_folder, "*.JPG")))
        
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hist, _ = compute_histogram(img)
            global_y_max = max(global_y_max, hist.max())
    
    print(f"Global Y-max: {global_y_max}")
    
    # Process angle orientations
    for angle in config["orientation"]["angle_range"]:
        orientation = f"{angle}_deg"
        angle_name = f"{angle}"
        
        process_orientation(sample_folder, set_num, orientation, output_base, 
                          sample_name, results_list, sample_id, angle_name, global_y_max)

def main():
    config = load_config(CONFIG_FILE)
    base_path = config["base_path"]
    output_base = config["output_path"]
    csv_path = config["csv_path"]

    sample_df = pd.read_csv(csv_path)
    results_list = []

    # Handle sample_num as list or single value
    sample_nums = config["sample_num"]
    if not isinstance(sample_nums, list):
        sample_nums = [sample_nums]
    
    # Check if head_on is enabled in config
    head_on = config.get("head_on", False)
    
    # Process all samples
    for sample_id in sample_nums:
        if head_on:
            # Process head-on images
            process_sample_head_on(sample_id, base_path, output_base, sample_df, results_list)
        else:
            # Process orientation images
            process_sample_orientation(sample_id, base_path, output_base, sample_df, results_list, config)

    results_df = pd.DataFrame(results_list)
    
    # Reorder columns based on mode
    if head_on:
        # For head-on mode: no set_number or orientation_angle_deg
        results_df = results_df[["sample_number", "image_name", "shutter_speed", "aperture", "iso",
                             "peak_start", "peak", "peak_end"]]
    else:
        # For orientation mode: include set_number and orientation_angle_deg
        results_df = results_df[["sample_number", "set_number", "orientation_angle_deg",
                             "image_name", "shutter_speed", "aperture", "iso",
                             "peak_start", "peak", "peak_end"]]

    # Determine output CSV path
    if head_on:
        # Use csv_output path from config if provided, otherwise default
        if "csv_output" in config:
            output_csv = config["csv_output"]
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        else:
            csv_output_folder = os.path.join(output_base, "headon_closeup")
            os.makedirs(csv_output_folder, exist_ok=True)
            output_csv = os.path.join(csv_output_folder, "peak_analysis_results.csv")
    else:
        # For orientation mode, use the first sample to determine path
        row = sample_df[sample_df["Sample Number"] == sample_nums[0]]
        length = row["Length"].values[0]
        breadth = row["Breadth"].values[0]
        sample_name = f"sample_{sample_nums[0]}_{length}x{breadth}"
        set_num = config["orientation"]["set_num"]
        
        # Save CSV in set folder
        csv_output_folder = os.path.join(output_base, sample_name, f"set_{set_num}")
        os.makedirs(csv_output_folder, exist_ok=True)
        output_csv = os.path.join(csv_output_folder, "peak_analysis_results.csv")
    
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(f"Total images processed: {len(results_list)}")

if __name__ == "__main__":
    main()