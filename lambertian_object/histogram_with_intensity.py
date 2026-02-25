import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ExifTags
from fractions import Fraction
from PIL.TiffImagePlugin import IFDRational

CONFIG_FILE = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram_with_intensity.json"

def load_config(filename):
    """Load JSON configuration file"""
    with open(filename, "r") as f:
        return json.load(f)

def read_exif_basic(img_path):
    """Extract shutter speed, aperture, and ISO from image EXIF data"""
    try:
        exif = Image.open(img_path)._getexif()
        if exif is None:
            return "Unknown", "Unknown", "Unknown"
        
        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        
        # Parse shutter speed
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
    """Compute histogram with low-intensity threshold mask"""
    mask = (img > 40).astype(np.uint8) * 255  # Mask pixels above intensity 40
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
    return hist, mask

def compute_peak_support(hist):
    """Find peak position and its support boundaries (left/right support)"""
    hist = hist.flatten()
    hist_smooth = gaussian_filter1d(hist, sigma=2)  # Smooth histogram
    
    # Find peak
    peak_idx = np.argmax(hist_smooth)
    peak_value = hist_smooth[peak_idx]
    
    # Calculate support threshold (2% above background)
    background_level = np.percentile(hist_smooth, 5)
    support_threshold = background_level + 0.02 * (peak_value - background_level)
    
    # Find left support boundary
    left_support = peak_idx
    for i in range(peak_idx, 0, -1):
        if hist_smooth[i] <= support_threshold:
            left_support = i
            break
    
    # Find right support boundary
    right_support = peak_idx
    for i in range(peak_idx, len(hist_smooth)):
        if hist_smooth[i] <= support_threshold:
            right_support = i
            break
    
    return {"peak_idx": peak_idx, "left_support": left_support, "right_support": right_support}

def process_image(img_path, output_folder, global_y_max):
    """Process single image: compute histogram, detect peak, save plot"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hist, _ = compute_histogram(img)
    peak_info = compute_peak_support(hist)
    shutter, aperture, iso = read_exif_basic(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Create histogram plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(hist, color='blue', linewidth=1.5, label='Histogram')
    
    # Mark peak boundaries and center
    ax.axvline(peak_info["left_support"], color='purple', linestyle='--', linewidth=1.5, label='Peak Start')
    ax.axvline(peak_info["peak_idx"], color='red', linestyle=':', linewidth=1.5, label='Peak Center')
    ax.axvline(peak_info["right_support"], color='orange', linestyle='--', linewidth=1.5, label='Peak End')
    
    # Add value labels
    y_pos = global_y_max * 0.97
    for val in [peak_info["left_support"], peak_info["peak_idx"], peak_info["right_support"]]:
        ax.text(val, y_pos, str(val), 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=0.8),
                ha='center', va='top', fontsize=8)
    
    ax.set_ylim(0, global_y_max)
    ax.set_xlabel("Intensity", fontsize=11)
    ax.set_ylabel("Pixel Count", fontsize=11)
    ax.set_title(f"Grayscale Histogram with Peak Support", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_folder, f"{img_name}.png")
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

def get_sample_name(sample_id, sample_df):
    """Get formatted sample name from CSV"""
    row = sample_df[sample_df["Sample Number"] == sample_id]
    if row.empty:
        raise ValueError(f"Sample {sample_id} not found in CSV")
    length = row["Length"].values[0]
    breadth = row["Breadth"].values[0]
    return f"sample_{sample_id}_{length}x{breadth}"

def compute_global_y_max(image_paths):
    """Compute maximum histogram height across all images for consistent y-axis scaling"""
    global_y_max = 0
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist, _ = compute_histogram(img)
        global_y_max = max(global_y_max, hist.max())
    return global_y_max

def process_head_on(sample_id, base_path, output_base, sample_df, results_list):
    """Process head-on closeup images for a sample"""
    sample_name = get_sample_name(sample_id, sample_df)
    head_on_folder = os.path.join(base_path, sample_name, "head_on")
    
    if not os.path.exists(head_on_folder):
        print(f"Head-on folder not found: {head_on_folder}")
        return
    
    print(f"\nProcessing head-on images for {sample_name}")
    
    # Get all images
    image_paths = sorted(glob.glob(os.path.join(head_on_folder, "*.JPG")))
    if not image_paths:
        print(f"  No images found")
        return
    
    print(f"  Found {len(image_paths)} images")
    
    # Compute global y-axis maximum for consistent scaling
    global_y_max = compute_global_y_max(image_paths)
    print(f"  Global Y-max: {global_y_max}")
    
    # Create output folder
    output_folder = os.path.join(output_base, "headon_closeup", sample_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image
    for img_path in image_paths:
        result = process_image(img_path, output_folder, global_y_max)
        result["sample_number"] = sample_id
        results_list.append(result)

def process_orientation(sample_id, base_path, output_base, sample_df, config, results_list):
    """Process orientation-varying images for a sample"""
    sample_name = get_sample_name(sample_id, sample_df)
    set_num = config["set_num"]
    light_nums = config["orientation"]["light_num"]
    angles = config["orientation"]["angle_range"]
    
    print(f"\nProcessing orientation images for {sample_name}")
    
    # Collect all image paths for global y-max calculation
    all_image_paths = []
    for light_num in light_nums:
        for angle in angles:
            # Input path: base_path/sample_name/set_X/light_Y/Z_deg/*.JPG
            angle_folder = os.path.join(base_path, sample_name, f"set_{set_num}", f"light_{light_num}", f"{angle}_deg")
            image_paths = sorted(glob.glob(os.path.join(angle_folder, "*.JPG")))
            all_image_paths.extend(image_paths)
    
    if not all_image_paths:
        print(f"  No images found")
        return
    
    # Compute global y-max for consistent scaling
    global_y_max = compute_global_y_max(all_image_paths)
    print(f"  Found {len(all_image_paths)} total images, Global Y-max: {global_y_max}")
    
    # Process each light/angle combination
    for light_num in light_nums:
        for angle in angles:
            # Input path
            angle_folder = os.path.join(base_path, sample_name, f"set_{set_num}", f"light_{light_num}", f"{angle}_deg")
            
            # Output path: output_base/sample_name/set_X/light_Y/Z_deg/
            output_folder = os.path.join(output_base, sample_name, f"set_{set_num}", f"light_{light_num}", f"{angle}_deg")
            os.makedirs(output_folder, exist_ok=True)
            
            # Get images for this specific angle/light combination
            image_paths = sorted(glob.glob(os.path.join(angle_folder, "*.JPG")))
            
            if not image_paths:
                print(f"  set_{set_num}/light_{light_num}/{angle}_deg: No images")
                continue
            
            print(f"  set_{set_num}/light_{light_num}/{angle}_deg: {len(image_paths)} images")
            
            # Process each image
            for img_path in image_paths:
                result = process_image(img_path, output_folder, global_y_max)
                result["sample_number"] = sample_id
                result["set_number"] = set_num
                result["light_number"] = light_num
                result["orientation_angle_deg"] = angle
                results_list.append(result)

def main():
    # Load configuration
    config = load_config(CONFIG_FILE)
    base_path = config["base_path"]
    output_base = config["output_path"]
    csv_path = config["csv_path"]
    sample_nums = config["sample_num"] if isinstance(config["sample_num"], list) else [config["sample_num"]]
    head_on = config.get("head_on", False)
    
    # Load sample measurements
    sample_df = pd.read_csv(csv_path)
    results_list = []
    
    # Process all samples
    for sample_id in sample_nums:
        if head_on:
            process_head_on(sample_id, base_path, output_base, sample_df, results_list)
        else:
            process_orientation(sample_id, base_path, output_base, sample_df, config, results_list)
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Reorder columns based on mode
    if head_on:
        results_df = results_df[["sample_number", "image_name", "shutter_speed", "aperture", "iso",
                                 "peak_start", "peak", "peak_end"]]
        # Determine output CSV path
        if "csv_output" in config:
            output_csv = config["csv_output"]
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        else:
            csv_output_folder = os.path.join(output_base, "headon_closeup")
            os.makedirs(csv_output_folder, exist_ok=True)
            output_csv = os.path.join(csv_output_folder, "peak_analysis_results.csv")
    else:
        results_df = results_df[["sample_number", "set_number", "light_number", "orientation_angle_deg",
                                 "image_name", "shutter_speed", "aperture", "iso",
                                 "peak_start", "peak", "peak_end"]]
        # Save CSV in set folder
        sample_name = get_sample_name(sample_nums[0], sample_df)
        csv_output_folder = os.path.join(output_base, sample_name, f"set_{config['set_num']}")
        os.makedirs(csv_output_folder, exist_ok=True)
        output_csv = os.path.join(csv_output_folder, "peak_analysis_results.csv")
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(f"Total images processed: {len(results_list)}")

if __name__ == "__main__":
    main()