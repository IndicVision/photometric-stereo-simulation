import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ---------------- CONFIG ----------------
CONFIG_FILE = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram_comparison.json"

def load_config(p):
    with open(p, "r") as f:
        return json.load(f)

def get_sample_name(sample_id, sample_df):
    """Get formatted sample name from CSV"""
    row = sample_df[sample_df["Sample Number"] == sample_id]
    if row.empty:
        raise ValueError(f"Sample {sample_id} not found in CSV")
    length = row["Length"].values[0]
    breadth = row["Breadth"].values[0]
    return f"sample_{sample_id}_{length}x{breadth}"

def get_matching_image_name(analysis_df, cfg, sample_num, set_num, light_num, angle):
    """Get image filename from analysis CSV based on parameters and angle"""
    matched = analysis_df[
        (analysis_df["sample_number"] == sample_num) &  # Added missing '&'
        (analysis_df["shutter_speed"] == cfg["shutter_speed"]) &
        (analysis_df["aperture"] == cfg["aperture"]) &
        (analysis_df["iso"] == cfg["iso"]) &
        (analysis_df["set_number"] == set_num) &
        (analysis_df["light_number"] == light_num) &
        (analysis_df["orientation_angle_deg"] == angle)
    ]
    
    if matched.empty:
        return None
    
    return matched.iloc[0]["image_name"] if "image_name" in matched.columns else None

def compute_peak_support(hist):
    """Calculate peak center and support boundaries"""
    hist = hist.flatten()
    hist_smooth = gaussian_filter1d(hist, sigma=2)

    peak_idx = np.argmax(hist_smooth)
    peak_value = hist_smooth[peak_idx]

    background_level = np.percentile(hist_smooth, 5)
    support_threshold = background_level + 0.02 * (peak_value - background_level)

    left_support = 0
    for i in range(peak_idx, 0, -1):
        if hist_smooth[i] <= support_threshold:
            left_support = i
            break

    right_support = 255
    for i in range(peak_idx, len(hist_smooth)):
        if hist_smooth[i] <= support_threshold:
            right_support = i
            break

    return {
        "peak_idx": peak_idx,
        "left_support": left_support,
        "right_support": right_support
    }

def compute_hist(img):
    """Compute histogram with masking"""
    mask = (img > 40).astype(np.uint8) * 255
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256]).flatten()
    return hist

def plot_histograms_per_light(matched_data, hists, cfg, sample_folder, light_num, set_num):
    """Plot all orientation histograms for one light source"""
    out_dir = os.path.join(cfg["output_path"], "comparison_analysis", sample_folder, f"set_{set_num}")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, d in enumerate(matched_data):
        if d["light"] == light_num and d["set"] == set_num:
            ax.plot(hists[i], label=f'{d["angle"]}°', alpha=0.8)

    ax.set_title(
        f"Histogram Comparison - Light {light_num}, Set {set_num}\n"
        f"Shutter {cfg['shutter_speed']} | f/{cfg['aperture']} | ISO {cfg['iso']}"
    )
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixel Count")
    ax.legend(title="Orientation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"light_{light_num}_histograms.png"),
        dpi=150
    )
    plt.close()

def save_individual_plot(img_path, hist, peak_info, angle_deg, light_num, set_num, sample_num, y_max, out_root, sample_folder):
    """Save individual histogram plot with peak analysis"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    name = os.path.splitext(os.path.basename(img_path))[0]

    set_dir = os.path.join(out_root, "comparison_analysis", sample_folder, "individual_analysis", f"set_{set_num}", f"light_{light_num}")
    os.makedirs(set_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Image
    ax[0].imshow(img, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title(f"{name}\nOrientation = {angle_deg}° | Light {light_num} | Set {set_num}")

    # Histogram
    ax[1].plot(hist, label="Histogram")
    ax[1].axvline(peak_info["left_support"], linestyle="--", label="Peak Start")
    ax[1].axvline(peak_info["peak_idx"], linestyle=":", label="Peak Center")
    ax[1].axvline(peak_info["right_support"], linestyle="--", label="Peak End")

    y_label_pos = y_max * 0.95
    label_props = dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8)

    ax[1].text(peak_info["left_support"], y_label_pos, peak_info["left_support"],
               bbox=label_props, ha="center", fontsize=8)
    ax[1].text(peak_info["peak_idx"], y_label_pos, peak_info["peak_idx"],
               bbox=label_props, ha="center", fontsize=8)
    ax[1].text(peak_info["right_support"], y_label_pos, peak_info["right_support"],
               bbox=label_props, ha="center", fontsize=8)

    ax[1].set_ylim(0, y_max * 1.1)
    ax[1].set_title(f"Peak Analysis (Orientation = {angle_deg}°)")
    ax[1].set_xlabel("Pixel Intensity")
    ax[1].set_ylabel("Frequency")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(set_dir, f"{name}_{angle_deg}deg_analysis.png"),
        dpi=150
    )
    plt.close()

def save_peak_csv(matched_data, peak_results, cfg,sample_folder):
    """Save peak analysis results to CSV"""
    rows = []

    for i, d in enumerate(matched_data):
        peak_start = peak_results[i]["left_support"]
        peak = peak_results[i]["peak_idx"]
        peak_end = peak_results[i]["right_support"]
        img_filename = os.path.basename(d["path"])
        rows.append({
            "sample_number": d["sample"],  # Fixed: use d["sample"] instead of cfg
            "set_number": d["set"],
            "light_number": d["light"],
            "orientation_angle_deg": d["angle"],
            "image_name": img_filename,
            "shutter_speed": cfg["shutter_speed"],
            "aperture": cfg["aperture"],
            "iso": cfg["iso"],
            "peak_start": peak_start,
            "peak": peak,
            "peak_end": peak_end,
            "peak_width": peak_end - peak_start
        })

    df_out = pd.DataFrame(rows)
    out_dir = os.path.join(cfg["output_path"], "comparison_analysis", sample_folder)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "comparison_analysis_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved peak analysis to: {out_csv}")

def main():
    cfg = load_config(CONFIG_FILE)
    df = pd.read_csv(cfg["csv_path"])
    analysis_df = pd.read_csv(cfg["analysis_csv_path"])
    
    # Convert all to lists if not already
    sample_list = cfg["sample_num"] if isinstance(cfg["sample_num"], list) else [cfg["sample_num"]]
    set_list = cfg["set_num"] if isinstance(cfg["set_num"], list) else [cfg["set_num"]]
    light_list = cfg["light_num"] if isinstance(cfg["light_num"], list) else [cfg["light_num"]]
    angle_list = cfg["orientation"]["angle_range"]

    matched_data = []
        
    for sample_num in sample_list:
        sample_folder = get_sample_name(sample_num, df)
        sample_path = os.path.join(cfg["base_path"], sample_folder)
        
        for set_num in set_list:
            for light_num in light_list:
                for angle in angle_list:
                    # Get matching image name for this specific combination
                    target_image = get_matching_image_name(analysis_df, cfg, sample_num, set_num, light_num, angle)
                    
                    if not target_image:
                        print(f"No matching image in CSV for: Sample {sample_num}, Set {set_num}, Light {light_num}, Angle {angle}°")
                        continue
                    
                    # Construct path
                    angle_path = os.path.join(
                        sample_path, 
                        f"set_{set_num}", 
                        f"light_{light_num}", 
                        f"{angle}_deg"
                    )
                    
                    if not os.path.exists(angle_path):
                        print(f"Path not found: {angle_path}")
                        continue
                    
                    # Find matching image
                    # Check if the target_image already has an extension
                    if not target_image.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(angle_path, target_image + ".JPG")
                    else:
                        img_path = os.path.join(angle_path, target_image)
                    
                    if os.path.exists(img_path):
                        matched_data.append({
                            "path": img_path,
                            "angle": angle,
                            "set": set_num,
                            "light": light_num,
                            "sample": sample_num
                        })
                        print(f"Found: Sample {sample_num}, Set {set_num}, Light {light_num}, Angle {angle}° -> {target_image}")
                    else:
                        print(f"Image not found: {img_path}")

    if not matched_data:
        print("No matching images found!")
        return

    print(f"\nProcessing {len(matched_data)} images...")

    # Compute histograms and peaks
    hists, peak_results, y_max = [], [], 0

    for item in matched_data:
        gray = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Failed to read: {item['path']}")
            continue
            
        hist = compute_hist(gray)
        peak_info = compute_peak_support(hist)

        hists.append(hist)
        peak_results.append(peak_info)
        y_max = max(y_max, hist.max())

    # Save individual plots
    ind_root = cfg["output_path"]
    for i, item in enumerate(matched_data):
        sample_folder = get_sample_name(item["sample"], df)
        save_individual_plot(
            item["path"], hists[i], peak_results[i],
            item["angle"], item["light"], item["set"], item["sample"], y_max, ind_root, sample_folder
        )

    # Plot comparison for each sample, set, and light combination
    for sample_num in sample_list:
        sample_folder = get_sample_name(sample_num, df)
        for set_num in set_list:
            for light_num in light_list:
                plot_histograms_per_light(matched_data, hists, cfg, sample_folder, light_num, set_num)

    # Save CSV for each sample
    for sample_num in sample_list:
        sample_folder = get_sample_name(sample_num, df)
        # Filter matched data for this sample
        sample_matched = [d for d in matched_data if d["sample"] == sample_num]
        sample_hists_idx = [i for i, d in enumerate(matched_data) if d["sample"] == sample_num]
        sample_peaks = [peak_results[i] for i in sample_hists_idx]
        
        if sample_matched:
            save_peak_csv(sample_matched, sample_peaks, cfg, sample_folder)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()