import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- CONFIGURATION LOADING ---
CONFIG_FILE = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\img_contour_cfg.json"

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_contour(img_path, output_dir, config):
    # 1. Load Image
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"Error: File not found - {img_path}")
        return

    # Load as grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image - {img_path}")
        return

    # 2. Preprocessing (Smoothing)
    if config["preprocessing"]["enable_blur"]:
        k_size = config["preprocessing"]["blur_kernel_size"]
        # Kernel size must be odd
        if k_size % 2 == 0: k_size += 1
        img_proc = cv2.GaussianBlur(img, (k_size, k_size), 0)
    else:
        img_proc = img

    # 3. Determine Contour Levels
    lvl_config = config["levels"]
    
    if lvl_config["mode"] == "manual":
        # Use user-defined range
        v_min = lvl_config["manual_range"]["min"]
        v_max = lvl_config["manual_range"]["max"]
    else:
        # Auto-detect range from image min/max
        v_min = float(np.min(img_proc))
        v_max = float(np.max(img_proc))

    # Generate linearly spaced levels
    levels = np.linspace(v_min, v_max, lvl_config["number_of_levels"])

    # 4. Visualization Setup
    viz = config["visualization"]
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Show base image (faded)
    ax.imshow(img, cmap="gray", alpha=0.6)

    # Draw Contours
    cp = ax.contour(
        img_proc, 
        levels=levels, 
        cmap=viz["colormap"], 
        linewidths=viz["line_width"]
    )

    # 5. Formatting
    ax.axis("off")
    ax.set_title(f"Contour Map: {img_path.name}\nRange: {int(v_min)} - {int(v_max)}", fontsize=12)

    # Add Colorbar if requested
    if viz["show_colorbar"]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cp, cax=cax)
        cbar.set_label('Pixel Intensity', rotation=270, labelpad=15)
        
        # Format ticks to be integers for readability
        cbar.set_ticks(levels[::2]) # Show every second tick to avoid crowding
        cbar.set_ticklabels([f"{int(l)}" for l in levels[::2]])

    # 6. Save Output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_name = output_dir / f"contour_{img_path.stem}.png"
    plt.savefig(save_name, bbox_inches='tight', dpi=viz["dpi"])
    plt.close(fig)
    
    print(f"Success: Saved to {save_name}")

def main():
    try:
        config = load_config(CONFIG_FILE)
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        return

    print("--- Starting Contour Generation ---")
    
    inputs = config["input_files"]
    out_dir = config["output_directory"]
    
    if isinstance(inputs, str):
        inputs = [inputs]

    for img_file in inputs:
        generate_contour(img_file, out_dir, config)
        
    print("--- Processing Complete ---")

if __name__ == "__main__":
    main()