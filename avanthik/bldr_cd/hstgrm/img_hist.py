import cv2 as cv
import matplotlib.pyplot as plt
import os

# --- Provided Functions ---

def calculate_histograms(patch, mask):
    histograms = []
    for channel in range(3):
        # OpenCV expects the image in a list: [patch]
        hist = cv.calcHist([patch], [channel], mask, [256], [0, 256], accumulate=False)
        histograms.append(hist)
    return histograms

def plot_histogram(ax, histograms, sample_num, y_limit):
    colors = ('b', 'g', 'r')
    for hist, col in zip(histograms, colors):
        ax.plot(hist, color=col, linewidth=2)
    ax.set_title(f"Sample: {sample_num}", fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 256])
    ax.set_ylim([0, y_limit])
    ax.set_xlabel('Pixel Value', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)

# --- Main Processing Loop ---

def process_directory(input_dir, output_dir, y_limit=5000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Supported extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_dir, filename)
            
            # Load image
            img = cv.imread(img_path)
            if img is None:
                print(f"Skipping {filename}: Could not read image.")
                continue

            # Calculate Histograms (No mask used here, passing None)
            hists = calculate_histograms(img, None)

            # Create a figure for plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use the filename (without extension) as the sample identifier
            sample_name = os.path.splitext(filename)[0]
            plot_histogram(ax, hists, sample_name, y_limit)

            # Save the plot
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, bbox_inches='tight')
            
            # Close the plot to free up memory
            plt.close(fig)
            print(f"Processed: {filename}")

# --- Configuration ---

input_folder = r"C:\Users\vishn\Desktop\avanthik\blender_outputs\main_storage\best_exposure_rendered_outputs\samples_16\0.00_60.00_0.00_0.00_0.00_9.57\4_30.00_area_rectangle_60.0_45.00_1.17"
output_folder = r"C:\Users\vishn\Desktop\avanthik\blender_outputs\main_storage\best_exposure_rendered_outputs\samples_16\0.00_60.00_0.00_0.00_0.00_9.57\4_30.00_area_rectangle_60.0_45.00_1.17\histogram"

# Run the process
process_directory(input_folder, output_folder, y_limit=10000)