import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

CONFIG_FILE = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram.json"


def load_config(filename):
    with open(filename, "r") as f:
        return json.load(f)


def compute_histogram(img):
    low = 10
    mask = (img > low).astype(np.uint8) * 255
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
    return hist, mask


def process_image(img_path, output_folder, global_y_max):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    hist, _ = compute_histogram(img)

    plt.figure(figsize=(12, 6))
    plt.plot(hist)
    plt.ylim(0, global_y_max)   

    plt.title("Grayscale Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_folder, f"{img_name}_hist.png")

    plt.savefig(save_path)
    plt.close()


def process_sample(sample_id, base_path, output_base,sample_df):
    
    row = sample_df[sample_df["Sample Number"] == sample_id]
    length = row["Length"].values[0]
    breadth = row["Breadth"].values[0]
    
    sample_name = f"sample_{sample_id}_{length}x{breadth}"
    sample_folder = os.path.join(base_path, sample_name)
    output_folder = os.path.join(output_base, f"sample_{sample_id}")

    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(sample_folder, "*.JPG")))
    print(f"sample_{sample_id}: {len(image_paths)} images")

    
    global_y_max = 0

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        hist, _ = compute_histogram(img)
        global_y_max = max(global_y_max, hist.max())

    
    for img_path in image_paths:
        process_image(img_path, output_folder, global_y_max)


def main():
    config = load_config(CONFIG_FILE)

    base_path = config["base_path"]
    output_base = config["output_path"]
    csv_path = config["csv_path"]

    start = config["sample_num"]["start"]
    stop = config["sample_num"]["stop"]
    step = config["sample_num"]["step"]
    
    sample_df = pd.read_csv(csv_path)

    for sample_id in range(start, stop, step):
        process_sample(sample_id, base_path, output_base,sample_df)


if __name__ == "__main__":
    main()
