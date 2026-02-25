import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np



low = 40
img1 = cv2.imread(r"C:\Users\Deva_pg\Pictures\2026_01_13\IMG_0006.JPG",cv2.IMREAD_GRAYSCALE)
mask = (img1 > low).astype(np.uint8) * 255
gray_hist_1 = cv2.calcHist([img1],[0],mask,[256],[0,256])

# img2 = cv2.imread(r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\dataset\sample_2_2.6x2.0\set_3\20_deg\IMG_0198.JPG",cv2.IMREAD_GRAYSCALE)
# low = 40
# mask = (img2 > low).astype(np.uint8) * 255
# gray_hist_2 = cv2.calcHist([img2],[0],mask,[256],[0,256])

plt.figure(figsize=(11,8))
plt.plot(gray_hist_1)

# fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# axes[0].plot(gray_hist_1, color='y')
# axes[0].set_title("Histogram")
# axes[0].set_xlabel("Intensity")
# axes[0].set_ylabel("Count")

# axes[1].plot(gray_hist_2, color='y')
# axes[1].set_title("With Ambient Light")
# axes[1].set_xlabel("Intensity")

plt.tight_layout()
plt.show()



"""import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------
# INPUT
# --------------------------------------------------
image_path = r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\sample_2_2.6x2.0\0_deg\level_1\IMG_0001.JPG"

img = cv.imread(image_path)

# --------------------------------------------------
# CHANNEL SELECTION
# --------------------------------------------------
B = img[:, :, 0]

# --------------------------------------------------
# MASK: ignore saturated pixels (B == 255)
# --------------------------------------------------
_, mask = cv.threshold(B, 254, 255, cv.THRESH_BINARY_INV)

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
B_hist = cv.calcHist([img], [0], mask, [256], [0, 256])
hist = B_hist.flatten()

# --------------------------------------------------
# SMOOTH HISTOGRAM
# --------------------------------------------------
hist_smooth = gaussian_filter1d(hist, sigma=2)

# --------------------------------------------------
# PEAK CENTER (dominant mode)
# --------------------------------------------------
peak_idx = np.argmax(hist_smooth)
peak_value = hist_smooth[peak_idx]

# --------------------------------------------------
# LOCAL BACKGROUND ESTIMATION
# --------------------------------------------------
# Robust background: lower percentile ignores the peak
background_level = np.percentile(hist_smooth, 5)

# Define how far above background the peak "exists"
# 2–5% works well for your data
support_fraction = 0.02
support_threshold = background_level + support_fraction * (peak_value - background_level)

# --------------------------------------------------
# FIND PEAK SUPPORT (YOUR ARROW REGION)
# --------------------------------------------------

# Left support
left_support = peak_idx
for i in range(peak_idx, 0, -1):
    if hist_smooth[i] <= support_threshold:
        left_support = i
        break

# Right support
right_support = peak_idx
for i in range(peak_idx, len(hist_smooth)):
    if hist_smooth[i] <= support_threshold:
        right_support = i
        break

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
print(f"Peak center intensity     : {peak_idx}")
print(f"Background level          : {background_level:.2f}")
print(f"Support threshold         : {support_threshold:.2f}")
print(f"Peak support range        : [{left_support}, {right_support}]")
print(f"Peak support width (bins) : {right_support - left_support}")

# --------------------------------------------------
# VISUALIZATION (VERIFICATION)
# --------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(hist_smooth, label="Smoothed Histogram")
plt.axvline(left_support, color='purple', linestyle='--', label="Peak Start (support)")
plt.axvline(right_support, color='purple', linestyle='--', label="Peak End (support)")
plt.axvline(peak_idx, color='red', linestyle=':', label="Peak Center")
plt.axhline(support_threshold, color='gray', linestyle=':', label="Support Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("img_without_amb.png")
plt.show()"""