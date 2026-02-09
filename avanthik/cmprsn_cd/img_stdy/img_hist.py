import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CRITICAL FIX: Set Env Var BEFORE importing cv2 ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# NOW import cv2
import cv2 as cv

# Try rawpy
try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    print("Warning: 'rawpy' not found. RAW files will be skipped.")
    RAWPY_AVAILABLE = False

class HistogramGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.input_dir = self.cfg['paths']['input_dir']
        self.output_dir = self.cfg['paths']['output_dir']
        self.use_mask = self.cfg['settings']['use_mask']
        self.combine_plots = self.cfg['settings']['combine_plots']
        self.y_limit = self.cfg['settings']['y_limit']
        
        # New Setting: Default to 'rgb' if missing
        self.hist_mode = self.cfg['settings'].get('histogram_mode', 'rgb').lower()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_image_robust(self, img_path):
        """Loads and normalizes image to 16-bit or 8-bit, handling HDR/RAW outliers."""
        ext = os.path.splitext(img_path)[1].lower()
        img = None
        
        # 1. Handle RAW
        if ext in ['.cr2', '.nef', '.dng', '.arw', '.orf', '.rw2']:
            if RAWPY_AVAILABLE:
                try:
                    with rawpy.imread(img_path) as raw:
                        # output_bps=16 for high fidelity
                        rgb = raw.postprocess(output_bps=16, user_sat=None) 
                        img = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Error reading RAW {img_path}: {e}")
                    return None

        # 2. Standard / EXR Load
        else:
            try:
                img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                return None

        if img is None: return None

        # 3. Handle Floating Point (EXR) -> Normalize to 16-bit
        if img.dtype == np.float32 or img.dtype == np.float16:
            # Robust scaling: map 99.5th percentile to max to ignore specular highlights
            limit_val = np.percentile(img, 99.5)
            if limit_val <= 0: limit_val = img.max()
            
            scale_factor = 65535.0 / limit_val
            img = np.clip(img * scale_factor, 0, 65535)
            img = img.astype('uint16')

        # 4. Handle Low-Dynamic Range 16-bit
        elif img.dtype == np.uint16 and img.max() < 1000:
            # Stretch if data is compressed in low range
            img = cv.normalize(img, None, 0, 65535, cv.NORM_MINMAX)

        # 5. Channel Cleanup
        if len(img.shape) == 3 and img.shape[2] == 4: # Drop Alpha
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        if len(img.shape) == 2: # Gray to Color
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        return img

    def get_otsu_mask(self, img):
        """Generates a binary mask of the object using Otsu's method."""
        # Otsu requires 8-bit single channel
        if img.dtype == np.uint16:
            # Downscale to 8-bit just for mask calculation
            img_8bit = (img / 256).astype('uint8')
        else:
            img_8bit = img.astype('uint8')
            
        gray = cv.cvtColor(img_8bit, cv.COLOR_BGR2GRAY)
        
        # Apply Otsu
        thresh_val, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return mask

    def calculate_histograms(self, image, mask):
        histograms = []
        is_16bit = (image.dtype == np.uint16)
        
        # Parameters
        hist_size = [65536] if is_16bit else [256]
        hist_range = [0, 65536] if is_16bit else [0, 256]

        if self.hist_mode == 'grayscale':
            # Convert BGR (from load_robust) to Grayscale
            gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Calculate single histogram
            hist = cv.calcHist([gray_img], [0], mask, hist_size, hist_range, accumulate=False)
            histograms.append(hist)
        else:
            # Standard RGB (BGR) Calculation
            for channel in range(3):
                hist = cv.calcHist([image], [channel], mask, hist_size, hist_range, accumulate=False)
                histograms.append(hist)
            
        return histograms, image.dtype

    def process(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.exr', '.hdr', '.cr2', '.nef', '.dng')
        
        # Store data for combined plotting
        all_plots_data = [] 

        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_extensions)]
        print(f"Found {len(files)} images to process in {self.hist_mode.upper()} mode...")

        for filename in files:
            img_path = os.path.join(self.input_dir, filename)
            img = self.load_image_robust(img_path)
            
            if img is None: continue

            # --- Masking Logic ---
            mask = None
            if self.use_mask:
                mask = self.get_otsu_mask(img)
                # Check if mask is empty
                if cv.countNonZero(mask) == 0:
                    print(f"Warning: {filename} resulted in empty mask. Using full image.")
                    mask = None

            # --- Calculate ---
            hists, bit_depth = self.calculate_histograms(img, mask)
            sample_name = os.path.splitext(filename)[0]

            if self.combine_plots:
                all_plots_data.append({
                    'name': sample_name,
                    'hists': hists,
                    'depth': bit_depth
                })
                print(f"Analyzed: {filename}")
            else:
                self.save_single_plot(hists, sample_name, bit_depth)
                print(f"Processed & Saved: {filename}")

        if self.combine_plots and all_plots_data:
            self.save_combined_plot(all_plots_data)

    def save_single_plot(self, hists, sample_name, bit_depth):
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_on_axes(ax, hists, sample_name, bit_depth)
        
        output_path = os.path.join(self.output_dir, f"{sample_name}_hist.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def save_combined_plot(self, data_list):
        print("Generating combined plot...")
        fig, ax = plt.subplots(figsize=(15, 8))
        
        is_any_16bit = any(d['depth'] == np.uint16 for d in data_list)
        max_val = 65536 if is_any_16bit else 256
        
        for i, data in enumerate(data_list):
            num_channels = len(data['hists'])
            
            # Determine colors based on mode (RGB or Gray)
            if num_channels == 3:
                colors = ('b', 'g', 'r')
            else:
                colors = ('k',) # Black for grayscale

            linestyles = ['-', '--', ':'] 
            ls = linestyles[i % len(linestyles)]
            
            for hist, col in zip(data['hists'], colors):
                # Logic for legend:
                # If RGB: Label only the Blue channel to avoid 3 labels per image.
                # If Gray: Label the only channel available.
                if num_channels == 3:
                    lbl = data['name'] if col == 'b' else None 
                else:
                    lbl = data['name']

                ax.plot(hist, color=col, linestyle=ls, linewidth=1, alpha=0.7, label=lbl)

        ax.set_title(f"Combined {self.hist_mode.upper()} Histogram ({'Masked' if self.use_mask else 'Full Image'})")
        ax.set_xlim([0, max_val])
        if self.y_limit: ax.set_ylim([0, self.y_limit])
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        
        if len(data_list) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        output_path = os.path.join(self.output_dir, "Combined_Histogram.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved combined plot to {output_path}")

    def _plot_on_axes(self, ax, histograms, title, bit_depth):
        num_channels = len(histograms)
        
        if num_channels == 3:
            colors = ('b', 'g', 'r')
        else:
            colors = ('k',) # Black

        is_16bit = (bit_depth == np.uint16)
        max_val = 65536 if is_16bit else 256
        
        for hist, col in zip(histograms, colors):
            ax.plot(hist, color=col, linewidth=1)

        ax.set_title(f"{title} ({'Masked' if self.use_mask else 'Full'}) - {self.hist_mode.upper()}")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, max_val])
        if self.y_limit: ax.set_ylim([0, self.y_limit])
        ax.set_xlabel(f'Pixel Value (0-{max_val})')
        ax.set_ylabel('Frequency')

if __name__ == "__main__":
    # Point this to your config file
    config_file = r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\img_hist_cfg.json" 
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
    else:
        processor = HistogramGenerator(config_file)
        processor.process()