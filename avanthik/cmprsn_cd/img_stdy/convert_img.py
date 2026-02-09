import os
import json
import numpy as np
import rawpy
import imageio.v3 as iio

# --- CRITICAL: Enable OpenEXR for OpenCV ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 

class UniversalConverter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.img_data = None  
        self.is_linear_data = False # Tracks the mathematical state of the data

    def run(self):
        print(f"{'='*40}\nSTARTING CONVERSION (LOSSLESS PIPELINE)\n{'='*40}")
        try:
            self._load_image()
            self._process_image()
            self._save_image()
            print(f"\n{'-'*40}\nSUCCESS: Conversion Complete.\n{'-'*40}")
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _load_image(self):
        path = self.cfg['io_settings']['input_path']
        ext = os.path.splitext(path)[1].lower()
        print(f"Loading: {os.path.basename(path)} ({ext})")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

        # --- CASE A: Camera Raw (ALWAYS LINEAR LOAD) ---
        if ext in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.raf']:
            with rawpy.imread(path) as raw:
                if self.cfg['processing_pipeline']['demosaic_raw']:
                    print("Status: Demosaicing Raw to RGB (Force Linear 16-bit)...")
                    
                    # FORCE LINEARITY: gamma=(1,1) disables the hidden sRGB curve.
                    rgb = raw.postprocess(
                        use_camera_wb=True, 
                        no_auto_bright=True, 
                        bright=1.0, 
                        user_sat=None, 
                        output_bps=16,
                        gamma=(1, 1) # <--- THE KEY to Zero Data Loss
                    )
                    
                    self.img_data = rgb.astype(np.float32) / 65535.0
                    self.is_linear_data = True # Flag as Linear
                    del rgb 

                else:
                    print("Status: Reading Raw Bayer Pattern (No Demosaic)...")
                    raw_data = raw.raw_image.astype(np.float32)
                    white_level = raw.white_level
                    self.img_data = raw_data / float(white_level)
                    self.img_data = np.expand_dims(self.img_data, axis=2)
                    self.is_linear_data = True

        # --- CASE B: High-End Float (Always Linear) ---
        elif ext in ['.exr', '.hdr']:
            data = cv2.imread(path, flags=-1)
            if data is None: raise ValueError("Failed to read EXR/HDR.")
            
            if data.ndim == 3:
                if data.shape[2] == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                elif data.shape[2] == 4:
                    data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
            
            self.img_data = data.astype(np.float32)
            self.is_linear_data = True 

        # --- CASE C: Blind Binary ---
        elif ext == '.raw':
            print("Status: Reading Blind Binary Stream...")
            bs = self.cfg['blind_raw_settings']
            dtype_map = {'uint8': np.uint8, 'uint16': np.uint16, 'float32': np.float32}
            
            raw_data = np.fromfile(path, dtype=dtype_map[bs['dtype']])
            expected_pixels = bs['width'] * bs['height'] * bs['channels']
            
            if len(raw_data) != expected_pixels:
                raise ValueError(f"Size mismatch! Expected {expected_pixels}, got {len(raw_data)}.")

            raw_data = raw_data.reshape((bs['height'], bs['width'], bs['channels']))
            
            if bs['dtype'] == 'uint8':
                self.img_data = raw_data.astype(np.float32) / 255.0
            elif bs['dtype'] == 'uint16':
                self.img_data = raw_data.astype(np.float32) / 65535.0
            else:
                self.img_data = raw_data.astype(np.float32)
            
            # Assume Binary dumps are linear unless proven otherwise
            self.is_linear_data = True 

        # --- CASE D: Standard Image (JPG/PNG) - Gamma Encoded ---
        else:
            data = cv2.imread(path, flags=-1)
            if data is None: raise ValueError("Failed to read standard image.")
            
            if data.ndim == 3:
                if data.shape[2] == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                elif data.shape[2] == 4:
                    data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)
            elif data.ndim == 2:
                data = np.expand_dims(data, axis=2)

            if data.dtype == np.uint8:
                self.img_data = data.astype(np.float32) / 255.0
            elif data.dtype == np.uint16:
                self.img_data = data.astype(np.float32) / 65535.0
            else:
                self.img_data = data.astype(np.float32)

            self.is_linear_data = False # These are sRGB (Gamma 2.2)

        print(f"Internal Data Loaded. Shape: {self.img_data.shape}, Linear State: {self.is_linear_data}")

    def _process_image(self):
        pipeline = self.cfg['processing_pipeline']
        cm = pipeline['color_management']
        gamma = cm['gamma_value']

        # A. LINEARIZATION (sRGB -> Linear)
        if cm['mode'] == 'srgb_to_linear':
            # SMART CHECK: If data is already linear (Raw/EXR), DON'T double-linearize.
            if self.is_linear_data:
                print("Status: Input is naturally Linear. Skipping 'srgb_to_linear' step.")
            else:
                print(f"Status: Linearizing sRGB input (Pixel ^ {gamma})...")
                if self.img_data.shape[2] >= 3:
                    self.img_data[:,:,:3] = np.power(self.img_data[:,:,:3], gamma)
                else:
                    self.img_data = np.power(self.img_data, gamma)
                self.is_linear_data = True

        # B. TONE MAPPING (Works best on Linear Data)
        if pipeline['tone_mapping']['enable']:
            print(f"Status: Applying Tone Mapping ({pipeline['tone_mapping']['method']})...")
            
            if not self.is_linear_data:
                print("WARNING: Tone Mapping Non-Linear Data. Results may be inaccurate.")

            if self.img_data.shape[2] >= 3:
                lum = 0.2126 * self.img_data[:,:,0] + 0.7152 * self.img_data[:,:,1] + 0.0722 * self.img_data[:,:,2]
                lum_compressed = lum / (1.0 + lum)
                ratio = lum_compressed / (lum + 1e-6)
                self.img_data[:,:,0] *= ratio
                self.img_data[:,:,1] *= ratio
                self.img_data[:,:,2] *= ratio
            else:
                self.img_data = self.img_data / (1 + self.img_data)
        
        # C. GAMMA ENCODING (Linear -> sRGB)
        if cm['mode'] == 'linear_to_srgb':
            print(f"Status: Applying Gamma Encoding (Pixel ^ 1/{gamma})...")
            target_channels = 3 if self.img_data.shape[2] >= 3 else 1
            safe_data = np.maximum(self.img_data[:,:,:target_channels], 1e-6)
            self.img_data[:,:,:target_channels] = np.power(safe_data, 1.0 / gamma)
            self.is_linear_data = False

        # D. SAFETY CLIPPING
        tgt_dtype = self.cfg['target_format']['dtype']
        if tgt_dtype in ['uint8', 'uint16']:
            self.img_data = np.clip(self.img_data, 0.0, 1.0)
        else:
            self.img_data = np.maximum(self.img_data, 0.0)

    def _save_image(self):
        tgt = self.cfg['target_format']
        io_set = self.cfg['io_settings']
        
        if not os.path.exists(io_set['output_folder']):
            os.makedirs(io_set['output_folder'])
        
        out_path = os.path.join(io_set['output_folder'], f"{io_set['output_filename']}{tgt['extension']}")
        print(f"Saving to: {out_path}")

        final_data = None
        
        if tgt['dtype'] == 'uint8':
            final_data = (self.img_data * 255.0).astype(np.uint8)
        elif tgt['dtype'] == 'uint16':
            final_data = (self.img_data * 65535.0).astype(np.uint16)
        elif tgt['dtype'] == 'float32':
            final_data = self.img_data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported target dtype: {tgt['dtype']}")

        if tgt['extension'] == '.raw':
            print(f"Status: Dumping Binary Stream ({final_data.nbytes} bytes)")
            final_data.tofile(out_path)
        else:
            save_img = final_data
            if final_data.ndim == 3:
                if final_data.shape[2] == 3:
                    save_img = cv2.cvtColor(final_data, cv2.COLOR_RGB2BGR)
                elif final_data.shape[2] == 4:
                    save_img = cv2.cvtColor(final_data, cv2.COLOR_RGBA2BGRA)
            
            success = cv2.imwrite(out_path, save_img)
            if not success:
                iio.imwrite(out_path, final_data)

if __name__ == "__main__":
    converter = UniversalConverter(r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\img_stdy\convert_img_cfg.json")
    converter.run()