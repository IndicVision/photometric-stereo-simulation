import os
import json
import numpy as np
import OpenEXR
import Imath
import pandas as pd
from pathlib import Path

class ExrToCsvConverter:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.input_root = Path(self.config['paths']['input_root_dir'])
        self.output_root = Path(self.config['paths']['output_csv_dir'])
        
        # Settings
        self.SEP = self.config['processing'].get('csv_separator', ',')
        self.VALID_ONLY = self.config['processing'].get('valid_pixels_only', True)

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def read_exr(self, exr_path):
        """Reads an EXR file and returns (height, width, r, g, b, a)."""
        if not os.path.exists(exr_path):
            raise FileNotFoundError(f"EXR file not found: {exr_path}")

        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        channel_names = header['channels'].keys()
        
        # Map Channels (R, G, B, A)
        r_chan = 'R' if 'R' in channel_names else list(channel_names)[0]
        g_chan = 'G' if 'G' in channel_names else list(channel_names)[1]
        b_chan = 'B' if 'B' in channel_names else list(channel_names)[2]
        a_chan = 'A' if 'A' in channel_names else 'Alpha' # Try 'A' then 'Alpha'

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Helper to read channel safely
        def get_chan(name):
            if name in channel_names:
                return np.frombuffer(exr_file.channel(name, FLOAT), dtype=np.float32).reshape(height, width)
            # Fallback for Alpha if missing (treat as fully opaque 1.0)
            if name == 'Alpha' or name == 'A':
                return np.ones((height, width), dtype=np.float32)
            raise ValueError(f"Channel {name} not found in EXR")

        r = get_chan(r_chan)
        g = get_chan(g_chan)
        b = get_chan(b_chan)
        a = get_chan(a_chan)
        
        return height, width, r, g, b, a

    def process_file(self, exr_path, relative_path):
        """Converts a single EXR to CSV with Un-Premultiplication."""
        print(f"Processing: {exr_path.name}...")
        
        try:
            h, w, r, g, b, a = self.read_exr(exr_path)
            
            # --- MATH FIX: Un-premultiply Alpha ---
            # EXR stores data as (Value * Alpha). We must divide by Alpha to get true Value.
            # 1. Avoid division by zero
            valid_mask = a > 0.00001
            
            x_world = np.zeros_like(r)
            y_world = np.zeros_like(g)
            z_world = np.zeros_like(b)
            
            # 2. Restore original coordinates
            x_world[valid_mask] = r[valid_mask] / a[valid_mask]
            y_world[valid_mask] = g[valid_mask] / a[valid_mask]
            z_world[valid_mask] = b[valid_mask] / a[valid_mask]
            
            # Create coordinate grids
            v_coords, u_coords = np.indices((h, w))
            
            data = {
                'pixel_u': u_coords.flatten(),
                'pixel_v': v_coords.flatten(),
                'x_world': x_world.flatten(),
                'y_world': y_world.flatten(),
                'z_world': z_world.flatten(),
                'alpha': a.flatten()
            }
            
            df = pd.DataFrame(data)
            
            # --- LOGIC FIX: Filter by Alpha, NOT Coordinate Value ---
            if self.VALID_ONLY:
                # Discard pixels that are mostly transparent (background)
                # Using 0.5 threshold ensures we keep edge pixels but drop background
                df = df[df['alpha'] > 0.5]
            
            # Define output path
            output_subdir = self.output_root / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            csv_filename = exr_path.stem + ".csv"
            output_path = output_subdir / csv_filename
            
            df.to_csv(output_path, index=False, sep=self.SEP)
            print(f"  -> Saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Failed to convert {exr_path}: {e}")
            return False

    def run(self):
        print(f"Searching for EXR files in: {self.input_root}")
        
        count = 0
        for root, dirs, files in os.walk(self.input_root):
            for file in files:
                if file.lower().endswith('.exr'):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self.input_root)
                    self.process_file(full_path, rel_path)
                    count += 1
        
        print("="*60)
        print(f"Extraction Complete. Processed {count} files.")

if __name__ == "__main__":
    # --- UPDATE PATH HERE ---
    CONFIG_PATH = r"C:\Users\vishn\Desktop\avanthik\bldr_cd\grl_rndr\exrTOcsv_px_gbl_cord_cfg.json"
    
    converter = ExrToCsvConverter(CONFIG_PATH)
    converter.run()