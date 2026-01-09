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
        self.THRESHOLD = self.config['processing'].get('background_threshold', 1e-6)
        self.VALID_ONLY = self.config['processing'].get('valid_pixels_only', True)
        self.SEP = self.config['processing'].get('csv_separator', ',')

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def read_exr(self, exr_path):
        """Reads an EXR file and returns (height, width, channels_dict)."""
        if not os.path.exists(exr_path):
            raise FileNotFoundError(f"EXR file not found: {exr_path}")

        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        
        # Parse data window to get dimensions
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Determine channels (usually R, G, B for XYZ)
        # Note: Blender might name them R, G, B or similar.
        channel_names = header['channels'].keys()
        
        # Map common channel names for Position
        # Priority: explicit R/G/B, then any 3 channels found
        r_chan = 'R' if 'R' in channel_names else list(channel_names)[0]
        g_chan = 'G' if 'G' in channel_names else list(channel_names)[1]
        b_chan = 'B' if 'B' in channel_names else list(channel_names)[2]

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Read raw bytes
        r_str = exr_file.channel(r_chan, FLOAT)
        g_str = exr_file.channel(g_chan, FLOAT)
        b_str = exr_file.channel(b_chan, FLOAT)
        
        # Convert to numpy arrays
        r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
        g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
        b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
        
        return height, width, r, g, b

    def process_file(self, exr_path, relative_path):
        """Converts a single EXR to CSV."""
        print(f"Processing: {exr_path.name}...")
        
        try:
            h, w, x_map, y_map, z_map = self.read_exr(exr_path)
            
            # Create coordinate grids
            # v is row index (y-pixel), u is col index (x-pixel)
            v_coords, u_coords = np.indices((h, w))
            
            # Flatten arrays for DataFrame creation
            data = {
                'pixel_u': u_coords.flatten(),
                'pixel_v': v_coords.flatten(), # v=0 is usually top or bottom depending on software, Blender is bottom-left origin in UV, top-left in raster.
                'x_world': x_map.flatten(),
                'y_world': y_map.flatten(),
                'z_world': z_map.flatten()
            }
            
            df = pd.DataFrame(data)
            
            # Filter background if requested
            if self.VALID_ONLY:
                # Check magnitude of vector to see if it's effectively zero (background)
                magnitude = np.abs(df['x_world']) + np.abs(df['y_world']) + np.abs(df['z_world'])
                df = df[magnitude > self.THRESHOLD]
            
            # Define output path
            # We mirror the input folder structure inside the output folder
            # e.g. input/500_500/conf_01/world.exr -> output/500_500/conf_01/world.csv
            
            output_subdir = self.output_root / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            csv_filename = exr_path.stem + ".csv"
            output_path = output_subdir / csv_filename
            
            # Save
            df.to_csv(output_path, index=False, sep=self.SEP)
            print(f"  -> Saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Failed to convert {exr_path}: {e}")
            return False

    def run(self):
        print(f"Searching for EXR files in: {self.input_root}")
        
        # Walk through the directory tree
        count = 0
        for root, dirs, files in os.walk(self.input_root):
            for file in files:
                if file.lower().endswith('.exr'):
                    full_path = Path(root) / file
                    
                    # Calculate relative path to maintain folder structure
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