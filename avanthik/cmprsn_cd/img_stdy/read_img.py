import rawpy
import imageio.v3 as iio
import os
import numpy as np
import exifread

# --- FIX: Set this BEFORE importing cv2 ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 

def get_image_details(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    # Camera Raw formats
    camera_raw_exts = ['.cr2', '.nef', '.arw', '.dng', '.orf', '.raf']
    
    print(f"{'='*30}")
    print(f"ANALYSIS: {filename}")
    print(f"{'='*30}")

    try:
        # --- CASE 1: High-End Float Formats (EXR, HDR) ---
        if ext in ['.exr', '.hdr']:
            # flags=-1 is the same as cv2.IMREAD_UNCHANGED
            data = cv2.imread(file_path, flags=-1)
            
            if data is None:
                raise ValueError("OpenCV failed to load. File might be corrupted or path is wrong.")

            print(f"Extension   : {ext}")
            print(f"Structure   : {data.ndim}D Array (High Dynamic Range)")
            print(f"Resolution  : {data.shape[1]} x {data.shape[0]} (Width x Height)")
            print(f"Array Shape : {data.shape} (Height, Width, Channels)")
            print(f"Data Type   : {data.dtype}")
            
            # Detailed Bit Depth Check
            if data.dtype == np.float32:
                print(f"Bit Depth   : 32-bit Float (True HDR)")
            elif data.dtype == np.float16:
                print(f"Bit Depth   : 16-bit Float (Half Float)")
            else:
                print(f"Bit Depth   : {data.itemsize * 8}-bit (Converted)")

        # --- CASE 2: Camera Raw (CR2, NEF) ---
        elif ext in camera_raw_exts:
            with rawpy.imread(file_path) as raw:
                data = raw.raw_image
                bit_depth = raw.white_level.bit_length()
                
                print(f"Extension   : {ext}")
                print(f"Structure   : 2D Array (Bayer Mosaic)")
                print(f"Resolution  : {data.shape[1]} x {data.shape[0]} (Width x Height)")
                print(f"Array Shape : {data.shape} (Height, Width)")
                print(f"Data Type   : {data.dtype}")
                print(f"Bit Depth   : {bit_depth}-bit (Sensor Data)")

        # --- CASE 3: Pure Binary (.raw) ---
        elif ext == '.raw':
            data = np.fromfile(file_path, dtype=np.uint8)
            print(f"Extension   : {ext}")
            print(f"Structure   : 1D Flat Array (Binary Stream)")
            print(f"Total Bytes : {len(data)}")
            print(f"Data Type   : {data.dtype}")
            print("Note: Resolution is unknown without a header.")

        # --- CASE 4: Standard Images (JPG, PNG, TIF) ---
        else:
            # TRY OPENCV FIRST WITH 'UNCHANGED' FLAG
            # This is critical for 16-bit PNG/TIFF detection
            data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            # Fallback to imageio if OpenCV fails (e.g. for some gifs or webp)
            if data is None:
                data = iio.imread(file_path)

            print(f"Extension   : {ext}")
            print(f"Structure   : {data.ndim}D Array (Standard)")
            
            # Handle grayscale vs color shapes
            if data.ndim == 2:
                h, w = data.shape
                c = 1
            else:
                h, w, c = data.shape
                
            print(f"Resolution  : {w} x {h}")
            print(f"Array Shape : {data.shape}")
            print(f"Data Type   : {data.dtype}")
            
            # Accurate Bit Depth Calculation
            bit_depth = data.itemsize * 8
            print(f"Bit Depth   : {bit_depth}-bit")
            
            if bit_depth == 16:
                print("STATUS: CONFIRMED 16-BIT IMAGE")
            elif bit_depth == 8:
                print("STATUS: 8-BIT IMAGE (Standard)")

    except Exception as e:
        print(f"ERROR: {e}")

def check_orientation(file_path):
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f)
        # Orientation Tag IDs:
        # 1 = Horizontal (Landscape)
        # 6 = Rotated 90 CW (Portrait)
        # 8 = Rotated 270 CW (Portrait)
        orientation = tags.get('Image Orientation')
        print(f"Metadata Tag: {orientation}")


# --- USER INPUT ---
# Make sure to use raw string (r"path") for Windows paths to avoid backslash errors
image_path = r"C:\Users\vishn\Desktop\avanthik\cmr_op\basler\simulation_003\cr2_cmr_op\light_001.CR2"

if os.path.exists(image_path):
    get_image_details(image_path)
    check_orientation(image_path)
    print("\n")
else:
    print("File not found.")