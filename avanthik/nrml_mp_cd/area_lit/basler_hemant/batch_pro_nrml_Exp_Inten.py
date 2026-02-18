import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cupy as cp
import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

class GeneralizedNormalProcessor:
    def __init__(self, config_path):
        """Initializes processing environment from JSON configuration."""
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.verify_dir = self.output_dir / "verification"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verify_dir.mkdir(parents=True, exist_ok=True)

        self.mask_path = Path(self.cfg['paths']['input_mask_path'])
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask not found at: {self.mask_path}")
        
    def load_image(self, img_path, gamma, bit_depth):
        path_str = str(img_path)
        ext = img_path.suffix.lower()
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {path_str}")

        # --- CASE 1: Headerless RAW (.raw) ---
        if ext == '.raw':
            w = self.cfg['resolution']['width']
            h = self.cfg['resolution']['height']
            file_size = os.path.getsize(path_str)
            
            # Expected sizes, basically checking if it's RGB 8-bit, Grayscale 16-bit, or Grayscale 8-bit
            size_rgb_8bit = w * h * 3 
            size_gray_16bit = w * h * 2
            size_gray_8bit = w * h

            if file_size == size_rgb_8bit:
                raw_data = np.fromfile(path_str, dtype=np.uint8)
                img_rgb = raw_data.reshape((h, w, 3)).astype(np.float32) / 255.0
                img = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
            elif file_size == size_gray_16bit:
                img = np.fromfile(path_str, dtype=np.uint16).reshape((h, w))
                img = img.astype(np.float32) / 65535.0
            elif file_size == size_gray_8bit:
                img = np.fromfile(path_str, dtype=np.uint8).reshape((h, w))
                img = img.astype(np.float32) / 255.0
            else:
                raise ValueError(f"File {path_str} size {file_size} does not match resolution {w}x{h}")

        # --- CASE 2: Camera RAW Formats ---
        elif ext in ['.cr2', '.nef', '.dng']:
            import rawpy
            with rawpy.imread(path_str) as raw:
                # user_flip=0 ensures we ignore rotation tags and get raw sensor data
                rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True, user_flip=0)
                img = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2] # these values are based on the human eye sensitivity to different colors, and are commonly used for converting RGB to Grayscale.
                img = img.astype(np.float32) / 65535.0

        # --- CASE 3: Standard Image Formats ---
        else:
            img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if img is None: raise ValueError(f"Failed to decode: {path_str}")
            if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img.dtype != np.float32: 
                current_depth = 16 if img.dtype == np.uint16 else 8
                img = img.astype(np.float32) / (2**current_depth - 1)
            
            if gamma != 1.0: img = np.power(img, gamma)
            
        if self.cfg['global_settings'].get('apply_dark_threshold', False):
            img[img < self.cfg['global_settings']['dark_threshold_value']] = 0.0
            
        return img
    
    
    def get_light_samples_gpu(self, l_cfg):
        center = cp.array(l_cfg['pos_m'], dtype=cp.float32)
        normal = cp.array(l_cfg['norm_dir'], dtype=cp.float32)
        normal /= cp.linalg.norm(normal)
        
        up = cp.array([0, 0, 1], dtype=cp.float32)
        if cp.abs(cp.dot(up, normal)) > 0.99: up = cp.array([1, 0, 0], dtype=cp.float32)
        right = cp.cross(up, normal); right /= cp.linalg.norm(right)
        up_loc = cp.cross(normal, right)

        if l_cfg['shape'] == "RECT":
            w, h = l_cfg['dims_m']
            nx, ny = l_cfg['sampling']
            # sx, sy basically give us the center of a single grid formed by the sampling
            sx = cp.linspace(-0.5 + 1.0/(2*nx), 0.5 - 1.0/(2*nx), nx) * w 
            sy = cp.linspace(-0.5 + 1.0/(2*ny), 0.5 - 1.0/(2*ny), ny) * h
            # using sx and sy, we create a grid of the center points across the area of light
            ii, jj = cp.meshgrid(sx, sy)
            # Finally, we calculate the sample points in 3d by offsetting from the center in the right and up directions
            samples = center + (right[None,None,:] * ii[:,:,None]) + (up_loc[None,None,:] * jj[:,:,None])
        # for now disk litee, but it too is more or less similar, just the discretization is angular and radial
        elif l_cfg['shape'] == "DISK":
            r_max = l_cfg['radius_m']; n_r, n_theta = l_cfg['sampling']
            r_steps = cp.linspace(0, r_max, n_r + 1); theta_steps = cp.linspace(0, 2*cp.pi, n_theta + 1)
            sl = []
            for i in range(n_r):
                for j in range(n_theta):
                    r1, r2 = r_steps[i], r_steps[i+1]; t1, t2 = theta_steps[j], theta_steps[j+1]
                    rc = (2/3) * (r2**3 - r1**3) / (r2**2 - r1**2 + 1e-9); tc = (t1 + t2) / 2.0
                    sl.append(center + (rc * cp.cos(tc) * right) + (rc * cp.sin(tc) * up_loc))
            samples = cp.stack(sl)
        return samples.reshape(-1, 3)

    def detect_plane_corners(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # give the outline of the masked image
        cnt = max(cnts, key=cv2.contourArea) #looks thru the contours in cnts and then judges the size of contour based on physical area, basically to take out the lines generated by noise
        rect = cv2.minAreaRect(cnt)# finds the min area rectangle that fits the contour
        pts = cv2.boxPoints(rect) # calculates the vertices
        s = pts.sum(axis=1) # now we have the 4 vertices, we need to order them in a consistent way (tl, tr, br, bl)
        diff = np.diff(pts, axis=1) # eg(w,0), x is smallest, y is largest -> top left, and so on
        ordered = np.zeros((4, 2), dtype=np.float32)# order the points in tl, tr, br, bl order
        ordered[0] = pts[np.argmin(s)]      # TL
        ordered[2] = pts[np.argmax(s)]      # BR
        ordered[1] = pts[np.argmin(diff)]   # TR
        ordered[3] = pts[np.argmax(diff)]   # BL
        return ordered

    def compute_geometry_with_offset(self, corners, mask, h, w):
        plane_cfg = self.cfg['plane']
        cam_cfg = self.cfg['camera']
        
        # --- FIXED: Read Center Position from Config ---
        # This injects the 2.7cm offset into the geometry
        center_pos = np.array(plane_cfg.get('center_position_cm', [0.0, 0.0, 0.0])) # gets the plane center position in cm, defaulting to (0,0,0) if not specified
        if self.cfg['global_settings']['system_units'] == 'm':
            center_pos = center_pos * 0.01
        # -----------------------------------------------

        center_px = np.array([w/2, h/2]) if cam_cfg.get('use_auto_center', True) else np.array(cam_cfg['manual_center_pixel']) # finding center of the image
        scale = 0.01 if self.cfg['global_settings']['system_units'] == 'm' else 1.0
        w_real, h_real = plane_cfg['dimensions_cm'][0] * scale, plane_cfg['dimensions_cm'][1] * scale # converting plane dimensions to real world units
        
        obj_corners_2d = np.array([[-w_real/2, -h_real/2], [w_real/2, -h_real/2], [w_real/2, h_real/2], [-w_real/2, h_real/2]], dtype=np.float32)
        H, _ = cv2.findHomography(corners, obj_corners_2d) #we know object vertices in image, as well as in the real life, so we find the H matrix using that
        center_h = (H @ np.array([center_px[0], center_px[1], 1.0])) # from H, we find out real world center from the image center, basically [X', Y', W]
        offset_2d = center_h[:2] / center_h[2] #it is de-homogenisation, basically to get rale world coordinates, Real_X= X'/W, Real_Y= Y'/W.....the actual 2D location of that point

        y_grid, x_grid = np.indices((h, w)) # we create a grid of pixel coordinates for the entire image, basically to get the real world coordinates of each pixel in the image
        ones = np.ones_like(x_grid.flatten()) # for z coordinate matching
        coords_homo = np.stack([x_grid.flatten(), y_grid.flatten(), ones]) 
        mapped_homo = H @ coords_homo #mapped x,y,z coordinates in the real world
        p_world_2d = (mapped_homo[:2] / mapped_homo[2]).T # this .T is there for reformatting, before this, it is (2,N), after it becomes (N,2), basically to match everything
        p_world_2d -= offset_2d # making sure worlld origin and camera z axis intersect
        p_world_3d = np.column_stack([p_world_2d, np.zeros(len(p_world_2d))])

        z_rad = np.deg2rad(90 - plane_cfg['elevation_deg'])
        a_rad = np.deg2rad(plane_cfg['azimuth_deg'])
        Rz = np.array([[np.cos(a_rad), -np.sin(a_rad), 0], [np.sin(a_rad), np.cos(a_rad), 0], [0,0,1]])
        Ry = np.array([[np.cos(z_rad), 0, np.sin(z_rad)], [0, 1, 0], [-np.sin(z_rad), 0, np.cos(z_rad)]])
        R = Rz @ Ry #first elevation and then tilt, it kinda gurantees elevation means pitch up
        
        p_world_final = p_world_3d @ R.T #getting the final coordinates after all rotations
        
        # --- FIXED: Apply Global Offset (Z=2.7cm) ---
        p_world_final += center_pos # we want the global center and the camera z axis to be aligned perfecctly, thats' why we shift it
        # --------------------------------------------

        P_map = p_world_final.reshape(h, w, 3).astype(np.float32) # it is now a map, like if we look at say P_map[30,20], we will get the xyz coordinate of that pixel in the real world, it is basically a look up table for pixel to world coordinates
        true_n = cp.array(np.array([0, 0, 1.0]) @ R.T, dtype=cp.float32)

        try:
            #we do all of this to calculate exactly where the center of the plane is in the image, 
            #....and then we can compare it to the camera center to see how much offset there is, 
            #....this is just for verification and visualization purposes, it does not affect the actual normal calculation
            H_inv = np.linalg.inv(H)
            origin_world_pt = np.array([[[0.0, 0.0]]], dtype=np.float32)
            plane_center_pixel = cv2.perspectiveTransform(origin_world_pt, H_inv)[0][0]
        except:
            plane_center_pixel = center_px

        vis_data = {'corners': corners, 'cam_origin': center_px, 'plane_center_proj': plane_center_pixel}
        return cp.array(P_map), true_n, vis_data

    def visualize_coordinate_offset(self, vis_data, mask):
        #we kinda plot the figure to see where the camera thinks the center is (red x),
        #....where the plane center actually is (green +),
        #  and the corners of the plane (cyan dashed line), this is just to verify that our geometry calculations are correct and that the offset is what we expect, ideally the red x and green + should be very close if our geometry is correct and our offset is correctly applied
        plt.figure(figsize=(12, 10))
        plt.imshow(mask, cmap='gray', alpha=0.3)
        cx, cy = vis_data['cam_origin']
        plt.scatter(cx, cy, c='red', marker='x', s=200, linewidth=3, label='Camera Center (Pixel)')
        px, py = vis_data['plane_center_proj']
        plt.scatter(px, py, c='lime', marker='+', s=200, linewidth=3, label='Object Physical Center')
        corn = vis_data['corners']
        plt.plot(np.vstack([corn, corn[0]])[:, 0], np.vstack([corn, corn[0]])[:, 1], 'c--', linewidth=2, label='Boundary')
        plt.title(f"Geometry Verification\nPixel Shift: ({px-cx:.1f}, {py-cy:.1f})")
        plt.legend(); plt.axis('off')
        plt.savefig(self.verify_dir / "geometry_offset_verification.png", bbox_inches='tight')
        plt.close()

    def process(self):
        print("Starting Generalized Normal Processing with Fixed Physics...")
        
        # 1. Load the Mask (This determines our target Geometry)
        mask_cpu = cv2.imread(str(self.mask_path), 0)
        h_ref, w_ref = mask_cpu.shape
        print(f"Reference Geometry (from Mask): {w_ref}x{h_ref}")

        # 2. Geometry & Coordinates
        corners = self.detect_plane_corners(mask_cpu)
        P_full_gpu, true_n, vis_data = self.compute_geometry_with_offset(corners, mask_cpu, h_ref, w_ref)
        self.visualize_coordinate_offset(vis_data, mask_cpu)
        
        df_coords = pd.read_csv(self.cfg['paths']['world_coordinate_csv'])
        df_coords = df_coords[df_coords['alpha'] > self.cfg['global_settings']['alpha_threshold']]
        
        u_coords = df_coords['pixel_u'].values.astype(int)
        v_coords = df_coords['pixel_v'].values.astype(int)
        
        # Clip coordinates to be safe
        u_coords = np.clip(u_coords, 0, w_ref - 1)
        v_coords = np.clip(v_coords, 0, h_ref - 1)

        P_surf = P_full_gpu[v_coords, u_coords]

        # --- FIXED: Aligning Geometry to Light Frame (Y-Inversion) ---
        # Image Y is down, World Y is up. We usually need to flip one.
        P_surf[:, 0] *= 1.0
        P_surf[:, 1] *= -1.0  # <--- Inverted Y-axis
        # -------------------------------------------------------------

        intensities, G_stack = [], []
        
        # 3. Load Images with Safe Orientation Logic
        for light_idx, l_cfg in enumerate(self.cfg['lights']):
            img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
            
            img = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
            h_img, w_img = img.shape
            
            # --- Check and Fix Orientation/Size ---
            if (h_img != h_ref) or (w_img != w_ref):
                if (h_img == w_ref) and (w_img == h_ref):
                    print(f"  -> Fixing orientation for {l_cfg['file_name']} (Rotated 90 deg)")
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                else:
                    print(f"  -> Warning: Resizing {l_cfg['file_name']} to match mask")
                    img = cv2.resize(img, (w_ref, h_ref))
            # -------------------------------------

            intensities.append(cp.array(img[v_coords, u_coords])) # mass data extraction
            
            l_samples = self.get_light_samples_gpu(l_cfg)
            G_accum = cp.zeros((P_surf.shape[0], 3), dtype=cp.float32)
            n_A = cp.array(l_cfg['norm_dir'], dtype=cp.float32)
            cos_half_spread = np.cos(np.deg2rad(l_cfg['spread_deg']/2))

            for pt in l_samples:
                # v = Light_Pos - Surface_Pos
                v = pt - P_surf; dist_sq = cp.sum(v**2, axis=1) + 1e-9; l_k = v / cp.sqrt(dist_sq)[:, None] 
                # v-> actual vector from surface to the area light, dist_sq-> 1/r**2 term for each sample
                cos_emit = -cp.sum(n_A * l_k, axis=1) #the alpha angle in between the light normal and the light vector,
                G_accum += (cp.maximum(0, cos_emit) / dist_sq)[:, None] * l_k * (cos_emit >= cos_half_spread)[:, None]
            G_stack.append(G_accum / l_samples.shape[0])
        
        # Debug: Check Light #1 (after all lights processed)
        if len(G_stack) > 1:
            light_idx = 1
            G_vector = G_stack[light_idx]  # Shape: (N_pixels, 3)
            I_real = intensities[light_idx] # Shape: (N_pixels,)

            # 1. Ground Truth Normal (Flat Board = Up)
            N_gt = cp.array([0, 0, 1.0], dtype=cp.float32)

            # 2. Predicted Intensity (Dot Product)
            # Since N_gt is just [0,0,1], this is just the Z component of G
            I_pred = G_vector[:, 2] 

            # Print measured vs theoretical intensities
            print(f"\n  -> Light {light_idx} Intensity Comparison:")
            print(f"     Measured (I_real):   Mean={cp.mean(I_real):.6f}, Std={cp.std(I_real):.6f}")
            print(f"     Theoretical (I_pred): Mean={cp.mean(I_pred):.6f}, Std={cp.std(I_pred):.6f}")

            # 3. Calculate Ratio (avoid divide by zero)
            ratio_map = I_real / (I_pred + 1e-6)

            # 4. Normalize for visualization (Scale so mean is 1.0)
            #ratio_map /= cp.mean(ratio_map)
            print(f"     Ratio (I_real/I_pred): Mean={cp.mean(ratio_map):.3f}, Std={cp.std(ratio_map):.3f}")
            # 5. Create full image and place values at valid coordinates
            ratio_img = np.zeros((h_ref, w_ref), dtype=np.float32)
            ratio_img[v_coords, u_coords] = cp.asnumpy(ratio_map)
            plt.figure(figsize=(12, 8))
            plt.imshow(ratio_img, cmap='viridis')
            plt.colorbar(label='Intensity Ratio (Measured/Theoretical)')
            plt.title(f'Light {light_idx} - Intensity Ratio Map')
            plt.axis('off')
            plt.show()
            

        # 4. Solve Normals
        I = cp.stack(intensities, axis=1)[:, :, None]
        G = cp.stack(G_stack, axis=1)
        GT = G.transpose(0, 2, 1)
        
        # Regularization to prevent singularities
        GTG_inv = cp.linalg.inv(cp.matmul(GT, G) + cp.eye(3)*1e-4) 
        
        N_raw = cp.matmul(GTG_inv, cp.matmul(GT, I)).squeeze(2)
        Normals = N_raw / (cp.linalg.norm(N_raw, axis=1)[:, None] + 1e-9)
        
        dot = cp.sum(Normals * true_n, axis=1)
        angular_errors = cp.degrees(cp.arccos(cp.clip(dot, -1.0, 1.0)))
        
        self.export_to_csv(u_coords, v_coords, Normals, angular_errors)
        self.generate_outputs(Normals, angular_errors, u_coords, v_coords, h_ref, w_ref)

    def export_to_csv(self, u, v, normals, errors):
        print("Exporting results to CSV...")
        n_cpu = cp.asnumpy(normals)
        err_cpu = cp.asnumpy(errors)
        
        results_df = pd.DataFrame({
            'pixel_u': u,
            'pixel_v': v,
            'normal_x': n_cpu[:, 0],
            'normal_y': n_cpu[:, 1],
            'normal_z': n_cpu[:, 2],
            'angular_error_deg': err_cpu
        })
        
        csv_path = self.output_dir / "pixelwise_normals_errors.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")

    def generate_outputs(self, normals, errors, u, v, h, w):
        err_cpu = cp.asnumpy(errors)
        stats = {k: float(v) for k, v in {"Mean": np.mean(err_cpu), "Median": np.median(err_cpu), "Min": np.min(err_cpu), "Max": np.max(err_cpu), "RMSE": np.sqrt(np.mean(err_cpu**2))}.items()}

        print("\n" + "="*40 + "\n NORMAL RECONSTRUCTION STATISTICS\n" + "="*40)
        for k, val in stats.items(): print(f" {k:<10}: {val:.4f} degrees")
        
        with open(self.output_dir / "error_stats.json", 'w') as f: json.dump(stats, f, indent=4)
        n_map = np.zeros((h, w, 3), dtype=np.float32); n_map[v, u] = cp.asnumpy(normals)
        imageio.imwrite(self.output_dir / "normal_map.tif", n_map)
        
        err_map = np.full((h, w), np.nan, dtype=np.float32); err_map[v, u] = err_cpu
        plt.figure(figsize=(12, 10)); plt.cm.inferno.set_bad(color='black')
        im = plt.imshow(err_map, cmap='inferno', vmin=0, vmax=np.percentile(err_cpu, 98))
        plt.colorbar(im, label='Angular Error'); plt.title(f"Mean: {stats['Mean']:.2f}°")
        plt.axis('off'); plt.savefig(self.output_dir / "error_heatmap_detailed.png", bbox_inches='tight'); plt.close()
        
        vis_map = (n_map + 1.0) / 2.0; vis_map[vis_map == 0.5] = 0
        plt.imsave(self.output_dir / "normal_vis.png", np.clip(vis_map, 0, 1))
        print(f"Done. Visual outputs saved to {self.output_dir}")

if __name__ == "__main__":
    import sys
    default_cfg = r"C:\Users\chand\OneDrive\Desktop\cropped_png_jpg_5th_feb\upd_bat_proc_nrml_grl_cfg.json"
    config_file = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    if os.path.exists(config_file): GeneralizedNormalProcessor(config_file).process()
    else: print(f"Config file not found: {config_file}")