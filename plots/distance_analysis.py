from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Optional

#specify type of analysis
config_path = Path(r"D:\Chandana\Photometric_Stereo\blender\scripts\analysis\analysis_input_parameters.json")


def parse_obj_plane(name: str):
    """Parse: <azimuth>_<elevation>_<x>_<y>_<z>_<area>"""
    p = name.split('_')
    return tuple(map(float, p)) + (name,) if len(p) == 6 else None


def parse_light(name: str):
    """Parse: <num_lights>_<distance>_<light_type>_<params>_<psi>_<energy>"""
    p = name.split('_')
    if len(p) < 6:
        return None
    try:
        n, d, t = int(p[0]), float(p[1]), p[2]
        psi, e = float(p[-2]), float(p[-1])
        params = '_'.join(p[3:-2])
        # Parse area params: rectangle_0.0_30.25 or rectangle_0.0_dim_a_dim_b
        sp, sh, da, db, area_val = None, None, None, None, None
        if t == "area" and params:
            pr = params.split('_')
            if len(pr) >= 2:
                sh, sp = pr[0], float(pr[1])
                if len(pr) >= 3:
                    if len(pr) == 3:
                        # Area is given directly: rectangle_0.0_30.25
                        area_val = float(pr[2])
                    elif len(pr) >= 4:
                        # Dimensions given: rectangle_0.0_dim_a_dim_b
                        da, db = float(pr[2]), float(pr[3])
        return (n, d, t, params, psi, e, name, sp, sh, da, db, area_val)
    except:
        return None


def build_plot_filename(lc):
    """
    Build plot filename excluding distance (since we're analyzing across distances):
    Format: <num_lights>_<light_type>_<shape>_<spread_value>_<area_value>_<psi>_<energy>
    Example: 4_area_rectangle_30.0_30.25_45.00_1.17
    """
    n, d, t, params, psi, e = lc[0], lc[1], lc[2], lc[3], lc[4], lc[5]
    sp, sh, da, db, area_val = lc[7], lc[8], lc[9], lc[10], lc[11]
    
    # Format: num_lights_light_type_shape_spread_area_psi_energy (excluding distance)
    if t == "area":
        if sp is not None and sh is not None:
            if area_val is not None:
                # Format: num_lights_area_shape_spread_area_value_psi_energy
                filename = f"{n}_{t}_{sh}_{sp}_{area_val}_{psi:.2f}_{e}.png"
            elif da is not None and db is not None:
                # Calculate area from dimensions
                if sh == "rectangle":
                    area_calc = da * db
                elif sh == "ellipse":
                    import math
                    area_calc = math.pi * da * db
                else:
                    area_calc = da * db  # Fallback
                filename = f"{n}_{t}_{sh}_{sp}_{area_calc}_{psi:.2f}_{e}.png"
            else:
                # Fallback: use spread and shape
                filename = f"{n}_{t}_{sh}_{sp}_unknown_{psi:.2f}_{e}.png"
        else:
            # Fallback to basic format
            filename = f"{n}_{t}_{params}_{psi:.2f}_{e}.png"
    elif t == "sun":
        # For sun light: num_lights_sun_angle_psi_energy
        angle_str = params if params else "angle"
        filename = f"{n}_{t}_{angle_str}_{psi:.2f}_{e}.png"
    elif t == "spot":
        # For spot light: num_lights_spot_beam_angle_psi_energy
        beam_angle_str = params if params else "beam_angle"
        filename = f"{n}_{t}_{beam_angle_str}_{psi:.2f}_{e}.png"
    else:
        # Fallback
        filename = f"{n}_{t}_{params}_{psi:.2f}_{e}.png"
    
    return filename


def parse_img(name: str):
    """Parse: <source>_<exposure>.png"""
    p = Path(name).stem.split('_')
    return (int(p[0]), float(p[1])) if len(p) == 2 else None


def expand(v):
    """Convert range/array/single to list"""
    if isinstance(v, list):
        return v
    if isinstance(v, dict) and "start" in v:
        s, e, st = float(v["start"]), float(v["end"]), float(v.get("step", 1.0))
        return [s + i * st for i in range(int((e - s) / st) + 1) if s + i * st <= e + 1e-6]
    return [float(v)]


def load_light_matrix(file_path: Path) -> np.ndarray:
    """Load light matrix from YAML file."""
    fs = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_READ)
    S = fs.getNode("Lights").mat()
    fs.release()
    
    if S is None:
        raise ValueError(f"Could not read 'Lights' from {file_path}")
    
    return S.astype(np.float32)


def load_ground_truth_normal(file_path: Path) -> np.ndarray:
    """Load ground truth normal from YAML file."""
    fs = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_READ)
    normal = fs.getNode("GroundTruthNormal").mat()
    fs.release()
    
    if normal is None:
        raise ValueError(f"Could not read 'GroundTruthNormal' from {file_path}")
    
    normal = normal.flatten().astype(np.float32)
    normal = normal / np.linalg.norm(normal)  # Ensure unit vector
    return normal


def photometric_stereo_rgb(images: np.ndarray, light_matrix: np.ndarray, use_mask: bool = True) -> np.ndarray:
    """Compute surface normals using photometric stereo on RGB images."""
    num_lights, height, width, _ = images.shape
    
    # Create mask
    if use_mask:
        max_intensity = np.max(images, axis=0)
        object_mask = np.sum(max_intensity, axis=2) > 0.01
    else:
        object_mask = np.ones((height, width), dtype=bool)
    
    # Compute pseudo-inverse
    if num_lights == 3:
        S_pinv = np.linalg.inv(light_matrix)
    else:
        S_pinv = np.linalg.pinv(light_matrix)
    
    normals_per_channel = []
    
    # Process each color channel
    for channel_idx in range(3):
        intensities = images[:, :, :, channel_idx]
        intensities_flat = intensities.reshape(num_lights, -1)
        
        normals_flat = S_pinv @ intensities_flat
        norms = np.linalg.norm(normals_flat, axis=0)
        valid = (norms > 1e-6) & object_mask.ravel()
        
        unit_normals_flat = np.zeros_like(normals_flat)
        if np.sum(valid) > 0:
            unit_normals_flat[:, valid] = normals_flat[:, valid] / norms[valid]
        
        normals_channel = unit_normals_flat.reshape(3, height, width).transpose(1, 2, 0)
        normals_per_channel.append(normals_channel)
    
    # Average across channels
    normals_avg = np.mean(normals_per_channel, axis=0)
    norms_final = np.linalg.norm(normals_avg, axis=2, keepdims=True)
    valid_final = (norms_final > 1e-6) & object_mask[:, :, np.newaxis]
    
    normals_final = np.where(valid_final, normals_avg / norms_final, 0)
    return normals_final


class Analyzer:
    def __init__(self, num_lights=4):
        self.num_lights = num_lights
    
    def load_images(self, folder: Path, exp_value=None):
        """Load images 001-004 with specified exposure"""
        if exp_value is None:
            raise ValueError("Exposure value must be provided")
        
        # Load images 001 to num_lights with the specified exposure (3 decimal places)
        frames = []
        for source_num in range(1, self.num_lights + 1):
            # Format: 001_exposure.png with 3 decimal places (e.g., 001_0.006.png)
            img_name = f"{source_num:03d}_{exp_value:.3f}.png"
            img_path = folder / img_name
            if not img_path.exists():
                raise ValueError(f"Image not found: {img_name} in {folder}")
            
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            frames.append(img)
        
        if len(frames) < self.num_lights:
            raise ValueError(f"Not enough images found in {folder}. Need {self.num_lights}, found {len(frames)}")
        
        return np.stack(frames, axis=0)
    
    def process(self, folder: Path, exp_value=None, use_mask=True):
        """Process folder and return error, errors, normals"""
        imgs = self.load_images(folder, exp_value)
        
        # Load light matrix and ground truth from YAML files in the folder
        light_matrix_path = folder / "light_matrix.yml"
        ground_truth_path = folder / "ground_truth_normal.yml"
        
        if not light_matrix_path.exists():
            raise ValueError(f"Light matrix file not found: {light_matrix_path}")
        if not ground_truth_path.exists():
            raise ValueError(f"Ground truth normal file not found: {ground_truth_path}")
        
        S = load_light_matrix(light_matrix_path)
        gt_normal = load_ground_truth_normal(ground_truth_path)
        
        # Compute normals
        n = photometric_stereo_rgb(imgs, S, use_mask)
        
        # Compute error
        mask = np.sum(np.max(imgs, axis=0), axis=2) > 0.01 if use_mask else np.ones((imgs.shape[1], imgs.shape[2]), dtype=bool)
        dp = np.dot(n, gt_normal)
        v = np.linalg.norm(n, axis=2) > 1e-6
        err = np.zeros_like(v, dtype=np.float32)
        err[v] = np.arccos(np.clip(dp[v], -1.0, 1.0)) * 180.0 / np.pi
        err[~mask] = np.nan
        return float(np.nanmean(err)), err[mask], n


def matches(lc, filters, _):
    """Check if config matches filters"""
    if not filters:
        return True
    if "num_lights" in filters and lc[0] not in expand(filters["num_lights"]):
        return False
    if "light_distance_cm" in filters and lc[1] not in expand(filters["light_distance_cm"]):
        return False
    if "light_type" in filters and lc[2] != filters["light_type"]:
        return False
    if "psi_angle_deg" in filters and lc[4] not in expand(filters["psi_angle_deg"]):
        return False
    if "energy" in filters and lc[5] not in expand(filters["energy"]):
        return False
    if lc[2] == "area":
        if "spread_angle_deg" in filters and (lc[7] is None or lc[7] not in expand(filters["spread_angle_deg"])):
            return False
        # Check area_of_light_cm2
        if "area_of_light_cm2" in filters:
            area_match = False
            # Check if area is given directly (lc[11])
            if len(lc) > 11 and lc[11] is not None:
                if lc[11] in expand(filters["area_of_light_cm2"]):
                    area_match = True
            # Or compute from dimensions (lc[9] * lc[10])
            elif lc[9] is not None and lc[10] is not None:
                area = lc[9] * lc[10]
                if area in expand(filters["area_of_light_cm2"]):
                    area_match = True
            if not area_match:
                return False
    return True


def discover(base_dir: Path, config_params, filters=None, planes=None):
    """Build config name and discover configurations"""
    o, c, a = config_params["plane_orientations"][0], config_params["plane_center_coordinates"][0], config_params["plane_area_cm2"][0]
    config_name = f"{o['azimuth_deg']:.2f}_{o['elevation_deg']:.2f}_{c['x']:.2f}_{c['y']:.2f}_{c['z']:.2f}_{a:.2f}"
    config_folder = base_dir / config_name
    if not config_folder.exists():
        return {}
    oc = parse_obj_plane(config_name)
    lcs = [lc for lf in config_folder.iterdir() if lf.is_dir() and (lc := parse_light(lf.name)) and matches(lc, filters, lf)]
    return {oc: lcs} if lcs else {}


def process_config(oc, lcs, base_dir, out_dir, num_lights, exposure_value, plot_params=None, use_mask=True):
    """Process configuration"""
    # Create folder with config name (e.g., 0.00_60.00_0.00_0.00_0.00_9.57)
    config_folder = out_dir / oc[6]
    config_folder.mkdir(parents=True, exist_ok=True)
    
    # Create distanceanalysis subfolder
    dist_analysis_folder = config_folder / "distanceanalysis"
    dist_analysis_folder.mkdir(parents=True, exist_ok=True)
    
    groups = {}
    for lc in lcs:
        k = f"{lc[0]}_{lc[2]}_{lc[3]}_{lc[4]}_{lc[5]}"
        groups.setdefault(k, []).append(lc)
    a = Analyzer(num_lights)
    for glcs in groups.values():
        glcs.sort(key=lambda x: x[1])
        res = []
        for lc in glcs:
            f = base_dir / oc[6] / lc[6]
            if not f.exists():
                continue
            try:
                me, es, ns = a.process(f, exposure_value, use_mask)
                res.append((lc[1], me, ns))
                print(f"    Distance {lc[1]:.2f} cm: Error = {me:.4f} deg")
            except Exception as ex:
                print(f"    Error at {lc[1]:.2f} cm: {ex}")
                continue
        if res:
            ds, es, ns = zip(*res)
            min_dist, max_dist = min(ds), max(ds)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(ds, es, marker="o", markersize=8, linestyle="-", linewidth=2)
            ax.set_xlabel(plot_params.get("plot_x_label", "Distance (cm)") if plot_params else "Distance (cm)", fontsize=12)
            ax.set_ylabel(plot_params.get("plot_y_label", "Error Magnitude (deg)") if plot_params else "Error Magnitude (deg)", fontsize=12)
            ax.set_title(plot_params.get("plot_title", "Distance Analysis") if plot_params else "Distance Analysis", fontsize=14, fontweight="bold")
            ax.set_xlim(min_dist - 5, max_dist + 5)
            ax.grid(True, alpha=0.4)
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            
            # Build filename excluding distance (since we're analyzing across distances)
            # All folders in the group have the same structure except distance
            first_lc = glcs[0]
            plot_filename = build_plot_filename(first_lc)
            fig.savefig(dist_analysis_folder / plot_filename, facecolor="white", dpi=150)
            plt.close(fig)
            


def main():
    """Main function"""
    # Load from analysis_input_parameters.json
    with open(config_path) as f:
        c = json.load(f)["analysis_config"]
    base_dir = Path(c.get("base_image_dir"))
    output_base_dir = Path(c.get("base_output_dir"))
    config_params = c.get("config_parameters")
    analysis_params = c.get("analysis_parameters")
    plot_params = c.get("plot_parameters", {})
    images_params = c.get("images_parameters", {})
    grid_params = c.get("grid_parameters", {})
    
    # Get exposure value from config
    exposure_value = images_params.get("exposure_value")
    if exposure_value is None:
        print("Error: exposure_value not found in images_parameters")
        return
    
    # Get use_mask from grid_parameters
    use_mask = grid_params.get("use_mask", True)
    
    filters = {k: analysis_params[k] for k in ["num_lights", "light_distance_cm", "light_type", "psi_angle_deg", "energy", "spread_angle_deg", "area_of_light_cm2"] if k in analysis_params}
    configs = discover(base_dir, config_params, filters, config_params.get("plane_orientations"))
    
    if not configs:
        print("No matching configurations found!")
        return
    
    config_name = next(iter(configs.keys()))[6]
    print(f"Found configuration: {config_name}")
    
    # Get num_lights from the discovered folder name (first light config's num_lights)
    num_lights = next(iter(configs.values()))[0][0]
    
    for oc, lcs in configs.items():
        print(f"\nProcessing: {oc[6]} ({len(lcs)} light configs)")
        process_config(oc, lcs, base_dir, output_base_dir, num_lights, exposure_value, plot_params, use_mask)


if __name__ == "__main__":
    main()
