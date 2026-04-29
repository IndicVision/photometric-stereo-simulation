import os
import time
import psutil

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# =============================================================================
# THREADING & HARDWARE INITIALIZATION
# These MUST be set before importing Numba, NumPy, SciPy, or PyAMG
# =============================================================================





# FIX BUG 3: Restrict C-math libraries to 1 thread to prevent PyAMG cache thrashing. 
# This must happen before imports. Added MKL as well for conda environments.


import json
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go

# FIX BUG 5: Force matplotlib to headless mode for server/background execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import coo_matrix
import pyamg
from scipy.ndimage import uniform_filter, binary_erosion

# --- CPU MULTI-PROCESSING IMPORTS ---
from numba import njit, prange

# --- HARDWARE DETECTION (GPU vs CPU) ---
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        # Force a tiny test compilation AND execution to wake up the compiler
        _test_code = 'extern "C" __global__ void test() {}'
        _test_kernel = cp.RawKernel(_test_code, 'test')
        _test_kernel((1,), (1,), ())  # <-- THIS forces the compiler to run immediately
        
        # Force a basic array operation to check internal indexing kernels
        _ = cp.array([1.0]) * 2.0
        
        HAS_GPU = True
    else:
        HAS_GPU = False
except Exception as e:
    # If cupy is missing, OR if the CUDA driver version is mismatched/broken, fall back gracefully
    print(f"  [HARDWARE WARNING] GPU detected but CUDA compiler failed: {e}")
    print(f"  [HARDWARE WARNING] Forcing safe fallback to CPU.")
    HAS_GPU = False

# =============================================================================
# BOOLEAN FLAGS
# =============================================================================
DEBUG            = True   
FLAT_PLANE_ERROR = True  
COMPUTE_COND_NUM = False  
# =============================================================================

# =============================================================================
# 1. CPU KERNELS (NUMBA)
# =============================================================================
@njit(parallel=True, fastmath=True)
def integrate_area_light_cpu(P_surf, light_samples, light_norm, cos_half_spread):
    num_pixels = P_surf.shape[0]
    num_samples = light_samples.shape[0]
    G_out = np.zeros((num_pixels, 3), dtype=np.float32)

    for idx in prange(num_pixels):
        px = P_surf[idx, 0]
        py = P_surf[idx, 1]
        pz = P_surf[idx, 2]
        
        gx, gy, gz = 0.0, 0.0, 0.0
        
        for s in range(num_samples):
            lx = light_samples[s, 0]
            ly = light_samples[s, 1]
            lz = light_samples[s, 2]
            
            vx = lx - px
            vy = ly - py
            vz = lz - pz
            
            dist_sq = vx*vx + vy*vy + vz*vz
            if dist_sq < 1e-16: dist_sq = 1e-16
            
            inv_dist = 1.0 / np.sqrt(dist_sq)
            dx = vx * inv_dist
            dy = vy * inv_dist
            dz = vz * inv_dist
            cos_emit = -(light_norm[0]*dx + light_norm[1]*dy + light_norm[2]*dz)
            
            if cos_emit >= cos_half_spread:
                cos_emit = max(cos_emit, 0.0) 
                weight = cos_emit * (inv_dist * inv_dist)
                gx += dx * weight
                gy += dy * weight
                gz += dz * weight
                
        G_out[idx, 0] = gx / num_samples
        G_out[idx, 1] = gy / num_samples
        G_out[idx, 2] = gz / num_samples

    return G_out


@njit(parallel=True, fastmath=True)
def batched_3x3_solve_numba(GTG, GTI):
    """
    Massively parallel 3x3 matrix solver bypassing OpenBLAS bottleneck.
    """
    N = GTG.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)
    
    for i in prange(N):
        a11, a12, a13 = GTG[i, 0, 0], GTG[i, 0, 1], GTG[i, 0, 2]
        a21, a22, a23 = GTG[i, 1, 0], GTG[i, 1, 1], GTG[i, 1, 2]
        a31, a32, a33 = GTG[i, 2, 0], GTG[i, 2, 1], GTG[i, 2, 2]
        
        b1, b2, b3 = GTI[i, 0], GTI[i, 1], GTI[i, 2]
        
        det = (a11 * (a22 * a33 - a23 * a32) -
               a12 * (a21 * a33 - a23 * a31) +
               a13 * (a21 * a32 - a22 * a31))
               
        # FIX BUG 1: Adjusted threshold from 1e-8 to 1e-6 for safe float32 comparison
        if abs(det) < 1e-6:
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 1.0
            continue
            
        inv_det = 1.0 / det
        
        out[i, 0] = inv_det * (b1 * (a22 * a33 - a23 * a32) + 
                               b2 * (a13 * a32 - a12 * a33) + 
                               b3 * (a12 * a23 - a13 * a22))
                               
        out[i, 1] = inv_det * (b1 * (a23 * a31 - a21 * a33) + 
                               b2 * (a11 * a33 - a13 * a31) + 
                               b3 * (a13 * a21 - a11 * a23))
                               
        out[i, 2] = inv_det * (b1 * (a21 * a32 - a22 * a31) + 
                               b2 * (a12 * a31 - a11 * a32) + 
                               b3 * (a11 * a22 - a12 * a21))
    return out


# =============================================================================
# 2. GPU KERNEL (CUDA)
# =============================================================================
CUDA_KERNEL_SOURCE = r'''
extern "C" __global__
void integrate_area_light(
    const float* __restrict__ P_surf,
    const float* __restrict__ light_samples,
    const float* __restrict__ light_norm,
    float cos_half_spread,
    int num_pixels,
    int num_samples,
    float* __restrict__ G_out
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sh_lx[128];
    __shared__ float sh_ly[128];
    __shared__ float sh_lz[128];

    float nx = light_norm[0];
    float ny = light_norm[1];
    float nz = light_norm[2];

    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (idx < num_pixels) {
        px = P_surf[idx * 3 + 0];
        py = P_surf[idx * 3 + 1];
        pz = P_surf[idx * 3 + 2];
    }

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    int num_tiles = (num_samples + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < num_tiles; ++t) {
        int sample_idx = t * blockDim.x + threadIdx.x;
        if (sample_idx < num_samples) {
            sh_lx[threadIdx.x] = light_samples[sample_idx * 3 + 0];
            sh_ly[threadIdx.x] = light_samples[sample_idx * 3 + 1];
            sh_lz[threadIdx.x] = light_samples[sample_idx * 3 + 2];
        }
        __syncthreads();

        int num_samples_in_tile = min(blockDim.x, num_samples - t * blockDim.x);

        if (idx < num_pixels) {
            #pragma unroll 4
            for (int s = 0; s < num_samples_in_tile; ++s) {
                float vx = sh_lx[s] - px;
                float vy = sh_ly[s] - py;
                float vz = sh_lz[s] - pz;
                float dist_sq = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
                if (dist_sq < 1e-16f) dist_sq = 1e-16f;

                float inv_dist = rsqrtf(dist_sq);
                float dx = vx * inv_dist;
                float dy = vy * inv_dist;
                float dz = vz * inv_dist;
                float cos_emit = -(nx*dx + ny*dy + nz*dz);

                if (cos_emit >= cos_half_spread) {
                    cos_emit = fmaxf(cos_emit, 0.0f);
                    float weight = cos_emit * (inv_dist * inv_dist);
                    gx = fmaf(dx, weight, gx);
                    gy = fmaf(dy, weight, gy);
                    gz = fmaf(dz, weight, gz);
                }
            }
        }
        __syncthreads();
    }

    if (idx < num_pixels) {
        float inv_num_samples = 1.0f / (float)num_samples;
        G_out[idx * 3 + 0] = gx * inv_num_samples;
        G_out[idx * 3 + 1] = gy * inv_num_samples;
        G_out[idx * 3 + 2] = gz * inv_num_samples;
    }
}
'''


class AutoIterativePipeline:

    # -------------------------------------------------------------------------
    # DEBUG HELPERS 
    # -------------------------------------------------------------------------
    def _dbp(self, section, msg):
        if DEBUG:
            print(f"  [DBG|{section:12s}] {msg}")

    def _flat_plane_error(self, normals_nx3, label=""):
        if not FLAT_PLANE_ERROR:
            return
        nz = np.clip(normals_nx3[:, 2], -1.0, 1.0)
        ae = np.degrees(np.arccos(nz))
        pct_5  = 100.0 * np.mean(ae <=  5.0)
        pct_10 = 100.0 * np.mean(ae <= 10.0)
        pct_20 = 100.0 * np.mean(ae <= 20.0)
        tag = f" [{label}]" if label else ""
        print(f"  [FPE{tag}] Angular error vs flat plane [0,0,1]  (n={len(ae):,}):")
        print(f"  [FPE{tag}]   Mean  = {np.mean(ae):7.3f}°   Std    = {np.std(ae):7.3f}°   Median = {np.median(ae):7.3f}°")
        print(f"  [FPE{tag}]   P25   = {np.percentile(ae,25):7.3f}°   P75    = {np.percentile(ae,75):7.3f}°   Max    = {np.max(ae):7.3f}°")
        print(f"  [FPE{tag}]   Within  5°: {pct_5:5.1f}%   Within 10°: {pct_10:5.1f}%   Within 20°: {pct_20:5.1f}%")

    def _flat_plane_error_from_Z(self, Z, mask, v_idx, u_idx, dx, dy, label=""):
        if not FLAT_PLANE_ERROR:
            return
        Z_nan = np.where(mask, Z, np.nan)
        dZdy_grid, dZdx_grid = np.gradient(Z_nan, dy, dx)
        p_r = dZdx_grid[v_idx, u_idx]
        q_r = dZdy_grid[v_idx, u_idx]
        valid = ~np.isnan(p_r) & ~np.isnan(q_r)
        pr = p_r[valid]; qr = q_r[valid]
        if len(pr) == 0:
            print(f"  [FPE|{label}] No valid pixels for back-derivation — skipped.")
            return
        recon_n = np.stack([-pr, -qr, np.ones(len(pr))], axis=1)
        norms   = np.linalg.norm(recon_n, axis=1, keepdims=True)
        recon_n = recon_n / np.maximum(norms, 1e-8)
        full_tag = f"{label} | back-derived from Z"
        self._flat_plane_error(recon_n, label=full_tag)

    def _get_memory_usage(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        vram_mb = cp.get_default_memory_pool().used_bytes() / (1024 * 1024) if HAS_GPU else 0.0
        swap_mb = psutil.swap_memory().used / (1024 * 1024)
        return ram_mb, vram_mb, swap_mb
    
    # -------------------------------------------------------------------------
    def __init__(self, config_input):
        print("\n" + "="*80)
        if HAS_GPU:
            print(f"  [HARDWARE] NVIDIA GPU detected and active. Using CUDA Acceleration.")
            print(f"  AREA-LIGHT PS Pipeline (GPU EDITION)")
        else:
            print(f"  [HARDWARE] No NVIDIA GPU detected. Triggering Numba CPU Fallback.")
            print(f"  [HARDWARE] CPU detected: {psutil.cpu_count(logical=True)} Logical Threads.")
            print(f"  [HARDWARE] Numba engine is using default parallel thread configuration.")
            print(f"  AREA-LIGHT PS Pipeline (CPU PARALLEL EDITION)")
        print("="*80 + "\n")

        if isinstance(config_input, dict):
            self.cfg = config_input
        else:
            with open(config_input, 'r') as f:
                self.cfg = json.load(f)
        
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = self.cfg.get('max_iterations', 15)
        self.convergence_threshold = self.cfg.get('convergence_threshold', 1e-5)

        if HAS_GPU:
            self.cuda_kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light', options=('-use_fast_math',))
        else:
            print("  [STARTUP] Pre-compiling CPU kernel (one-time cost)...")
            _dummy_P = np.zeros((1, 3), dtype=np.float32)
            _dummy_S = np.zeros((1, 3), dtype=np.float32)
            _dummy_N = np.zeros(3, dtype=np.float32)
            integrate_area_light_cpu(_dummy_P, _dummy_S, _dummy_N, np.float32(0.0))
            
            _dummy_GTG = np.zeros((1, 3, 3), dtype=np.float32)
            _dummy_GTI = np.zeros((1, 3), dtype=np.float32)
            batched_3x3_solve_numba(_dummy_GTG, _dummy_GTI)
            print("  [STARTUP] CPU kernel ready.")

        self.timers = {
            '1_initialization': 0.0,
            '2_estimate_normals': 0.0,
            '3_repair_normals': 0.0,
            '4_reconstruct_surface': 0.0,
            '5_save_visualizations': 0.0,
            '6_df_and_convergence': 0.0,
            'total_pipeline_time': 0.0
        }

        self.mem_stats = {
            '1_initialization': [0.0, 0.0, 0.0],
            '2_estimate_normals': [0.0, 0.0, 0.0],
            '3_repair_normals': [0.0, 0.0, 0.0],
            '4_reconstruct_surface': [0.0, 0.0, 0.0],
            '5_save_visualizations': [0.0, 0.0, 0.0]
        }

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(f"Missing image: {img_path}")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        if gamma != 1.0: img = np.power(img, gamma)
        return img 


    # =========================================================================
    # GPU METHODS (Active if NVIDIA GPU detected)
    # =========================================================================
    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        pos    = np.array(light_cfg['pos_m'])
        dims   = np.array(light_cfg['dims_m'])
        norm   = np.array(light_cfg['norm_dir'])
        samples    = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)

        if 'ax_u' in light_cfg and 'ax_v' in light_cfg:
            ax_u = np.array(light_cfg['ax_u'], dtype=np.float32)
            ax_v = np.array(light_cfg['ax_v'], dtype=np.float32)
        else:
            v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
            ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
            ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v

        sample_pts_gpu   = cp.array(sample_pts, dtype=cp.float32)
        norm_gpu         = cp.array(norm, dtype=cp.float32)
        cos_half_spread  = np.cos(np.deg2rad(spread_deg / 2.0))
        num_pixels       = P_surf_full.shape[0]
        num_samples      = sample_pts_gpu.shape[0]
        G_eff_full       = cp.zeros((num_pixels, 3), dtype=cp.float32)

        chunk_size        = 50000
        threads_per_block = 128
        for start_idx in range(0, num_pixels, chunk_size):
            end_idx       = min(start_idx + chunk_size, num_pixels)
            current_chunk = end_idx - start_idx
            P_surf_chunk  = P_surf_full[start_idx:end_idx]
            G_out_chunk   = G_eff_full[start_idx:end_idx]
            blocks_per_grid = (current_chunk + threads_per_block - 1) // threads_per_block
            self.cuda_kernel((blocks_per_grid,), (threads_per_block,),
                (P_surf_chunk, sample_pts_gpu, norm_gpu, cp.float32(cos_half_spread),
                 cp.int32(current_chunk), cp.int32(num_samples), G_out_chunk))
            cp.cuda.Device(0).synchronize()

        return G_eff_full, sample_pts   

    def estimate_normals_gpu(self, df):
        u_idx  = df['pixel_u'].values.astype(int)
        v_idx  = df['pixel_v'].values.astype(int)

        elev_m = self.cfg.get('camera', {}).get('object_elevation_m', 0.0)
        xyz_cpu = df[['x_world', 'y_world', 'z_world']].values.copy().astype(np.float32)
        xyz_cpu[:, 2] += elev_m   

        if DEBUG:
            print(f"  [DBG|ELEVATION  ] object_elevation_m = {elev_m*1000:.3f} mm  (added to z_world before kernel)")

        P_surf = cp.ascontiguousarray(cp.array(xyz_cpu, dtype=cp.float32))

        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)
        obj_center = np.array([df['x_world'].mean(), df['y_world'].mean(), df['z_world'].mean() + elev_m])  

        self._dbp("NORMALS", f"Pixel count: {num_pixels:,}  |  Lights: {num_lights}")

        global_cfg   = self.cfg.get('global_settings', {})
        apply_thresh = global_cfg.get('apply_dark_threshold', True)
        thresh_val   = global_cfg.get('dark_threshold_value', 0.025)
        sat_thresh   = global_cfg.get('saturation_threshold_value', 1.01)  # >1.0 = disabled by default

        if not hasattr(self, 'I_w_cache_gpu'):
            self._dbp("SPEED", "Caching image intensities to VRAM (One-time cost)...")
            cal_factors = self.cfg.get('intensity_correction', None)

            # ── Calibration debug (printed once on cache build) ───────────────
            if DEBUG:
                if cal_factors is not None:
                    print(f"  [DBG|CALIB      ] Intensity correction ENABLED — factors: "
                          + "  ".join(f"L{j+1}:{cal_factors[j]:.4f}" for j in range(num_lights)))
                else:
                    print(f"  [DBG|CALIB      ] Intensity correction DISABLED (no 'intensity_correction' key in config)")

            I_raw = cp.zeros((num_pixels, num_lights), dtype=cp.float32)
            for j, l_cfg in enumerate(self.cfg['lights']):
                img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
                img_np   = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
                if cal_factors is not None:
                    img_np = img_np * float(cal_factors[j])
                img_gpu  = cp.array(img_np)
                I_raw[:, j] = img_gpu[v_idx, u_idx]
                del img_gpu
                cp.get_default_memory_pool().free_all_blocks()
            if apply_thresh:
                W = (I_raw >= thresh_val).astype(cp.float32)
            else:
                W = cp.ones_like(I_raw)

            # Saturation mask: if ANY light saturates for a pixel, zero out ALL
            # lights for that pixel so its corrupted intensity ratios never enter
            # the LS solve. It will be filled by repair_normals neighbourhood avg.
            if sat_thresh <= 1.0:
                sat_pixels = cp.any(I_raw >= sat_thresh, axis=1)  # shape: (N_pixels,)
                n_sat = int(cp.sum(sat_pixels).item())
                W[sat_pixels, :] = 0.0
                if DEBUG:
                    print(f"  [DBG|SAT_MASK   ] sat_thresh={sat_thresh:.3f} → "
                          f"{n_sat:,} pixels excluded ({100.0*n_sat/num_pixels:.2f}%) "
                          f"— will be filled by neighbourhood averaging")
            else:
                if DEBUG:
                    print(f"  [DBG|SAT_MASK   ] Saturation masking disabled "
                          f"(sat_thresh={sat_thresh:.3f} > 1.0)")

            # Store sat_pixels mask so we can stamp sentinel after the solve
            self.sat_pixels_gpu     = sat_pixels if sat_thresh <= 1.0 else cp.zeros(num_pixels, dtype=cp.bool_)
            self.I_w_cache_gpu      = I_raw * W
            self.W_cache_gpu        = W
            self.I_raw_cache_gpu    = I_raw   

        for j, l_cfg in enumerate(self.cfg['lights']):
            G_j_gpu, sample_pts = self.get_light_samples_gpu(l_cfg, P_surf)
            G[:, j, :] = G_j_gpu

            if DEBUG:
                I_j_cpu   = cp.asnumpy(self.I_raw_cache_gpu[:, j])
                pct_above = 100.0 * np.mean(I_j_cpu >= thresh_val)

                G_j_cpu  = cp.asnumpy(G_j_gpu)
                G_j_mag  = np.linalg.norm(G_j_cpu, axis=1)
                G_j_dir  = G_j_cpu / np.maximum(G_j_mag[:, None], 1e-10)

                mean_G_weighted  = np.mean(G_j_cpu, axis=0)
                mean_G_mag_total = np.linalg.norm(mean_G_weighted)
                mean_G_dir_norm  = mean_G_weighted / max(mean_G_mag_total, 1e-10)

                cos_to_mean = np.clip(np.sum(G_j_dir * mean_G_dir_norm[None, :], axis=1), -1.0, 1.0)
                angular_spread_per_pixel = np.degrees(np.arccos(cos_to_mean))

                print(f"  [DBG|LIGHT {j:02d}    ] ── Light {l_cfg['id']} ──────────────────────────────────")
                print(f"  [DBG|LIGHT {j:02d}    ] G mean dir : [{mean_G_dir_norm[0]:+.4f}  {mean_G_dir_norm[1]:+.4f}  {mean_G_dir_norm[2]:+.4f}]  |  |G| mean={G_j_mag.mean():.4e}  std={G_j_mag.std():.4e}")
                print(f"  [DBG|LIGHT {j:02d}    ] Angular spread (pixel-to-pixel): {angular_spread_per_pixel.mean():.3f}°")
                print(f"  [DBG|LIGHT {j:02d}    ] Intensity mean={I_j_cpu.mean():.4f}  above_thresh({thresh_val}): {pct_above:.1f}%")
                del G_j_cpu, G_j_mag, G_j_dir

            del G_j_gpu
            cp.get_default_memory_pool().free_all_blocks()

        if apply_thresh:
            G_w = G * self.W_cache_gpu[:, :, None]
            I_w = self.I_w_cache_gpu
        else:
            G_w = G
            I_w = self.I_w_cache_gpu

        GT_w = G_w.transpose(0, 2, 1)
        GTG  = cp.matmul(GT_w, G_w) + (cp.eye(3, dtype=cp.float32) * 1.0)
        GTI  = cp.matmul(GT_w, I_w[:, :, None])

        n_est   = cp.linalg.solve(GTG, GTI).squeeze(-1)   
        albedo  = cp.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / cp.where(albedo == 0, 1, albedo)

        nan_mask = cp.any(cp.isnan(normals), axis=1)
        if cp.any(nan_mask):
            normals[nan_mask] = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)

        # Stamp saturated pixels with sentinel [0,0,-1].
        # This overrides whatever the LS solver returned for these corrupted pixels.
        # repair_normals_gpu already handles nz<0 pixels via neighbourhood averaging,
        # so no changes are needed there — the sentinel piggybacks on that path.
        if cp.any(self.sat_pixels_gpu):
            normals[self.sat_pixels_gpu] = cp.array([0.0, 0.0, -1.0], dtype=cp.float32)
            if DEBUG:
                n_stamped = int(cp.sum(self.sat_pixels_gpu).item())
                print(f"  [DBG|SAT_MASK   ] Stamped {n_stamped:,} saturated pixels with "
                      f"sentinel [0,0,-1] → will be filled by repair_normals")

        if DEBUG:
            normals_cpu  = cp.asnumpy(normals)
            nz_cpu       = normals_cpu[:, 2]
            print(f"  [DBG|NORMALS    ] --- Estimated normal statistics ---")
            for comp_name, comp_vals in [("nx", normals_cpu[:,0]), ("ny", normals_cpu[:,1]), ("nz", normals_cpu[:,2])]:
                print(f"  [DBG|NORMALS    ]   {comp_name}: min={comp_vals.min():+.4f}  max={comp_vals.max():+.4f}  mean={comp_vals.mean():+.4f}")
            print(f"  [DBG|NORMALS    ]   nz < 0 : {100.0 * np.mean(nz_cpu < 0):5.1f}%  "
                  f"(includes {int(cp.sum(self.sat_pixels_gpu).item()):,} sat-sentinel pixels)")

        del G, G_w, GT_w, GTG, GTI, n_est
        cp.get_default_memory_pool().free_all_blocks()
        
        return cp.asnumpy(normals)

    def repair_normals_gpu(self, normals_np, v_idx_np, u_idx_np, M_img, N_img):
        import cupyx.scipy.ndimage as nd_gpu
        NZ_THRESH = 0.0          
        WIN       = 5

        normals_gpu = cp.array(normals_np)
        nx_in, ny_in, nz_in = normals_gpu[:, 0], normals_gpu[:, 1], normals_gpu[:, 2]

        v_idx_gpu = cp.array(v_idx_np)
        u_idx_gpu = cp.array(u_idx_np)

        nx_g = cp.zeros((M_img, N_img), dtype=cp.float32)
        ny_g = cp.zeros((M_img, N_img), dtype=cp.float32)
        nz_g = cp.zeros((M_img, N_img), dtype=cp.float32)
        mask_g = cp.zeros((M_img, N_img), dtype=cp.bool_)
        
        nx_g[v_idx_gpu, u_idx_gpu] = nx_in
        ny_g[v_idx_gpu, u_idx_gpu] = ny_in
        nz_g[v_idx_gpu, u_idx_gpu] = nz_in
        mask_g[v_idx_gpu, u_idx_gpu] = True

        good_g = mask_g & (nz_g >= NZ_THRESH)
        bad_g  = mask_g & (nz_g <  NZ_THRESH)
        good_f = good_g.astype(cp.float32)
        
        nx_sum = nd_gpu.uniform_filter(nx_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        ny_sum = nd_gpu.uniform_filter(ny_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        nz_sum = nd_gpu.uniform_filter(nz_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        w_sum  = nd_gpu.uniform_filter(good_f,         size=WIN, mode='constant') * (WIN * WIN)

        safe_w   = cp.maximum(w_sum, 1e-8)
        nx_fill  = nx_sum / safe_w
        ny_fill  = ny_sum / safe_w
        nz_fill  = nz_sum / safe_w

        mag_fill = cp.sqrt(nx_fill**2 + ny_fill**2 + nz_fill**2)
        safe_mag = cp.maximum(mag_fill, 1e-8)
        nx_fill /= safe_mag
        ny_fill /= safe_mag
        nz_fill /= safe_mag

        has_neighbor = bad_g & (w_sum > 0)
        no_neighbor  = bad_g & (w_sum == 0)

        nx_g[has_neighbor] = nx_fill[has_neighbor]
        ny_g[has_neighbor] = ny_fill[has_neighbor]
        nz_g[has_neighbor] = nz_fill[has_neighbor]

        if cp.sum(no_neighbor) > 0:
            nx_g[no_neighbor] = 0.0
            ny_g[no_neighbor] = 0.0
            nz_g[no_neighbor] = 1.0

        normals_repaired = cp.stack([
            nx_g[v_idx_gpu, u_idx_gpu], ny_g[v_idx_gpu, u_idx_gpu], nz_g[v_idx_gpu, u_idx_gpu]
        ], axis=1)

        res = cp.asnumpy(normals_repaired)
        del nx_g, ny_g, nz_g, mask_g, good_g, bad_g, good_f, nx_sum, ny_sum, nz_sum, w_sum, normals_repaired
        cp.get_default_memory_pool().free_all_blocks()
        return res

    # =========================================================================
    # CPU METHODS (Active if NVIDIA GPU missing)
    # =========================================================================
    def get_light_samples_cpu(self, light_cfg, P_surf_full):
        pos    = np.array(light_cfg['pos_m'])
        dims   = np.array(light_cfg['dims_m'])
        norm   = np.array(light_cfg['norm_dir'])
        samples    = light_cfg['sampling']
        spread_deg = light_cfg.get('spread_deg', 180.0)

        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)

        if 'ax_u' in light_cfg and 'ax_v' in light_cfg:
            ax_u = np.array(light_cfg['ax_u'], dtype=np.float32)
            ax_v = np.array(light_cfg['ax_v'], dtype=np.float32)
        else:
            v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
            ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
            ax_v = np.cross(norm, ax_u)

        sample_pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v

        sample_pts_np   = sample_pts.astype(np.float32)
        norm_np         = norm.astype(np.float32)
        cos_half_spread  = np.float32(np.cos(np.deg2rad(spread_deg / 2.0)))

        G_eff_full = integrate_area_light_cpu(P_surf_full, sample_pts_np, norm_np, cos_half_spread)
        return G_eff_full, sample_pts   

    def estimate_normals_cpu(self, df):
        u_idx  = df['pixel_u'].values.astype(int)
        v_idx  = df['pixel_v'].values.astype(int)

        elev_m = self.cfg.get('camera', {}).get('object_elevation_m', 0.0)
        xyz_cpu = df[['x_world', 'y_world', 'z_world']].values.copy().astype(np.float32)
        xyz_cpu[:, 2] += elev_m   

        if DEBUG:
            print(f"  [DBG|ELEVATION  ] object_elevation_m = {elev_m*1000:.3f} mm  (added to z_world before kernel)")

        P_surf = np.ascontiguousarray(xyz_cpu)

        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])
        G = np.zeros((num_pixels, num_lights, 3), dtype=np.float32)
        obj_center = np.array([df['x_world'].mean(), df['y_world'].mean(), df['z_world'].mean() + elev_m])  

        self._dbp("NORMALS", f"Pixel count: {num_pixels:,}  |  Lights: {num_lights}")

        global_cfg   = self.cfg.get('global_settings', {})
        apply_thresh = global_cfg.get('apply_dark_threshold', True)
        thresh_val   = global_cfg.get('dark_threshold_value', 0.025)
        sat_thresh   = global_cfg.get('saturation_threshold_value', 1.01)  # >1.0 = disabled by default

        if not hasattr(self, 'I_w_cache_cpu'):
            self._dbp("SPEED", "Caching image intensities to RAM (One-time cost)...")
            cal_factors = self.cfg.get('intensity_correction', None)

            # ── Calibration debug (printed once on cache build) ───────────────
            if DEBUG:
                if cal_factors is not None:
                    print(f"  [DBG|CALIB      ] Intensity correction ENABLED — factors: "
                          + "  ".join(f"L{j+1}:{cal_factors[j]:.4f}" for j in range(num_lights)))
                else:
                    print(f"  [DBG|CALIB      ] Intensity correction DISABLED (no 'intensity_correction' key in config)")

            I_raw = np.zeros((num_pixels, num_lights), dtype=np.float32)
            for j, l_cfg in enumerate(self.cfg['lights']):
                img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
                img_np  = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
                if cal_factors is not None:
                    img_np = img_np * float(cal_factors[j])
                I_raw[:, j] = img_np[v_idx, u_idx]
            if apply_thresh:
                W = (I_raw >= thresh_val).astype(np.float32)
            else:
                W = np.ones_like(I_raw)

            # Saturation mask: if ANY light saturates for a pixel, zero out ALL
            # lights for that pixel so its corrupted intensity ratios never enter
            # the LS solve. It will be filled by repair_normals neighbourhood avg.
            if sat_thresh <= 1.0:
                sat_pixels = np.any(I_raw >= sat_thresh, axis=1)  # shape: (N_pixels,)
                n_sat = int(np.sum(sat_pixels))
                W[sat_pixels, :] = 0.0
                if DEBUG:
                    print(f"  [DBG|SAT_MASK   ] sat_thresh={sat_thresh:.3f} → "
                          f"{n_sat:,} pixels excluded ({100.0*n_sat/num_pixels:.2f}%) "
                          f"— will be filled by neighbourhood averaging")
            else:
                sat_pixels = np.zeros(num_pixels, dtype=bool)
                if DEBUG:
                    print(f"  [DBG|SAT_MASK   ] Saturation masking disabled "
                          f"(sat_thresh={sat_thresh:.3f} > 1.0)")

            self.sat_pixels_cpu     = sat_pixels
            self.I_w_cache_cpu      = I_raw * W
            self.W_cache_cpu        = W
            self.I_raw_cache_cpu    = I_raw   

        for j, l_cfg in enumerate(self.cfg['lights']):
            G_j_np, sample_pts = self.get_light_samples_cpu(l_cfg, P_surf)
            G[:, j, :] = G_j_np

            if DEBUG:
                I_j_cpu   = self.I_raw_cache_cpu[:, j]
                pct_above = 100.0 * np.mean(I_j_cpu >= thresh_val)

                G_j_cpu  = G_j_np
                G_j_mag  = np.linalg.norm(G_j_cpu, axis=1)
                G_j_dir  = G_j_cpu / np.maximum(G_j_mag[:, None], 1e-10)

                mean_G_weighted  = np.mean(G_j_cpu, axis=0)
                mean_G_mag_total = np.linalg.norm(mean_G_weighted)
                mean_G_dir_norm  = mean_G_weighted / max(mean_G_mag_total, 1e-10)

                cos_to_mean = np.clip(np.sum(G_j_dir * mean_G_dir_norm[None, :], axis=1), -1.0, 1.0)
                angular_spread_per_pixel = np.degrees(np.arccos(cos_to_mean))

                print(f"  [DBG|LIGHT {j:02d}    ] ── Light {l_cfg['id']} ──────────────────────────────────")
                print(f"  [DBG|LIGHT {j:02d}    ] G mean dir : [{mean_G_dir_norm[0]:+.4f}  {mean_G_dir_norm[1]:+.4f}  {mean_G_dir_norm[2]:+.4f}]  |  |G| mean={G_j_mag.mean():.4e}  std={G_j_mag.std():.4e}")
                print(f"  [DBG|LIGHT {j:02d}    ] Angular spread (pixel-to-pixel): {angular_spread_per_pixel.mean():.3f}°")
                print(f"  [DBG|LIGHT {j:02d}    ] Intensity mean={I_j_cpu.mean():.4f}  above_thresh({thresh_val}): {pct_above:.1f}%")

        if apply_thresh:
            G_w = G * self.W_cache_cpu[:, :, None]
            I_w = self.I_w_cache_cpu
        else:
            G_w = G
            I_w = self.I_w_cache_cpu

        GT_w = G_w.transpose(0, 2, 1)
        GTG  = np.matmul(GT_w, G_w) + (np.eye(3, dtype=np.float32) * 1.0)
        GTI  = np.matmul(GT_w, I_w[:, :, None])

        n_est   = batched_3x3_solve_numba(GTG, GTI.squeeze(-1))

        albedo  = np.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / np.where(albedo == 0, 1, albedo)

        nan_mask = np.any(np.isnan(normals), axis=1)
        if np.any(nan_mask):
            normals[nan_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Stamp saturated pixels with sentinel [0,0,-1].
        # repair_normals_cpu already handles nz<0 via neighbourhood averaging.
        if np.any(self.sat_pixels_cpu):
            normals[self.sat_pixels_cpu] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            if DEBUG:
                n_stamped = int(np.sum(self.sat_pixels_cpu))
                print(f"  [DBG|SAT_MASK   ] Stamped {n_stamped:,} saturated pixels with "
                      f"sentinel [0,0,-1] → will be filled by repair_normals")

        if DEBUG:
            normals_cpu  = normals
            nz_cpu       = normals_cpu[:, 2]
            print(f"  [DBG|NORMALS    ] --- Estimated normal statistics ---")
            for comp_name, comp_vals in [("nx", normals_cpu[:,0]), ("ny", normals_cpu[:,1]), ("nz", normals_cpu[:,2])]:
                print(f"  [DBG|NORMALS    ]   {comp_name}: min={comp_vals.min():+.4f}  max={comp_vals.max():+.4f}  mean={comp_vals.mean():+.4f}")
            print(f"  [DBG|NORMALS    ]   nz < 0 : {100.0 * np.mean(nz_cpu < 0):5.1f}%  "
                  f"(includes {int(np.sum(self.sat_pixels_cpu)):,} sat-sentinel pixels)")
        
        return normals

    def repair_normals_cpu(self, normals_np, v_idx_np, u_idx_np, M_img, N_img):
        NZ_THRESH = 0.0          
        WIN       = 5

        nx_in, ny_in, nz_in = normals_np[:, 0], normals_np[:, 1], normals_np[:, 2]

        nx_g = np.zeros((M_img, N_img), dtype=np.float32)
        ny_g = np.zeros((M_img, N_img), dtype=np.float32)
        nz_g = np.zeros((M_img, N_img), dtype=np.float32)
        mask_g = np.zeros((M_img, N_img), dtype=bool)
        
        nx_g[v_idx_np, u_idx_np] = nx_in
        ny_g[v_idx_np, u_idx_np] = ny_in
        nz_g[v_idx_np, u_idx_np] = nz_in
        mask_g[v_idx_np, u_idx_np] = True

        good_g = mask_g & (nz_g >= NZ_THRESH)
        bad_g  = mask_g & (nz_g <  NZ_THRESH)
        good_f = good_g.astype(np.float32)
        
        nx_sum = uniform_filter(nx_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        ny_sum = uniform_filter(ny_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        nz_sum = uniform_filter(nz_g * good_f, size=WIN, mode='constant') * (WIN * WIN)
        w_sum  = uniform_filter(good_f,         size=WIN, mode='constant') * (WIN * WIN)

        safe_w   = np.maximum(w_sum, 1e-8)
        nx_fill  = nx_sum / safe_w
        ny_fill  = ny_sum / safe_w
        nz_fill  = nz_sum / safe_w

        mag_fill = np.sqrt(nx_fill**2 + ny_fill**2 + nz_fill**2)
        safe_mag = np.maximum(mag_fill, 1e-8)
        nx_fill /= safe_mag
        ny_fill /= safe_mag
        nz_fill /= safe_mag

        has_neighbor = bad_g & (w_sum > 0)
        no_neighbor  = bad_g & (w_sum == 0)

        nx_g[has_neighbor] = nx_fill[has_neighbor]
        ny_g[has_neighbor] = ny_fill[has_neighbor]
        nz_g[has_neighbor] = nz_fill[has_neighbor]

        if np.sum(no_neighbor) > 0:
            nx_g[no_neighbor] = 0.0
            ny_g[no_neighbor] = 0.0
            nz_g[no_neighbor] = 1.0

        normals_repaired = np.stack([
            nx_g[v_idx_np, u_idx_np], ny_g[v_idx_np, u_idx_np], nz_g[v_idx_np, u_idx_np]
        ], axis=1)

        return normals_repaired

    # =========================================================================
    # RECONSTRUCTION (Unified PyAMG - Handles both natively)
    # =========================================================================
    def reconstruct_surface(self, df, normals_np):
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)

        if not hasattr(self, '_recon_cache'):
            M_img = np.max(v_idx) + 1
            N_img = np.max(u_idx) + 1
            mask  = np.zeros((M_img, N_img), dtype=bool)
            mask[v_idx, u_idx] = True

            erosion_px = int(self.cfg.get('global_settings', {}).get('mask_erosion_pixels', 0))
            keep = np.ones(len(u_idx), dtype=bool)   
            if erosion_px > 0:
                struct = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=bool)
                mask_eroded = binary_erosion(mask, structure=struct)
                n_removed   = int(np.sum(mask)) - int(np.sum(mask_eroded))
                if DEBUG:
                    print(f"  [DBG|MASK_EROSION] Eroding mask by {erosion_px} px  Removed: {n_removed:,} pixels")
                mask = mask_eroded
                keep = mask[v_idx, u_idx]

            u_idx_e = u_idx[keep]
            v_idx_e = v_idx[keep]
            df_e    = df[keep].reset_index(drop=True)

            X_grid = np.full((M_img, N_img), np.nan)
            Y_grid = np.full((M_img, N_img), np.nan)
            X_grid[v_idx_e, u_idx_e] = df_e['x_world'].values
            Y_grid[v_idx_e, u_idx_e] = df_e['y_world'].values

            r_idx, c_idx = np.where(mask)
            center_r, center_c = int(np.mean(r_idx)), int(np.mean(c_idx))

            try: dx = np.abs(X_grid[center_r, center_c+1] - X_grid[center_r, center_c])
            except IndexError: dx = 0.0001
            try: dy = np.abs(Y_grid[center_r+1, center_c] - Y_grid[center_r, center_c])
            except IndexError: dy = 0.0001
            if np.isnan(dx) or dx == 0: dx = 0.0001
            if np.isnan(dy) or dy == 0: dy = 0.0001

            node_ids     = np.zeros((M_img, N_img), dtype=int)
            num_unknowns = int(np.sum(mask))
            node_ids[mask] = np.arange(num_unknowns)

            mask_H = mask[:, :-1] & mask[:, 1:]
            r_H, c_H = np.where(mask_H)
            id_self_H  = node_ids[r_H, c_H]
            id_right_H = node_ids[r_H, c_H + 1]

            mask_V = mask[:-1, :] & mask[1:, :]
            r_V, c_V = np.where(mask_V)
            id_self_V = node_ids[r_V, c_V]
            id_down_V = node_ids[r_V + 1, c_V]

            num_H  = len(id_self_H)
            num_V  = len(id_self_V)
            num_eq = num_H + num_V + 1

            center_node_id = node_ids[center_r, center_c]
            I_list = np.concatenate([np.arange(num_H), np.arange(num_H), np.arange(num_H, num_H + num_V), np.arange(num_H, num_H + num_V), [num_eq - 1]])
            J_list = np.concatenate([id_right_H, id_self_H, id_down_V,  id_self_V, [center_node_id]])
            V_list = np.concatenate([np.ones(num_H), -np.ones(num_H), np.ones(num_V), -np.ones(num_V), np.array([1.0])])
            A = coo_matrix((V_list, (I_list, J_list)), shape=(num_eq, num_unknowns)).tocsr()
            
            C   = A.T @ A
            A_T = A.T

            self._dbp("SOLVER", f"Building PyAMG multigrid hierarchy (smoothed aggregation)...")
            t_factor_start = time.time()
            
            ml = pyamg.smoothed_aggregation_solver(C, coarse_solver='lu')
            self._dbp("SOLVER", f"AMG hierarchy built ✓  |  time: {time.time() - t_factor_start:.3f} s  |  levels: {len(ml.levels)}")

            self._recon_cache = dict(
                keep=keep, u_idx_e=u_idx_e, v_idx_e=v_idx_e, M_img=M_img, N_img=N_img, mask=mask,
                center_r=center_r, center_c=center_c, dx=dx, dy=dy, X_grid=X_grid, Y_grid=Y_grid,
                r_H=r_H, c_H=c_H, r_V=r_V, c_V=c_V, A=A, A_T=A_T, ml=ml,
                num_unknowns=num_unknowns, num_eq=num_eq
            )

        rc = self._recon_cache
        normals_e_np = normals_np[rc['keep']]

        self._dbp("RECONSTRUCT", f"Grid: {rc['M_img']}×{rc['N_img']}  |  Valid pixels (mask): {rc['num_unknowns']:,}")
        self._dbp("RECONSTRUCT", f"Physical spacing: dx={rc['dx']*1e3:.4f} mm  dy={rc['dy']*1e3:.4f} mm")

        nx_grid = np.zeros((rc['M_img'], rc['N_img']), dtype=np.float32)
        ny_grid = np.zeros((rc['M_img'], rc['N_img']), dtype=np.float32)
        nz_grid = np.zeros((rc['M_img'], rc['N_img']), dtype=np.float32)
        
        nx_grid[rc['v_idx_e'], rc['u_idx_e']] = normals_e_np[:, 0]
        ny_grid[rc['v_idx_e'], rc['u_idx_e']] = normals_e_np[:, 1]
        nz_grid[rc['v_idx_e'], rc['u_idx_e']] = normals_e_np[:, 2]  

        eps_nz = np.float32(0.15)
        valid_np  = rc['mask']
        
        nz_abs_clamped = np.maximum(np.abs(nz_grid), eps_nz)
        nz_safe = np.where(nz_grid >= 0, nz_abs_clamped, -nz_abs_clamped)
        nz_safe[~valid_np] = 1.0

        p = np.zeros((rc['M_img'], rc['N_img']), dtype=np.float32)
        q = np.zeros((rc['M_img'], rc['N_img']), dtype=np.float32)
        p[valid_np] = -(nx_grid[valid_np] / nz_safe[valid_np]) * np.float32(rc['dx'])
        q[valid_np] = +(ny_grid[valid_np] / nz_safe[valid_np]) * np.float32(rc['dy'])

        self._dbp("SOLVER", f"System size: {rc['num_eq']:,} equations  |  {rc['num_unknowns']:,} unknowns")

        val_p = p[rc['r_H'], rc['c_H']]
        val_q = q[rc['r_V'], rc['c_V']]
        
        b = np.concatenate([val_p, val_q, np.array([0.0], dtype=np.float32)]).astype(np.float64)

        d = rc['A_T'] @ b
        t_solve = time.time()
        z = rc['ml'].solve(d, tol=1e-10, accel='cg')
        self._dbp("SOLVER", f"AMG solve complete ✓  |  time: {time.time() - t_solve:.3f} s")

        Z = np.full((rc['M_img'], rc['N_img']), np.nan)
        Z[rc['mask']] = z

        # ── Detrending (unchanged from Code 1) ────────────────────────────────
        # ── Detrending (unchanged from Code 1) ────────────────────────────────
        detrend_mode = self.cfg.get('global_settings', {}).get('detrending_mode', 'none').lower()

        valid_X, valid_Y, valid_Z = rc['X_grid'][rc['mask']], rc['Y_grid'][rc['mask']], Z[rc['mask']]

        # FIX: Use rc['mask'] instead of mask
        Z_before = np.ptp(Z[rc['mask']]) * 1000.0

        if detrend_mode == 'quadratic':
            self._dbp("DETREND", "Mode = QUADRATIC — removing macro bowl/tilt for flat objects")
            A_quad = np.c_[valid_X**2, valid_Y**2, valid_X * valid_Y,
                           valid_X, valid_Y, np.ones_like(valid_X)]
            C_quad, _, _, _ = np.linalg.lstsq(A_quad, valid_Z, rcond=None)
            
            # FIX: Use rc['X_grid'] and rc['Y_grid']
            Detrend_Z = (C_quad[0] * rc['X_grid']**2 + C_quad[1] * rc['Y_grid']**2 +
                         C_quad[2] * rc['X_grid'] * rc['Y_grid'] + C_quad[3] * rc['X_grid'] +
                         C_quad[4] * rc['Y_grid'] + C_quad[5])
            self._dbp("DETREND", f"Quadratic coefficients: a={C_quad[0]:.4e}  b={C_quad[1]:.4e}  "
                                  f"c={C_quad[2]:.4e}  d={C_quad[3]:.4e}  e={C_quad[4]:.4e}  f={C_quad[5]:.4e}")
            
            # FIX: Use rc['mask']
            Z[rc['mask']] = Z[rc['mask']] - Detrend_Z[rc['mask']]

        elif detrend_mode == 'linear':
            self._dbp("DETREND", "Mode = LINEAR — removing planar tilt only")
            A_plane = np.c_[valid_X, valid_Y, np.ones_like(valid_X)]
            C_plane, _, _, _ = np.linalg.lstsq(A_plane, valid_Z, rcond=None)
            
            # FIX: Use rc['X_grid'] and rc['Y_grid']
            Detrend_Z = (C_plane[0] * rc['X_grid'] + C_plane[1] * rc['Y_grid'] + C_plane[2])
            self._dbp("DETREND", f"Plane coefficients: nx={C_plane[0]:.4e}  ny={C_plane[1]:.4e}  d={C_plane[2]:.4e}")
            
            # FIX: Use rc['mask']
            Z[rc['mask']] = Z[rc['mask']] - Detrend_Z[rc['mask']]

        else:
            self._dbp("DETREND", f"Mode = NONE (or unknown: '{detrend_mode}') — keeping raw Poisson data.")

        # FIX: Use rc['mask']
        Z_after = np.ptp(Z[rc['mask']]) * 1000.0
        if detrend_mode in ['linear', 'quadratic']:
            self._dbp("DETREND", f"Z range before detrend: {Z_before:.4f} mm  ->  after: {Z_after:.4f} mm")
        else:
            self._dbp("DETREND", f"Z range remains: {Z_after:.4f} mm")

        # ── Floor anchor (unchanged from Code 1) ──────────────────────────────
        # FIX: Use rc['mask']
        surface_floor = np.nanmin(Z[rc['mask']])
        Z[rc['mask']] = Z[rc['mask']] - surface_floor
        self._dbp("DETREND", f"Floor anchor (min -> 0): shifted by {surface_floor*1000:.4f} mm")

        if DEBUG:
            # FIX: Use rc['mask']
            Z_valid_post = Z[rc['mask']] * 1000.0
            self._dbp("DETREND", f"Final Z: min={Z_valid_post.min():.4f} mm  max={Z_valid_post.max():.4f} mm  "
                                  f"mean={Z_valid_post.mean():.4f} mm  std={Z_valid_post.std():.4f} mm  "
                                  f"range={np.ptp(Z_valid_post):.4f} mm")

        # Flat plane error — AFTER Poisson + detrend (back-derived from Z)
        # FIX: Replace bare variables with rc variants
        self._flat_plane_error_from_Z(Z, rc['mask'], rc['v_idx_e'], rc['u_idx_e'], rc['dx'], rc['dy'],
                                       label="reconstructed | post-Poisson+detrend")

        return Z, rc['dx'], rc['dy'], rc['mask']

    # -------------------------------------------------------------------------
    def save_visualizations(self, iter_num, normals_np, df, Z, dx, dy, mask):
        iter_dir = self.output_dir / f"iteration_{iter_num:02d}"
        iter_dir.mkdir(exist_ok=True)

        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)

        dx_mm = dx * 1000.0
        dy_mm = dy * 1000.0

        h_res, w_res = self.cfg['resolution']['height'], self.cfg['resolution']['width']
        n_map = np.zeros((h_res, w_res, 3), dtype=np.float32)
        n_map[v_idx, u_idx] = normals_np
        n_vis_uint16 = ((n_map + 1.0) / 2.0 * 65535).astype(np.uint16)
        cv2.imwrite(str(iter_dir / "normal_map.png"), cv2.cvtColor(n_vis_uint16, cv2.COLOR_RGB2BGR))

        r_idx, c_idx = np.where(mask)
        r_min, r_max = np.min(r_idx), np.max(r_idx)
        c_min, c_max = np.min(c_idx), np.max(c_idx)

        Z_crop_mm  = Z[r_min:r_max+1, c_min:c_max+1] * 1000.0
        x_plot_mm  = np.arange(Z_crop_mm.shape[1]) * dx_mm
        y_plot_mm  = np.arange(Z_crop_mm.shape[0]) * dy_mm
        Z_crop_vis = Z_crop_mm   

        fig, ax1 = plt.subplots(figsize=(8, 7))
        im = ax1.imshow(Z_crop_vis, extent=[x_plot_mm[0], x_plot_mm[-1], y_plot_mm[-1], y_plot_mm[0]], cmap='viridis')
        ax1.set_title(f"2D Depth Map - Iteration {iter_num}", fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        fig.colorbar(im, ax=ax1, label='Depth (mm)', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(iter_dir / "2D_depth_map.png", dpi=200)
        plt.close(fig)

        z_valid_for_color = Z_crop_vis[~np.isnan(Z_crop_vis)]
        cmin_val = float(np.nanmin(z_valid_for_color)) if len(z_valid_for_color) > 0 else 0.0
        cmax_val = float(np.nanmax(z_valid_for_color)) if len(z_valid_for_color) > 0 else 1.0

        fig3d = go.Figure(data=[go.Surface(
            z=Z_crop_vis, x=x_plot_mm, y=y_plot_mm,
            colorscale=[[0.0, 'limegreen'], [0.5, 'greenyellow'], [1.0, 'yellow']],
            cmin=cmin_val, cmax=cmax_val,
            lighting=dict(ambient=0.2, diffuse=1.0, roughness=0.8, specular=0.1, fresnel=0.0),
            lightposition=dict(x=-1000, y=0, z=1), colorbar=dict(title='Z (mm)')
        )])

        range_x_mm = float(np.ptp(x_plot_mm)) if len(x_plot_mm) > 1 else 1.0
        range_y_mm = float(np.ptp(y_plot_mm)) if len(y_plot_mm) > 1 else 1.0
        z_valid_mm = Z_crop_mm[~np.isnan(Z_crop_mm)]
        range_z_mm = float(np.ptp(z_valid_mm)) if len(z_valid_mm) > 0 else 0.1
        if range_z_mm < 0.1: range_z_mm = 0.1

        Z_EXAGGERATION = float(self.cfg.get('global_settings', {}).get('z_exaggeration', 1.0))

        fig3d.update_layout(
            title=f"3D Surface - Iteration {iter_num}  [Z ×{Z_EXAGGERATION:.1f}]",
            scene=dict(
                xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title=f'Z (mm)',
                yaxis=dict(autorange="reversed"),
                aspectmode='manual',
                aspectratio=dict(x=1.0, y=range_y_mm / range_x_mm, z=(range_z_mm / range_x_mm) * Z_EXAGGERATION),
                camera=dict(eye=dict(x=1.4, y=-1.4, z=1.2), up=dict(x=0, y=0, z=1))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig3d.write_html(str(iter_dir / "3D_surface_interactive.html"))

        df_out = df.copy()
        z_vals = Z[v_idx, u_idx]
        df_out['z_world'] = z_vals
        z_vals_valid = z_vals[~np.isnan(z_vals)]
        surface_floor_csv = np.nanmin(z_vals_valid) if len(z_vals_valid) > 0 else 0.0
        df_out['z_extrusion'] = z_vals - surface_floor_csv
        out_csv = iter_dir / f"mapping_iter{iter_num}.csv"
        df_out.to_csv(out_csv, index=False)

        return out_csv, Z[v_idx, u_idx]

    # -------------------------------------------------------------------------
    def run(self):
        t_pipeline_start = time.time()

        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        df = pd.read_csv(csv_path)
        previous_z_vals = None

        def record_mem(step_name):
            ram, vram, swap = self._get_memory_usage()
            self.mem_stats[step_name][0] = max(self.mem_stats[step_name][0], ram)
            self.mem_stats[step_name][1] = max(self.mem_stats[step_name][1], vram)
            self.mem_stats[step_name][2] = max(self.mem_stats[step_name][2], swap)

        final_normals, final_Z_map, final_dx, final_dy, final_mask = None, None, None, None, None
        final_iter_num = 0

        for i in range(self.max_iterations):
            print(f"\n{'='*65}")
            print(f"  ITERATION {i}")
            print(f"{'='*65}")
            t_iter_start = time.time()

            t_init = time.time()
            if i == 0:
                cam_cfg         = self.cfg.get('camera', {})
                sensor_width_mm = cam_cfg.get('sensor_width_mm', 35.9)
                focal_length_mm = cam_cfg.get('focal_length_mm', 50.0)
                object_dist_m   = cam_cfg.get('object_distance_m', 0.5)
                res_w           = self.cfg['resolution']['width']
                res_h           = self.cfg['resolution']['height']

                sensor_pixel_size_mm = sensor_width_mm / res_w
                dx_meters = sensor_pixel_size_mm * (object_dist_m / focal_length_mm)
                dy_meters = dx_meters

                if cam_cfg.get('use_auto_center', True):
                    cam_u = res_w / 2.0
                    cam_v = res_h / 2.0
                else:
                    cam_u, cam_v = cam_cfg.get('manual_center_pixel', [0, 0])

                df['x_world'] = (df['pixel_u'] - cam_u) * dx_meters
                df['y_world'] =  -(df['pixel_v'] - cam_v) * dy_meters
                df['z_world'] = 0.0

                if DEBUG:
                    x_range = df['x_world'].max() - df['x_world'].min()
                    y_range = df['y_world'].max() - df['y_world'].min()
                    print(f"  [DBG|WORLD_COORDS] Pinhole model applied (iteration 0):")
                    print(f"  [DBG|WORLD_COORDS]   pixel size on object : {dx_meters*1e3:.4f} mm/px")
                    print(f"  [DBG|WORLD_COORDS]   x_world range        : [{df['x_world'].min()*1e3:.2f}, {df['x_world'].max()*1e3:.2f}] mm  (span {x_range*1e3:.2f} mm)")
                    print(f"  [DBG|WORLD_COORDS]   y_world range        : [{df['y_world'].min()*1e3:.2f}, {df['y_world'].max()*1e3:.2f}] mm  (span {y_range*1e3:.2f} mm)")
                    print(f"  [DBG|WORLD_COORDS]   principal point      : pixel ({cam_u:.1f}, {cam_v:.1f})")

            self.timers['1_initialization'] += (time.time() - t_init)
            record_mem('1_initialization')

            print(f"\n  > Estimating Normals (AREA LIGHT)...")
            t_norm = time.time()
            if HAS_GPU:
                normals_np = self.estimate_normals_gpu(df)
            else:
                normals_np = self.estimate_normals_cpu(df)
            
            t_normals_dur = time.time() - t_norm
            self.timers['2_estimate_normals'] += t_normals_dur
            record_mem('2_estimate_normals')
            self._dbp("TIMING", f"estimate_normals: {t_normals_dur:.3f} s")

            print(f"\n  > Repairing Normals...")
            t_rep = time.time()
            u_idx_r = df['pixel_u'].values.astype(int)
            v_idx_r = df['pixel_v'].values.astype(int)
            M_img_r = int(np.max(v_idx_r)) + 1
            N_img_r = int(np.max(u_idx_r)) + 1
            
            if HAS_GPU:
                normals_np = self.repair_normals_gpu(normals_np, v_idx_r, u_idx_r, M_img_r, N_img_r)
            else:
                normals_np = self.repair_normals_cpu(normals_np, v_idx_r, u_idx_r, M_img_r, N_img_r)
            
            self.timers['3_repair_normals'] += (time.time() - t_rep)
            record_mem('3_repair_normals')

            print(f"\n  > Reconstructing Surface...")
            t_rec = time.time()
            Z_map, dx, dy, mask = self.reconstruct_surface(df, normals_np)
            t_recon_dur = time.time() - t_rec
            self.timers['4_reconstruct_surface'] += t_recon_dur
            record_mem('4_reconstruct_surface')
            self._dbp("TIMING", f"reconstruct_surface: {t_recon_dur:.3f} s")

            final_normals, final_Z_map, final_dx, final_dy, final_mask = normals_np, Z_map, dx, dy, mask
            final_iter_num = i

            t_conv = time.time()
            z_vals_raw = Z_map[df['pixel_v'].values.astype(int), df['pixel_u'].values.astype(int)]
            nan_count  = int(np.sum(np.isnan(z_vals_raw)))
            if nan_count > 0:
                print(f"  [WARN|RUN       ] {nan_count:,} NaN Z values replaced with 0 to prevent cascade failure.")
                z_vals_raw = np.nan_to_num(z_vals_raw, nan=0.0)
            df['z_world'] = z_vals_raw
            
            z_world_clean = df['z_world'].values.copy()

            if DEBUG:
                z_valid = z_world_clean[~np.isnan(z_world_clean)]
                z_range_mm = (z_valid.max() - z_valid.min()) * 1e3 if len(z_valid) > 0 else 0.0
                z_mean_mm  = z_valid.mean() * 1e3 if len(z_valid) > 0 else 0.0
                print(f"  [DBG|Z_ITER {i:02d}   ] z_world stats:  range={z_range_mm:.4f} mm  mean={z_mean_mm:.4f} mm  valid_px={len(z_valid):,}")

            t_total = time.time() - t_iter_start
            
            is_converged = False
            if previous_z_vals is not None:
                mad = np.mean(np.abs(z_world_clean - previous_z_vals))
                print(f"\n  [CONVERGENCE] Iteration {i}: MAD = {mad:.6e} m  (threshold = {self.convergence_threshold:.0e} m)")
                if np.isnan(mad):
                    print(f"  [CONVERGENCE] ✗ MAD is NaN — Z solve failed. Restoring previous valid z_world.")
                    df['z_world'] = previous_z_vals
                elif mad < self.convergence_threshold:
                    print(f"  [CONVERGENCE] ✓ Converged after {i} iterations!")
                    is_converged = True
            else:
                print(f"\n  [CONVERGENCE] Iteration 0 — baseline established.")

            self.timers['6_df_and_convergence'] += (time.time() - t_conv)
            print(f"  [TIMING     ] Total iteration wall time: {t_total:.2f} s")
            previous_z_vals = z_world_clean
            
            if is_converged:
                break

        print(f"\n  > Saving FINAL Visualizations...")
        t_vis = time.time()
        
        current_csv, current_z_vals = self.save_visualizations(final_iter_num, final_normals, df, final_Z_map, final_dx, final_dy, final_mask)
        
        t_vis_dur = time.time() - t_vis
        self.timers['5_save_visualizations'] += t_vis_dur
        record_mem('5_save_visualizations')
        self._dbp("TIMING", f"save_visualizations: {t_vis_dur:.3f} s")

        self.timers['total_pipeline_time'] = time.time() - t_pipeline_start
        print("\n" + "="*65)
        print("  PROFILING SUMMARY (Cumulative Time Across All Iterations)")
        print("="*65)
        for key, value in sorted(self.timers.items()):
            if "total" in key: print("-" * 65)
            print(f"  {key:25s} : {value:8.3f} seconds")
        print("="*65 + "\n")

        print("\n" + "="*80)
        print("  MEMORY PROFILING SUMMARY (Peak Memory Observed Per Step)")
        print("="*80)
        print(f"  {'Step':<25s} | {'Peak CPU RAM':<15s} | {'Peak GPU VRAM':<15s} | {'Peak System Swap':<15s}")
        print("-" * 80)
        for key, values in sorted(self.mem_stats.items()):
            print(f"  {key:<25s} | {values[0]:7.1f} MB       | {values[1]:7.1f} MB       | {values[2]:7.1f} MB")
        print("="*80 + "\n")

        print(f"  Pipeline finished. Outputs: {self.output_dir}")
        print("="*65 + "\n")

        html_candidate = self.output_dir / f"iteration_{final_iter_num:02d}" / "3D_surface_interactive.html"
        if html_candidate.exists():
            return str(html_candidate)
        return None


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\chand\OneDrive\Desktop\all data\codes\6th_april_code\upd_basis_flipped.json"
    print(f"Config path: {cfg_path}")
    if os.path.exists(cfg_path):
        AutoIterativePipeline(cfg_path).run()
    else:
        print(f"File not found: {cfg_path}")