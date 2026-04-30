import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import cupy as cp
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized

# Note: cupyx.scipy.sparse imports removed — they were imported in Code 1 but
# never used anywhere in the pipeline. Removing them is pure cleanup.

# =============================================================================
# BOOLEAN FLAGS — toggle here, no need to touch anything else in the file
# =============================================================================
DEBUG            = True   # Master switch: all [DBG] and [SOLVER] print blocks
FLAT_PLANE_ERROR = True   # Compute mean angular error vs flat plane [0,0,1]
COMPUTE_COND_NUM = False  # Estimate condition number of C=AᵀA (diagonal heuristic,
                          # never hangs). Kept False by default — informational only.
#   - When FLAT_PLANE_ERROR=True, error is computed at TWO points per iteration:
#     (1) On estimated normals, BEFORE Poisson integration
#     (2) On normals back-derived from the reconstructed Z map, AFTER integration
#   This lets you see how much the Poisson step changes the effective normal field.
# =============================================================================

# =============================================================================
# WHAT THIS COMBINED CODE DOES vs THE TWO SOURCE CODES
#
# Base: check_surface_recon.py (Code 1) — all accuracy decisions kept exactly.
#
# Added from check_autom_w_debug.py (Code 2) — speed only, zero accuracy change:
#   1. Chunked CUDA dispatch in get_light_samples_gpu (50k pixels/chunk, explicit sync)
#   2. Image caching in estimate_normals — images loaded once into VRAM, reused
#      every iteration. G is always recomputed (depends on z_world which changes).
#   3. Matrix caching in reconstruct_surface — Poisson system A and its SuperLU
#      factorization built once, reused from iteration 1 onward. b (RHS) recomputed
#      every iteration from updated normals/gradients as normal.
#      Safe because: mask, erosion params, and pixel coordinates are fixed.
#
# Visual fix (agreed):
#   4. cmin/cmax in go.Surface now computed from actual Z data range each iteration.
#      Previously hardcoded to -0.1 / 0.7, clipping the top half of feature height.
#   5. Dead stride variables (Z_plot, x_plot_3d, y_plot_3d, plotly_stride) removed.
#      These were computed in Code 1 but never passed to go.Surface — dead code.
#
# Everything else: identical to Code 1, line for line.
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
    # DEBUG HELPERS  (unchanged from Code 1)
    # -------------------------------------------------------------------------
    def _dbp(self, section, msg):
        """Print one debug line, gated by DEBUG flag."""
        if DEBUG:
            print(f"  [DBG|{section:12s}] {msg}")

    def _flat_plane_error(self, normals_nx3, label=""):
        """
        Compute and print angular error of each normal vs the flat plane [0,0,1].
        Formula: angle_i = arccos(nz_i)   (normals assumed unit-length)
        No abs() — a normal pointing [0,0,-1] has 180° error, which is real info.
        Gated by FLAT_PLANE_ERROR flag.
        """
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
        """
        Back-derive surface normals from a height map using finite differences,
        then compute angular error vs [0,0,1].

        For a height field z = Z(x, y):
          surface tangent along x : [1, 0, dZ/dx]
          surface tangent along y : [0, 1, dZ/dy]
          outward normal (upward)  : cross product = [-dZ/dx, -dZ/dy, 1]  (unnorm)

        np.gradient(Z, dy, dx) returns [dZ/d(row), dZ/d(col)] = [dZ/dy, dZ/dx].
        """
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

    # -------------------------------------------------------------------------
    def __init__(self, config_input):
        # Accept either a cfg dict (from process_pipeline) or a path string
        if isinstance(config_input, dict):
            self.cfg = config_input
        else:
            with open(config_input, 'r') as f:
                self.cfg = json.load(f)
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')
        self.max_iterations = self.cfg.get('max_iterations', 15)
        self.convergence_threshold = self.cfg.get('convergence_threshold', 1e-5)

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(f"Missing image: {img_path}")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        if gamma != 1.0: img = np.power(img, gamma)
        return cp.array(img)

    # -------------------------------------------------------------------------
    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        """
        CHANGE vs Code 1: Chunked CUDA dispatch (from Code 2).
        Processes pixels in batches of 50,000 with explicit GPU sync between chunks.
        More memory-safe for large pixel counts. Mathematical result is identical
        to Code 1's single-dispatch — same kernel, same arithmetic, just split.
        """
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

        # --- SPEED CHANGE: Chunked dispatch (Code 2) ---
        chunk_size        = 50000
        threads_per_block = 128
        for start_idx in range(0, num_pixels, chunk_size):
            end_idx       = min(start_idx + chunk_size, num_pixels)
            current_chunk = end_idx - start_idx
            P_surf_chunk  = P_surf_full[start_idx:end_idx]
            G_out_chunk   = G_eff_full[start_idx:end_idx]
            blocks_per_grid = (current_chunk + threads_per_block - 1) // threads_per_block
            self.kernel((blocks_per_grid,), (threads_per_block,),
                (P_surf_chunk, sample_pts_gpu, norm_gpu, cp.float32(cos_half_spread),
                 cp.int32(current_chunk), cp.int32(num_samples), G_out_chunk))
            cp.cuda.Device(0).synchronize()

        return G_eff_full, sample_pts   # also return sample_pts for debug use

    # -------------------------------------------------------------------------
    def estimate_normals(self, df):
        """
        CHANGE vs Code 1: Image caching (from Code 2).
        Images (I_raw) and the thresholded versions (I_w_cache, W_cache) are loaded
        once on the first call and cached on VRAM. Every subsequent call skips
        image loading entirely.
        G is always recomputed — it depends on z_world which changes each iteration.
        All accuracy decisions (regularisation=1.0, gradient signs, etc.) unchanged.
        """
        u_idx  = df['pixel_u'].values.astype(int)
        v_idx  = df['pixel_v'].values.astype(int)

        # ── Elevation offset (unchanged from Code 1) ───────────────────────────
        # z_world is stored as *relative* Poisson depth (0 at anchor, updated each
        # iteration). Light coordinates are in the *absolute* frame whose Z=0 is
        # the ground/camera origin, not the object surface.
        # We must shift z_world into that absolute frame before computing G vectors,
        # otherwise the kernel sees the surface at Z=0 instead of Z=elevation.
        # NOTE: only the copy fed to the kernel is shifted — df['z_world'] is NOT
        # modified here, so Poisson integration and convergence checks are unaffected.
        elev_m = self.cfg.get('camera', {}).get('object_elevation_m', 0.0)
        xyz_cpu = df[['x_world', 'y_world', 'z_world']].values.copy().astype(np.float32)
        xyz_cpu[:, 2] += elev_m   # shift Z into absolute frame for G computation

        if DEBUG:
            print(f"  [DBG|ELEVATION  ] object_elevation_m = {elev_m*1000:.3f} mm  "
                  f"(added to z_world before kernel; df unchanged)")
            print(f"  [DBG|ELEVATION  ] z fed to kernel: "
                  f"min={xyz_cpu[:,2].min()*1000:.3f} mm  "
                  f"max={xyz_cpu[:,2].max()*1000:.3f} mm  "
                  f"mean={xyz_cpu[:,2].mean()*1000:.3f} mm")

        P_surf = cp.ascontiguousarray(cp.array(xyz_cpu, dtype=cp.float32))

        num_pixels = len(df)
        num_lights = len(self.cfg['lights'])
        G = cp.zeros((num_pixels, num_lights, 3), dtype=cp.float32)

        # For point-light comparison we need the object center
        obj_center = np.array([df['x_world'].mean(),
                                df['y_world'].mean(),
                                df['z_world'].mean() + elev_m])  # absolute frame

        self._dbp("NORMALS", f"Pixel count: {num_pixels:,}  |  Lights: {num_lights}")
        self._dbp("NORMALS", f"Object center (world): x={obj_center[0]:.6f} m  y={obj_center[1]:.6f} m  z={obj_center[2]:.6f} m")

        global_cfg   = self.cfg.get('global_settings', {})
        apply_thresh = global_cfg.get('apply_dark_threshold', True)
        thresh_val   = global_cfg.get('dark_threshold_value', 0.025)

        # --- SPEED CHANGE: Image caching (from Code 2) ---
        # Images are static across all iterations. Load them once on first call,
        # apply dark threshold, cache I_w_cache (thresholded intensities) and
        # W_cache (threshold mask) on VRAM. Also cache I_raw_cache for debug
        # intensity prints (pct_above, mean, std, min, max).
        if not hasattr(self, 'I_w_cache'):
            self._dbp("SPEED", "Caching image intensities to VRAM (One-time cost)...")
            I_raw = cp.zeros((num_pixels, num_lights), dtype=cp.float32)
            for j, l_cfg in enumerate(self.cfg['lights']):
                img_path = Path(self.cfg['paths']['image_dir']) / l_cfg['file_name']
                img_gpu  = self.load_image(img_path, l_cfg['gamma'], l_cfg['bit_depth'])
                I_raw[:, j] = img_gpu[v_idx, u_idx]
                del img_gpu
                cp.get_default_memory_pool().free_all_blocks()
            if apply_thresh:
                W = (I_raw >= thresh_val).astype(cp.float32)
                self.I_w_cache  = I_raw * W
                self.W_cache    = W
            else:
                self.I_w_cache  = I_raw
                self.W_cache    = cp.ones_like(I_raw)
            self.I_raw_cache = I_raw   # kept for debug intensity stats only

        # ---- Per-light loop: G always recomputed (z_world changes each iter) ----
        for j, l_cfg in enumerate(self.cfg['lights']):
            G_j_gpu, sample_pts = self.get_light_samples_gpu(l_cfg, P_surf)
            G[:, j, :] = G_j_gpu

            if DEBUG:
                light_pos = np.array(l_cfg['pos_m'])

                # Intensity stats — use cached raw intensities (same every iteration)
                I_j_cpu   = cp.asnumpy(self.I_raw_cache[:, j])
                pct_above = 100.0 * np.mean(I_j_cpu >= thresh_val)

                # Area-light effective direction stats
                G_j_cpu  = cp.asnumpy(G_j_gpu)                    # (N, 3)
                G_j_mag  = np.linalg.norm(G_j_cpu, axis=1)        # (N,)
                G_j_dir  = G_j_cpu / np.maximum(G_j_mag[:, None], 1e-10)

                mean_G_weighted  = np.mean(G_j_cpu, axis=0)
                mean_G_mag_total = np.linalg.norm(mean_G_weighted)
                mean_G_dir_norm  = mean_G_weighted / max(mean_G_mag_total, 1e-10)

                cos_to_mean = np.clip(np.sum(G_j_dir * mean_G_dir_norm[None, :], axis=1), -1.0, 1.0)
                angular_spread_per_pixel = np.degrees(np.arccos(cos_to_mean))

                pt_dir_raw  = light_pos - obj_center
                pt_dist     = np.linalg.norm(pt_dir_raw)
                pt_dir_norm = pt_dir_raw / max(pt_dist, 1e-10)

                cos_pt_vs_area = np.clip(np.dot(pt_dir_norm, mean_G_dir_norm), -1.0, 1.0)
                ang_pt_vs_area = np.degrees(np.arccos(cos_pt_vs_area))

                s_min = sample_pts.min(axis=0)
                s_max = sample_pts.max(axis=0)

                print(f"  [DBG|LIGHT {j:02d}    ] ── Light {l_cfg['id']} ─────────────────────────────────────────")
                print(f"  [DBG|LIGHT {j:02d}    ] pos={list(np.round(light_pos,5))} m  norm_dir={l_cfg['norm_dir']}")
                print(f"  [DBG|LIGHT {j:02d}    ] dims={l_cfg['dims_m']} m  spread={l_cfg.get('spread_deg',180)}°  samples={l_cfg['sampling']} ({len(sample_pts):,} pts)")
                print(f"  [DBG|LIGHT {j:02d}    ] Sample bounding box: min={np.round(s_min,5).tolist()}  max={np.round(s_max,5).tolist()}")
                print(f"  [DBG|LIGHT {j:02d}    ]")
                print(f"  [DBG|LIGHT {j:02d}    ] >>> AREA-LIGHT mean eff. direction (weighted): [{mean_G_dir_norm[0]:+.4f}  {mean_G_dir_norm[1]:+.4f}  {mean_G_dir_norm[2]:+.4f}]")
                print(f"  [DBG|LIGHT {j:02d}    ] >>> POINT-LIGHT approx direction  (trad. PS) : [{pt_dir_norm[0]:+.4f}  {pt_dir_norm[1]:+.4f}  {pt_dir_norm[2]:+.4f}]  dist={pt_dist:.5f} m")
                print(f"  [DBG|LIGHT {j:02d}    ] >>> Angular difference area vs point:  {ang_pt_vs_area:.3f}°  {'⚠ significant' if ang_pt_vs_area > 2.0 else '✓ small'}")
                print(f"  [DBG|LIGHT {j:02d}    ]")
                print(f"  [DBG|LIGHT {j:02d}    ] G magnitude (inv-sq weighted sum/n_samples):")
                print(f"  [DBG|LIGHT {j:02d}    ]   mean={G_j_mag.mean():.4e}  std={G_j_mag.std():.4e}  min={G_j_mag.min():.4e}  max={G_j_mag.max():.4e}")
                print(f"  [DBG|LIGHT {j:02d}    ] Spatial variation of G direction across pixels:")
                print(f"  [DBG|LIGHT {j:02d}    ]   mean_angular_spread={angular_spread_per_pixel.mean():.3f}°  std={angular_spread_per_pixel.std():.3f}°  max={angular_spread_per_pixel.max():.3f}°")
                print(f"  [DBG|LIGHT {j:02d}    ]   (This is zero for a true point light — non-zero = area-light effect)")
                print(f"  [DBG|LIGHT {j:02d}    ] Intensity: mean={I_j_cpu.mean():.4f}  std={I_j_cpu.std():.4f}  min={I_j_cpu.min():.4f}  max={I_j_cpu.max():.4f}  above_thresh({thresh_val}): {pct_above:.1f}%")
                del G_j_cpu, G_j_mag, G_j_dir

            del G_j_gpu
            cp.get_default_memory_pool().free_all_blocks()

        # ---- Dark threshold / weighting — use cached W and I_w ----
        if apply_thresh:
            G_w = G * self.W_cache[:, :, None]
            I_w = self.I_w_cache
            if DEBUG:
                W_cpu = cp.asnumpy(self.W_cache)
                for j in range(num_lights):
                    pct_w = 100.0 * W_cpu[:, j].mean()
                    self._dbp("THRESH", f"Light {j}: {pct_w:.1f}% pixels above threshold → included in LS solve")
        else:
            G_w = G
            I_w = self.I_w_cache
            self._dbp("THRESH", "Dark threshold disabled — all pixels used.")

        # ---- LS solve (unchanged from Code 1) ----
        # Regularisation = 1.0 (not 1e-4).
        # G magnitudes are ~100, so GTG diagonal is ~10,000-40,000.
        # 1e-4 is numerically zero relative to GTG and allows near-singular
        # matrices to produce NaN from cp.linalg.inv on degenerate pixels.
        # 1.0 is still <0.01% of GTG for well-lit pixels but stabilises them.
        GT_w = G_w.transpose(0, 2, 1)
        GTG  = cp.matmul(GT_w, G_w) + (cp.eye(3, dtype=cp.float32) * 1.0)
        GTI  = cp.matmul(GT_w, I_w[:, :, None])

        # Use solve() not inv() — more numerically stable; handles ill-conditioned
        # 3×3 batches via LU pivoting rather than explicit inversion.
        n_est   = cp.linalg.solve(GTG, GTI).squeeze(-1)   # (N, 3)

        albedo  = cp.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / cp.where(albedo == 0, 1, albedo)

        # Hard NaN guard — replace any remaining NaN normals with [0,0,1].
        nan_mask = cp.any(cp.isnan(normals), axis=1)
        if cp.any(nan_mask):
            n_nan = int(cp.sum(nan_mask))
            normals[nan_mask] = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
            print(f"  [WARN|NORMALS] {n_nan} NaN normals clamped to [0,0,1]")

        # Move to CPU for debug
        normals_cpu = cp.asnumpy(normals)
        albedo_cpu  = cp.asnumpy(albedo).squeeze()

        if DEBUG:
            nz_cpu       = normals_cpu[:, 2]
            pct_nz_pos   = 100.0 * np.mean(nz_cpu > 0)
            pct_nz_neg   = 100.0 * np.mean(nz_cpu < 0)
            pct_nz_small = 100.0 * np.mean(np.abs(nz_cpu) < 0.1)
            print(f"  [DBG|NORMALS    ] --- Estimated normal statistics ---")
            for comp_name, comp_vals in [("nx", normals_cpu[:,0]), ("ny", normals_cpu[:,1]), ("nz", normals_cpu[:,2])]:
                print(f"  [DBG|NORMALS    ]   {comp_name}: "
                      f"min={comp_vals.min():+.4f}  max={comp_vals.max():+.4f}  "
                      f"mean={comp_vals.mean():+.4f}  std={comp_vals.std():.4f}")
            print(f"  [DBG|NORMALS    ]   nz > 0 : {pct_nz_pos:5.1f}%  |  nz < 0 : {pct_nz_neg:5.1f}%  |  |nz| < 0.1 : {pct_nz_small:5.1f}%")
            print(f"  [DBG|NORMALS    ]   Albedo (pre-norm magnitude): "
                  f"min={albedo_cpu.min():.4f}  max={albedo_cpu.max():.4f}  "
                  f"mean={albedo_cpu.mean():.4f}  std={albedo_cpu.std():.4f}")

        self._flat_plane_error(normals_cpu, label="estimated | pre-Poisson")

        del G, G_w, GT_w, GTG, GTI, n_est
        cp.get_default_memory_pool().free_all_blocks()
        return normals_cpu

    # -------------------------------------------------------------------------
    def repair_normals(self, normals, v_idx, u_idx, M_img, N_img):
        """
        UNCHANGED from Code 1.
        Fix 1: Replace physically impossible normals (nz < 0) by inpainting
        from valid neighbors in a 5×5 window.
        """
        from scipy.ndimage import uniform_filter

        NZ_THRESH = 0.0          # below this → repair
        WIN       = 5

        nx_in = normals[:, 0]
        ny_in = normals[:, 1]
        nz_in = normals[:, 2]

        nx_g = np.zeros((M_img, N_img), dtype=np.float64)
        ny_g = np.zeros((M_img, N_img), dtype=np.float64)
        nz_g = np.zeros((M_img, N_img), dtype=np.float64)
        mask_g = np.zeros((M_img, N_img), dtype=bool)
        nx_g[v_idx, u_idx] = nx_in
        ny_g[v_idx, u_idx] = ny_in
        nz_g[v_idx, u_idx] = nz_in
        mask_g[v_idx, u_idx] = True

        good_g = mask_g & (nz_g >= NZ_THRESH)
        bad_g  = mask_g & (nz_g <  NZ_THRESH)
        n_bad_neg  = int(np.sum(mask_g & (nz_g < 0)))
        n_bad_all  = int(np.sum(bad_g))

        print(f"  [DBG|REPAIR] ── Normal Repair ──────────────────────────────")
        if n_bad_all == 0:
            print(f"  [DBG|REPAIR] No bad normals (nz < {NZ_THRESH}) — skipping.")
            return normals

        print(f"  [DBG|REPAIR] Bad pixels total : {n_bad_all:,}  ({100*n_bad_all/len(nx_in):.3f}% of mask)")
        print(f"  [DBG|REPAIR]   nz < 0         : {n_bad_neg:,}")
        print(f"  [DBG|REPAIR]   0 ≤ nz < {NZ_THRESH}  : {n_bad_all - n_bad_neg:,}")

        good_f = good_g.astype(np.float64)
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

        nx_new = nx_g.copy()
        ny_new = ny_g.copy()
        nz_new = nz_g.copy()

        nx_new[has_neighbor] = nx_fill[has_neighbor]
        ny_new[has_neighbor] = ny_fill[has_neighbor]
        nz_new[has_neighbor] = nz_fill[has_neighbor]

        n_fallback = int(np.sum(no_neighbor))
        if n_fallback > 0:
            nx_new[no_neighbor] = 0.0
            ny_new[no_neighbor] = 0.0
            nz_new[no_neighbor] = 1.0
            print(f"  [DBG|REPAIR] {n_fallback:,} pixels had no valid neighbor in {WIN}×{WIN} → set to [0,0,1]")

        nz_after = nz_new[v_idx, u_idx]
        n_still_bad = int(np.sum(nz_after < NZ_THRESH))
        print(f"  [DBG|REPAIR] After repair: still bad (nz<{NZ_THRESH}): {n_still_bad:,}")
        print(f"  [DBG|REPAIR] Repaired nz  — min={nz_after.min():+.4f}  mean={nz_after.mean():+.4f}  max={nz_after.max():+.4f}")

        normals_repaired = np.stack([
            nx_new[v_idx, u_idx],
            ny_new[v_idx, u_idx],
            nz_new[v_idx, u_idx]
        ], axis=1).astype(np.float32)

        return normals_repaired

    # -------------------------------------------------------------------------
    def reconstruct_surface(self, df, normals):
        """
        CHANGE vs Code 1: Matrix caching (from Code 2).
        On the first call: builds the mask (with erosion), constructs Poisson
        system A, factorizes C = AᵀA using SuperLU, and caches everything static.
        On every subsequent call: unpacks cache, skips all matrix building and
        factorization, goes straight to building b (RHS) from the new gradients
        and back-substituting through the cached factorization.
        Safe because mask, erosion parameters, and x_world/y_world are fixed
        across all iterations.

        OLS solver and all gradient decisions (sign, eps_nz, valid=mask) are
        UNCHANGED from Code 1.
        """
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)

        # =====================================================================
        # FIRST CALL ONLY: full setup, matrix build, and factorization
        # =====================================================================
        if not hasattr(self, '_recon_cache'):
            M_img = np.max(v_idx) + 1
            N_img = np.max(u_idx) + 1
            mask  = np.zeros((M_img, N_img), dtype=bool)
            mask[v_idx, u_idx] = True

            # ── Mask erosion (unchanged from Code 1) ──────────────────────────
            erosion_px = int(self.cfg.get('global_settings', {}).get('mask_erosion_pixels', 0))
            keep = np.ones(len(u_idx), dtype=bool)   # default: keep all pixels
            if erosion_px > 0:
                from scipy.ndimage import binary_erosion
                struct = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=bool)
                mask_eroded = binary_erosion(mask, structure=struct)
                n_removed   = int(np.sum(mask)) - int(np.sum(mask_eroded))
                if DEBUG:
                    print(f"  [DBG|MASK_EROSION] Eroding mask by {erosion_px} px  "
                          f"(struct {erosion_px*2+1}×{erosion_px*2+1})  "
                          f"Removed: {n_removed:,} boundary pixels")
                mask = mask_eroded
                keep = mask[v_idx, u_idx]

            u_idx_e = u_idx[keep]
            v_idx_e = v_idx[keep]
            df_e    = df[keep].reset_index(drop=True)
            num_valid_px = int(np.sum(mask))

            # ── Coordinate grids (x_world, y_world don't change) ──────────────
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

            # ── Build Poisson system ───────────────────────────────────────────
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
            I_list = np.concatenate([
                np.arange(num_H), np.arange(num_H),
                np.arange(num_H, num_H + num_V), np.arange(num_H, num_H + num_V),
                [num_eq - 1]
            ])
            J_list = np.concatenate([
                id_right_H, id_self_H,
                id_down_V,  id_self_V,
                [center_node_id]
            ])
            V_list = np.concatenate([
                np.ones(num_H), -np.ones(num_H),
                np.ones(num_V), -np.ones(num_V),
                [1.0]
            ])
            A = coo_matrix((V_list, (I_list, J_list)), shape=(num_eq, num_unknowns)).tocsr()

            # OLS normal equations: C = AᵀA
            C   = A.T @ A
            A_T = A.T

            # ── Factorize once — SPEED CHANGE (from Code 2) ───────────────────
            self._dbp("SOLVER", f"Factoring Poisson Matrix using Direct Solver (SuperLU)...")
            t_factor_start = time.time()
            solver = factorized(C)
            t_factor = time.time() - t_factor_start
            self._dbp("SOLVER", f"Direct solve complete ✓  |  time: {t_factor:.3f} s")

            # ── Cache everything static ────────────────────────────────────────
            self._recon_cache = dict(
                keep=keep, u_idx_e=u_idx_e, v_idx_e=v_idx_e,
                M_img=M_img, N_img=N_img, mask=mask,
                center_r=center_r, center_c=center_c,
                dx=dx, dy=dy, X_grid=X_grid, Y_grid=Y_grid,
                r_H=r_H, c_H=c_H, r_V=r_V, c_V=c_V,
                A=A, A_T=A_T, solver=solver,
                num_unknowns=num_unknowns,
                num_H=num_H, num_V=num_V, num_eq=num_eq,
            )

        # =====================================================================
        # EVERY CALL: unpack cache, build RHS from current normals, solve
        # =====================================================================
        rc           = self._recon_cache
        keep         = rc['keep']
        u_idx_e      = rc['u_idx_e']
        v_idx_e      = rc['v_idx_e']
        M_img        = rc['M_img']
        N_img        = rc['N_img']
        mask         = rc['mask']
        center_r     = rc['center_r']
        center_c     = rc['center_c']
        dx           = rc['dx']
        dy           = rc['dy']
        X_grid       = rc['X_grid']
        Y_grid       = rc['Y_grid']
        r_H          = rc['r_H']
        c_H          = rc['c_H']
        r_V          = rc['r_V']
        c_V          = rc['c_V']
        A            = rc['A']
        A_T          = rc['A_T']
        solver       = rc['solver']
        num_unknowns = rc['num_unknowns']
        num_H        = rc['num_H']
        num_V        = rc['num_V']
        num_eq       = rc['num_eq']

        # Apply same erosion keep to normals (identical to Code 1 behavior)
        normals_e = normals[keep]

        # Debug prints every iteration (same as Code 1)
        self._dbp("RECONSTRUCT", f"Grid: {M_img}×{N_img}  |  Valid pixels (mask): {num_unknowns:,}")
        self._dbp("RECONSTRUCT", f"Physical spacing: dx={dx*1e3:.4f} mm  dy={dy*1e3:.4f} mm")
        self._dbp("RECONSTRUCT", f"Center anchor pixel: row={center_r}  col={center_c}")

        # ── Normal grids from current iteration's normals ─────────────────────
        nx_grid = np.zeros((M_img, N_img))
        ny_grid = np.zeros((M_img, N_img))
        nz_grid = np.zeros((M_img, N_img))
        nx_grid[v_idx_e, u_idx_e] = normals_e[:, 0]
        ny_grid[v_idx_e, u_idx_e] = normals_e[:, 1]
        nz_grid[v_idx_e, u_idx_e] = normals_e[:, 2]   # raw, no abs()

        if DEBUG:
            nz_on_mask   = nz_grid[mask]
            pct_nz_small = 100.0 * np.mean(np.abs(nz_on_mask) < 0.05)
            pct_nz_neg   = 100.0 * np.mean(nz_on_mask < 0)
            print(f"  [DBG|RECONSTRUCT] Normal components on mask:")
            for comp_name, comp_grid in [("nx", nx_grid), ("ny", ny_grid), ("nz", nz_grid)]:
                v = comp_grid[mask]
                print(f"  [DBG|RECONSTRUCT]   {comp_name}: min={v.min():+.4f}  max={v.max():+.4f}  mean={v.mean():+.4f}  std={v.std():.4f}")
            print(f"  [DBG|RECONSTRUCT]   nz < 0        : {pct_nz_neg:.1f}%")
            print(f"  [DBG|RECONSTRUCT]   |nz| < 0.05   : {pct_nz_small:.1f}%  (near-grazing — can cause large gradients)")

        # ── Gradients p, q (unchanged from Code 1) ────────────────────────────
        # Correct photometric stereo gradient convention:
        # For a surface z=Z(x,y), outward normal n = (-dZ/dx, -dZ/dy, 1)/|...|
        # => dZ/dx = -nx/nz  and  dZ/dy = -ny/nz
        # The minus signs are required — without them the Poisson integration
        # produces a mold impression (inverted relief).
        # NOTE: Code 1 uses positive signs (+nx/nz) here because x_world was
        # defined with a leading negative: x_world = -(pixel_u - cam_u)*dx.
        # The two negations cancel, giving the correct outward-pointing surface.
        eps_nz = 0.15
        valid  = mask
        # Clamp magnitude to eps while PRESERVING sign.
        # Pixels with nz==0 treated as slightly positive (upward-facing surface).
        nz_abs_clamped = np.maximum(np.abs(nz_grid), eps_nz)
        nz_safe = np.where(nz_grid >= 0, nz_abs_clamped, -nz_abs_clamped)
        nz_safe[~valid] = 1.0

        p = np.zeros((M_img, N_img))
        q = np.zeros((M_img, N_img))
        p[valid] = -(nx_grid[valid] / nz_safe[valid]) * dx
        q[valid] = +(ny_grid[valid] / nz_safe[valid]) * dy

        if DEBUG:
            pv = p[valid]; qv = q[valid]
            pct_p_large = 100.0 * np.mean(np.abs(pv) > 10)
            pct_q_large = 100.0 * np.mean(np.abs(qv) > 10)
            print(f"  [DBG|RECONSTRUCT] Gradient fields (eps_nz={eps_nz:.0e}):")
            print(f"  [DBG|RECONSTRUCT]   p: min={pv.min():+.4e}  max={pv.max():+.4e}  mean={pv.mean():+.4e}  std={pv.std():.4e}  |p|>10: {pct_p_large:.2f}%")
            print(f"  [DBG|RECONSTRUCT]   q: min={qv.min():+.4e}  max={qv.max():+.4e}  mean={qv.mean():+.4e}  std={qv.std():.4e}  |q|>10: {pct_q_large:.2f}%")

        # System size debug (printed every iteration, same as Code 1)
        self._dbp("SOLVER", f"System size: {num_eq:,} equations  |  {num_unknowns:,} unknowns")
        self._dbp("SOLVER", f"Horizontal eqs (p): {num_H:,}  |  Vertical eqs (q): {num_V:,}  |  Anchor: 1")
        self._dbp("SOLVER", f"Overdetermination ratio: {num_eq/num_unknowns:.3f}×")

        # ── Build RHS b from current gradients ────────────────────────────────
        val_p = p[r_H, c_H]
        val_q = q[r_V, c_V]
        b     = np.concatenate([val_p, val_q, [0.0]])

        self._dbp("SOLVER", f"RHS vector b: ||b||₂ = {np.linalg.norm(b):.4f}")
        if DEBUG:
            print(f"  [DBG|SOLVER      ] Equation weights: OLS (uniform = 1.0)")

        # ── Solve: d = Aᵀb, z = solver(d) using cached factorization ─────────
        d = A_T @ b
        t_solve = time.time()
        z = solver(d)
        t_solve = time.time() - t_solve

        if DEBUG:
            self._dbp("SOLVER", f"Direct solve complete ✓  |  time: {t_solve:.3f} s")
            residual     = A @ z - b
            res_norm     = np.linalg.norm(residual)
            b_norm       = np.linalg.norm(b)
            rel_residual = res_norm / b_norm if b_norm > 0 else float('inf')
            self._dbp("SOLVER", f"Residual  ||Az - b||₂         = {res_norm:.4e}")
            self._dbp("SOLVER", f"Relative  ||Az - b||₂/||b||₂  = {rel_residual:.4e}  "
                                 f"{'✓ good' if rel_residual < 1e-4 else '⚠ check normals'}")

        if DEBUG:
            z_mm = z * 1000.0
            self._dbp("SOLVER", f"z solution (all {num_unknowns:,} unknowns):")
            self._dbp("SOLVER", f"  min={z_mm.min():.4f} mm   max={z_mm.max():.4f} mm   mean={z_mm.mean():.4f} mm   std={z_mm.std():.4f} mm   range={np.ptp(z_mm):.4f} mm")

        # ── Map back to full grid ──────────────────────────────────────────────
        Z = np.full((M_img, N_img), np.nan)
        Z[mask] = z

        if DEBUG:
            Z_valid = Z[mask] * 1000.0
            nan_pct = 100.0 * np.sum(np.isnan(Z)) / Z.size
            self._dbp("RECONSTRUCT", f"Z map statistics:")
            self._dbp("RECONSTRUCT", f"  min={Z_valid.min():.4f} mm   max={Z_valid.max():.4f} mm   mean={Z_valid.mean():.4f} mm")
            self._dbp("RECONSTRUCT", f"  std={Z_valid.std():.4f} mm   range={np.ptp(Z_valid):.4f} mm   NaN pixels: {nan_pct:.1f}%")

        # ── Detrending (unchanged from Code 1) ────────────────────────────────
        detrend_mode = self.cfg.get('global_settings', {}).get('detrending_mode', 'none').lower()

        valid_X = X_grid[mask]
        valid_Y = Y_grid[mask]
        valid_Z = Z[mask]

        Z_before = np.ptp(Z[mask]) * 1000.0

        if detrend_mode == 'quadratic':
            self._dbp("DETREND", "Mode = QUADRATIC — removing macro bowl/tilt for flat objects")
            A_quad = np.c_[valid_X**2, valid_Y**2, valid_X * valid_Y,
                           valid_X, valid_Y, np.ones_like(valid_X)]
            C_quad, _, _, _ = np.linalg.lstsq(A_quad, valid_Z, rcond=None)
            Detrend_Z = (C_quad[0] * X_grid**2 + C_quad[1] * Y_grid**2 +
                         C_quad[2] * X_grid * Y_grid + C_quad[3] * X_grid +
                         C_quad[4] * Y_grid + C_quad[5])
            self._dbp("DETREND", f"Quadratic coefficients: a={C_quad[0]:.4e}  b={C_quad[1]:.4e}  "
                                  f"c={C_quad[2]:.4e}  d={C_quad[3]:.4e}  e={C_quad[4]:.4e}  f={C_quad[5]:.4e}")
            Z[mask] = Z[mask] - Detrend_Z[mask]

        elif detrend_mode == 'linear':
            self._dbp("DETREND", "Mode = LINEAR — removing planar tilt only")
            A_plane = np.c_[valid_X, valid_Y, np.ones_like(valid_X)]
            C_plane, _, _, _ = np.linalg.lstsq(A_plane, valid_Z, rcond=None)
            Detrend_Z = (C_plane[0] * X_grid + C_plane[1] * Y_grid + C_plane[2])
            self._dbp("DETREND", f"Plane coefficients: nx={C_plane[0]:.4e}  ny={C_plane[1]:.4e}  d={C_plane[2]:.4e}")
            Z[mask] = Z[mask] - Detrend_Z[mask]

        else:
            self._dbp("DETREND", f"Mode = NONE (or unknown: '{detrend_mode}') — keeping raw Poisson data.")

        Z_after = np.ptp(Z[mask]) * 1000.0
        if detrend_mode in ['linear', 'quadratic']:
            self._dbp("DETREND", f"Z range before detrend: {Z_before:.4f} mm  ->  after: {Z_after:.4f} mm")
        else:
            self._dbp("DETREND", f"Z range remains: {Z_after:.4f} mm")

        # ── Floor anchor (unchanged from Code 1) ──────────────────────────────
        surface_floor = np.nanmin(Z[mask])
        Z[mask] = Z[mask] - surface_floor
        self._dbp("DETREND", f"Floor anchor (min -> 0): shifted by {surface_floor*1000:.4f} mm")

        if DEBUG:
            Z_valid_post = Z[mask] * 1000.0
            self._dbp("DETREND", f"Final Z: min={Z_valid_post.min():.4f} mm  max={Z_valid_post.max():.4f} mm  "
                                  f"mean={Z_valid_post.mean():.4f} mm  std={Z_valid_post.std():.4f} mm  "
                                  f"range={np.ptp(Z_valid_post):.4f} mm")

        # Flat plane error — AFTER Poisson + detrend (back-derived from Z)
        self._flat_plane_error_from_Z(Z, mask, v_idx_e, u_idx_e, dx, dy,
                                       label="reconstructed | post-Poisson+detrend")

        return Z, dx, dy, mask

    # -------------------------------------------------------------------------
    def save_visualizations(self, iter_num, normals, df, Z, dx, dy, mask):
        """
        CHANGES vs Code 1:
          1. cmin / cmax: computed from actual Z data range each iteration.
             Code 1 had hardcoded cmin=-0.1, cmax=0.7 — clipping the top half
             of feature height. Now auto-scaled to the real min/max of Z_crop_vis.
          2. Dead stride variables removed: Z_plot, x_plot_3d, y_plot_3d, and
             plotly_stride were computed in Code 1 but never passed to go.Surface
             (which already used Z_crop_vis, x_plot_mm, y_plot_mm directly).
             Removing them is pure cleanup — no behavioural change.
        Everything else unchanged from Code 1.
        """
        iter_dir = self.output_dir / f"iteration_{iter_num:02d}"
        iter_dir.mkdir(exist_ok=True)

        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)

        dx_mm = dx * 1000.0
        dy_mm = dy * 1000.0

        # Normal map (unchanged)
        h_res, w_res = self.cfg['resolution']['height'], self.cfg['resolution']['width']
        n_map = np.zeros((h_res, w_res, 3), dtype=np.float32)
        n_map[v_idx, u_idx] = normals
        n_vis_uint16 = ((n_map + 1.0) / 2.0 * 65535).astype(np.uint16)
        cv2.imwrite(str(iter_dir / "normal_map.png"), cv2.cvtColor(n_vis_uint16, cv2.COLOR_RGB2BGR))

        # Crop region (unchanged)
        r_idx, c_idx = np.where(mask)
        r_min, r_max = np.min(r_idx), np.max(r_idx)
        c_min, c_max = np.min(c_idx), np.max(c_idx)

        Z_crop_mm  = Z[r_min:r_max+1, c_min:c_max+1] * 1000.0
        x_plot_mm  = np.arange(Z_crop_mm.shape[1]) * dx_mm
        y_plot_mm  = np.arange(Z_crop_mm.shape[0]) * dy_mm
        Z_crop_vis = Z_crop_mm   # no smoothing — raw Z used directly

        # Debug print — dead stride variables removed, full resolution noted
        if DEBUG:
            self._dbp("VIS", f"Plotly surface grid: {Z_crop_vis.shape[1]}×{Z_crop_vis.shape[0]} pts (full resolution)")

        # 2D depth map (unchanged)
        fig, ax1 = plt.subplots(figsize=(8, 7))
        im = ax1.imshow(
            Z_crop_vis,
            extent=[x_plot_mm[0], x_plot_mm[-1], y_plot_mm[-1], y_plot_mm[0]],
            cmap='viridis'
        )
        ax1.set_title(f"2D Depth Map - Iteration {iter_num}", fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        fig.colorbar(im, ax=ax1, label='Depth (mm)', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(iter_dir / "2D_depth_map.png", dpi=200)
        plt.close(fig)

        # Angular error heatmap (unchanged)
        if FLAT_PLANE_ERROR:
            ae_map = np.full((h_res, w_res), np.nan, dtype=np.float32)
            nz_vals = np.clip(normals[:, 2], -1.0, 1.0)
            ae_vals = np.degrees(np.arccos(nz_vals))
            ae_map[v_idx, u_idx] = ae_vals
            ae_crop = ae_map[r_min:r_max+1, c_min:c_max+1]

            fig, ax = plt.subplots(figsize=(9, 7))
            cmap_ae = matplotlib.colormaps['hot_r'].copy()
            cmap_ae.set_bad(color='#404040')
            im_ae = ax.imshow(
                ae_crop,
                extent=[x_plot_mm[0], x_plot_mm[-1], y_plot_mm[-1], y_plot_mm[0]],
                cmap=cmap_ae,
                vmin=0,
                vmax=max(np.nanpercentile(ae_crop, 99), 1.0)
            )
            ax.set_title(
                f"Angular Error vs Flat Plane [0,0,1] — Iteration {iter_num}\n"
                f"Mean = {np.nanmean(ae_crop):.2f}°   Median = {np.nanmedian(ae_crop):.2f}°   "
                f"P95 = {np.nanpercentile(ae_crop, 95):.2f}°",
                fontsize=12, fontweight='bold'
            )
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            cb = fig.colorbar(im_ae, ax=ax, label='Angular error (°)', fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=9)
            plt.tight_layout()
            plt.savefig(iter_dir / "angular_error_heatmap.png", dpi=200)
            plt.close(fig)

        # 3D interactive surface
        # CHANGE: cmin/cmax now computed from actual Z data (fixed from hardcoded -0.1/0.7)
        z_valid_for_color = Z_crop_vis[~np.isnan(Z_crop_vis)]
        cmin_val = float(np.nanmin(z_valid_for_color)) if len(z_valid_for_color) > 0 else 0.0
        cmax_val = float(np.nanmax(z_valid_for_color)) if len(z_valid_for_color) > 0 else 1.0

        fig3d = go.Figure(data=[go.Surface(
            z=Z_crop_vis,
            x=x_plot_mm,
            y=y_plot_mm,
            colorscale=[[0.0, 'limegreen'], [0.5, 'greenyellow'], [1.0, 'yellow']],
            cmin=cmin_val,
            cmax=cmax_val,
            lighting=dict(ambient=0.2, diffuse=1.0, roughness=0.8, specular=0.1, fresnel=0.0),
            lightposition=dict(x=-1000, y=0, z=1),
            colorbar=dict(title='Z (mm)')
        )])

        range_x_mm = float(np.ptp(x_plot_mm)) if len(x_plot_mm) > 1 else 1.0
        range_y_mm = float(np.ptp(y_plot_mm)) if len(y_plot_mm) > 1 else 1.0
        z_valid_mm = Z_crop_mm[~np.isnan(Z_crop_mm)]
        range_z_mm = float(np.ptp(z_valid_mm)) if len(z_valid_mm) > 0 else 0.1
        if range_z_mm < 0.1: range_z_mm = 0.1

        Z_EXAGGERATION = float(self.cfg.get('global_settings', {}).get('z_exaggeration', 1.0))
        self._dbp("VIS", f"Z exaggeration factor (from JSON 'z_exaggeration'): {Z_EXAGGERATION}×  "
                         f"(True Z range: {range_z_mm:.4f} mm  →  Visual Z range: {range_z_mm * Z_EXAGGERATION:.4f} mm)")

        fig3d.update_layout(
            title=f"3D Surface - Iteration {iter_num}  [Z ×{Z_EXAGGERATION:.1f}]",
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title=f'Z (mm)  [×{Z_EXAGGERATION:.1f} exaggerated]',
                yaxis=dict(autorange="reversed"),
                aspectmode='manual',
                aspectratio=dict(
                    x=1.0,
                    y=range_y_mm / range_x_mm,
                    z=(range_z_mm / range_x_mm) * Z_EXAGGERATION
                ),
                camera=dict(
                    eye=dict(x=1.4, y=-1.4, z=1.2),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig3d.write_html(str(iter_dir / "3D_surface_interactive.html"))

        # CSV (unchanged from Code 1)
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
        """UNCHANGED from Code 1."""
        print("\n" + "="*65)
        print("  AREA-LIGHT PS Pipeline (COMBINED)  [DEBUG={}, FLAT_PLANE_ERROR={}]".format(DEBUG, FLAT_PLANE_ERROR))
        print("="*65)

        csv_path = Path(self.cfg['paths']['world_coordinate_csv'])
        df = pd.read_csv(csv_path)
        previous_z_vals = None

        for i in range(self.max_iterations):
            print(f"\n{'='*65}")
            print(f"  ITERATION {i}")
            print(f"{'='*65}")
            t_iter_start = time.time()

            # ---- Coordinate initialisation ----
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
                    print(f"  [DBG|INIT       ] sensor_width={sensor_width_mm} mm  focal_length={focal_length_mm} mm  obj_dist={object_dist_m} m")
                    print(f"  [DBG|INIT       ] Physical pixel size: dx=dy={dx_meters*1e3:.4f} mm")
                    print(f"  [DBG|INIT       ] Principal point: cam_u={cam_u}  cam_v={cam_v}")
                    x_range_mm = (df['x_world'].max() - df['x_world'].min()) * 1e3
                    y_range_mm = (df['y_world'].max() - df['y_world'].min()) * 1e3
                    print(f"  [DBG|INIT       ] Scene footprint: X span={x_range_mm:.3f} mm  Y span={y_range_mm:.3f} mm")
                    print(f"  [DBG|INIT       ] z_world initialised to: 0.0 m (flat plane assumption for iter 0)")

            # ---- Normals ----
            print(f"\n  > Estimating Normals (AREA LIGHT)...")
            t0 = time.time()
            normals = self.estimate_normals(df)
            t_normals = time.time() - t0
            self._dbp("TIMING", f"estimate_normals: {t_normals:.3f} s")

            # ---- Normal Repair ----
            print(f"\n  > Repairing Normals...")
            u_idx_r = df['pixel_u'].values.astype(int)
            v_idx_r = df['pixel_v'].values.astype(int)
            M_img_r = int(np.max(v_idx_r)) + 1
            N_img_r = int(np.max(u_idx_r)) + 1
            normals = self.repair_normals(normals, v_idx_r, u_idx_r, M_img_r, N_img_r)

            # ---- Reconstruction ----
            print(f"\n  > Reconstructing Surface...")
            t0 = time.time()
            Z_map, dx, dy, mask = self.reconstruct_surface(df, normals)
            t_recon = time.time() - t0
            self._dbp("TIMING", f"reconstruct_surface: {t_recon:.3f} s")

            # ---- Update df ----
            z_vals_raw = Z_map[df['pixel_v'].values.astype(int), df['pixel_u'].values.astype(int)]
            nan_count  = int(np.sum(np.isnan(z_vals_raw)))
            if nan_count > 0:
                print(f"  [WARN|RUN       ] {nan_count:,} NaN Z values replaced with 0 to prevent cascade failure.")
                z_vals_raw = np.nan_to_num(z_vals_raw, nan=0.0)
            df['z_world'] = z_vals_raw
            if DEBUG:
                zw = df['z_world'].values * 1e3
                self._dbp("RUN", f"Updated df z_world (mm): min={zw.min():.4f}  max={zw.max():.4f}  mean={zw.mean():.4f}  std={zw.std():.4f}")

            # Snapshot z_world AFTER nan_to_num for convergence tracking
            z_world_clean = df['z_world'].values.copy()

            # ---- Save ----
            print(f"\n  > Saving Visualizations...")
            t0 = time.time()
            current_csv, current_z_vals = self.save_visualizations(i, normals, df, Z_map, dx, dy, mask)
            self._dbp("TIMING", f"save_visualizations: {time.time()-t0:.3f} s")

            # ---- Convergence ----
            t_total = time.time() - t_iter_start
            if previous_z_vals is not None:
                mad = np.mean(np.abs(z_world_clean - previous_z_vals))
                print(f"\n  [CONVERGENCE] Iteration {i}: MAD = {mad:.6e} m  (threshold = {self.convergence_threshold:.0e} m)")
                if np.isnan(mad):
                    print(f"  [CONVERGENCE] ✗ MAD is NaN — Z solve failed. Restoring previous valid z_world.")
                    df['z_world'] = previous_z_vals
                elif mad < self.convergence_threshold:
                    print(f"  [CONVERGENCE] ✓ Converged after {i} iterations!")
                    break
            else:
                print(f"\n  [CONVERGENCE] Iteration 0 — baseline established.")

            print(f"  [TIMING     ] Total iteration wall time: {t_total:.2f} s")
            previous_z_vals = z_world_clean

        print("\n" + "="*65)
        print(f"  Pipeline finished. Outputs: {self.output_dir}")
        print("="*65 + "\n")

        # Return the HTML path from the last iteration for the server
        last_iter = self.max_iterations - 1
        for it in range(self.max_iterations - 1, -1, -1):
            html_candidate = self.output_dir / f"iteration_{it:02d}" / "3D_surface_interactive.html"
            if html_candidate.exists():
                return str(html_candidate)
        return None


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\chand\OneDrive\Desktop\all data\codes\6th_march_code\upd_basis_copy.json"
    print(f"Config path: {cfg_path}")
    if os.path.exists(cfg_path):
        AutoIterativePipeline(cfg_path).run()
    else:
        print(f"File not found: {cfg_path}")
