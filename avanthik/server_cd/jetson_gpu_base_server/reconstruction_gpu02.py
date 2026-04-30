import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import cupy as cp
import cupyx.scipy.sparse as cpsp          # GPU sparse matrix
import cupyx.scipy.sparse.linalg as cpspla  # GPU sparse solvers
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') # Add this line to force background rendering
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix        # CPU — only used to BUILD A (no solve)

# =============================================================================
# BOOLEAN FLAGS
# =============================================================================
DEBUG            = True   
FLAT_PLANE_ERROR = True   
COMPUTE_COND_NUM = False  

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
    def _dbp(self, section, msg):
        if DEBUG: print(f"  [DBG|{section:12s}] {msg}")

    def _flat_plane_error(self, normals_nx3, label=""):
        if not FLAT_PLANE_ERROR: return
        nz = np.clip(normals_nx3[:, 2], -1.0, 1.0)
        ae = np.degrees(np.arccos(nz))
        tag = f" [{label}]" if label else ""
        print(f"  [FPE{tag}] Mean  = {np.mean(ae):7.3f}°   Std = {np.std(ae):7.3f}°")

    def _flat_plane_error_from_Z(self, Z_gpu, mask_gpu, v_idx, u_idx, dx, dy, label=""):
        if not FLAT_PLANE_ERROR: return
        Z = cp.asnumpy(Z_gpu); mask = cp.asnumpy(mask_gpu)
        Z_nan = np.where(mask, Z, np.nan)
        dZdy_grid, dZdx_grid = np.gradient(Z_nan, dy, dx)
        p_r = dZdx_grid[v_idx, u_idx]; q_r = dZdy_grid[v_idx, u_idx]
        valid = ~np.isnan(p_r) & ~np.isnan(q_r)
        pr = p_r[valid]; qr = q_r[valid]
        if len(pr) == 0: return
        recon_n = np.stack([-pr, -qr, np.ones(len(pr))], axis=1)
        recon_n /= np.maximum(np.linalg.norm(recon_n, axis=1, keepdims=True), 1e-8)
        self._flat_plane_error(recon_n, label=f"{label} | back-derived")

    def __init__(self, config_input):
        if isinstance(config_input, dict): self.cfg = config_input
        else:
            with open(config_input, 'r') as f: self.cfg = json.load(f)
        self.output_dir = Path(self.cfg['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, 'integrate_area_light')
        self.max_iterations = self.cfg.get('max_iterations', 15)
        self.convergence_threshold = self.cfg.get('convergence_threshold', 1e-5)

    def load_image(self, img_path, gamma, bit_depth):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(f"Missing: {img_path}")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / (2**bit_depth - 1)
        if gamma != 1.0: img = np.power(img, gamma)
        return cp.array(img)

    def get_light_samples_gpu(self, light_cfg, P_surf_full):
        pos = np.array(light_cfg['pos_m']); dims = np.array(light_cfg['dims_m'])
        norm = np.array(light_cfg['norm_dir']); samples = light_cfg['sampling']
        u = np.linspace(-dims[0]/2, dims[0]/2, samples[0])
        v = np.linspace(-dims[1]/2, dims[1]/2, samples[1])
        uu, vv = np.meshgrid(u, v)
        v_dummy = np.array([0, 1, 0]) if abs(norm[1]) < 0.9 else np.array([1, 0, 0])
        ax_u = np.cross(v_dummy, norm); ax_u /= np.linalg.norm(ax_u)
        ax_v = np.cross(norm, ax_u)
        pts = pos + uu.flatten()[:, None] * ax_u + vv.flatten()[:, None] * ax_v
        pts_gpu = cp.array(pts, dtype=cp.float32); norm_gpu = cp.array(norm, dtype=cp.float32)
        num_pixels = P_surf_full.shape[0]; num_samples = pts_gpu.shape[0]
        G_eff = cp.zeros((num_pixels, 3), dtype=cp.float32)
        chunk, tpb = 50000, 128
        for i in range(0, num_pixels, chunk):
            end = min(i + chunk, num_pixels); cur = end - i
            self.kernel(((cur + tpb - 1) // tpb,), (tpb,),
                (P_surf_full[i:end], pts_gpu, norm_gpu, cp.float32(np.cos(np.deg2rad(light_cfg.get('spread_deg', 180)/2))),
                 cp.int32(cur), cp.int32(num_samples), G_eff[i:end]))
        cp.cuda.Device(0).synchronize()
        return G_eff, pts

    def estimate_normals(self, df):
        u_idx = df['pixel_u'].values.astype(int); v_idx = df['pixel_v'].values.astype(int)
        elev_m = self.cfg.get('camera', {}).get('object_elevation_m', 0.0)
        xyz = df[['x_world', 'y_world', 'z_world']].values.copy().astype(np.float32)
        xyz[:, 2] += elev_m
        P_surf = cp.ascontiguousarray(cp.array(xyz, dtype=cp.float32))
        num_pix, num_lights = len(df), len(self.cfg['lights'])
        G = cp.zeros((num_pix, num_lights, 3), dtype=cp.float32)
        apply_thresh = self.cfg.get('global_settings', {}).get('apply_dark_threshold', True)
        thresh_val = self.cfg.get('global_settings', {}).get('dark_threshold_value', 0.025)

        if not hasattr(self, 'I_w_cache'):
            I_raw = cp.zeros((num_pix, num_lights), dtype=cp.float32)
            for j, l_cfg in enumerate(self.cfg['lights']):
                img_gpu = self.load_image(Path(self.cfg['paths']['image_dir']) / l_cfg['file_name'], l_cfg['gamma'], l_cfg['bit_depth'])
                I_raw[:, j] = img_gpu[v_idx, u_idx]
            W = (I_raw >= thresh_val).astype(cp.float32) if apply_thresh else cp.ones_like(I_raw)
            self.I_w_cache, self.W_cache = I_raw * W, W

        for j, l_cfg in enumerate(self.cfg['lights']):
            G_j, _ = self.get_light_samples_gpu(l_cfg, P_surf)
            G[:, j, :] = G_j
        
        GT_w = (G * self.W_cache[:, :, None]).transpose(0, 2, 1)
        GTG = cp.matmul(GT_w, (G * self.W_cache[:, :, None])) + (cp.eye(3, dtype=cp.float32) * 1.0)
        GTI = cp.matmul(GT_w, self.I_w_cache[:, :, None])
        n_est = cp.linalg.solve(GTG, GTI).squeeze(-1)
        albedo = cp.linalg.norm(n_est, axis=1, keepdims=True)
        normals = n_est / cp.where(albedo == 0, 1, albedo)
        self._flat_plane_error(cp.asnumpy(normals), label="pre-Poisson")
        return cp.asnumpy(normals)

    def repair_normals(self, normals, v_idx, u_idx, M_img, N_img):
        from scipy.ndimage import uniform_filter
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        nx_g, ny_g, nz_g, mask_g = [np.zeros((M_img, N_img)) for _ in range(3)] + [np.zeros((M_img, N_img), dtype=bool)]
        nx_g[v_idx, u_idx], ny_g[v_idx, u_idx], nz_g[v_idx, u_idx], mask_g[v_idx, u_idx] = nx, ny, nz, True
        good = mask_g & (nz_g >= 0.0)
        nx_s = uniform_filter(nx_g * good, size=5) * 25; ny_s = uniform_filter(ny_g * good, size=5) * 25
        nz_s = uniform_filter(nz_g * good, size=5) * 25; w_s = uniform_filter(good.astype(float), size=5) * 25
        sw = np.maximum(w_s, 1e-8)
        nxf, nyf, nzf = nx_s/sw, ny_s/sw, nz_s/sw
        mag = np.maximum(np.sqrt(nxf**2 + nyf**2 + nzf**2), 1e-8)
        repaired = np.stack([nx_g, ny_g, nz_g], axis=2)
        fill = np.stack([nxf/mag, nyf/mag, nzf/mag], axis=2)
        repaired[mask_g & (nz_g < 0)] = fill[mask_g & (nz_g < 0)]
        return repaired[v_idx, u_idx].astype(np.float32)

    def reconstruct_surface(self, df, normals):
        u_idx, v_idx = df['pixel_u'].values.astype(int), df['pixel_v'].values.astype(int)
        if not hasattr(self, '_recon_cache'):
            M_img, N_img = np.max(v_idx) + 1, np.max(u_idx) + 1
            mask = np.zeros((M_img, N_img), dtype=bool); mask[v_idx, u_idx] = True
            X_grid, Y_grid = np.full((M_img, N_img), np.nan), np.full((M_img, N_img), np.nan)
            X_grid[v_idx, u_idx], Y_grid[v_idx, u_idx] = df['x_world'].values, df['y_world'].values
            r_idx, c_idx = np.where(mask); cr, cc = int(np.mean(r_idx)), int(np.mean(c_idx))
            dx, dy = np.abs(X_grid[cr, cc+1]-X_grid[cr, cc]), np.abs(Y_grid[cr+1, cc]-Y_grid[cr, cc])
            node_ids = np.zeros((M_img, N_img), dtype=int); n_un = int(np.sum(mask)); node_ids[mask] = np.arange(n_un)
            mH, mV = mask[:, :-1] & mask[:, 1:], mask[:-1, :] & mask[1:, :]
            rH, cH = np.where(mH); rV, cV = np.where(mV)
            id_sH, id_rH = node_ids[rH, cH], node_ids[rH, cH+1]
            id_sV, id_dV = node_ids[rV, cV], node_ids[rV+1, cV]
            n_eq = len(id_sH) + len(id_sV) + 1
            I = np.concatenate([np.arange(len(id_sH)), np.arange(len(id_sH)), np.arange(len(id_sH), len(id_sH)+len(id_sV)), np.arange(len(id_sH), len(id_sH)+len(id_sV)), [n_eq-1]])
            J = np.concatenate([id_rH, id_sH, id_dV, id_sV, [node_ids[cr, cc]]])
            V = np.concatenate([np.ones(len(id_sH)), -np.ones(len(id_sH)), np.ones(len(id_sV)), -np.ones(len(id_sV)), [1.0]])
            A_g = cpsp.csr_matrix(coo_matrix((V, (I, J)), shape=(n_eq, n_un)).tocsr())
            self._recon_cache = dict(M_img=M_img, N_img=N_img, mask=mask, dx=dx, dy=dy, A_gpu=A_g, AT_gpu=A_g.T, C_gpu=A_g.T.dot(A_g), rH=rH, cH=cH, rV=rV, cV=cV, X_grid=X_grid, Y_grid=Y_grid)

        rc = self._recon_cache
        M, N, mask, dx, dy = rc['M_img'], rc['N_img'], rc['mask'], rc['dx'], rc['dy']
        mask_g = cp.asarray(mask)
        nx_g, ny_g, nz_g = [cp.zeros((M, N)) for _ in range(3)]
        nx_g[v_idx, u_idx], ny_g[v_idx, u_idx], nz_g[v_idx, u_idx] = cp.asarray(normals[:,0]), cp.asarray(normals[:,1]), cp.asarray(normals[:,2])
        nz_s = cp.where(nz_g >= 0, cp.maximum(cp.abs(nz_g), 0.15), -cp.maximum(cp.abs(nz_g), 0.15))
        p_g, q_g = cp.zeros((M, N)), cp.zeros((M, N))
        p_g[mask_g], q_g[mask_g] = -(nx_g[mask_g]/nz_s[mask_g])*dx, (ny_g[mask_g]/nz_s[mask_g])*dy
        b_g = cp.concatenate([p_g[rc['rH'], rc['cH']], q_g[rc['rV'], rc['cV']], cp.zeros(1)])
        z_g, _ = cpspla.cg(rc['C_gpu'], rc['AT_gpu'].dot(b_g), tol=1e-8, maxiter=5000)

        # ── MEMORY-SAFE CPU DETRENDING ──
        mode = self.cfg.get('global_settings', {}).get('detrending_mode', 'none').lower()
        if mode in ['linear', 'quadratic']:
            vx_c, vy_c, vz_c = cp.asnumpy(cp.asarray(rc['X_grid'])[mask_g]), cp.asnumpy(cp.asarray(rc['Y_grid'])[mask_g]), cp.asnumpy(z_g)
            if mode == 'quadratic':
                A_t = np.stack([vx_c**2, vy_c**2, vx_c*vy_c, vx_c, vy_c, np.ones_like(vx_c)], axis=1)
                c_t, _, _, _ = np.linalg.lstsq(A_t, vz_c, rcond=None)
                XG, YG = cp.asarray(rc['X_grid']), cp.asarray(rc['Y_grid'])
                z_g -= (c_t[0]*XG[mask_g]**2 + c_t[1]*YG[mask_g]**2 + c_t[2]*XG[mask_g]*YG[mask_g] + c_t[3]*XG[mask_g] + c_t[4]*YG[mask_g] + c_t[5])
            elif mode == 'linear':
                A_t = np.stack([vx_c, vy_c, np.ones_like(vx_c)], axis=1)
                c_t, _, _, _ = np.linalg.lstsq(A_t, vz_c, rcond=None)
                XG, YG = cp.asarray(rc['X_grid']), cp.asarray(rc['Y_grid'])
                z_g -= (c_t[0]*XG[mask_g] + c_t[1]*YG[mask_g] + c_t[2])
        
        z_g -= cp.nanmin(z_g); Z_out = cp.full((M, N), cp.nan); Z_out[mask_g] = z_g
        self._flat_plane_error_from_Z(Z_out, mask_g, v_idx, u_idx, dx, dy, label="post-Poisson")
        return cp.asnumpy(Z_out), dx, dy, mask

    def save_visualizations(self, iter_num, normals, df, Z, dx, dy, mask):
        iter_dir = self.output_dir / f"iteration_{iter_num:02d}"
        iter_dir.mkdir(exist_ok=True)
        
        u_idx = df['pixel_u'].values.astype(int)
        v_idx = df['pixel_v'].values.astype(int)
        h = self.cfg['resolution']['height']
        w = self.cfg['resolution']['width']
        
        # 1. Normal Map
        n_map = np.zeros((h, w, 3), dtype=np.float32)
        n_map[v_idx, u_idx] = normals
        cv2.imwrite(str(iter_dir / "normal_map.png"), cv2.cvtColor(((n_map+1)/2*65535).astype(np.uint16), cv2.COLOR_RGB2BGR))
        
        # 2. 2D Depth Map
        r, c = np.where(mask)
        Zc = Z[np.min(r):np.max(r)+1, np.min(c):np.max(c)+1] * 1000.0
        plt.figure()
        plt.imshow(Zc, cmap='viridis')
        plt.colorbar(label='Depth (mm)')
        plt.savefig(iter_dir / "2D_depth_map.png")
        plt.close()
        
        # 3. CSV Mapping
        df_out = df.copy()
        df_out['z_world'] = Z[v_idx, u_idx]
        df_out.to_csv(iter_dir / f"mapping_iter{iter_num}.csv", index=False)

        # 4. NEW: 3D Interactive HTML (Plotly)
        x_plot = np.arange(Zc.shape[1]) * dx * 1000.0
        y_plot = np.arange(Zc.shape[0]) * dy * 1000.0
        
        fig = go.Figure(data=[go.Surface(
            z=Zc, 
            x=x_plot, 
            y=y_plot,
            colorscale='Viridis',
            colorbar=dict(title='Z (mm)')
        )])
        
        fig.update_layout(
            title=f"3D Surface - Iteration {iter_num}",
            scene=dict(
                aspectmode='data', 
                yaxis=dict(autorange="reversed")
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Save the HTML file so the server can actually find it!
        fig.write_html(str(iter_dir / "3D_surface_interactive.html"))

        return str(iter_dir / f"mapping_iter{iter_num}.csv"), Z[v_idx, u_idx]

    def run(self):
        df = pd.read_csv(Path(self.cfg['paths']['world_coordinate_csv'])); prev_z = None
        for i in range(self.max_iterations):
            if i == 0:
                cam = self.cfg.get('camera', {}); res_w = self.cfg['resolution']['width']
                dx = (cam.get('sensor_width_mm', 35.9)/res_w) * (cam.get('object_distance_m', 0.5)/cam.get('focal_length_mm', 50.0))
                cu, cv = (res_w/2, self.cfg['resolution']['height']/2) if cam.get('use_auto_center', True) else cam.get('manual_center_pixel', [0,0])
                df['x_world'], df['y_world'], df['z_world'] = (df['pixel_u']-cu)*dx, -(df['pixel_v']-cv)*dx, 0.0
            normals = self.estimate_normals(df)
            normals = self.repair_normals(normals, df['pixel_v'].values.astype(int), df['pixel_u'].values.astype(int), int(df['pixel_v'].max())+1, int(df['pixel_u'].max())+1)
            Z_map, dx, dy, mask = self.reconstruct_surface(df, normals)
            df['z_world'] = np.nan_to_num(Z_map[df['pixel_v'].values.astype(int), df['pixel_u'].values.astype(int)], nan=0.0)
            self.save_visualizations(i, normals, df, Z_map, dx, dy, mask)
            if prev_z is not None:
                mad = np.mean(np.abs(df['z_world'].values - prev_z))
                if mad < self.convergence_threshold: break
            prev_z = df['z_world'].values.copy()
        print("\n" + "="*65)
        print(f"  Pipeline finished. Outputs: {self.output_dir}")
        print("="*65 + "\n")

        # Return the HTML path from the last iteration for the server
        for it in range(self.max_iterations - 1, -1, -1):
            html_candidate = self.output_dir / f"iteration_{it:02d}" / "3D_surface_interactive.html"
            if html_candidate.exists():
                return str(html_candidate)
        
        # Fallback if something went wrong
        return str(self.output_dir)

if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    if os.path.exists(cfg): AutoIterativePipeline(cfg).run()