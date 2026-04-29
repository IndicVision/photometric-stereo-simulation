import os
import shutil
import json
import subprocess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List

app = FastAPI(title="Photometric 3D Server")
WORKSPACE_DIR = Path("temp_workspace")

@app.post("/process-3d")
# Notice we are now accepting a LIST of files!
async def process_3d_scan(files: List[UploadFile] = File(...)):
    print(f"\n--- [API] Received {len(files)} individual files ---")

    # A. Clean up old workspace
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    WORKSPACE_DIR.mkdir(exist_ok=True)
    
    extract_dir = WORKSPACE_DIR / "extracted_files"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # B. Save all files directly to the folder (NO ZIPPING REQUIRED!)
    for incoming_file in files:
        file_path = extract_dir / incoming_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(incoming_file.file, buffer)
        print(f"  > Saved: {incoming_file.filename}")

    mask_json_path = extract_dir / "mask_config.json"
    recon_json_path = extract_dir / "recon_config.json"

    if not mask_json_path.exists() or not recon_json_path.exists():
        return {"error": "Missing JSON configuration files."}

    # C. OVERRIDE MASK CONFIGURATION
    print("  > Configuring AI Masking Pipeline...")
    with open(mask_json_path, 'r') as f:
        mask_cfg = json.load(f)
    
    mask_output_dir = WORKSPACE_DIR / "mask_output"
    mask_output_dir.mkdir(exist_ok=True)
    mask_cfg['output_dir'] = str(mask_output_dir)
    mask_cfg['image_paths'] = [str(extract_dir / fname) for fname in mask_cfg['image_paths']]
    
    with open(mask_json_path, 'w') as f:
        json.dump(mask_cfg, f, indent=4)

    # D. RUN MASKING SCRIPT
    print("  > Executing Hybrid AI Masking...")
    try:
        subprocess.run(["python", "trad_guidedFilter_u2netp_msk.py", str(mask_json_path)], check=True)
    except subprocess.CalledProcessError:
        return {"error": "The masking script failed to run."}

    csv_path = mask_output_dir / "valid_pixels_hybrid_u2netp.csv"
    if not csv_path.exists():
        return {"error": "Masking finished but CSV was not generated."}
    print("  > Masking Complete! CSV generated.")

    # E. OVERRIDE 3D RECONSTRUCTION CONFIGURATION
    print("  > Configuring 3D Reconstruction Pipeline...")
    with open(recon_json_path, 'r') as f:
        recon_cfg = json.load(f)
    
    recon_output_dir = WORKSPACE_DIR / "recon_output"
    recon_output_dir.mkdir(exist_ok=True)
    
    recon_cfg['paths']['output_dir'] = str(recon_output_dir)
    recon_cfg['paths']['image_dir'] = str(extract_dir) 
    recon_cfg['paths']['world_coordinate_csv'] = str(csv_path) 
    
    with open(recon_json_path, 'w') as f:
        json.dump(recon_cfg, f, indent=4)

    # F. RUN 3D RECONSTRUCTION SCRIPT
    print("  > Executing 3D Surface Reconstruction...")
    try:
        subprocess.run(["python", "rdm_obj_nrml_pls_surf.py", str(recon_json_path)], check=True)
    except subprocess.CalledProcessError:
        return {"error": "The 3D reconstruction script failed to run."}

    # G. FIND AND RETURN THE RESULT
    html_files = list(recon_output_dir.rglob("*.html"))
    if not html_files:
        return {"error": "Reconstruction finished, but no 3D HTML file was found."}

    final_html = html_files[-1] 
    print(f"  > Success! Sending {final_html.name} back to user.")
    return FileResponse(path=final_html, media_type='text/html', filename="final_3d_surface.html")

@app.get("/")
def read_root():
    return {"message": "Hello! The Photometric Server is awake and ready."}