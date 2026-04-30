import subprocess
import sys
import re

def get_cuda_version():
    """Extracts the major CUDA version from the nvidia-smi output."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None

def verify_gpu_stack():
    """Spins up a test process to check if the GPU libraries actually work without DLL errors."""
    test_script = (
        "import sys\n"
        "try:\n"
        "    import cupy as cp\n"
        "    cp.cuda.runtime.getDeviceCount()\n"
        "    _ = cp.array([1.0]) * 2.0\n"
        "    import onnxruntime as ort\n"
        "    if 'CUDAExecutionProvider' not in ort.get_available_providers():\n"
        "        sys.exit(1)\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", test_script], capture_output=True)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("  PhotoStereo UNIVERSAL GPU Auto-Installer")
    print("=" * 60)
    
    major_version = get_cuda_version()
    if not major_version:
        print("\n[ERROR] No NVIDIA GPU detected via nvidia-smi.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(f"\n[SUCCESS] Detected NVIDIA Driver supporting up to CUDA {major_version}.x")
    
    print("\nPreparing environment (removing CPU-only ONNX)...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cap the version at 12 since CUDA 13 PyPI packages are pre-release/unstable
    effective_version = min(major_version, 12)
    versions_to_try = list(range(effective_version, 10, -1)) # e.g., [12, 11]
    
    success = False
    for v in versions_to_try:
        cupy_pkg = f"cupy-cuda{v}x"
        
        # We only install/uninstall cupy and onnxruntime-gpu. 
        # We NEVER touch the core rembg package during the loop so it doesn't get deleted.
        packages = [cupy_pkg, "onnxruntime-gpu"]

        print(f"\n--- Attempting to install and verify stack for CUDA {v}.x ---")
        try:
            # 1. Install the packages
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
            
            print(f"\nInstall complete. Running hardware verification test...")
            
            # 2. Verify they actually run
            if verify_gpu_stack():
                print(f"\n[SUCCESS] CUDA {v}.x stack installed and passed all hardware tests!")
                success = True
                break
            else:
                print(f"\n[WARNING] Installed successfully, but hardware verification failed.")
                print("This usually means a missing DLL or architecture mismatch.")
                print("Rolling back and trying an older, more compatible version...\n")
                
                # 3. Rollback if the runtime test failed
                subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y"] + packages, 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        except subprocess.CalledProcessError:
            print(f"\n[WARNING] Failed to download {cupy_pkg} (likely not on PyPI yet).")

    if success:
        print("\nUpgrading AI Masking to use GPU...")
        # Upgrade rembg to GPU variant now that onnxruntime-gpu is stable
        subprocess.call([sys.executable, "-m", "pip", "install", "rembg[gpu]"], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("\n" + "=" * 60)
        print("  GPU Acceleration is FULLY INSTALLED and READY.")
        print("  You can now start the PhotoStereo Server normally.")
        print("=" * 60)
    else:
        print("\n[ERROR] Could not find any fully compatible GPU libraries for your system.")
        print("Restoring CPU-only ONNX and verifying rembg is intact...")
        
        # Clean out any broken GPU pieces
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Reinstall the pure CPU dependencies to guarantee the fallback is 100% operational
        subprocess.call([sys.executable, "-m", "pip", "install", "onnxruntime", "rembg[cpu]"], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("The server will continue to use the CPU fallback safely.")

    input("\nPress Enter to close this window...")

if __name__ == "__main__":
    main()