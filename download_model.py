"""
download_model.py - Fast multi-threaded model downloader with progress tracking.
Allows choosing between lightweight ColSmol-500M (~800MB) and Flagship ColPali-v1.2 (~3.5GB).
"""

import sys
import os

print("=" * 60)
print("     COLPALI MULTIMODAL MODEL FAST DOWNLOADER")
print("=" * 60)
print("1. vidore/colpali-v1.2   [~3.5 GB]  (Flagship: PaliGemma-3B backbone)")
print("2. vidore/colqwen2-v0.1  [~3.8 GB]  (Alternative: Qwen2-VL backbone)")
print("=" * 60)

choice = input("Enter choice [1 or 2] (default 1): ").strip()
if choice == "2":
    model_id = "vidore/colqwen2-v0.1"
else:
    model_id = "vidore/colpali-v1.2"

print(f"\n[+] Preparing to download: {model_id}")
print("[+] Installing / enabling hf-transfer for high-speed multi-threaded download...")

try:
    import huggingface_hub
except ImportError:
    print("Please install requirements first: pip install -r requirements.txt")
    sys.exit(1)

# Enable fast rust-based multi-threaded downloader
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import snapshot_download
    print(f"[+] Downloading {model_id} from Hugging Face Hub with progress bar...\n")
    local_dir = snapshot_download(
        repo_id=model_id,
        resume_download=True
    )
    print("\n" + "=" * 60)
    print(f"SUCCESS: {model_id} is downloaded and cached locally!")
    print(f"Cache location: {local_dir}")
    print("=" * 60)
    print("\nYou can now run your app without waiting:")
    print("   streamlit run app.py")
    print("=" * 60)
except Exception as e:
    print(f"\nStandard download fallback due to: {e}")
    from huggingface_hub import snapshot_download
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    snapshot_download(repo_id=model_id, resume_download=True)
