"""
Utility script to automatically download the OpenCV Caffe SSD face detector model files:
1. deploy.prototxt
2. res10_300x300_ssd_iter_140000.caffemodel

Uses Python's standard library (urllib.request) so no external dependencies are needed.
"""

import os
import sys
import urllib.request
import hashlib

MODELS = {
    "deploy.prototxt": {
        "urls": [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt"
        ],
        "min_size": 2000,  # ~28 KB
    },
    "res10_300x300_ssd_iter_140000.caffemodel": {
        "urls": [
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000.caffemodel"
        ],
        "min_size": 10000000,  # ~10.6 MB
    }
}


def download_file_with_progress(url, dest_path):
    print(f"Connecting to: {url}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {percent:5.1f}% [{mb_downloaded:5.1f}MB / {mb_total:5.1f}MB]")
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {mb_downloaded:5.1f}MB")
        sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print()


def ensure_models_downloaded(target_dir="models/face_detector"):
    os.makedirs(target_dir, exist_ok=True)
    all_success = True

    for filename, info in MODELS.items():
        dest = os.path.join(target_dir, filename)
        
        # Check if file exists and has sufficient size
        if os.path.exists(dest) and os.path.getsize(dest) >= info["min_size"]:
            print(f"[OK] {filename} already exists ({os.path.getsize(dest):,} bytes).")
            continue

        print(f"\n[DOWNLOAD] Fetching {filename} into {target_dir}...")
        downloaded = False
        
        for url in info["urls"]:
            try:
                download_file_with_progress(url, dest)
                size = os.path.getsize(dest)
                if size >= info["min_size"]:
                    print(f"[SUCCESS] {filename} downloaded successfully ({size:,} bytes).")
                    downloaded = True
                    break
                else:
                    print(f"[WARN] File downloaded from {url} is smaller than expected ({size} bytes). Trying fallback...")
                    if os.path.exists(dest):
                        os.remove(dest)
            except Exception as e:
                print(f"[ERROR] Failed downloading from {url}: {e}")
                if os.path.exists(dest):
                    os.remove(dest)

        if not downloaded:
            print(f"[FAILED] Could not download {filename} from any mirror.")
            all_success = False

    return all_success


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(script_dir, "models", "face_detector")
    print(f"Target directory: {target}")
    success = ensure_models_downloaded(target)
    if success:
        print("\nAll face detector assets are verified and ready.")
        sys.exit(0)
    else:
        print("\nModel asset download failed. Check network connectivity or manual download instructions.")
        sys.exit(1)
