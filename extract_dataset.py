"""
extract_dataset.py - Helper script to find and unpack the Kaggle Face Mask Detection archive.
Looks for 'archive.zip', 'face-mask-detection.zip', or any .zip in the repository or parent directories,
and extracts the 853 images and 853 XML annotations into:
  - data/images/
  - data/annotations/
"""

import os
import glob
import zipfile
import shutil

def extract_archive():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Search for potential zip files
    search_dirs = [base_dir, os.path.dirname(base_dir), data_dir, os.path.expanduser("~/Downloads")]
    found_zip = None

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for fname in ["archive.zip", "face-mask-detection.zip", "*mask*.zip"]:
            matches = glob.glob(os.path.join(d, fname))
            if matches:
                found_zip = matches[0]
                break
        if found_zip:
            break

    if not found_zip:
        print("[!] No dataset zip archive found.")
        print(f"Please download the dataset from: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
        print(f"Place 'archive.zip' into: {base_dir}/ and rerun this script, or directly copy:")
        print(f"  - 853 .png images into:      {images_dir}/")
        print(f"  - 853 .xml annotations into: {annotations_dir}/")
        return False

    print(f"[FOUND] Extracting dataset archive: {found_zip}...")
    with zipfile.ZipFile(found_zip, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            
            # Route images to data/images/ and xmls to data/annotations/
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                dest = os.path.join(images_dir, filename)
                with zip_ref.open(member) as source, open(dest, "wb") as target:
                    shutil.copyfileobj(source, target)
            elif filename.lower().endswith('.xml'):
                dest = os.path.join(annotations_dir, filename)
                with zip_ref.open(member) as source, open(dest, "wb") as target:
                    shutil.copyfileobj(source, target)

    img_count = len(glob.glob(os.path.join(images_dir, "*.*")))
    xml_count = len(glob.glob(os.path.join(annotations_dir, "*.xml")))
    print(f"[SUCCESS] Extracted {img_count} images and {xml_count} annotations.")
    return True

if __name__ == "__main__":
    extract_archive()
