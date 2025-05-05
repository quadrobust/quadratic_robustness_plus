#!/usr/bin/env python3
"""
ImageNet Validation Set Organizer

This script restructures the ImageNet validation dataset by:

1. Reading the ground-truth labels for the 50,000 validation images.
2. Sorting image files alphabetically.
3. Moving each image into a folder named by its WordNet ID (WNID).
4. Creating a unified "val" directory containing all WNID subfolders.
5. Loading the devkit metadata to map numeric synset IDs (n1, n2, ..., n1000) to correct WNIDs.
6. Renaming each n<id> directory to its proper synset name.

All user-facing messages and log output are in English.

Dependencies:
    - Python 3.6+
    - numpy
    - scipy

Usage:
    Ensure the following structure under your working directory:
        data/imagenet/
            ILSVRC2012_devkit_t12/
                data/meta.mat
                data/ILSVRC2012_validation_ground_truth.txt
            *.JPEG (50k validation images)

    Then run:
        python organize_imagenet_val.py

"""
import os
import sys
import glob
import shutil
import pathlib
import numpy as np
import scipy.io

# Constants for dataset paths
VAL_DIR = pathlib.Path("data/imagenet")
GT_FILE = VAL_DIR / "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
DEVKIT_DIR = VAL_DIR / "ILSVRC2012_devkit_t12"
META_MAT = DEVKIT_DIR / "data/meta.mat"

# Step 1: Load the list of 50,000 WNIDs (with prefix 'n')
wnids = [f"n{line.strip()}" for line in GT_FILE.open()]
assert len(wnids) == 50000, "Ground-truth file does not contain 50,000 entries"

# Step 2: Sort all validation images alphabetically
image_files = sorted(VAL_DIR.glob("*.JPEG"))
assert len(image_files) == 50000, f"Found {len(image_files)} image files; expected 50,000"

print("→ Moving files… (this may take < 20s on NVMe)")
for src, wnid in zip(image_files, wnids):
    target_dir = VAL_DIR / wnid
    target_dir.mkdir(exist_ok=True)
    shutil.move(str(src), target_dir / src.name)
print("✔️ Done: files moved into 1000 WNID subfolders.")

# Step 3: Consolidate all WNID directories under 'val/'
os.chdir(VAL_DIR)
os.makedirs('val', exist_ok=True)
for fname in glob.glob('n*'):
    shutil.move(fname, 'val')
# Return to project root
os.chdir(os.path.join('..', '..'))

# Step 4: Load meta.mat to map numeric synset IDs to actual WNIDs
print("→ Loading meta.mat…")
mat = scipy.io.loadmat(META_MAT, squeeze_me=True)
synsets = mat.get('synsets')
# Flatten to a Python list
entries = synsets.flatten().tolist() if isinstance(synsets, np.ndarray) else list(synsets)

# Build a mapping: numeric ID → WNID
id_to_wnid = {}
for record in entries:
    try:
        # Each record is a tuple/list: (ID, WNID, ...)
        idx, wnid = record[0], record[1]
        idx = int(idx)
    except Exception:
        continue
    if 1 <= idx <= 1000:
        id_to_wnid[idx] = str(wnid)

# Verify mapping completeness
if len(id_to_wnid) != 1000:
    print(f"❌ Error: expected 1000 synset mappings, found {len(id_to_wnid)}")
    sys.exit(1)

# Step 5: Rename directories n1…n1000 to actual synset names
val_dir = pathlib.Path("data/imagenet/val")
print("→ Renaming directories n1…n1000 to correct synset names…")
for idx in range(1, 1001):
    src_dir = val_dir / f"n{idx}"
    if not src_dir.exists():
        print(f"! Warning: {src_dir} does not exist – skipping.")
        continue
    dst_dir = val_dir / id_to_wnid[idx]
    print(f"{src_dir.name} → {dst_dir.name}")
    dst_dir.mkdir(exist_ok=True)
    for img in src_dir.iterdir():
        shutil.move(str(img), dst_dir / img.name)
    src_dir.rmdir()
print("✔️ Done – 'val' directory now contains 1000 correctly named synset folders.")
