# dataset_audit.py

import os
import json
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

from paths import IMAGE_SEG_DIR, LABELME_DIR

# ==================================================
# PATHS
# ==================================================

SEG_DIR = IMAGE_SEG_DIR
MASK_DIR = os.path.join(SEG_DIR, "masks")

# ==================================================
# COUNTERS
# ==================================================

label_counter = Counter()
shape_counter = Counter()

images_per_label = defaultdict(set)

hf_per_image = defaultdict(int)

json_count = 0

# ==================================================
# JSON ANALYSIS
# ==================================================

for eye_folder in tqdm(os.listdir(LABELME_DIR)):

    eye_path = os.path.join(LABELME_DIR, eye_folder)

    if not os.path.isdir(eye_path):
        continue

    for file in os.listdir(eye_path):

        if not file.endswith(".json"):
            continue

        json_count += 1

        json_path = os.path.join(eye_path, file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id = file.replace(".json", "")

        for shape in data["shapes"]:

            label = shape["label"]
            shape_type = shape["shape_type"]

            label_counter[label] += 1
            shape_counter[shape_type] += 1

            images_per_label[label].add(image_id)

            if label == "HF":
                hf_per_image[image_id] += 1

print("\n========================")
print("JSON ANALYSIS")
print("========================")

print("JSON Files:", json_count)

for label in sorted(label_counter.keys()):

    print(
        f"{label:10s}"
        f" annotations={label_counter[label]:6d}"
        f" images={len(images_per_label[label]):6d}"
    )

print("\nShape Types")

for shape_type in shape_counter:
    print(shape_type, shape_counter[shape_type])

# ==================================================
# HF STATISTICS
# ==================================================

hf_counts = list(hf_per_image.values())

print("\n========================")
print("HF STATISTICS")
print("========================")

print("Images with HF:", len(hf_counts))

if len(hf_counts) > 0:

    print("Total HF:", sum(hf_counts))
    print("Mean HF/Image:", np.mean(hf_counts))
    print("Max HF/Image:", np.max(hf_counts))

# ==================================================
# MASK ANALYSIS
# ==================================================

pixel_count = defaultdict(int)

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
]

for file in tqdm(mask_files):

    mask_path = os.path.join(MASK_DIR, file)

    mask = cv2.imread(mask_path, 0)

    unique_vals, counts = np.unique(mask, return_counts=True)

    for val, cnt in zip(unique_vals, counts):

        pixel_count[int(val)] += int(cnt)

print("\n========================")
print("MASK PIXEL DISTRIBUTION")
print("========================")

for k in sorted(pixel_count.keys()):

    print(
        f"Class {k}: {pixel_count[k]:,}"
    )

# ==================================================
# TRAIN TEST CHECK
# ==================================================

train_file = os.path.join(SEG_DIR, "train.txt")
test_file = os.path.join(SEG_DIR, "test.txt")

if os.path.exists(train_file):

    train_ids = open(train_file).read().splitlines()
    print("\nTrain Images:", len(train_ids))

if os.path.exists(test_file):

    test_ids = open(test_file).read().splitlines()
    print("Test Images:", len(test_ids))