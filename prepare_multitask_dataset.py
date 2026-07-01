import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import (
    ANATOMY_MASK_DIR,
    IMAGE_SEG_DIR,
    LABELME_DIR,
    PATHOLOGY_MASK_DIR,
    PROCESSED_DIR,
)

# =====================================================
# PATHS
# =====================================================

SEG_DIR = IMAGE_SEG_DIR
MASK_DIR = os.path.join(SEG_DIR, "masks")

OUTPUT_DIR = PROCESSED_DIR

os.makedirs(PATHOLOGY_MASK_DIR, exist_ok=True)
os.makedirs(ANATOMY_MASK_DIR, exist_ok=True)

# =====================================================
# STORAGE
# =====================================================

hf_records = []
burden_records = []

# =====================================================
# STEP 1
# Process Masks
# =====================================================

mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

print("\nProcessing masks...")

for file in tqdm(mask_files):

    mask_path = os.path.join(MASK_DIR, file)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = mask.shape

    # -----------------------------------------
    # Pathology Mask
    # 0 Background
    # 1 SRF
    # 2 IRF
    # -----------------------------------------

    pathology_mask = np.zeros_like(mask)

    pathology_mask[mask == 1] = 1
    pathology_mask[mask == 2] = 2

    cv2.imwrite(
        os.path.join(PATHOLOGY_MASK_DIR, file),
        pathology_mask
    )

    # -----------------------------------------
    # Anatomy Mask
    # 0 Background
    # 1 ELM
    # 2 EZ
    # -----------------------------------------

    anatomy_mask = np.zeros_like(mask)

    anatomy_mask[mask == 3] = 1
    anatomy_mask[mask == 4] = 2

    cv2.imwrite(
        os.path.join(ANATOMY_MASK_DIR, file),
        anatomy_mask
    )

    # -----------------------------------------
    # Burden Score
    # -----------------------------------------

    irf_pixels = np.sum(mask == 2)

    srf_pixels = np.sum(mask == 1)

    lesion_pixels = irf_pixels + srf_pixels

    burden = lesion_pixels / (h * w)

    burden_records.append({
        "image": file.replace(".png", ".jpg"),
        "irf_pixels": int(irf_pixels),
        "srf_pixels": int(srf_pixels),
        "lesion_pixels": int(lesion_pixels),
        "burden": float(burden)
    })

# =====================================================
# STEP 2
# Extract HF Annotations
# =====================================================

print("\nExtracting HF annotations...")

BOX_SIZE = 10

for eye_folder in tqdm(os.listdir(LABELME_DIR)):

    eye_path = os.path.join(
        LABELME_DIR,
        eye_folder
    )

    if not os.path.isdir(eye_path):
        continue

    for file in os.listdir(eye_path):

        if not file.endswith(".json"):
            continue

        json_path = os.path.join(
            eye_path,
            file
        )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        image_name = file.replace(
            ".json",
            ".jpg"
        )

        for shape in data["shapes"]:

            if shape["label"] != "HF":
                continue

            x, y = shape["points"][0]

            xmin = max(0, x - BOX_SIZE / 2)
            ymin = max(0, y - BOX_SIZE / 2)

            xmax = x + BOX_SIZE / 2
            ymax = y + BOX_SIZE / 2

            hf_records.append({
                "image": image_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "class": 0
            })

# =====================================================
# SAVE CSV FILES
# =====================================================

hf_df = pd.DataFrame(hf_records)

hf_csv = os.path.join(OUTPUT_DIR, "hf_annotations.csv")

hf_df.to_csv(
    hf_csv,
    index=False
)

burden_df = pd.DataFrame(
    burden_records
)

burden_csv = os.path.join(OUTPUT_DIR, "burden_scores.csv")

burden_df.to_csv(
    burden_csv,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n====================")
print("Finished")
print("====================")

print(
    f"Pathology Masks: "
    f"{len(mask_files)}"
)

print(
    f"Anatomy Masks: "
    f"{len(mask_files)}"
)

print(
    f"HF Boxes: "
    f"{len(hf_records)}"
)

print(
    f"Burden Records: "
    f"{len(burden_records)}"
)

print("\nSaved:")
print(hf_csv)
print(burden_csv)