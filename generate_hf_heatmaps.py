import os
import cv2
import json
import numpy as np

from tqdm import tqdm

from paths import HF_HEATMAP_DIR, LABELME_DIR

# ------------------------------------
# PATHS
# ------------------------------------

SAVE_DIR = HF_HEATMAP_DIR

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_W = 570
IMG_H = 380

SIGMA = 5

# ------------------------------------
# GAUSSIAN FUNCTION
# ------------------------------------

def add_gaussian(
    heatmap,
    x,
    y,
    sigma=5
):

    radius = sigma * 3

    x_min = max(
        0,
        int(x - radius)
    )

    x_max = min(
        IMG_W,
        int(x + radius)
    )

    y_min = max(
        0,
        int(y - radius)
    )

    y_max = min(
        IMG_H,
        int(y + radius)
    )

    xs = np.arange(
        x_min,
        x_max
    )

    ys = np.arange(
        y_min,
        y_max
    )

    yy, xx = np.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    gaussian = np.exp(
        -(
            (xx - x) ** 2
            +
            (yy - y) ** 2
        ) / (2 * sigma ** 2)
    )

    heatmap[
        y_min:y_max,
        x_min:x_max
    ] = np.maximum(
        heatmap[
            y_min:y_max,
            x_min:x_max
        ],
        gaussian
    )

# ------------------------------------
# PROCESS JSONS
# ------------------------------------

json_files = []

for eye_folder in os.listdir(
    LABELME_DIR
):

    eye_path = os.path.join(
        LABELME_DIR,
        eye_folder
    )

    if not os.path.isdir(
        eye_path
    ):
        continue

    for file in os.listdir(
        eye_path
    ):

        if file.endswith(".json"):

            json_files.append(
                os.path.join(
                    eye_path,
                    file
                )
            )

print(
    f"JSON files found: {len(json_files)}"
)

# ------------------------------------
# GENERATE HEATMAPS
# ------------------------------------

total_hf = 0

for json_path in tqdm(json_files):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    heatmap = np.zeros(
        (
            IMG_H,
            IMG_W
        ),
        dtype=np.float32
    )

    hf_count = 0

    for shape in data["shapes"]:

        if shape["label"] != "HF":
            continue

        x, y = shape["points"][0]

        add_gaussian(
            heatmap,
            x,
            y,
            sigma=SIGMA
        )

        hf_count += 1

    total_hf += hf_count

    image_name = os.path.basename(
        json_path
    ).replace(
        ".json",
        ".png"
    )

    heatmap = (
        heatmap * 255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        os.path.join(
            SAVE_DIR,
            image_name
        ),
        heatmap
    )

print()

print(
    "HF Heatmaps:",
    len(json_files)
)

print(
    "Total HF:",
    total_hf
)