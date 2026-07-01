import os
import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

IMG_WIDTH = 384
IMG_HEIGHT = 256

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

class RVOMultiTaskDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train"
    ):

        self.root = root_dir

        self.image_dir = os.path.join(
            root_dir,
            "Image_Seg",
            "images"
        )

        # Get the parent directory (RVO folder)
        parent_dir = os.path.dirname(root_dir)

        self.pathology_dir = os.path.join(
            parent_dir,
            "processed",
            "pathology_masks"
        )

        self.anatomy_dir = os.path.join(
            parent_dir,
            "processed",
            "anatomy_masks"
        )

        # ---------------------------------
        # TRAIN / TEST SPLIT
        # ---------------------------------

        split_file = os.path.join(
            root_dir,
            "Image_Seg",
            f"{split}.txt"
        )

        with open(split_file, "r") as f:

            self.image_names = [
                line.strip()
                for line in f.readlines()
            ]

        # ---------------------------------
        # BURDEN
        # ---------------------------------

        burden_csv = os.path.join(
            parent_dir,
            "processed",
            "burden_scores.csv"
        )

        burden_df = pd.read_csv(
            burden_csv
        )

        self.burden_dict = {

            row["image"]: row["burden"]

            for _, row in burden_df.iterrows()
        }

        # ---------------------------------
        # HF BOXES
        # ---------------------------------

        hf_csv = os.path.join(
            parent_dir,
            "processed",
            "hf_annotations.csv"
        )

        hf_df = pd.read_csv(
            hf_csv
        )

        self.box_dict = {}

        for image_name, group in hf_df.groupby(
            "image"
        ):

            boxes = group[
                ["xmin",
                 "ymin",
                 "xmax",
                 "ymax"]
            ].values

            self.box_dict[
                image_name
            ] = boxes

        # Original image size

        self.orig_w = 570
        self.orig_h = 380

    def __len__(self):

        return len(
            self.image_names
        )

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        # ---------------------------------
        # IMAGE
        # ---------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # ---------------------------------
        # PATHOLOGY MASK
        # ---------------------------------

        pathology_path = os.path.join(
            self.pathology_dir,
            image_name.replace(
                ".jpg",
                ".png"
            )
        )

        pathology_mask = cv2.imread(
            pathology_path,
            0
        )

        # ---------------------------------
        # ANATOMY MASK
        # ---------------------------------

        anatomy_path = os.path.join(
            self.anatomy_dir,
            image_name.replace(
                ".jpg",
                ".png"
            )
        )

        anatomy_mask = cv2.imread(
            anatomy_path,
            0
        )

        # ---------------------------------
        # RESIZE
        # ---------------------------------

        image = cv2.resize(
            image,
            (IMG_WIDTH, IMG_HEIGHT)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        pathology_mask = cv2.resize(
            pathology_mask,
            (IMG_WIDTH,
             IMG_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        )

        anatomy_mask = cv2.resize(
            anatomy_mask,
            (IMG_WIDTH,
             IMG_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        )

        # ---------------------------------
        # HF BOXES
        # ---------------------------------

        boxes = self.box_dict.get(
            image_name,
            np.zeros((0,4))
        )

        scale_x = (
            IMG_WIDTH /
            self.orig_w
        )

        scale_y = (
            IMG_HEIGHT /
            self.orig_h
        )

        boxes = boxes.copy()

        if len(boxes) > 0:

            boxes[:, [0,2]] *= scale_x
            boxes[:, [1,3]] *= scale_y

        labels = np.zeros(
            len(boxes),
            dtype=np.int64
        )

        # ---------------------------------
        # BURDEN
        # ---------------------------------

        burden = self.burden_dict[
            image_name
        ]

        # ---------------------------------
        # TO TENSOR
        # ---------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        ) / 255.0

        pathology_mask = torch.tensor(
            pathology_mask,
            dtype=torch.long
        )

        anatomy_mask = torch.tensor(
            anatomy_mask,
            dtype=torch.long
        )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        burden = torch.tensor(
            [burden],
            dtype=torch.float32
        )

        return {

            "image":
                image,

            "pathology_mask":
                pathology_mask,

            "anatomy_mask":
                anatomy_mask,

            "boxes":
                boxes,

            "labels":
                labels,

            "burden":
                burden,

            "image_name":
                image_name
        }

#=========================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = RVOMultiTaskDataset(
        root_dir=r"C:\Users\Priya\Downloads\RVO\RVO-Lesion",
        split="train"
    )

    print(len(dataset))

    sample = dataset[0]

    for k,v in sample.items():

        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, v)