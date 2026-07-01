import os
import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

IMG_W = 384
IMG_H = 256


class MultiTaskDataset(Dataset):

    def __init__(self, root_dir, split="train"):

        self.root_dir = root_dir

        self.image_dir = os.path.join(
            root_dir,
            "Image_Seg",
            "images"
        )

        split_file = os.path.join(
            root_dir,
            "Image_Seg",
            f"{split}.txt"
        )

        with open(split_file) as f:

            self.image_names = [
                x.strip()
                for x in f.readlines()
            ]

        parent = os.path.dirname(root_dir)

        self.pathology_dir = os.path.join(
            parent,
            "processed",
            "pathology_masks"
        )

        self.anatomy_dir = os.path.join(
            parent,
            "processed",
            "anatomy_masks"
        )

        self.hf_dir = os.path.join(
            parent,
            "processed",
            "hf_heatmaps"
        )

        burden_csv = os.path.join(
            parent,
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

    def __len__(self):

        return len(
            self.image_names
        )

    def __getitem__(self, idx):

        image_name = self.image_names[idx]

        image = cv2.imread(

            os.path.join(
                self.image_dir,
                image_name
            ),

            0
        )

        pathology = cv2.imread(

            os.path.join(
                self.pathology_dir,
                image_name.replace(
                    ".jpg",
                    ".png"
                )
            ),

            0
        )

        anatomy = cv2.imread(

            os.path.join(
                self.anatomy_dir,
                image_name.replace(
                    ".jpg",
                    ".png"
                )
            ),

            0
        )

        hf = cv2.imread(

            os.path.join(
                self.hf_dir,
                image_name.replace(
                    ".jpg",
                    ".png"
                )
            ),

            0
        )

        image = cv2.resize(
            image,
            (IMG_W, IMG_H)
        )

        pathology = cv2.resize(
            pathology,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_NEAREST
        )

        anatomy = cv2.resize(
            anatomy,
            (IMG_W, IMG_H),
            interpolation=cv2.INTER_NEAREST
        )

        hf = cv2.resize(
            hf,
            (IMG_W, IMG_H)
        )

        hf = hf.astype(
            np.float32
        ) / 255.0

        burden = self.burden_dict[
            image_name
        ]

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        pathology = torch.tensor(
            pathology,
            dtype=torch.long
        )

        anatomy = torch.tensor(
            anatomy,
            dtype=torch.long
        )

        hf = torch.tensor(
            hf,
            dtype=torch.float32
        ).unsqueeze(0)

        burden = torch.tensor(
            [burden],
            dtype=torch.float32
        )

        return (
            image,
            pathology,
            anatomy,
            hf,
            burden
        )