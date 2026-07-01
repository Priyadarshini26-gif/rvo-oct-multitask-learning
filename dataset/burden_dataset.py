import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset

IMG_WIDTH = 384
IMG_HEIGHT = 256


class BurdenDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train"
    ):

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
                line.strip()
                for line in f
            ]

        burden_csv = os.path.join(
            os.path.dirname(root_dir),
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

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (IMG_WIDTH, IMG_HEIGHT)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        burden = torch.tensor(
            [self.burden_dict[image_name]],
            dtype=torch.float32
        )

        return image, burden