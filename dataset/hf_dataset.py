import os
import cv2
import torch

from torch.utils.data import Dataset

IMG_W = 384
IMG_H = 256


class HFDataset(Dataset):

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

        parent_dir = os.path.dirname(
            root_dir
        )

        self.heatmap_dir = os.path.join(
            parent_dir,
            "processed",
            "hf_heatmaps"
        )

        split_file = os.path.join(
            root_dir,
            "Image_Seg",
            f"{split}.txt"
        )

        with open(split_file) as f:

            self.images = [
                line.strip()
                for line in f
            ]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image = cv2.imread(
            os.path.join(
                self.image_dir,
                image_name
            ),
            0
        )

        heatmap = cv2.imread(
            os.path.join(
                self.heatmap_dir,
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

        heatmap = cv2.resize(
            heatmap,
            (IMG_W, IMG_H)
        )

        image = (
            image.astype("float32")
            / 255.0
        )

        heatmap = (
            heatmap.astype("float32")
            / 255.0
        )

        image = torch.tensor(
            image
        ).unsqueeze(0)

        heatmap = torch.tensor(
            heatmap
        ).unsqueeze(0)

        return image, heatmap