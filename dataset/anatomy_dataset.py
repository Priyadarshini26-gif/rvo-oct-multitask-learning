import torch

from dataset.RVO_MultiTask_Dataset import RVOMultiTaskDataset


class AnatomyDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir,
        split="train"
    ):

        self.dataset = RVOMultiTaskDataset(
            root_dir=root_dir,
            split=split
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        return (
            sample["image"],
            sample["anatomy_mask"]
        )