import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.softmax(
            pred,
            dim=1
        )

        target_onehot = F.one_hot(
            target,
            num_classes=3
        ).permute(
            0,3,1,2
        ).float()

        smooth = 1e-6

        intersection = (
            pred *
            target_onehot
        ).sum((2,3))

        union = (
            pred +
            target_onehot
        ).sum((2,3))

        dice = (
            2*intersection + smooth
        ) / (
            union + smooth
        )

        return 1 - dice.mean()
    
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()

dice_loss = DiceLoss()


def segmentation_loss(
    pred,
    target
):

    ce = ce_loss(
        pred,
        target
    )

    dice = dice_loss(
        pred,
        target
    )

    return ce + dice

def dice_score(
    pred,
    target
):

    pred = pred.argmax(1)

    pred = pred.flatten()

    target = target.flatten()

    intersection = (
        pred == target
    ).sum()

    return (
        intersection.float()
        /
        len(target)
    )
