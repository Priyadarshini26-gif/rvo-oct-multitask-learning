import torch
import torch.nn as nn

from torchvision.models import (
    resnet34,
    ResNet34_Weights
)

class HFLocalizer(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1
        )

        old_weight = backbone.conv1.weight

        backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        with torch.no_grad():
            backbone.conv1.weight[:] = old_weight.mean(
                dim=1,
                keepdim=True
            )

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                256,
                128,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                16,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                1,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x