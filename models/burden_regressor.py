import torch
import torch.nn as nn
import torchvision.models as models


class BurdenRegressor(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
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
            *list(backbone.children())[:-1]
        )

        self.regressor = nn.Sequential(

            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(
            x.size(0),
            -1
        )

        return self.regressor(x)