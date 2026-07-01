import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskRVO(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
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

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # PATHOLOGY

        self.pathology_head = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(),

            # 5th upsample
            nn.ConvTranspose2d(32, 3, 2, 2)
        )

        # ANATOMY

        self.anatomy_head = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(),

            # 5th upsample
            nn.ConvTranspose2d(32, 3, 2, 2)
        )

        # HF

        self.hf_head = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(),

            # 5th upsample
            nn.ConvTranspose2d(32, 1, 2, 2),

            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.burden_head = nn.Sequential(

            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        feat = self.encoder(x)

        pathology = self.pathology_head(feat)

        anatomy = self.anatomy_head(feat)

        hf = self.hf_head(feat)

        burden = self.pool(feat)

        burden = burden.flatten(1)

        burden = self.burden_head(burden)

        return (
            pathology,
            anatomy,
            hf,
            burden
        )