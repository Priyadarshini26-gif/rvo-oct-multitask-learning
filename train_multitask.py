import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset.multitask_dataset import (
    MultiTaskDataset
)

from models.multitask_model import (
    MultiTaskRVO
)

from paths import RAW_DATA_DIR

ROOT_DIR = RAW_DATA_DIR

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

train_dataset = MultiTaskDataset(
    ROOT_DIR,
    "train"
)

test_dataset = MultiTaskDataset(
    ROOT_DIR,
    "test"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

model = MultiTaskRVO().to(device)

seg_loss = nn.CrossEntropyLoss()

hf_loss_fn = nn.MSELoss()

burden_loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

best_loss = 999

EPOCHS = 25

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for (

        image,
        pathology,
        anatomy,
        hf,
        burden

    ) in train_loader:

        image = image.to(device)

        pathology = pathology.to(device)

        anatomy = anatomy.to(device)

        hf = hf.to(device)

        burden = burden.to(device)

        optimizer.zero_grad()

        pred_pathology,\
        pred_anatomy,\
        pred_hf,\
        pred_burden = model(
            image
        )

        pathology_loss = seg_loss(
            pred_pathology,
            pathology
        )

        anatomy_loss = seg_loss(
            pred_anatomy,
            anatomy
        )

        hf_loss = hf_loss_fn(
            pred_hf,
            hf
        )

        burden_loss = burden_loss_fn(
            pred_burden,
            burden
        )

        # consistency loss

        irf_mask = (
            pathology == 2
        ).float().unsqueeze(1)

        consistency = hf_loss_fn(
            pred_hf,
            irf_mask
        )

        loss = (

            pathology_loss

            + anatomy_loss

            + hf_loss

            + burden_loss

            + 0.1 * consistency
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )

    print(

        f"Epoch {epoch+1}/{EPOCHS}"

        f" | Loss={train_loss:.4f}"
    )

    if train_loss < best_loss:

        best_loss = train_loss

        torch.save(

            model.state_dict(),

            "best_multitask_model.pth"
        )

        print(
            "Best Model Saved!"
        )