import torch

from torch.utils.data import DataLoader

from dataset.anatomy_dataset import (
    AnatomyDataset
)

from models.pathology_unet import (
    build_model
)

from losses import (
    segmentation_loss
)

from paths import RAW_DATA_DIR

ROOT_DIR = RAW_DATA_DIR

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)

train_dataset = AnatomyDataset(
    root_dir=ROOT_DIR,
    split="train"
)

test_dataset = AnatomyDataset(
    root_dir=ROOT_DIR,
    split="test"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

model = build_model()

model = model.to(device)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-4

)

best_dice = 0
def multiclass_dice(pred, target, num_classes=3):

    pred = torch.argmax(pred, dim=1)

    dices = []

    for cls in range(1, num_classes):

        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()

        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            continue

        dice = (
            2 * intersection + 1e-6
        ) / (
            union + 1e-6
        )

        dices.append(
            dice.item()
        )

    if len(dices) == 0:
        return 0.0

    return sum(dices) / len(dices)

scaler = torch.amp.GradScaler('cuda')

EPOCHS = 25

for epoch in range(EPOCHS):

    # ==================
    # TRAIN
    # ==================

    model.train()

    train_loss = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            pred = model(images)

            loss = segmentation_loss(
                pred,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ==================
    # VALIDATION
    # ==================

    model.eval()

    val_loss = 0
    val_dice = 0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)

            loss = segmentation_loss(
                pred,
                masks
            )

            val_loss += loss.item()

            val_dice += multiclass_dice(
                pred,
                masks
            )

    val_loss /= len(test_loader)
    val_dice /= len(test_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # ==================
    # SAVE BEST MODEL
    # ==================

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            "best_anatomy_model.pth"
        )

        print(
            f"Best Model Saved! Dice={best_dice:.4f}"
        )