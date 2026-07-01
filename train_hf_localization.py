import torch
import numpy as np

from torch.utils.data import DataLoader

from dataset.hf_dataset import HFDataset
from models.hf_localizer import HFLocalizer

from paths import RAW_DATA_DIR

# =====================================================
# CONFIG
# =====================================================

ROOT_DIR = RAW_DATA_DIR

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# DATASET
# =====================================================

train_dataset = HFDataset(
    ROOT_DIR,
    split="train"
)

test_dataset = HFDataset(
    ROOT_DIR,
    split="test"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# =====================================================
# MODEL
# =====================================================

model = HFLocalizer().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

best_rmse = 999

EPOCHS = 25

# =====================================================
# TRAINING
# =====================================================

for epoch in range(EPOCHS):

    # -----------------------------------
    # TRAIN
    # -----------------------------------

    model.train()

    train_loss = 0

    for images, heatmaps in train_loader:

        images = images.to(device)
        heatmaps = heatmaps.to(device)

        optimizer.zero_grad()

        preds = model(images)

        loss = criterion(
            preds,
            heatmaps
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------------------
    # VALIDATION
    # -----------------------------------

    model.eval()

    val_loss = 0

    batch_maes = []
    batch_mses = []

    with torch.no_grad():

        for images, heatmaps in test_loader:

            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)

            loss = criterion(
                preds,
                heatmaps
            )

            val_loss += loss.item()

            batch_preds = preds.cpu().numpy().flatten()
            batch_targets = heatmaps.cpu().numpy().flatten()

            batch_maes.append(
                np.mean(
                    np.abs(batch_preds - batch_targets)
                )
            )

            batch_mses.append(
                np.mean(
                    (batch_preds - batch_targets) ** 2
                )
            )

    val_loss /= len(test_loader)

    mae = np.mean(batch_maes)

    rmse = np.sqrt(np.mean(batch_mses))

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss={train_loss:.6f} | "
        f"Val Loss={val_loss:.6f} | "
        f"MAE={mae:.6f} | "
        f"RMSE={rmse:.6f}"
    )

    # -----------------------------------
    # SAVE BEST
    # -----------------------------------

    if rmse < best_rmse:

        best_rmse = rmse

        torch.save(
            model.state_dict(),
            "best_hf_localizer.pth"
        )

        print(
            f"Best Model Saved! "
            f"RMSE={best_rmse:.6f}"
        )