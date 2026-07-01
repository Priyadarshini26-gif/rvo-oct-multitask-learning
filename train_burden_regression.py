import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from dataset.burden_dataset import BurdenDataset
from models.burden_regressor import BurdenRegressor

from paths import RAW_DATA_DIR

ROOT_DIR = RAW_DATA_DIR

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ----------------------------------
# DATASET
# ----------------------------------

train_dataset = BurdenDataset(
    ROOT_DIR,
    "train"
)

test_dataset = BurdenDataset(
    ROOT_DIR,
    "test"
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

# ----------------------------------
# MODEL
# ----------------------------------

model = BurdenRegressor().to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

best_rmse = 999

EPOCHS = 25

# ----------------------------------
# TRAINING
# ----------------------------------

for epoch in range(EPOCHS):

    # ==========================
    # TRAIN
    # ==========================

    model.train()

    train_loss = 0

    for images, burdens in train_loader:

        images = images.to(device)
        burdens = burdens.to(device)

        optimizer.zero_grad()

        preds = model(images)

        loss = criterion(
            preds,
            burdens
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    val_loss = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for images, burdens in test_loader:

            images = images.to(device)
            burdens = burdens.to(device)

            preds = model(images)

            loss = criterion(
                preds,
                burdens
            )

            val_loss += loss.item()

            all_preds.extend(
                preds.cpu().numpy().flatten()
            )

            all_targets.extend(
                burdens.cpu().numpy().flatten()
            )

    val_loss /= len(test_loader)

    mae = mean_absolute_error(
        all_targets,
        all_preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            all_targets,
            all_preds
        )
    )

    r2 = r2_score(
        all_targets,
        all_preds
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss={train_loss:.6f} | "
        f"Val Loss={val_loss:.6f} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f}"
    )

    # ==========================
    # SAVE BEST MODEL
    # ==========================

    if rmse < best_rmse:

        best_rmse = rmse

        torch.save(
            model.state_dict(),
            "best_burden_model.pth"
        )

        print(
            f"Best Model Saved! RMSE={best_rmse:.4f}"
        )