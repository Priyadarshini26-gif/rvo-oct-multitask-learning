import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from torch.utils.data import DataLoader

from dataset.multitask_dataset import (
    MultiTaskDataset
)

from models.multitask_model import (
    MultiTaskRVO
)

from paths import RAW_DATA_DIR, RESULTS_DIR

# ==================================================
# CONFIG
# ==================================================

ROOT_DIR = RAW_DATA_DIR

MODEL_PATH = "best_multitask_model.pth"

RESULT_DIR = RESULTS_DIR

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==================================================
# DATA
# ==================================================

test_dataset = MultiTaskDataset(
    ROOT_DIR,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ==================================================
# MODEL
# ==================================================

model = MultiTaskRVO().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

# ==================================================
# METRIC FUNCTIONS
# ==================================================

def dice_score(pred, target, cls):

    pred = (pred == cls)
    target = (target == cls)

    inter = (
        pred & target
    ).sum()

    union = (
        pred.sum()
        + target.sum()
    )

    if union == 0:
        return np.nan

    return (
        2.0 * inter
    ) / (
        union + 1e-6
    )


def iou_score(pred, target, cls):

    pred = (pred == cls)
    target = (target == cls)

    inter = (
        pred & target
    ).sum()

    union = (
        pred | target
    ).sum()

    if union == 0:
        return np.nan

    return (
        inter /
        (union + 1e-6)
    )

# ==================================================
# STORAGE
# ==================================================

pathology_dices = []
pathology_ious = []

anatomy_dices = []
anatomy_ious = []

# Batch-level accumulators for HF
hf_batch_maes = []
hf_batch_mses = []

# Batch-level accumulators for burden
burden_preds = []
burden_targets = []

saved = 0

# ==================================================
# LOOP
# ==================================================

with torch.no_grad():

    for idx, batch in enumerate(
        test_loader
    ):

        image,\
        pathology_gt,\
        anatomy_gt,\
        hf_gt,\
        burden_gt = batch

        image = image.to(device)

        pathology_pred,\
        anatomy_pred,\
        hf_pred,\
        burden_pred = model(
            image
        )

        # --------------------------------
        # PATHOLOGY
        # --------------------------------

        pathology_pred = torch.argmax(
            pathology_pred,
            dim=1
        )

        pathology_pred = pathology_pred\
            .cpu().numpy()[0]

        pathology_gt_np = pathology_gt\
            .numpy()[0]

        img_dices = []
        img_ious = []

        for cls in [1, 2]:

            d = dice_score(
                pathology_pred,
                pathology_gt_np,
                cls
            )

            i = iou_score(
                pathology_pred,
                pathology_gt_np,
                cls
            )

            if not np.isnan(d):
                img_dices.append(d)

            if not np.isnan(i):
                img_ious.append(i)

        if len(img_dices):
            pathology_dices.append(
                np.mean(img_dices)
            )

        if len(img_ious):
            pathology_ious.append(
                np.mean(img_ious)
            )

        # --------------------------------
        # ANATOMY
        # --------------------------------

        anatomy_pred = torch.argmax(
            anatomy_pred,
            dim=1
        )

        anatomy_pred = anatomy_pred\
            .cpu().numpy()[0]

        anatomy_gt_np = anatomy_gt\
            .numpy()[0]

        img_dices = []
        img_ious = []

        for cls in [1, 2]:

            d = dice_score(
                anatomy_pred,
                anatomy_gt_np,
                cls
            )

            i = iou_score(
                anatomy_pred,
                anatomy_gt_np,
                cls
            )

            if not np.isnan(d):
                img_dices.append(d)

            if not np.isnan(i):
                img_ious.append(i)

        if len(img_dices):
            anatomy_dices.append(
                np.mean(img_dices)
            )

        if len(img_ious):
            anatomy_ious.append(
                np.mean(img_ious)
            )

        # --------------------------------
        # HF — batch-level metrics
        # --------------------------------

        hf_pred_np = hf_pred.cpu().numpy().flatten()
        hf_gt_np = hf_gt.numpy().flatten()

        hf_batch_maes.append(
            np.mean(np.abs(hf_pred_np - hf_gt_np))
        )

        hf_batch_mses.append(
            np.mean((hf_pred_np - hf_gt_np) ** 2)
        )

        # --------------------------------
        # BURDEN — scalar per sample, safe to store
        # --------------------------------

        burden_preds.append(
            burden_pred.cpu().item()
        )

        burden_targets.append(
            burden_gt.item()
        )

        # --------------------------------
        # SAVE VISUAL RESULTS
        # --------------------------------

        if saved < 10:

            img = image.cpu().numpy()[0, 0]

            fig, ax = plt.subplots(
                2, 3,
                figsize=(14, 8)
            )

            ax[0, 0].imshow(img, cmap="gray")
            ax[0, 0].set_title("OCT")

            ax[0, 1].imshow(pathology_gt_np, cmap="jet")
            ax[0, 1].set_title("GT Pathology")

            ax[0, 2].imshow(pathology_pred, cmap="jet")
            ax[0, 2].set_title("Pred Pathology")

            ax[1, 0].imshow(anatomy_gt_np, cmap="jet")
            ax[1, 0].set_title("GT Anatomy")

            ax[1, 1].imshow(anatomy_pred, cmap="jet")
            ax[1, 1].set_title("Pred Anatomy")

            ax[1, 2].imshow(
                hf_pred.cpu().numpy()[0, 0],
                cmap="hot"
            )
            ax[1, 2].set_title(
                f"HF\n"
                f"GT Burden={burden_gt.item():.3f}\n"
                f"Pred Burden={burden_pred.item():.3f}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    RESULT_DIR,
                    f"sample_{saved}.png"
                )
            )

            plt.close()

            saved += 1

# ==================================================
# FINAL METRICS
# ==================================================

pathology_dice = np.mean(pathology_dices)
pathology_iou = np.mean(pathology_ious)

anatomy_dice = np.mean(anatomy_dices)
anatomy_iou = np.mean(anatomy_ious)

hf_mae = np.mean(hf_batch_maes)
hf_rmse = np.sqrt(np.mean(hf_batch_mses))

burden_preds = np.array(burden_preds)
burden_targets = np.array(burden_targets)

burden_mae = np.mean(np.abs(burden_preds - burden_targets))
burden_rmse = np.sqrt(np.mean((burden_preds - burden_targets) ** 2))
burden_r2 = r2_score(burden_targets, burden_preds)

# ==================================================
# PRINT
# ==================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Pathology Dice : {pathology_dice:.4f}")
print(f"Pathology IoU  : {pathology_iou:.4f}")
print(f"Anatomy Dice   : {anatomy_dice:.4f}")
print(f"Anatomy IoU    : {anatomy_iou:.4f}")
print(f"HF MAE         : {hf_mae:.6f}")
print(f"HF RMSE        : {hf_rmse:.6f}")
print(f"Burden MAE     : {burden_mae:.6f}")
print(f"Burden RMSE    : {burden_rmse:.6f}")
print(f"Burden R2      : {burden_r2:.4f}")

print("=" * 60)

print(
    f"\nSaved qualitative figures "
    f"to: {RESULT_DIR}"
)