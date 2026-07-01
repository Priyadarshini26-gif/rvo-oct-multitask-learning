import matplotlib.pyplot as plt
import numpy as np
import torch

from RVO_MultiTask_Dataset import RVOMultiTaskDataset

dataset = RVOMultiTaskDataset(
    root_dir=r"C:\Users\Priya\Downloads\RVO\RVO-Lesion",
    split="train"
)

sample = dataset[0]

image = sample["image"].squeeze().numpy()

pathology = sample["pathology_mask"].numpy()

anatomy = sample["anatomy_mask"].numpy()

boxes = sample["boxes"].numpy()

fig, ax = plt.subplots(1,3, figsize=(18,6))

# OCT

ax[0].imshow(
    image,
    cmap="gray"
)

ax[0].set_title("OCT")

# Pathology

ax[1].imshow(
    pathology,
    cmap="jet"
)

ax[1].set_title("Pathology")

# Anatomy

ax[2].imshow(
    anatomy,
    cmap="jet"
)

for box in boxes:

    x1, y1, x2, y2 = box

    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="red",
        linewidth=2
    )

    ax[0].add_patch(rect)

    # draw center point

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    ax[0].scatter(
        cx,
        cy,
        s=20,
        c="yellow"
    )

    ax[0].add_patch(rect)

for i in range(10):

    sample = dataset[i]

    print(
        sample["image_name"],
        torch.unique(sample["pathology_mask"]).tolist(),
        torch.unique(sample["anatomy_mask"]).tolist(),
        len(sample["boxes"])
    )
    
ax[2].set_title("Anatomy")

for i in range(len(dataset)):

    p = dataset[i]["pathology_mask"]

    classes = torch.unique(p).tolist()

    if 1 in classes:

        print(
            dataset[i]["image_name"],
            classes
        )

        break

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import torch

# from RVO_MultiTask_Dataset import RVOMultiTaskDataset

# # -----------------------------
# # Load Dataset
# # -----------------------------
# dataset = RVOMultiTaskDataset(
#     root_dir=r"C:\Users\Priya\Downloads\RVO\RVO-Lesion",
#     split="train"
# )

# # -----------------------------
# # Helper to visualize one sample
# # -----------------------------
# def visualize(idx):
#     sample = dataset[idx]

#     image = sample["image"].squeeze().numpy()
#     pathology = sample["pathology_mask"].numpy()
#     anatomy = sample["anatomy_mask"].numpy()
#     boxes = sample["boxes"].numpy()

#     fig, ax = plt.subplots(1, 3, figsize=(18, 6))

#     # -------------------------
#     # OCT IMAGE + HF BOXES
#     # -------------------------
#     ax[0].imshow(image, cmap="gray")
#     ax[0].set_title(f"OCT + HF Boxes (idx={idx})")

#     for box in boxes:
#         x1, y1, x2, y2 = box

#         rect = plt.Rectangle(
#             (x1, y1),
#             x2 - x1,
#             y2 - y1,
#             fill=False,
#             edgecolor="red",
#             linewidth=2
#         )
#         ax[0].add_patch(rect)

#     # -------------------------
#     # PATHOLOGY MASK
#     # -------------------------
#     ax[1].imshow(pathology, cmap="jet", vmin=0, vmax=2)
#     ax[1].set_title("Pathology Mask (0 BG, 1 SRF, 2 IRF)")

#     # -------------------------
#     # ANATOMY MASK
#     # -------------------------
#     ax[2].imshow(anatomy, cmap="jet", vmin=0, vmax=2)
#     ax[2].set_title("Anatomy Mask (1 ELM, 2 EZ)")

#     plt.tight_layout()
#     plt.show()


# # -----------------------------
# # Visualize multiple random samples
# # -----------------------------
# def visualize_random(n=10):
#     indices = random.sample(range(len(dataset)), n)

#     for idx in indices:
#         visualize(idx)


# # Run check
# visualize_random(10)