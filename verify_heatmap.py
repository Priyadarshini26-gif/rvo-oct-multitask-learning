import cv2
import matplotlib.pyplot as plt

from paths import HF_HEATMAP_DIR

heatmap = cv2.imread(
    str(HF_HEATMAP_DIR / "5_1.png"),
    0
)

plt.imshow(
    heatmap,
    cmap="hot"
)

plt.colorbar()
plt.show()