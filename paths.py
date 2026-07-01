from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "RVO-Lesion"
IMAGE_SEG_DIR = RAW_DATA_DIR / "Image_Seg"
LABELME_DIR = RAW_DATA_DIR / "RVO_Lesion_Labelme"
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
HF_HEATMAP_DIR = PROCESSED_DIR / "hf_heatmaps"
PATHOLOGY_MASK_DIR = PROCESSED_DIR / "pathology_masks"
ANATOMY_MASK_DIR = PROCESSED_DIR / "anatomy_masks"