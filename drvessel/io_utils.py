"""Loading DRIVE images, expert masks and FOV masks."""
import os
import glob
import re
import numpy as np
from PIL import Image

# Images reported by the DRIVE authors to show signs of mild early diabetic retinopathy.
DRIVE_EARLY_DR = {"03_test", "08_test", "14_test", "17_test",
                  "25_training", "26_training", "32_training"}


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def read_binary(path):
    return np.array(Image.open(path).convert("L")) > 127


def list_drive(root="data/DRIVE"):
    """Return a list of dicts with keys id, split, image, manual, mask."""
    items = []
    for split in ("training", "test"):
        for img in sorted(glob.glob(os.path.join(root, split, "images", "*.tif"))):
            num = re.match(r"(\d+)_", os.path.basename(img)).group(1)
            items.append({
                "id": f"{num}_{split}",
                "split": split,
                "image": img,
                "manual": os.path.join(root, split, "1st_manual", f"{num}_manual1.gif"),
                "mask": os.path.join(root, split, "mask", f"{num}_{split}_mask.gif"),
                "early_dr": f"{num}_{split}" in DRIVE_EARLY_DR,
            })
    return items
