"""Grid search of pipeline parameters on the 20 DRIVE TRAINING images only (mean Dice).

The method has no learned weights; this only fixes a handful of filter constants, and the test
split is never looked at during tuning.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drvessel.config import make_config  # noqa: E402
from drvessel.io_utils import list_drive, read_rgb, read_binary  # noqa: E402
from drvessel.pipeline import run_pipeline  # noqa: E402
from drvessel.metrics import segmentation_metrics  # noqa: E402

GRID = {
    "clahe_clip": [2.0, 3.0],
    "hef_d0": [15.0, 30.0],
    "frangi_beta": [0.5, 1.0],
    "frangi_c": [5.0, 10.0, 15.0],
    "otsu_gamma": [0.5, 0.7, 1.0],
}

_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        _DATA = [(read_rgb(i["image"]), read_binary(i["manual"]), read_binary(i["mask"]))
                 for i in list_drive() if i["split"] == "training"]
    return _DATA


def evaluate(params):
    cfg = make_config(**params)
    dices = []
    for rgb, gt, fov in _load():
        out = run_pipeline(rgb, cfg, fov=fov)
        dices.append(segmentation_metrics(out["mask"], gt, fov)["dice"])
    return params, float(np.mean(dices))


if __name__ == "__main__":
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} combinations on training split")
    with Pool() as pool:
        results = pool.map(evaluate, combos)
    results.sort(key=lambda r: -r[1])
    for p, d in results[:10]:
        print(f"{d:.4f}  {p}")
    os.makedirs("results", exist_ok=True)
    with open("results/tuning.json", "w") as fh:
        json.dump([{"params": p, "mean_dice_train": d} for p, d in results], fh, indent=2)
