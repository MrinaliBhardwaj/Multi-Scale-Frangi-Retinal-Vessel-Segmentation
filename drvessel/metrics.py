"""Pixel-level segmentation metrics, computed inside the field of view only."""
import numpy as np


def confusion(pred, gt, fov):
    p, g = pred[fov], gt[fov]
    tp = np.sum(p & g)
    tn = np.sum(~p & ~g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)
    return tp, tn, fp, fn


def segmentation_metrics(pred, gt, fov):
    tp, tn, fp, fn = (float(x) for x in confusion(pred, gt, fov))
    return {
        "sensitivity": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-12),
        "jaccard": tp / (tp + fp + fn + 1e-12),
    }
