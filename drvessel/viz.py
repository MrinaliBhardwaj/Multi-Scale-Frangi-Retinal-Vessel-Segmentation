"""Figures shared by the batch scripts and the Streamlit demo."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .fusion import FEATURES  # noqa: E402

PRETTY = {
    "vessel_density": "Vessel density",
    "tortuosity": "Tortuosity",
    "branch_points": "Branch points",
    "glcm_contrast": "GLCM contrast",
    "glcm_homogeneity": "GLCM homogeneity",
    "glcm_energy": "GLCM energy",
}
BAND_COLOURS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


def stage_images(out):
    """Ordered (title, image, cmap) list of every pipeline stage."""
    v = out["vesselness"]
    return [
        ("1. Input RGB", out["rgb"], None),
        ("1. Green channel", out["green"], "gray"),
        ("2. CLAHE", out["clahe"], "gray"),
        ("3. DFT high-emphasis", out["hef"], "gray"),
        ("4. Illumination corrected", out["illum"], "gray"),
        ("4. Median 3x3", out["restored"], "gray"),
        ("5. Frangi vesselness", v / (v.max() + 1e-12), "magma"),
        ("5. Otsu + morphology", out["mask"], "gray"),
        ("5. Zhang-Suen skeleton", ndi.binary_dilation(out["skeleton"]), "gray"),
    ]


def stage_montage(out, path=None, title=None):
    stages = stage_images(out)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12.5))
    for ax, (t, img, cmap) in zip(axes.ravel(), stages):
        ax.imshow(img, cmap=cmap)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return _finish(fig, path)


def error_overlay(pred, gt, fov):
    """Green = true positive, red = false positive, blue = false negative (missed vessel)."""
    rgb = np.zeros(pred.shape + (3,))
    rgb[fov] = 0.08
    rgb[pred & gt] = (0.1, 0.85, 0.2)
    rgb[pred & ~gt & fov] = (0.95, 0.15, 0.15)
    rgb[~pred & gt & fov] = (0.2, 0.45, 1.0)
    return rgb


def skeleton_overlay(out, top_k=8):
    """Fundus image with the skeleton (cyan), branch points (yellow) and the most tortuous segments (magenta)."""
    base = out["rgb"].astype(np.float64) / 255.0 * 0.75
    skel = ndi.binary_dilation(out["skeleton"])
    base[skel] = (0.0, 0.9, 0.9)
    segs = sorted(out["segments"], key=lambda s: -s["tortuosity"])[:top_k]
    lab = out["segment_labels"]
    for s in segs:
        m = ndi.binary_dilation(lab == s["label"], iterations=2)
        base[m] = (1.0, 0.1, 0.9)
    return np.clip(base, 0, 1), out["branch_centres"]


def plot_skeleton_overlay(out, ax):
    img, centres = skeleton_overlay(out)
    ax.imshow(img)
    if len(centres):
        ax.scatter(centres[:, 1], centres[:, 0], s=14, c="yellow", edgecolors="black", linewidths=0.4)
    ax.set_title("Skeleton (cyan), branch points (yellow),\nmost tortuous segments (magenta)", fontsize=10)
    ax.axis("off")


def contribution_chart(contrib, score, band, ax=None, path=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [PRETTY[f] for f in FEATURES]
    vals = [contrib[f] for f in FEATURES]
    ax.barh(names, vals, color="#5c6bc0")
    ax.invert_yaxis()
    ax.set_xlabel("Contribution w_i x f_i to risk score")
    ax.set_title(f"Risk score = {score:.3f}  ->  {band} risk", color=BAND_COLOURS[band], fontsize=12)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(max(vals) * 1.25, 0.05))
    if own:
        fig.tight_layout()
        return _finish(fig, path)


def _finish(fig, path):
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig
