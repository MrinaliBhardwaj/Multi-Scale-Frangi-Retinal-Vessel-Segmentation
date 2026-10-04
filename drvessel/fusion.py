"""Module 7: min-max normalisation, rule-based weighted fusion and risk banding."""
import json
import numpy as np

FEATURES = ["vessel_density", "tortuosity", "branch_points",
            "glcm_contrast", "glcm_homogeneity", "glcm_energy"]


def fit_reference(feature_rows, lo_pct=5, hi_pct=95):
    """Robust min-max range of each feature across a reference dataset (DRIVE, 40 images).

    The 5th/95th percentiles are used instead of the strict min/max so that one badly segmented
    image (e.g. hundreds of spurious branch points) does not compress every other image into a
    narrow band. Values outside the range are clipped to 0 or 1.
    """
    ref = {}
    for f in FEATURES:
        vals = np.array([r[f] for r in feature_rows], dtype=float)
        ref[f] = {"min": float(np.percentile(vals, lo_pct)), "max": float(np.percentile(vals, hi_pct))}
    return ref


def save_reference(ref, path):
    with open(path, "w") as fh:
        json.dump(ref, fh, indent=2)


def load_reference(path):
    with open(path) as fh:
        return json.load(fh)


def normalise(feats, ref, directions):
    """Map each feature to 0..1 where 1 always means 'more risk'."""
    out = {}
    for f in FEATURES:
        lo, hi = ref[f]["min"], ref[f]["max"]
        x = (feats[f] - lo) / (hi - lo + 1e-12)
        x = float(np.clip(x, 0.0, 1.0))
        out[f] = x if directions[f] > 0 else 1.0 - x
    return out


def band(score, bands=(0.3, 0.6)):
    if score < bands[0]:
        return "Low"
    if score <= bands[1]:
        return "Medium"
    return "High"


def score(feats, ref, cfg):
    """Return (risk score, band, per-feature contribution w_i * f_i)."""
    w = cfg["weights"]
    total_w = sum(w.values())
    norm = normalise(feats, ref, cfg["directions"])
    contrib = {f: w[f] * norm[f] / total_w for f in FEATURES}
    s = float(sum(contrib.values()))
    return s, band(s, cfg["bands"]), contrib, norm
