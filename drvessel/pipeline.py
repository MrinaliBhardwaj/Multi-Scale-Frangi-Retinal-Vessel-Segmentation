"""End-to-end pipeline: raw fundus RGB -> vessel mask, skeleton, six features."""
import numpy as np
from skimage.filters import threshold_otsu

from . import preprocess as pp
from .vessels import frangi, segment, thin
from .features import extract_features
from .config import DEFAULT_CONFIG


def run_pipeline(rgb, cfg=None, fov=None):
    """Run Modules 1-6. Returns a dict with every intermediate image (for figures and the demo).

    `fov` may be supplied (e.g. the official DRIVE mask); otherwise it is computed from the red channel.
    """
    cfg = cfg or DEFAULT_CONFIG
    out = {"rgb": rgb}
    # Module 1
    green = pp.green_channel(rgb)
    fov_full = fov if fov is not None else pp.fov_mask(rgb)
    fov_eroded = _erode(fov_full, cfg["fov_erode_px"])
    out.update(green=green, fov=fov_full, fov_eroded=fov_eroded)
    x = pp.fill_outside_fov(green, fov_full)
    # Module 2
    x = pp.clahe(x, cfg["clahe_clip"], cfg["clahe_tile"]).astype(np.float64) if cfg["use_clahe"] else x
    out["clahe"] = x
    # Module 3
    if cfg["use_hef"]:
        x = pp.high_emphasis(x, cfg["hef_a"], cfg["hef_b"], cfg["hef_d0"])
    out["hef"] = x
    # Module 4
    if cfg["use_restoration"]:
        x = pp.correct_illumination(x, fov_full, cfg["illum_kernel"])
        out["illum"] = x
        x = pp.denoise(x, cfg["median_kernel"])
    else:
        out["illum"] = x
    out["restored"] = x
    # Module 5
    sigmas = [cfg["frangi_sigmas"][len(cfg["frangi_sigmas"]) // 2]] if cfg["single_scale"] else cfg["frangi_sigmas"]
    V, best_sigma = frangi(x, sigmas, cfg["frangi_beta"], cfg["frangi_c"])
    V[~fov_eroded] = 0
    out.update(vesselness=V, scale_map=best_sigma)
    mask, t = segment(V, fov_eroded, cfg["open_radius"], cfg["close_radius"],
                      cfg["min_area"], cfg["otsu_gamma"])
    skel = thin(mask)
    out.update(mask=mask, threshold=t, skeleton=skel)
    # Module 6 (texture measured on the restored image: illumination flattened and impulse noise
    # removed, so GLCM reflects tissue structure rather than sensor/JPEG noise)
    feats, extras = extract_features(mask, skel, fov_eroded, out["restored"], cfg)
    out.update(features=feats, **extras)
    return out


def _erode(fov, px):
    if px <= 0:
        return fov
    return pp.ndi.binary_erosion(fov, structure=pp.disk(px))


def baseline_otsu(rgb, fov):
    """Baseline from the report: plain Otsu on the raw (inverted) green channel inside the FOV."""
    g = 255.0 - rgb[:, :, 1].astype(np.float64)
    t = threshold_otsu(g[fov])
    return (g > t) & fov
