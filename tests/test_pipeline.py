import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drvessel.vessels import frangi  # noqa: E402
from drvessel.features import tortuosity, branch_points, glcm_features, _arc_length  # noqa: E402
from drvessel.metrics import segmentation_metrics  # noqa: E402
from drvessel import fusion  # noqa: E402
from drvessel.config import DEFAULT_CONFIG  # noqa: E402
from drvessel.io_utils import list_drive, read_rgb, read_binary  # noqa: E402
from drvessel.pipeline import run_pipeline  # noqa: E402


def test_frangi_responds_to_dark_line_not_background():
    img = np.full((64, 64), 200.0)
    img[:, 30:33] = 120.0                         # dark vertical vessel, 3 px wide
    v, _ = frangi(img)
    assert v[32, 31] > 0.5 * v.max()
    assert v[32, 5] < 0.01 * v.max()


def test_frangi_ignores_bright_line():
    img = np.full((64, 64), 100.0)
    img[:, 30:33] = 200.0
    v, _ = frangi(img)
    assert v[32, 31] < 1e-6                       # centre of a bright ridge is not a (dark) vessel


def test_frangi_matches_skimage():
    from skimage.filters import frangi as sk_frangi
    rng = np.random.default_rng(0)
    img = np.full((80, 80), 180.0) + rng.normal(0, 2, (80, 80))
    img[20:23, :] -= 60
    img[:, 50:52] -= 50
    ours, _ = frangi(img, [1, 2, 3], 0.5, 15)
    ref = sk_frangi(img, sigmas=[1, 2, 3], alpha=0.5, beta=0.5, gamma=15, black_ridges=True)
    inner = (slice(5, -5), slice(5, -5))
    assert np.corrcoef(ours[inner].ravel(), ref[inner].ravel())[0, 1] > 0.9


def test_straight_segment_has_tortuosity_one():
    skel = np.zeros((40, 40), bool)
    skel[20, 5:35] = True
    t, segs, _ = tortuosity(skel, np.zeros_like(skel), min_len=5)
    assert t == pytest.approx(1.0, abs=0.05)


def test_diagonal_arc_length():
    seg = np.eye(11, dtype=bool)
    assert _arc_length(seg) == pytest.approx(10 * np.sqrt(2))


def test_branch_point_on_t_junction():
    skel = np.zeros((30, 30), bool)
    skel[15, 2:28] = True
    skel[2:15, 15] = True
    n, _, _ = branch_points(skel)
    assert n == 1


def test_glcm_uniform_tissue():
    img = np.full((50, 50), 128.0)
    c, h, e = glcm_features(img, np.ones_like(img, bool))
    assert c == 0 and h == pytest.approx(1) and e == pytest.approx(1)


def test_metrics_perfect():
    gt = np.zeros((10, 10), bool)
    gt[3:6] = True
    m = segmentation_metrics(gt, gt, np.ones_like(gt))
    assert m["dice"] == pytest.approx(1) and m["jaccard"] == pytest.approx(1)


def test_fusion_bounds_and_direction():
    rows = [{f: float(i) for f in fusion.FEATURES} for i in range(10)]
    ref = fusion.fit_reference(rows, 0, 100)
    hi = {f: (9.0 if DEFAULT_CONFIG["directions"][f] > 0 else 0.0) for f in fusion.FEATURES}
    lo = {f: (0.0 if DEFAULT_CONFIG["directions"][f] > 0 else 9.0) for f in fusion.FEATURES}
    assert fusion.score(hi, ref, DEFAULT_CONFIG)[0] == pytest.approx(1.0)
    assert fusion.score(lo, ref, DEFAULT_CONFIG)[0] == pytest.approx(0.0)
    assert fusion.band(0.1) == "Low" and fusion.band(0.45) == "Medium" and fusion.band(0.9) == "High"


@pytest.mark.skipif(not list_drive(), reason="DRIVE not downloaded")
def test_drive_image_reasonable_dice():
    it = next(i for i in list_drive() if i["id"] == "01_test")
    out = run_pipeline(read_rgb(it["image"]))
    m = segmentation_metrics(out["mask"], read_binary(it["manual"]), read_binary(it["mask"]))
    assert m["dice"] > 0.7
