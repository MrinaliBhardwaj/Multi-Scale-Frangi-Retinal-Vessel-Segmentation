"""Module 6: structural (skeleton) and texture (GLCM) features."""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import graycomatrix

_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def neighbour_count(skel):
    return ndi.convolve(skel.astype(np.uint8), _K8, mode="constant") * skel


def vessel_density(mask, fov):
    return mask[fov].sum() / fov.sum()


def branch_points(skel):
    """Skeleton pixels with >= 3 neighbours; adjacent junction pixels are merged into one junction."""
    junction = neighbour_count(skel) >= 3
    lab, n = ndi.label(junction, structure=np.ones((3, 3)))
    centres = ndi.center_of_mass(junction, lab, range(1, n + 1)) if n else []
    return n, junction, np.array(centres).reshape(-1, 2)


def _arc_length(seg):
    """Path length of an 8-connected thin curve: 1 per straight step, sqrt(2) per diagonal step."""
    s = seg.astype(np.int32)
    straight = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum()
    d1 = s[:-1, :-1] & s[1:, 1:]
    d2 = s[:-1, 1:] & s[1:, :-1]
    # a diagonal step is only taken if it is not already covered by an L-shaped straight pair
    corner1 = d1 & ~(s[:-1, 1:] | s[1:, :-1])
    corner2 = d2 & ~(s[:-1, :-1] | s[1:, 1:])
    return straight + np.sqrt(2) * (corner1.sum() + corner2.sum())


def tortuosity(skel, junction, min_len=10):
    """Arc/chord ratio per segment (branch points removed); returns length-weighted mean and per-segment list."""
    segs = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    lab, n = ndi.label(segs, structure=np.ones((3, 3)))
    nc = neighbour_count(segs)
    results = []
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        seg = lab[sl] == i
        arc = _arc_length(seg)
        if arc < min_len:
            continue
        ends = np.argwhere(seg & (nc[sl] == 1))
        if len(ends) < 2:                 # closed loop, no meaningful chord
            continue
        d = np.sqrt(((ends[:, None, :] - ends[None, :, :]) ** 2).sum(-1))
        chord = d.max()
        if chord < 1:
            continue
        results.append({"label": i, "arc": arc, "chord": chord, "tortuosity": max(arc / chord, 1.0),
                        "slice": sl})
    if not results:
        return 1.0, results, lab
    arcs = np.array([r["arc"] for r in results])
    tort = np.array([r["tortuosity"] for r in results])
    return float((arcs * tort).sum() / arcs.sum()), results, lab


def glcm_features(gray, tissue_mask, levels=32):
    """Contrast, homogeneity, energy of the GLCM of non-vessel tissue, distance 1, angles 0/45/90/135 averaged.

    Pixels outside the tissue mask are assigned an extra grey level 0, which is dropped from the
    co-occurrence matrix, so only tissue-tissue pairs are counted.
    """
    g = np.clip(gray, 0, 255)
    q = (g / 256.0 * levels).astype(np.uint8) + 1
    q[~tissue_mask] = 0
    P = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=levels + 1, symmetric=True, normed=False)
    P = P[1:, 1:, 0, :].astype(np.float64)
    P /= P.sum(axis=(0, 1), keepdims=True)
    i, j = np.ogrid[0:levels, 0:levels]
    diff2 = ((i - j) ** 2)[..., None]
    contrast = (P * diff2).sum(axis=(0, 1)).mean()
    homogeneity = (P / (1.0 + diff2)).sum(axis=(0, 1)).mean()
    energy = np.sqrt((P ** 2).sum(axis=(0, 1))).mean()
    return float(contrast), float(homogeneity), float(energy)


def extract_features(mask, skel, fov, texture_img, cfg):
    n_branch, junction, centres = branch_points(skel)
    tort, segments, seg_labels = tortuosity(skel, junction, cfg["min_segment_len"])
    margin = cfg["glcm_vessel_margin"]
    vessel_zone = ndi.binary_dilation(mask, iterations=margin) if margin else mask
    tissue = fov & ~vessel_zone
    contrast, homog, energy = glcm_features(texture_img, tissue, cfg["glcm_levels"])
    feats = {
        "vessel_density": float(vessel_density(mask, fov)),
        "tortuosity": tort,
        "branch_points": int(n_branch),
        "glcm_contrast": contrast,
        "glcm_homogeneity": homog,
        "glcm_energy": energy,
    }
    extras = {"junction": junction, "branch_centres": centres, "segments": segments,
              "segment_labels": seg_labels, "tissue": tissue}
    return feats, extras
