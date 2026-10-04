"""Module 5: multi-scale Hessian (Frangi) vesselness, Otsu, morphology, thinning."""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, disk


def hessian_eigenvalues(img, sigma):
    """Scale-normalised Hessian via Gaussian-derivative convolution; returns (l1, l2) with |l1| <= |l2|."""
    s2 = sigma ** 2
    Hxx = ndi.gaussian_filter(img, sigma, order=(0, 2)) * s2
    Hyy = ndi.gaussian_filter(img, sigma, order=(2, 0)) * s2
    Hxy = ndi.gaussian_filter(img, sigma, order=(1, 1)) * s2
    # closed-form eigenvalues of the symmetric 2x2 matrix
    tmp = np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2)
    mu1 = 0.5 * (Hxx + Hyy + tmp)
    mu2 = 0.5 * (Hxx + Hyy - tmp)
    swap = np.abs(mu1) > np.abs(mu2)
    l1 = np.where(swap, mu2, mu1)
    l2 = np.where(swap, mu1, mu2)
    return l1, l2


def frangi_single(img, sigma, beta=0.5, c=15.0):
    """V = exp(-Rb^2 / 2 beta^2) * (1 - exp(-S^2 / 2 c^2)) for dark vessels (l2 > 0)."""
    l1, l2 = hessian_eigenvalues(img, sigma)
    Rb = l1 / (l2 + 1e-10)
    S = np.sqrt(l1 ** 2 + l2 ** 2)
    V = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1 - np.exp(-(S ** 2) / (2 * c ** 2)))
    V[l2 <= 0] = 0                    # vessels are darker than background in the green channel
    return V


def frangi(img, sigmas=(1, 1.5, 2, 2.5, 3), beta=0.5, c=15.0):
    """Maximum vesselness response over scales; also returns the winning scale per pixel."""
    stack = np.stack([frangi_single(img, s, beta, c) for s in sigmas])
    return stack.max(axis=0), np.asarray(sigmas)[stack.argmax(axis=0)]


def compress(v, gamma=1.0):
    """Power-law (gamma) transform of the vesselness map before Otsu.

    Frangi responses are heavy-tailed: a few thick vessels dominate the histogram and Otsu then
    ignores thin vessels. gamma < 1 spreads the low responses (a point operation, not a learned step).
    """
    v = v / (v.max() + 1e-12)
    return v ** gamma


def remove_small(mask, min_area):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def segment(vesselness, fov, open_r=1, close_r=2, min_area=30, gamma=1.0):
    """Otsu threshold inside FOV, then opening, closing and small-component removal."""
    v = compress(vesselness, gamma)
    v[~fov] = 0
    t = threshold_otsu(v[fov])
    mask = (v > t) & fov
    if open_r > 0:
        mask = ndi.binary_opening(mask, structure=disk(open_r))
    if close_r > 0:
        mask = ndi.binary_closing(mask, structure=disk(close_r))
    mask = remove_small(mask, min_area)
    return mask & fov, t


def thin(mask):
    """Zhang-Suen thinning (skimage's 2-D skeletonize implements Zhang & Suen 1984)."""
    return skeletonize(mask)
