"""Modules 1-4: acquisition, spatial enhancement, frequency enhancement, restoration."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


# ---------------- Module 1 ----------------
def green_channel(rgb):
    """Green channel has the best vessel contrast (haemoglobin absorbs green)."""
    return rgb[:, :, 1].astype(np.uint8)


def fov_mask(rgb, erode_px=0):
    """Field-of-view mask from Otsu on the red channel (the camera aperture)."""
    red = rgb[:, :, 0].astype(np.float64)
    m = red > threshold_otsu(red) * 0.5          # aperture is much brighter than the black border
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=2))
    lab, n = ndi.label(m)
    if n > 1:                                      # keep largest component
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
    if erode_px > 0:
        m = ndi.binary_erosion(m, structure=disk(erode_px))
    return m


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def fill_outside_fov(img, fov):
    """Replace the black border with the mean FOV intensity so filters do not ring at the rim."""
    out = img.astype(np.float64).copy()
    out[~fov] = out[fov].mean()
    return out


# ---------------- Module 2 ----------------
def clahe(gray, clip=2.5, tile=8):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


# ---------------- Module 3 ----------------
def gaussian_highpass(shape, d0):
    rows, cols = shape
    u = np.arange(rows) - rows / 2
    v = np.arange(cols) - cols / 2
    V, U = np.meshgrid(v, u)
    D2 = U ** 2 + V ** 2
    return 1.0 - np.exp(-D2 / (2.0 * d0 ** 2))


def high_emphasis(gray, a=0.5, b=1.5, d0=30.0):
    """H(u,v) = a + b * H_HP(u,v) applied in the 2-D DFT domain (zero-padded)."""
    f = gray.astype(np.float64)
    rows, cols = f.shape
    P, Q = 2 * rows, 2 * cols
    padded = np.zeros((P, Q))
    padded[:rows, :cols] = f
    F = np.fft.fftshift(np.fft.fft2(padded))
    H = a + b * gaussian_highpass((P, Q), d0 * 2)  # d0 given for the unpadded image
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))[:rows, :cols]
    return rescale(g)


def rescale(x, fov=None):
    ref = x[fov] if fov is not None else x
    lo, hi = np.percentile(ref, 0.5), np.percentile(ref, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-9) * 255.0, 0, 255)


# ---------------- Module 4 ----------------
def correct_illumination(gray, fov, kernel=51):
    """Degradation model g = f * L + n : estimate L with a large median filter and divide it out."""
    g = np.clip(gray, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(g, kernel).astype(np.float64)
    flat = g.astype(np.float64) / (background + 1.0)
    flat = fill_outside_fov(flat, fov)
    return rescale(flat, fov)


def denoise(gray, kernel=3):
    return cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), kernel).astype(np.float64)
