"""All pipeline parameters in one place (Table 1 of the report)."""
import copy

DEFAULT_CONFIG = {
    # Module 1 - preprocessing
    "fov_erode_px": 6,              # shrink FOV so the aperture rim is not detected as a vessel
    # Module 2 - CLAHE
    "clahe_clip": 3.0,
    "clahe_tile": 8,
    # Module 3 - frequency-domain high-emphasis  H = a + b * H_gaussianHP
    "hef_a": 0.5,
    "hef_b": 1.5,
    "hef_d0": 15.0,                 # cutoff (frequency samples) of the Gaussian high-pass
    # Module 4 - restoration
    "illum_kernel": 51,
    "median_kernel": 3,
    # Module 5 - Frangi + segmentation
    "frangi_sigmas": [1.0, 1.5, 2.0, 2.5, 3.0],
    "frangi_beta": 0.5,
    "frangi_c": 15.0,
    "otsu_gamma": 0.7,              # power-law applied to vesselness before Otsu
    "open_radius": 0,                # opening with disc>=1 erases 1-2 px vessels; area filter removes specks instead
    "close_radius": 1,
    "min_area": 60,
    # Module 6 - features
    "min_segment_len": 10,
    "glcm_levels": 32,
    "glcm_vessel_margin": 2,
    # Module 7 - fusion
    # direction +1: larger value -> higher risk ; -1: smaller value -> higher risk
    "weights": {
        "tortuosity": 0.25,
        "glcm_contrast": 0.25,
        "branch_points": 0.15,
        "vessel_density": 0.15,
        "glcm_homogeneity": 0.10,
        "glcm_energy": 0.10,
    },
    "directions": {
        "tortuosity": +1,          # increased tortuosity is an early DR sign
        "glcm_contrast": +1,       # irregular tissue (haemorrhages, exudates) raises contrast
        "branch_points": +1,       # abnormal / new vessel branching
        "vessel_density": -1,      # capillary dropout / reduced vascular complexity
        "glcm_homogeneity": -1,    # irregular tissue lowers homogeneity
        "glcm_energy": -1,         # irregular tissue lowers energy (less uniform)
    },
    "bands": [0.3, 0.6],
    # ablation switches
    "use_clahe": True,
    "use_hef": True,
    "use_restoration": True,
    "single_scale": False,
}


def make_config(**overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg
