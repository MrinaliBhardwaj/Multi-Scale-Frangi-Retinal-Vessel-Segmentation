# Results on DRIVE (all metrics inside the official FOV)

## Segmentation: proposed vs baseline

| Method | Split | Sensitivity | Specificity | Accuracy | Dice | Jaccard |
|---|---|---|---|---|---|---|
| Proposed | test (20) | 0.7300 ± 0.0635 | 0.9690 ± 0.0144 | 0.9381 ± 0.0095 | 0.7492 ± 0.0346 | 0.6002 ± 0.0436 |
| Otsu on green (baseline) | test (20) | 0.6354 ± 0.3960 | 0.4684 ± 0.3657 | 0.4924 ± 0.2735 | 0.1994 ± 0.1234 | 0.1155 ± 0.0728 |
| Proposed | training (20) | 0.7328 ± 0.0458 | 0.9585 ± 0.0356 | 0.9297 ± 0.0290 | 0.7280 ± 0.0686 | 0.5762 ± 0.0758 |
| Otsu on green (baseline) | training (20) | 0.7620 ± 0.2756 | 0.3707 ± 0.2812 | 0.4157 ± 0.2149 | 0.2294 ± 0.0889 | 0.1321 ± 0.0531 |
| Proposed | all (40) | 0.7314 ± 0.0547 | 0.9637 ± 0.0273 | 0.9339 ± 0.0217 | 0.7386 ± 0.0547 | 0.5882 ± 0.0622 |
| Otsu on green (baseline) | all (40) | 0.6987 ± 0.3428 | 0.4196 ± 0.3258 | 0.4540 ± 0.2459 | 0.2144 ± 0.1073 | 0.1238 ± 0.0635 |

## Ablation (mean Dice, all 40 images)

| Variant | Dice | Change |
|---|---|---|
| Full pipeline | 0.7386 | – |
| w/o CLAHE | 0.6880 | -0.0507 |
| w/o DFT high-emphasis | 0.7363 | -0.0023 |
| w/o restoration | 0.7326 | -0.0060 |
| Single-scale Frangi (sigma=2) | 0.7240 | -0.0147 |

## JPEG robustness (all 40 images)

| Input | Mean Dice | Dice drop | Mean abs. risk-score drift | Images changing band |
|---|---|---|---|---|
| Original | 0.7386 | – | – | – |
| JPEG q=80 | 0.7328 | 0.0058 | 0.0539 | 9 |
| JPEG q=70 | 0.7298 | 0.0088 | 0.0655 | 9 |

## Features (mean ± std, all 40 images)

| Feature | Mean ± std | P5 (→0) | P95 (→1) | Weight | Direction |
|---|---|---|---|---|---|
| Vessel density | 0.1298 ± 0.0278 | 0.1004 | 0.1664 | 0.15 | lower = risk |
| Tortuosity | 1.1020 ± 0.0311 | 1.0785 | 1.1443 | 0.25 | higher = risk |
| Branch points | 127.5500 ± 72.1021 | 72.0000 | 202.9500 | 0.15 | higher = risk |
| GLCM contrast | 1.6613 ± 0.3931 | 1.1048 | 2.4002 | 0.25 | higher = risk |
| GLCM homogeneity | 0.6505 ± 0.0317 | 0.5982 | 0.7035 | 0.1 | lower = risk |
| GLCM energy | 0.2034 ± 0.0243 | 0.1657 | 0.2489 | 0.1 | lower = risk |

## Risk stratification

| Band | Images |
|---|---|
| Low | 7 |
| Medium | 27 |
| High | 6 |

Sanity check against DRIVE's own annotation (7 images with signs of mild early DR): mean score 0.508 (early DR) vs 0.415 (no DR); one-sided Mann-Whitney U = 156.0, p = 0.079. This is a sanity check, not a clinical validation.


## Weight sensitivity (±20 % on one weight, number of images changing band out of 40)

| Feature | ×0.8 | ×1.2 |
|---|---|---|
| Vessel density | 3 | 1 |
| Tortuosity | 0 | 4 |
| Branch points | 0 | 2 |
| GLCM contrast | 2 | 1 |
| GLCM homogeneity | 0 | 0 |
| GLCM energy | 1 | 0 |

Module 1 FOV vs official DRIVE mask: mean pixel agreement 0.9964.
