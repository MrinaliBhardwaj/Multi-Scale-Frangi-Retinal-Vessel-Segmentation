# Project Plan — Multi-Scale Frangi Vessel Segmentation + Rule-Based Fusion for Explainable DR Risk

DIP (BCSE403L) course project, Review I topic by Aviral Chaubey (23BCE0566).

## 1. Prior work on GitHub

I searched GitHub (repo search, Oct 2026). Some repos cover parts of this pipeline. None does the whole thing, and none has the fusion and risk-scoring stage.

| Repo | What it covers | What it lacks for this project |
|---|---|---|
| `mmd-ahd/Retinal-Vessel-Segmentation-DRIVE` | Green → CLAHE → Frangi → Otsu → morphology on DRIVE (notebook + script) | No DFT high-emphasis, no illumination restoration, no skeleton, no features, no risk score |
| `golzalexander666-bit/mcect_drive` | Green → CLAHE → bottom-hat → Frangi → min cross-entropy threshold | Segmentation only |
| `Gaurika-vats/...U-Net`, `LSaiko/Reti-ves-seg` | Frangi used as an input channel to a U-Net | Supervised and needs a GPU, which is the opposite of the proposal |
| `maz369/Retinal-Vessel-Tortuosity-Fundus-Optps`, `enrigrisan/RET-Tortuosity` | Tortuosity measures (MATLAB) | Not tied to segmentation or scoring |
| `TFMap/...Multi-GLCM-Energy-Threshold` | Uses GLCM to pick a segmentation threshold | GLCM is used for thresholding, not as a texture feature of the tissue |
| `LeadingIndiaAI/Digital-Retinal-Images-for-Vessel-Extraction` | Segmentation plus general morphological attributes | No texture fusion and no graded risk output |

**Conclusion:** stages 1–5 (preprocessing → Frangi → Otsu) have been done before. The contributions claimed in the report are new on GitHub: DFT high-emphasis plus illumination restoration in one chain, combined structural and GLCM features, and an explainable weighted risk score. I will build the project from scratch, so it is original work. The repos above serve only as sanity references for the Dice values I should expect.

## 2. Tech stack

- Python 3.10+, CPU only
- `numpy`, `scipy`, `opencv-python`, `scikit-image` (Frangi via `skimage.filters.frangi` plus a hand-written version for explainability, GLCM via `graycomatrix/graycoprops`, `skeletonize`), `pandas`, `matplotlib`
- `streamlit` for a one-page demo app (upload a fundus image and get the mask, skeleton, features and risk band)
- `pytest` for unit tests

## 3. Repository layout

```
dr_vessel/
  config.yaml            # every parameter from Table 1 of the report, in one place
  data/                  # DRIVE (git-ignored); download script provided
  src/
    io_utils.py          # load DRIVE images, manual masks, FOV masks
    m1_preprocess.py     # green channel + FOV mask (Otsu on red channel)
    m2_clahe.py          # CLAHE (clip 2.0–3.0, 8×8)
    m3_frequency.py      # DFT high-emphasis: H = a + b·H_GaussianHP
    m4_restoration.py    # illumination division (51×51 median) + 3×3 median
    m5_vessels.py        # Hessian → Frangi (σ=1..3, β=0.5, c=15), Otsu, morphology, area filter, Zhang–Suen
    m6_features.py       # density, tortuosity, branch points, GLCM contrast/homogeneity/energy
    m7_fusion.py         # min-max normalisation, weights, score, banding
    metrics.py           # Se, Sp, Acc, Dice, Jaccard (inside FOV only)
    baseline.py          # plain Otsu on raw green channel
    pipeline.py          # runs M1→M7 on one image and returns all intermediate outputs
  scripts/
    download_drive.py
    run_all.py           # all 40 images → results/metrics.csv, features.csv, risk.csv
    tune_params.py       # grid search over σ range, β, c, clip limit on the TRAINING split only
    jpeg_robustness.py   # rerun at JPEG quality 70/80, report the Dice drop
    make_figures.py      # stage-wise montage, ROC/bar charts, risk explanation plots
  app/streamlit_app.py
  tests/
  notebooks/walkthrough.ipynb   # stage-by-stage visual explanation for the viva
  results/
  report/                # final report figures and tables
```

## 4. Build order (milestones)

**M0 — Setup and data (day 1)**
- Get DRIVE: 40 images, each with a manual vessel mask and a FOV mask. The official site needs a grand-challenge login; Kaggle mirrors are a fallback. Add a script that checks folder structure and checksums.
- Note: DRIVE ships the FOV masks. I will still build my own Otsu-on-red FOV, as the report says, and compare it with the official one.

**M1 — Modules 1–4: enhancement chain (days 2–3)**
- Implement each module as a pure function `img → img`, each with a parameter dict.
- Save a 6-panel montage per image: RGB → green → CLAHE → high-emphasis → illumination-corrected → median.
- Implementation details:
  - DFT: zero-pad, `fftshift`, Gaussian HP with D0 as a parameter, and clip/renormalise after the inverse transform.
  - Illumination: divide by the background, guard against division by zero with ε, and restrict to the FOV.

**M2 — Module 5: Frangi + segmentation (days 4–6)**
- Hand-write the Hessian with Gaussian-derivative kernels and the eigenvalue sort |λ1| ≤ |λ2|. Vessels are dark on a bright background in the green channel, so keep only pixels with λ2 > 0 (equivalent to `black_ridges=True`).
- Cross-check my output against `skimage.filters.frangi` (correlation > 0.99).
- c = 15 assumes a 0–255 intensity scale. Fix the scale explicitly so the constant means what the report says.
- Then: max over scales, Otsu inside the FOV, opening (disc 2), closing (disc 3), remove components < 30 px, Zhang–Suen thinning.
- Erode the FOV by a few pixels before thresholding. Otherwise the aperture rim lights up as a "vessel".

**M3 — Evaluation and baseline (day 7)**
- Pixel metrics computed inside the FOV only (standard DRIVE practice): Se, Sp, Acc, Dice, Jaccard. Report per image plus mean ± std.
- Baseline: Otsu on the raw green channel (inverted).
- **Tuning protocol:** tune parameters on the 20 *training* images and report on the 20 *test* images, plus all 40. This avoids the criticism of tuning on test data, even though the method is training-free.
- Ablation table that removes one module at a time (no CLAHE, no DFT, no restoration, single-scale Frangi). This shows each DIP stage earns its place.
- Compare with published unsupervised numbers on DRIVE (Chaudhuri, Zana–Klein, Mendonça, B-COSFIRE), quoted from the papers.

**M4 — Module 6: features (days 8–9)**
- Density: vessel pixels / FOV pixels.
- Tortuosity:
  1. Remove branch points from the skeleton so it splits into segments.
  2. For each segment with length ≥ L_min, compute arc length (√2 for diagonal steps) divided by the chord between its endpoints.
  3. Report the length-weighted mean.
- Branch points: skeleton pixels with ≥ 3 neighbours. Merge clusters of adjacent junction pixels so one junction counts once.
- GLCM on non-vessel tissue:
  1. Quantise to 32 or 64 levels.
  2. Mask out vessels and outside-FOV pixels. One option is to compute the GLCM on tiles that are mostly tissue; the other is to use a masked co-occurrence count. I'll pick one and document it.
  3. Distance 1, angles 0/45/90/135, averaged.

**M5 — Module 7: fusion and scoring (day 10)**
- Min-max normalise each feature across the 40 images. Save the min/max to JSON so a single new image (e.g. in the Streamlit app) is scored against the same reference.
- **Feature direction:**
  - Homogeneity and energy fall as tissue gets more irregular, so they enter as `1 − f`.
  - Density direction is ambiguous: dropout lowers it, neovascularisation raises it. I'll document the choice and justify it in the report.
- Weights sum to 1, with tortuosity and GLCM contrast highest as the report states. Proposed start: tortuosity 0.25, contrast 0.25, branch points 0.15, density 0.15, (1−homogeneity) 0.10, (1−energy) 0.10.
- Bands: < 0.3 low, 0.3–0.6 medium, > 0.6 high.
- **Explainability output:** per image, a bar chart of the six wᵢ·fᵢ contributions, plus a skeleton overlay with branch points and the most tortuous segments highlighted.
- Weight-sensitivity check: perturb each weight by ±20% and count how many images change band.

**M6 — Robustness and validation of the score (day 11)**
- JPEG quality 70 and 80: report the Dice drop and the score drift per image.
- **Score sanity check:** DRIVE has no DR grades, only a note that 7 of the 40 images show signs of mild early DR. I will check whether those 7 score higher than the rest (Mann-Whitney U test). This is a sanity check, not a clinical validation, and the report must say so.
- Optional, if time allows: run the scoring on a small graded set (e.g. a few dozen IDRiD or Messidor images) and report the score vs. grade correlation. This is the strongest answer to "how do you know the score means anything?"

**M7 — Demo app, notebook, report (days 12–14)**
- Streamlit app: upload an image, see all intermediate stages, the mask, the skeleton, the six features, the contribution chart and the risk band. Plus a disclaimer that it is not a medical device.
- Walkthrough notebook for the viva: one image through every stage, with the equation next to each plot.
- Final report sections: results tables (metrics, baseline, ablation, literature comparison, JPEG), figures, limitations.

## 5. Deliverables

1. Working, tested code (`python scripts/run_all.py` reproduces every number).
2. `results/` with per-image CSVs and figures.
3. Streamlit demo.
4. Walkthrough notebook.
5. README with setup, run instructions and the results table.

## 6. Risks / things to raise with the faculty guide

- **The risk score has no ground truth on DRIVE.** The 7 "early DR" images are the only signal. This is the weakest point at review, and the optional graded-dataset check (M6) addresses it.
- **Fixed clinical weights.** Cite the sources behind tortuosity and contrast being weighted highest (Cheung et al. 2010), and show the sensitivity analysis.
- **Expected Dice.** Unsupervised Frangi pipelines usually reach about 0.70–0.78 Dice on DRIVE. Comparisons should be framed against unsupervised methods, not U-Net (~0.82).
- **Optic disc false positives.** If the eroded FOV and β tuning aren't enough, add an optional optic-disc suppression step (brightest-region detection plus a circular mask).

## 7. Open questions for you

1. Should the code be in Python (my default) or MATLAB?
2. Do you want the optional graded-dataset validation (IDRiD/Messidor)?
3. Is a Streamlit demo wanted, or only scripts plus a notebook?
