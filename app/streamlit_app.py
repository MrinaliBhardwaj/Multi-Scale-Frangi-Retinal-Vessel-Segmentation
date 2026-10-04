"""Streamlit demo: upload a fundus photograph (or pick a DRIVE image) and see every stage,
the vessel mask, the six features and the explainable risk score.

    streamlit run app/streamlit_app.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drvessel.config import make_config  # noqa: E402
from drvessel.pipeline import run_pipeline  # noqa: E402
from drvessel.io_utils import list_drive, read_rgb, read_binary  # noqa: E402
from drvessel.metrics import segmentation_metrics  # noqa: E402
from drvessel import fusion, viz  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REF_PATH = os.path.join(ROOT, "results", "feature_reference.json")

st.set_page_config(page_title="Retinal Vessel DR Risk", layout="wide")
st.title("Retinal vessel segmentation and explainable DR risk")
st.caption("Training-free pipeline: green channel → CLAHE → DFT high-emphasis → illumination correction → "
           "multi-scale Frangi → Otsu → morphology → Zhang–Suen skeleton → 6 features → weighted risk score. "
           "Academic project, not a medical device.")

# ---------------- sidebar: input + parameters ----------------
with st.sidebar:
    st.header("Input")
    drive = list_drive(os.path.join(ROOT, "data", "DRIVE"))
    choice = st.radio("Source", ["DRIVE image", "Upload"] if drive else ["Upload"])
    item, rgb = None, None
    if choice == "DRIVE image":
        ids = [d["id"] + (" (early DR)" if d["early_dr"] else "") for d in drive]
        idx = st.selectbox("Image", range(len(drive)), format_func=lambda i: ids[i],
                           index=next(i for i, d in enumerate(drive) if d["id"] == "01_test"))
        item = drive[idx]
        rgb = read_rgb(item["image"])
    else:
        up = st.file_uploader("Colour fundus image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if up:
            rgb = np.array(Image.open(io.BytesIO(up.read())).convert("RGB"))
            if max(rgb.shape[:2]) > 700:      # parameters are tuned for DRIVE resolution (~565 px wide)
                scale = 584 / rgb.shape[0]
                rgb = np.array(Image.fromarray(rgb).resize((int(rgb.shape[1] * scale), 584), Image.BILINEAR))
                st.info("Image resized to DRIVE height (584 px) because the filter scales are tuned for it.")

    st.header("Parameters")
    p = {}
    p["clahe_clip"] = st.slider("CLAHE clip limit", 1.0, 5.0, 3.0, 0.5)
    p["hef_d0"] = st.slider("High-pass cutoff D0", 5.0, 60.0, 15.0, 5.0)
    p["hef_b"] = st.slider("High-emphasis gain b", 0.5, 3.0, 1.5, 0.25)
    p["frangi_beta"] = st.slider("Frangi β", 0.25, 2.0, 0.5, 0.25)
    p["frangi_c"] = st.slider("Frangi c", 5.0, 30.0, 15.0, 1.0)
    p["otsu_gamma"] = st.slider("Vesselness γ before Otsu", 0.3, 1.0, 0.7, 0.1)
    p["min_area"] = st.slider("Min component area (px)", 0, 200, 60, 10)

if rgb is None:
    st.info("Pick a DRIVE image or upload a fundus photograph to start.")
    st.stop()


@st.cache_data(show_spinner="Running pipeline…")
def cached_run(img_bytes, shape, params):
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
    return run_pipeline(img, make_config(**dict(params)))


cfg = make_config(**p)
out = cached_run(rgb.tobytes(), rgb.shape, tuple(sorted(p.items())))

# ---------------- stages ----------------
st.subheader("Pipeline stages")
stages = viz.stage_images(out)
cols = st.columns(3)
for i, (title, img, cmap) in enumerate(stages):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    cols[i % 3].markdown(f"**{title}**")
    cols[i % 3].pyplot(fig, clear_figure=True)
    plt.close(fig)

# ---------------- evaluation vs ground truth ----------------
if item is not None:
    st.subheader("Comparison with the DRIVE expert annotation")
    gt, fov = read_binary(item["manual"]), read_binary(item["mask"])
    m = segmentation_metrics(out["mask"], gt, fov)
    c1, c2 = st.columns([1, 1])
    c1.image(viz.error_overlay(out["mask"], gt, fov), caption="green = TP, red = FP, blue = FN (missed)",
             width="stretch")
    c2.dataframe(pd.DataFrame({"Metric": [k.capitalize() for k in m], "Value": [f"{v:.4f}" for v in m.values()]}),
                 hide_index=True)

# ---------------- features & risk ----------------
st.subheader("Features and explainable risk score")
if not os.path.exists(REF_PATH):
    st.warning("results/feature_reference.json not found — run `python scripts/run_all.py` first.")
    st.stop()
ref = fusion.load_reference(REF_PATH)
score, band, contrib, norm = fusion.score(out["features"], ref, cfg)
c1, c2 = st.columns([1, 1])
with c1:
    fig, ax = plt.subplots(figsize=(6, 6))
    viz.plot_skeleton_overlay(out, ax)
    st.pyplot(fig, clear_figure=True)
with c2:
    colour = viz.BAND_COLOURS[band]
    st.markdown(f"### Risk score: **{score:.3f}** — <span style='color:{colour}'>{band} risk</span>",
                unsafe_allow_html=True)
    fig = viz.contribution_chart(contrib, score, band)
    st.pyplot(fig, clear_figure=True)
    rows = []
    for f in fusion.FEATURES:
        raw = out["features"][f]
        rows.append({"Feature": viz.PRETTY[f],
                     "Raw value": f"{raw:.4f}" if isinstance(raw, float) else str(raw),
                     "DRIVE range (P5–P95)": f"{ref[f]['min']:.3f} – {ref[f]['max']:.3f}",
                     "Normalised risk (0–1)": f"{norm[f]:.3f}",
                     "Weight": cfg["weights"][f],
                     "Contribution": f"{contrib[f]:.3f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Bands: Low < {cfg['bands'][0]} ≤ Medium ≤ {cfg['bands'][1]} < High. "
               "Each feature is mapped to 0–1 against the DRIVE reference range (1 = more risk) "
               "and multiplied by a fixed weight; the score is the sum.")
