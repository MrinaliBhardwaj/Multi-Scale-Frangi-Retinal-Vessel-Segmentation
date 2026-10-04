"""Reproduce every number and figure in the report.

    python scripts/run_all.py

Outputs go to results/ (CSV, JSON, markdown summary) and figures/ (PNG).
"""
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from drvessel.config import make_config, DEFAULT_CONFIG  # noqa: E402
from drvessel.io_utils import list_drive, read_rgb, read_binary  # noqa: E402
from drvessel.pipeline import run_pipeline, baseline_otsu  # noqa: E402
from drvessel.metrics import segmentation_metrics  # noqa: E402
from drvessel import fusion, viz  # noqa: E402
from drvessel.preprocess import fov_mask  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES, FIG = "results", "figures"
METRICS = ["sensitivity", "specificity", "accuracy", "dice", "jaccard"]
ABLATIONS = {
    "Full pipeline": {},
    "w/o CLAHE": {"use_clahe": False},
    "w/o DFT high-emphasis": {"use_hef": False},
    "w/o restoration": {"use_restoration": False},
    "Single-scale Frangi (sigma=2)": {"single_scale": True},
}


def jpeg(rgb, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def process(item):
    rgb, gt, fov = read_rgb(item["image"]), read_binary(item["manual"]), read_binary(item["mask"])
    row = {"id": item["id"], "split": item["split"], "early_dr": item["early_dr"]}
    out = run_pipeline(rgb)                         # FOV computed by Module 1 itself
    row.update({f"{k}": v for k, v in segmentation_metrics(out["mask"], gt, fov).items()})
    row.update(out["features"])
    row["fov_agreement"] = float((fov_mask(rgb) == fov).mean())
    for k, v in segmentation_metrics(baseline_otsu(rgb, fov), gt, fov).items():
        row[f"baseline_{k}"] = v
    for name, ov in ABLATIONS.items():
        if ov:
            o = run_pipeline(rgb, make_config(**ov))
            row[f"abl::{name}"] = segmentation_metrics(o["mask"], gt, fov)["dice"]
    for q in (80, 70):
        o = run_pipeline(jpeg(rgb, q))
        row[f"jpeg{q}_dice"] = segmentation_metrics(o["mask"], gt, fov)["dice"]
        for f, v in o["features"].items():
            row[f"jpeg{q}::{f}"] = v
    return row


def mean_std(df, cols):
    return {c: (df[c].mean(), df[c].std()) for c in cols}


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    items = list_drive()
    assert len(items) == 40, "DRIVE not found - run scripts/download_drive.py first"
    with Pool() as pool:
        rows = pool.map(process, items)
    df = pd.DataFrame(rows)

    # ---- Module 7: fusion ----
    ref = fusion.fit_reference(rows)
    fusion.save_reference(ref, os.path.join(RES, "feature_reference.json"))
    cfg = DEFAULT_CONFIG
    scores, bands, contribs = [], [], []
    for r in rows:
        s, b, c, _ = fusion.score(r, ref, cfg)
        scores.append(s)
        bands.append(b)
        contribs.append(c)
    df["risk_score"], df["risk_band"] = scores, bands
    for f in fusion.FEATURES:
        df[f"contrib::{f}"] = [c[f] for c in contribs]

    # JPEG drift of the risk score (same reference)
    for q in (80, 70):
        df[f"jpeg{q}_score"] = [fusion.score({f: r[f"jpeg{q}::{f}"] for f in fusion.FEATURES}, ref, cfg)[0]
                                for r in rows]

    # weight sensitivity: +-20 % on one weight at a time
    sens = []
    for f in fusion.FEATURES:
        for k in (0.8, 1.2):
            w = dict(cfg["weights"])
            w[f] *= k
            c2 = dict(cfg, weights=w)
            changed = sum(fusion.score(r, ref, c2)[1] != b for r, b in zip(rows, bands))
            sens.append({"feature": f, "factor": k, "images_changing_band": changed})
    sens = pd.DataFrame(sens)

    df.to_csv(os.path.join(RES, "per_image_results.csv"), index=False)
    sens.to_csv(os.path.join(RES, "weight_sensitivity.csv"), index=False)

    # ---- summary tables ----
    lines = ["# Results on DRIVE (all metrics inside the official FOV)\n"]
    lines.append("## Segmentation: proposed vs baseline\n")
    lines.append("| Method | Split | " + " | ".join(m.capitalize() for m in METRICS) + " |")
    lines.append("|---|---|" + "---|" * len(METRICS))
    for split, sub in (("test (20)", df[df.split == "test"]), ("training (20)", df[df.split == "training"]),
                       ("all (40)", df)):
        lines.append(f"| Proposed | {split} | " + " | ".join(fmt(*mean_std(sub, [m])[m]) for m in METRICS) + " |")
        lines.append(f"| Otsu on green (baseline) | {split} | " +
                     " | ".join(fmt(*mean_std(sub, [f"baseline_{m}"])[f"baseline_{m}"]) for m in METRICS) + " |")
    lines.append("\n## Ablation (mean Dice, all 40 images)\n")
    lines.append("| Variant | Dice | Change |")
    lines.append("|---|---|---|")
    full = df["dice"].mean()
    lines.append(f"| Full pipeline | {full:.4f} | – |")
    for name in list(ABLATIONS)[1:]:
        d = df[f"abl::{name}"].mean()
        lines.append(f"| {name} | {d:.4f} | {d - full:+.4f} |")
    lines.append("\n## JPEG robustness (all 40 images)\n")
    lines.append("| Input | Mean Dice | Dice drop | Mean abs. risk-score drift | Images changing band |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Original | {full:.4f} | – | – | – |")
    for q in (80, 70):
        d = df[f"jpeg{q}_dice"].mean()
        drift = (df[f"jpeg{q}_score"] - df.risk_score).abs().mean()
        changed = sum(fusion.band(s, cfg["bands"]) != b for s, b in zip(df[f"jpeg{q}_score"], df.risk_band))
        lines.append(f"| JPEG q={q} | {d:.4f} | {full - d:.4f} | {drift:.4f} | {changed} |")
    lines.append("\n## Features (mean ± std, all 40 images)\n")
    lines.append("| Feature | Mean ± std | P5 (→0) | P95 (→1) | Weight | Direction |")
    lines.append("|---|---|---|---|---|---|")
    for f in fusion.FEATURES:
        lines.append(f"| {viz.PRETTY[f]} | {df[f].mean():.4f} ± {df[f].std():.4f} | {ref[f]['min']:.4f} | "
                     f"{ref[f]['max']:.4f} | {cfg['weights'][f]} | {'higher = risk' if cfg['directions'][f] > 0 else 'lower = risk'} |")
    lines.append("\n## Risk stratification\n")
    counts = df.risk_band.value_counts()
    lines.append("| Band | Images |")
    lines.append("|---|---|")
    for b in ("Low", "Medium", "High"):
        lines.append(f"| {b} | {int(counts.get(b, 0))} |")
    e, n = df[df.early_dr].risk_score, df[~df.early_dr].risk_score
    u = stats.mannwhitneyu(e, n, alternative="greater")
    lines.append(f"\nSanity check against DRIVE's own annotation (7 images with signs of mild early DR): "
                 f"mean score {e.mean():.3f} (early DR) vs {n.mean():.3f} (no DR); "
                 f"one-sided Mann-Whitney U = {u.statistic:.1f}, p = {u.pvalue:.3f}. "
                 "This is a sanity check, not a clinical validation.\n")
    lines.append("\n## Weight sensitivity (±20 % on one weight, number of images changing band out of 40)\n")
    lines.append("| Feature | ×0.8 | ×1.2 |")
    lines.append("|---|---|---|")
    for f in fusion.FEATURES:
        s = sens[sens.feature == f].set_index("factor").images_changing_band
        lines.append(f"| {viz.PRETTY[f]} | {s[0.8]} | {s[1.2]} |")
    lines.append(f"\nModule 1 FOV vs official DRIVE mask: mean pixel agreement {df.fov_agreement.mean():.4f}.\n")
    summary = "\n".join(lines)
    with open(os.path.join(RES, "summary.md"), "w") as fh:
        fh.write(summary)
    print(summary)

    make_figures(df, items)


def make_figures(df, items):
    ref = fusion.load_reference(os.path.join(RES, "feature_reference.json"))
    by_id = {i["id"]: i for i in items}
    best = df.sort_values("dice", ascending=False).iloc[0].id
    demo_ids = ["01_test", "03_test", "08_test", best]
    for iid in dict.fromkeys(demo_ids):
        it = by_id[iid]
        rgb, gt, fov = read_rgb(it["image"]), read_binary(it["manual"]), read_binary(it["mask"])
        out = run_pipeline(rgb)
        viz.stage_montage(out, os.path.join(FIG, f"stages_{iid}.png"), f"DRIVE {iid}: pipeline stages")
        s, b, c, _ = fusion.score(out["features"], ref, DEFAULT_CONFIG)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(viz.error_overlay(out["mask"], gt, fov))
        m = segmentation_metrics(out["mask"], gt, fov)
        axes[0].set_title(f"vs expert mask  Dice={m['dice']:.3f}\ngreen=TP  red=FP  blue=FN", fontsize=10)
        axes[0].axis("off")
        viz.plot_skeleton_overlay(out, axes[1])
        viz.contribution_chart(c, s, b, ax=axes[2])
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"explain_{iid}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # proposed vs baseline per image
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.dice, 0.4, label="Proposed (Frangi pipeline)")
    ax.bar(x + 0.2, df.baseline_dice, 0.4, label="Baseline (Otsu on green)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.id, rotation=90, fontsize=7)
    ax.set_ylabel("Dice")
    ax.legend()
    ax.set_title("Per-image Dice on DRIVE")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dice_per_image.png"), dpi=110)
    plt.close(fig)

    # ablation
    names = list(ABLATIONS)
    vals = [df.dice.mean()] + [df[f"abl::{n}"].mean() for n in names[1:]]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(names, vals, color=["#3949ab"] + ["#9fa8da"] * (len(names) - 1))
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 0.05, max(vals) + 0.02)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center")
    ax.set_xlabel("Mean Dice (40 images)")
    ax.set_title("Ablation: removing one module at a time")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=110)
    plt.close(fig)

    # risk scores
    fig, ax = plt.subplots(figsize=(12, 4))
    d = df.sort_values("risk_score")
    colours = [viz.BAND_COLOURS[b] for b in d.risk_band]
    ax.bar(range(len(d)), d.risk_score, color=colours, edgecolor=["black" if e else "none" for e in d.early_dr],
           linewidth=[2 if e else 0 for e in d.early_dr])
    for t in DEFAULT_CONFIG["bands"]:
        ax.axhline(t, ls="--", c="grey", lw=1)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.id, rotation=90, fontsize=7)
    ax.set_ylabel("Risk score")
    ax.set_title("Risk score per image (black outline = DRIVE image with signs of mild early DR)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "risk_scores.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
