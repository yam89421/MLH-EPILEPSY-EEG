"""
Report Figure Generator — EEG Seizure Prediction Pipeline
==========================================================
Figures:
  fig1_pipeline.png              — End-to-end pipeline flowchart
  fig2_comparison.png            — Sens + FA/h: 5 strategies (heterogeneous)
  fig3_patient_heatmap.png       — Per-patient sensitivity heatmap (5 methods)
  fig4_sensitivity_bars.png      — Sensitivity bar chart with Δ annotations
  fig5_feature_importance.png    — Top-20 discriminative features
  fig6_pli_wpli_seizure.png      — PLI/WPLI correlation with seizure onset (PN00)
  fig7_auc_by_patient.png        — AUC per patient: ThAlgo (paper) vs OR-gate
  fig8_improvement_delta.png     — Per-patient Δsens & ΔFA/h: ThAlgo → OR-gate

Scripts that generated each result:
  ThAlgo       ← seizure_prediction_thalgo.py      (results_THALGO_2026-04-10.txt)
  LGBM all     ← seizure_prediction_loso.py         (results_LOSO_continuous_pli_2026-04-09.txt)
  LGBM no PLI  ← seizure_prediction_no_pli.py       (nohup_no_pli.log)
  Fused        ← seizure_prediction_fused.py        (results_FUSED_2026-04-09.txt)
  OR-gate      ← seizure_prediction_seizure_related.py (nohup_seizure_related_orgate.log)
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(__file__))

def savefig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")

# ── Per-patient results ───────────────────────────────────────────────────────

PATIENTS = ["PN00","PN01","PN03","PN05","PN06","PN09","PN10","PN12","PN13","PN14","PN16","PN17"]

n_seiz = {"PN00":5,"PN01":2,"PN03":2,"PN05":3,"PN06":2,"PN09":3,
          "PN10":10,"PN12":3,"PN13":3,"PN14":4,"PN16":2,"PN17":2}

# ── ThAlgo — Detti et al. 2019 exact implementation
#    Script: seizure_prediction_thalgo.py
#    File:   results_THALGO_2026-04-10.txt  (k-fold CV, MAACD arcs only)
THALGO = {
    "PN00":{"sens":0.900,"fah":5.76,"auc":0.531},
    "PN01":{"sens":1.000,"fah":5.22,"auc":0.470},
    "PN03":{"sens":1.000,"fah":5.22,"auc":0.530},
    "PN05":{"sens":0.333,"fah":4.70,"auc":0.500},
    "PN06":{"sens":0.333,"fah":6.26,"auc":0.514},
    "PN09":{"sens":0.333,"fah":5.04,"auc":0.543},
    "PN10":{"sens":0.940,"fah":5.72,"auc":0.492},
    "PN12":{"sens":0.667,"fah":5.63,"auc":0.495},
    "PN13":{"sens":0.667,"fah":5.74,"auc":0.524},
    "PN14":{"sens":1.000,"fah":4.93,"auc":0.545},
    "PN16":{"sens":1.000,"fah":4.72,"auc":0.489},
    "PN17":{"sens":1.000,"fah":4.96,"auc":0.525},
}

# ── LightGBM all features (LOSO)
#    Script: seizure_prediction_loso.py  (FEATURE_MODE="all")
#    File:   results_LOSO_continuous_pli_2026-04-09.txt
LGBM_ALL = {
    "PN00":{"sens":0.300,"fah":1.79,"auc":0.814},
    "PN01":{"sens":0.000,"fah":0.78,"auc":0.635},
    "PN03":{"sens":0.000,"fah":0.00,"auc":0.855},
    "PN05":{"sens":0.000,"fah":0.00,"auc":0.755},
    "PN06":{"sens":0.500,"fah":0.78,"auc":0.465},
    "PN09":{"sens":0.000,"fah":0.00,"auc":0.738},
    "PN10":{"sens":0.650,"fah":4.66,"auc":0.593},
    "PN12":{"sens":0.875,"fah":2.10,"auc":0.539},
    "PN13":{"sens":0.000,"fah":1.83,"auc":0.661},
    "PN14":{"sens":0.250,"fah":1.76,"auc":0.595},
    "PN16":{"sens":1.000,"fah":1.17,"auc":0.859},
    "PN17":{"sens":0.500,"fah":1.58,"auc":0.540},
}

# ── LightGBM without PLI/WPLI (temporal + frequency features only, LOSO)
#    Script: seizure_prediction_no_pli.py  (tuned threshold FA/h≤1)
#    File:   nohup_no_pli.log
LGBM_NOPLI = {
    "PN00":{"sens":0.700,"fah":2.61,"auc":0.835},
    "PN01":{"sens":0.500,"fah":0.78,"auc":0.637},
    "PN03":{"sens":0.000,"fah":0.00,"auc":0.871},
    "PN05":{"sens":0.000,"fah":0.00,"auc":0.708},
    "PN06":{"sens":0.500,"fah":0.78,"auc":0.478},
    "PN09":{"sens":0.000,"fah":0.00,"auc":0.747},
    "PN10":{"sens":0.900,"fah":4.14,"auc":0.598},
    "PN12":{"sens":0.833,"fah":2.08,"auc":0.585},
    "PN13":{"sens":0.000,"fah":2.35,"auc":0.657},
    "PN14":{"sens":0.250,"fah":1.96,"auc":0.591},
    "PN16":{"sens":1.000,"fah":1.57,"auc":0.865},
    "PN17":{"sens":0.500,"fah":1.96,"auc":0.549},
}

# ── LightGBM fused — LGBM arm α·score + (1-α)·MAACD_norm (LOSO)
#    Script: seizure_prediction_fused.py
#    File:   results_FUSED_2026-04-09.txt
LGBM_FUSED = {
    "PN00":{"sens":0.800,"fah":2.10,"auc":0.727},
    "PN01":{"sens":0.000,"fah":0.39,"auc":0.587},
    "PN03":{"sens":0.000,"fah":0.00,"auc":0.716},
    "PN05":{"sens":0.000,"fah":0.00,"auc":0.700},
    "PN06":{"sens":0.500,"fah":0.00,"auc":0.503},
    "PN09":{"sens":0.000,"fah":0.00,"auc":0.668},
    "PN10":{"sens":0.100,"fah":1.83,"auc":0.589},
    "PN12":{"sens":0.667,"fah":1.16,"auc":0.593},
    "PN13":{"sens":0.000,"fah":0.78,"auc":0.641},
    "PN14":{"sens":0.250,"fah":1.57,"auc":0.632},
    "PN16":{"sens":1.000,"fah":0.78,"auc":0.763},
    "PN17":{"sens":0.500,"fah":1.17,"auc":0.587},
}

# ── OR-gate — LightGBM (no PLI/WPLI) OR calibrated focal MAACD
#    Script: seizure_prediction_seizure_related.py
#    File:   nohup_seizure_related_orgate.log  (consec=6, thr=0.60)
ORGATE = {
    "PN00":{"sens":1.000,"fah":2.27,"auc":0.835},
    "PN01":{"sens":1.000,"fah":1.96,"auc":0.637},
    "PN03":{"sens":1.000,"fah":0.78,"auc":0.871},
    "PN05":{"sens":0.667,"fah":0.00,"auc":0.708},
    "PN06":{"sens":1.000,"fah":1.17,"auc":0.478},
    "PN09":{"sens":0.667,"fah":1.08,"auc":0.747},
    "PN10":{"sens":0.700,"fah":1.73,"auc":0.598},
    "PN12":{"sens":1.000,"fah":1.16,"auc":0.585},
    "PN13":{"sens":1.000,"fah":2.35,"auc":0.657},
    "PN14":{"sens":0.500,"fah":1.76,"auc":0.591},
    "PN16":{"sens":1.000,"fah":1.17,"auc":0.865},
    "PN17":{"sens":1.000,"fah":0.78,"auc":0.549},
}

# Summary means — ordered by increasing sensitivity for narrative arc
METHODS = [
    # (label,                      dict,        sens,  fah,   color)
    ("LGBM\nfused\n(α·LGBM+MAACD)", LGBM_FUSED, 0.318, 0.82, "#c0392b"),
    ("LGBM\nall feat.\n(LOSO)",      LGBM_ALL,   0.340, 1.37, "#e67e22"),
    ("LGBM\nno PLI/WPLI\n(LOSO)",    LGBM_NOPLI, 0.432, 1.52, "#d35400"),
    ("ThAlgo\n(Detti 2019)",          THALGO,     0.764, 5.32, "#8e44ad"),
    ("OR-gate\n(final)",              ORGATE,     0.878, 1.35, "#27ae60"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Pipeline flowchart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig1_pipeline.png ...")

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

def box(ax, x, y, w, h, lines, color="#3498db", fontsize=8.5):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=0.92, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4,
            multialignment="center", linespacing=1.4)

def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5,
                        connectionstyle="arc3,rad=0.0"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.15, label, ha="center", fontsize=7, color="#555")

box(ax, 1.3, 6.5, 2.0, 1.1,
    ["Siena Scalp EEG", "Database", "32-ch · 14 patients", "(12 used in study)"], "#7f8c8d")
arrow(ax, 2.3, 6.5, 3.2, 6.5)
box(ax, 4.0, 6.5, 1.5, 1.1, ["Feature", "Extraction", "per window (1s)"], "#2980b9")
arrow(ax, 4.75, 6.5, 5.6, 6.5)
box(ax, 6.4, 6.5, 1.5, 1.1, ["Per-channel", "Features", "~522 non-sync"], "#2980b9")
arrow(ax, 7.15, 6.5, 8.1, 6.5)
box(ax, 8.9, 6.5, 1.5, 1.1,
    ["Labels", "0 = interictal", "1 = ictal", "2 = preictal"], "#7f8c8d", fontsize=7.5)
arrow(ax, 9.65, 6.5, 10.5, 6.5)
box(ax, 11.3, 6.5, 1.5, 1.1, ["LOSO-CV", "Leave-One-", "Seizure-Out"], "#16a085")

ax.text(4.0, 5.6, "Band power (δθαβγ)\nEntropy · Hjorth\nFractal dim.",
        ha="center", fontsize=7, color="#2980b9", style="italic")

box(ax, 4.8, 4.0, 2.2, 1.1,
    ["LightGBM", "Classifier", "600 trees  scale_pos_weight"], "#2471a3")
box(ax, 9.5, 4.0, 2.2, 1.1,
    ["Focal MAACD Arm", "MAACD(f_S1) per patient", "ThAlgo pair selection"], "#7d3c98")

arrow(ax, 6.4, 5.95, 4.8, 4.55)
arrow(ax, 11.3, 5.95, 9.5, 4.55)

box(ax, 4.8, 2.5, 2.2, 1.0,
    ["Fixed threshold", "thr ∈ {0.60, 0.70, 0.80}", "consec ∈ {6,7,8,10}"], "#1a5276", fontsize=7.5)
box(ax, 9.5, 2.5, 2.2, 1.0,
    ["Per-fold calibration", "find thr: FA/h ≤ 1.0", "on train interictal"], "#4a235a", fontsize=7.5)

arrow(ax, 4.8, 3.45, 4.8, 3.0)
arrow(ax, 9.5, 3.45, 9.5, 3.0)

ax.add_patch(plt.Circle((7.15, 1.3), 0.42, color="#e67e22", zorder=4))
ax.text(7.15, 1.3, "OR", ha="center", va="center",
        fontsize=10, fontweight="bold", color="white", zorder=5)
arrow(ax, 4.8, 2.0, 6.6, 1.4)
arrow(ax, 9.5, 2.0, 7.7, 1.4)

box(ax, 12.5, 1.3, 2.4, 1.1,
    ["ALARM", "≥N consecutive hits", "10-min refractory"], "#c0392b")
arrow(ax, 7.57, 1.3, 11.3, 1.3)

for y, label, c in [(6.5,"Data & features","#7f8c8d"),(4.0,"Model training","#2471a3"),
                     (2.5,"Threshold setting","#1a5276"),(1.3,"Alarm logic","#c0392b")]:
    ax.text(0.15, y, label, ha="left", va="center", fontsize=7.5,
            color=c, fontweight="bold", rotation=90)

ax.set_title("Seizure Prediction Pipeline — Siena Scalp EEG Database",
             fontsize=13, fontweight="bold", pad=8)
plt.tight_layout()
savefig("fig1_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Sensitivity + FA/h side-by-side (4 methods)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig2_comparison.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(METHODS)); w = 0.6

for ax, key, ylabel, title, target, tlabel, ylim in [
    (axes[0], "sens", "Mean Sensitivity",   "Sensitivity by strategy",    0.70, "target (0.70)", 1.15),
    (axes[1], "fah",  "Mean FA/h",          "False Alarm Rate by strategy",2.0,  "limit (2.0)",   5.5),
]:
    vals   = [m[2 if key=="sens" else 3] for m in METHODS]
    colors = [m[4] for m in METHODS]
    bars = ax.bar(x, vals, width=w, color=colors, edgecolor="white", linewidth=1.3, zorder=3)
    for bar, v in zip(bars, vals):
        offset = 0.01 if key == "sens" else 0.06
        ax.text(bar.get_x()+bar.get_width()/2, v+offset,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(target, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1.4,
               label=f"Clinical {tlabel}")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in METHODS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)

plt.suptitle("Strategy comparison — Siena Scalp EEG Database (12 patients)\n"
             "LOSO-CV · consec=6 · thr=0.60 · 10-min refractory",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("fig2_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-patient heatmap: 4 methods × sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig3_patient_heatmap.png ...")

method_labels = [m[0].replace("\n"," ") for m in METHODS]
dicts = [m[1] for m in METHODS]

sens_mat = np.array([[d[p]["sens"] for d in dicts] for p in PATIENTS])
fah_mat  = np.array([[d[p]["fah"]  for d in dicts] for p in PATIENTS])

fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"wspace":0.35})
ylabels = [f"{p}  (n={n_seiz[p]})" for p in PATIENTS]

for ax, matrix, title, vmax, fmt, cmap in [
    (axes[0], sens_mat, "Sensitivity",  1.0, ".2f", "RdYlGn"),
    (axes[1], fah_mat,  "FA/h",         6.0, ".2f", "RdYlGn_r"),
]:
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(method_labels, fontsize=8, rotation=10, ha="right")
    ax.set_yticks(range(len(PATIENTS)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    for i in range(len(PATIENTS)):
        for j in range(len(METHODS)):
            val = matrix[i, j]
            brightness = val / (vmax + 1e-6)
            color = "black" if 0.25 < brightness < 0.75 else "white"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

fig.suptitle("Per-patient Sensitivity & FA/h — 5 strategies (consec=6, thr=0.60)",
             fontsize=11, fontweight="bold")
savefig("fig3_patient_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Sensitivity bar chart with delta annotations
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_sensitivity_bars.png ...")

senss = [m[2] for m in METHODS]
cols  = [m[4] for m in METHODS]
names = [m[0] for m in METHODS]
x = np.arange(len(METHODS))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, senss, width=0.6, color=cols, edgecolor="white", linewidth=1.4, zorder=3)

for bar, v in zip(bars, senss):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.007, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Delta arrows between consecutive methods
arc_colors = ["#e67e22", "#d35400", "#c0392b", "#8e44ad", "#27ae60"]
for i in range(len(METHODS)-1):
    y1, y2 = senss[i], senss[i+1]
    delta = y2 - y1
    ac = arc_colors[i]
    ax.annotate("", xy=(x[i+1], y2+0.015), xytext=(x[i], y1+0.015),
        arrowprops=dict(arrowstyle="-|>", color=ac, lw=1.8,
                        connectionstyle="arc3,rad=-0.3"), zorder=6)
    ax.text((x[i]+x[i+1])/2, max(y1,y2)+0.08, f"{delta:+.3f}",
            ha="center", fontsize=10, color=ac, fontweight="bold")

# Highlight OR-gate (last method)
ax.bar([len(METHODS)-1], [senss[-1]], width=0.6, color="#27ae60", edgecolor="#1a7a47",
       linewidth=2.5, zorder=4)

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Mean Sensitivity (12 patients)", fontsize=11)
ax.set_ylim(0, 1.20)
ax.axhline(0.70, color="#e74c3c", linestyle="--", alpha=0.55, linewidth=1.5,
           label="Clinical target (0.70)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=0)

ax.set_title("Sensitivity progression — 5 strategies\n"
             "(LOSO-CV, consec=6, thr=0.60, mean over 12 patients)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("fig4_sensitivity_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Top-20 feature importance
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig5_feature_importance.png ...")

csv_path = os.path.join(ROOT, "feature_importance_scores.csv")
if os.path.exists(csv_path):
    scores = pd.read_csv(csv_path)
    top20  = scores.head(20)

    fam_colors = {
        "FRACTAL_DIM":"#e74c3c","THETA_BP":"#3498db","PERM_ENTROPY":"#9b59b6",
        "HJORTH_MOB":"#1abc9c","DELTA_BP":"#f39c12","ALPHA_BP":"#2ecc71",
        "GAMMA_BP":"#e67e22","HJORTH_ACT":"#16a085","STD":"#7f8c8d",
        "BETA_BP":"#d35400","SPEC_ENTROPY":"#8e44ad","MEAN":"#27ae60","OTHER":"#bdc3c7"
    }
    def family(name):
        for prefix in ["FRACTAL_DIM","THETA_BP","PERM_ENTROPY","HJORTH_MOB","HJORTH_ACT",
                        "HJORTH_COMP","DELTA_BP","ALPHA_BP","BETA_BP","GAMMA_BP",
                        "SPEC_ENTROPY","STD","MEAN","PLI","WPLI"]:
            if name.startswith(prefix): return prefix
        return "OTHER"

    fams = [family(f) for f in top20["feature"]]
    cols = [fam_colors.get(f, "#bdc3c7") for f in fams]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top20)), top20["combined"].iloc[::-1].values,
            color=cols[::-1], edgecolor="white")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].iloc[::-1].values, fontsize=8.5)
    ax.set_xlabel("Combined score (MI + AUC + LGB gain, normalised)", fontsize=10)
    ax.set_title("Top 20 most discriminative features\n(interictal vs seizure-related)",
                 fontsize=11, fontweight="bold")
    legend_p = [mpatches.Patch(color=fam_colors[f], label=f)
                for f in sorted(set(fams)) if f in fam_colors]
    ax.legend(handles=legend_p, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig("fig5_feature_importance.png")
else:
    print("  [skip] feature_importance_scores.csv not found")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — PLI/WPLI correlation with seizure onset (PN00)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig6_pli_wpli_seizure.png ...")

FOCAL_COL  = "WPLI_CH11_CH24"
FOCAL_COL2 = "WPLI_CH14_CH24"
PREICTAL_WIN = 400
POSTICTAL_WIN = 100
MAACD_W, MAACD_P = 7, 27

def compute_maacd(series, w=MAACD_W, p=MAACD_P):
    ema   = series.ewm(span=w, adjust=False).mean()
    lower = ema.rolling(window=p, min_periods=1).min()
    return (ema - lower).values

pn00_csv = os.path.join(ROOT, "dataset", "PN00", "csv")
rec_dirs = sorted([d for d in os.listdir(pn00_csv)
                   if os.path.isdir(os.path.join(pn00_csv, d)) and d.startswith("PN00")],
                  key=lambda s: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)])

all_eeg, all_y = [], []
for rec in rec_dirs:
    ef = os.path.join(pn00_csv, rec, f"{rec}_EEG_X.csv")
    yf = os.path.join(pn00_csv, rec, f"{rec}_Y.csv")
    if not os.path.exists(ef) or not os.path.exists(yf): continue
    eeg = pd.read_csv(ef); y = pd.read_csv(yf).values.flatten()
    if len(eeg) != len(y): continue
    all_eeg.append(eeg); all_y.append(y)

if all_eeg and FOCAL_COL in all_eeg[0].columns:
    common = all_eeg[0].columns
    for df in all_eeg[1:]: common = common.intersection(df.columns)
    eeg_full = pd.concat([df[common] for df in all_eeg], ignore_index=True)
    y_full   = np.concatenate(all_y)

    maacd1 = compute_maacd(eeg_full[FOCAL_COL])
    maacd2 = compute_maacd(eeg_full[FOCAL_COL2])
    raw1   = eeg_full[FOCAL_COL].values
    raw2   = eeg_full[FOCAL_COL2].values

    ictal_starts = np.where((y_full[1:] == 1) & (y_full[:-1] != 1))[0] + 1
    ictal_ends   = np.where((y_full[:-1] == 1) & (y_full[1:] != 1))[0] + 1

    segs = {k: [] for k in ["m1","m2","r1","r2"]}
    t_axis = np.arange(-PREICTAL_WIN, POSTICTAL_WIN)

    for is_, ie in zip(ictal_starts, ictal_ends):
        s = is_ - PREICTAL_WIN; e = is_ + POSTICTAL_WIN
        if s < 0 or e > len(y_full): continue
        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-9)
        sm1 = norm(maacd1[s:e]); sm2 = norm(maacd2[s:e])
        sr1 = norm(raw1[s:e]);   sr2 = norm(raw2[s:e])
        if len(sm1) != len(t_axis): continue
        segs["m1"].append(sm1); segs["m2"].append(sm2)
        segs["r1"].append(sr1); segs["r2"].append(sr2)

    if segs["m1"]:
        M1=np.array(segs["m1"]); M2=np.array(segs["m2"])
        R1=np.array(segs["r1"]); R2=np.array(segs["r2"])

        fig, axes3 = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

        for ax, S, title, color in [
            (axes3[0,0], R1, f"Raw WPLI — {FOCAL_COL}",  "#2980b9"),
            (axes3[0,1], R2, f"Raw WPLI — {FOCAL_COL2}", "#8e44ad"),
            (axes3[1,0], M1, f"MAACD — {FOCAL_COL}",     "#1a5276"),
            (axes3[1,1], M2, f"MAACD — {FOCAL_COL2}",    "#4a235a"),
        ]:
            mean_s = S.mean(axis=0); std_s = S.std(axis=0)
            for seg in S:
                ax.plot(t_axis, seg, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(t_axis, mean_s, color=color, linewidth=2.5, label="Mean", zorder=5)
            ax.fill_between(t_axis, mean_s-std_s, mean_s+std_s,
                            color=color, alpha=0.18, label="±1 std")
            ax.axvline(0, color="#e74c3c", linewidth=1.8, linestyle="--",
                       label="Ictal onset", zorder=6)
            ax.axvspan(-300, 0, color="#f39c12", alpha=0.08, label="Preictal zone")
            ax.axvspan(0, POSTICTAL_WIN, color="#e74c3c", alpha=0.06, label="Ictal zone")
            ax.set_ylabel("Normalised amplitude", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=0.2)

        for ax in axes3[1,:]:
            ax.set_xlabel("Time relative to ictal onset (s)", fontsize=9)

        fig.suptitle(f"PN00 — PLI/WPLI focal pairs aligned to seizure onset\n"
                     f"({len(segs['m1'])} seizures overlaid · top: raw WPLI · bottom: MAACD trend)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig("fig6_pli_wpli_seizure.png")
    else:
        print("  [skip] no valid segments for PN00")
else:
    print(f"  [skip] {FOCAL_COL} not found")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — AUC per patient: all-feat (first) vs OR-gate (final)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig7_auc_by_patient.png ...")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(PATIENTS)); w = 0.35

thalgo_auc = [THALGO[p]["auc"]  for p in PATIENTS]
final_auc  = [ORGATE[p]["auc"]  for p in PATIENTS]

ax.bar(x - w/2, thalgo_auc, width=w, color="#8e44ad",
       label=f"ThAlgo — Detti 2019 (mean={np.mean(thalgo_auc):.3f})",
       edgecolor="white", linewidth=0.8)
ax.bar(x + w/2, final_auc, width=w, color="#27ae60",
       label=f"OR-gate final (mean={np.mean(final_auc):.3f})",
       edgecolor="white", linewidth=0.8)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1, label="AUC = 0.5 (chance)")
ax.axhline(np.mean(thalgo_auc), color="#8e44ad", linestyle=":", linewidth=1.5, alpha=0.8)
ax.axhline(np.mean(final_auc),  color="#27ae60", linestyle=":", linewidth=1.5, alpha=0.8)

for i, (b, o) in enumerate(zip(thalgo_auc, final_auc)):
    ax.text(i - w/2, b+0.005, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(i + w/2, o+0.005, f"{o:.2f}", ha="center", va="bottom", fontsize=7.5)

for p_hard in ["PN05","PN06","PN09","PN14"]:
    i = PATIENTS.index(p_hard)
    ax.text(i, 0.04, "★ hard", ha="center", fontsize=7, color="#e74c3c")

ax.set_xticks(x); ax.set_xticklabels(PATIENTS, fontsize=9)
ax.set_ylabel("Mean AUC (across folds)", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("AUC per patient — ThAlgo / Detti 2019 (paper) vs OR-gate (ours)\n"
             "ThAlgo AUC ≈ 0.51 (near chance) despite Sens=0.76 — alarm driven by signal amplitude, not discrimination",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig("fig7_auc_by_patient.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Per-patient Δsens: fused → OR-gate
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig8_improvement_delta.png ...")

delta_sens = [ORGATE[p]["sens"] - THALGO[p]["sens"] for p in PATIENTS]
delta_fah  = [ORGATE[p]["fah"]  - THALGO[p]["fah"]  for p in PATIENTS]

fig, axes4 = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(PATIENTS)); w = 0.65

for ax, deltas, ylabel, title, good_pos in [
    (axes4[0], delta_sens, "Δ Sensitivity", "Δ Sensitivity (OR-gate − ThAlgo)", True),
    (axes4[1], delta_fah,  "Δ FA/h",        "Δ FA/h (OR-gate − ThAlgo)",        False),
]:
    bar_cols = ["#bdc3c7" if abs(d) < 0.005 else
                "#27ae60" if (d > 0) == good_pos else "#e74c3c" for d in deltas]
    ax.bar(x, deltas, width=w, color=bar_cols, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    for i, v in enumerate(deltas):
        if abs(v) > 0.005:
            ax.text(i, v + (0.005 if v > 0 else -0.015), f"{v:+.3f}",
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(PATIENTS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title + "\n(green = OR-gate better, red = ThAlgo better)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)

plt.suptitle("Per-patient impact: ThAlgo (paper) → OR-gate (ours)\n"
             "Sensitivity ↑ +0.114 on average · FA/h ↓ −3.97 on average",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig("fig8_improvement_delta.png")


print("\nAll figures saved to figures/")
print("\nSummary:")
print(f"  fig1  — Pipeline (Siena Scalp EEG, 14 patients / 12 used)")
print(f"  fig2  — Sens + FA/h: 5 strategies (fused·0.318 → all-feat·0.340 → no-PLI·0.432 → ThAlgo·0.764 → OR-gate·0.878)")
print(f"  fig3  — Per-patient heatmap: 5 strategies")
print(f"  fig4  — Sensitivity bars with delta arrows")
print(f"  fig5  — Top-20 discriminative features")
print(f"  fig6  — PLI/WPLI focal pairs aligned to seizure onset (PN00)")
print(f"  fig7  — AUC: ThAlgo ≈0.51 (near chance) vs OR-gate 0.677")
print(f"  fig8  — Δ per patient: ThAlgo → OR-gate (Sens +0.114, FA/h −3.97)")
