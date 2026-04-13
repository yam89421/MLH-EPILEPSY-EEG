print("bouuu")
"""
Feature Importance Analysis
============================
Loads all patient data, pools interictal vs peri-ictal windows, then ranks
features by three complementary criteria:
  1. AUC (per-feature logistic regression)
  2. Mutual Information
  3. LightGBM feature importance (gain)

Saves:
  - feature_importance_scores.csv   — full ranked table
  - feature_importance_top40.png    — bar chart of top 40
  - feature_importance_by_type.png  — mean score grouped by feature family
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# ── reuse data loading from loso script ──────────────────────────────────────
DATASET_PATH = "./dataset/"

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])

PREICTAL_SEC         = 300
POSTICTAL_BUFFER_SEC = 1000
MAX_INTERICTAL_SEC   = 3600
STEP_SEC             = 1.0


import re as _re

def _recording_sort_key(name):
    return [int(c) if c.isdigit() else c for c in _re.split(r'(\d+)', name)]


def compute_maacd_features(eeg, w=7, p=27):
    sync_cols = [c for c in eeg.columns if
                 c.startswith("PLI_CH") or c.startswith("WPLI_CH")]
    new_cols = {}
    for col in sync_cols:
        vals  = eeg[col]
        ema   = vals.ewm(span=w, adjust=False).mean()
        lower = ema.rolling(window=p, min_periods=1).min()
        new_cols[f"MAACD_{col}"] = (ema - lower).values
    return eeg.assign(**new_cols)


def load_top_nonsync(k=30, csv="feature_importance_scores.csv"):
    """Top-k non-sync features by combined importance (from previous run)."""
    if not os.path.exists(csv):
        return []
    df = pd.read_csv(csv)
    df = df[~df["feature"].str.contains("PLI_CH|WPLI_CH|MAACD", regex=True)]
    df = df.sort_values("combined", ascending=False)
    return list(df["feature"].iloc[:k])


TOP_NONSYNC = load_top_nonsync()
print(f"Will compute MAACD on {len(TOP_NONSYNC)} top non-sync features.")


def compute_maacd_nonsync(eeg, top_features, w=7, p=27):
    """MAACD on top discriminative non-sync features."""
    new_cols = {}
    for col in top_features:
        if col not in eeg.columns:
            continue
        vals  = eeg[col]
        ema   = vals.ewm(span=w, adjust=False).mean()
        lower = ema.rolling(window=p, min_periods=1).min()
        new_cols[f"MAACD_FEAT_{col}"] = (ema - lower).values
    return eeg.assign(**new_cols)


def load_patient(patient):
    """Load per-recording CSVs, compute MAACD per recording, then concatenate."""
    csv_path = os.path.join(DATASET_PATH, patient, "csv")

    rec_dirs = sorted(
        [d for d in os.listdir(csv_path)
         if os.path.isdir(os.path.join(csv_path, d)) and d.startswith(patient)],
        key=_recording_sort_key,
    )
    if not rec_dirs:
        return None, None

    eeg_parts, y_parts = [], []
    for rec in rec_dirs:
        eeg_f = os.path.join(csv_path, rec, f"{rec}_EEG_X.csv")
        y_f   = os.path.join(csv_path, rec, f"{rec}_Y.csv")
        if not os.path.exists(eeg_f) or not os.path.exists(y_f):
            continue
        eeg = pd.read_csv(eeg_f)
        eeg = eeg.loc[:, eeg.std() > 0]
        y   = pd.read_csv(y_f).values.flatten()
        if len(eeg) != len(y):
            continue
        eeg = compute_maacd_features(eeg)
        eeg = compute_maacd_nonsync(eeg, TOP_NONSYNC)
        eeg_parts.append(eeg)
        y_parts.append(y)

    if not eeg_parts:
        return None, None

    common_cols = eeg_parts[0].columns
    for df in eeg_parts[1:]:
        common_cols = common_cols.intersection(df.columns)

    eeg_all = pd.concat([df[common_cols] for df in eeg_parts], ignore_index=True)
    y_all   = np.concatenate(y_parts)
    return eeg_all, y_all


def pool_data(patients):
    """
    Pool data across all patients: keep peri-ictal (label>0) and up to
    MAX_INTERICTAL_SEC of interictal before each seizure. Binary labels.
    """
    frames = []
    for p in patients:
        eeg, y = load_patient(p)
        if eeg is None:
            continue

        n = len(y)
        max_inter = int(MAX_INTERICTAL_SEC / STEP_SEC)
        post_buf  = int(POSTICTAL_BUFFER_SEC / STEP_SEC)
        keep = np.zeros(n, dtype=bool)

        # find seizure onsets (label transitions 0→>0)
        onsets = np.where((y[1:] > 0) & (y[:-1] == 0))[0] + 1
        ends   = np.where((y[:-1] > 0) & (y[1:] == 0))[0] + 1

        for onset in onsets:
            start = max(0, onset - max_inter)
            keep[start:onset] = True          # interictal before seizure

        for i in range(len(onsets)):
            start = onsets[i]
            end   = ends[i] if i < len(ends) else n
            keep[start:end] = True            # peri-ictal

        # mask postictal buffer
        for end in ends:
            post_end = min(n, end + post_buf)
            keep[end:post_end] = False

        sub = eeg[keep].copy()
        sub["_y"]       = (y[keep] > 0).astype(int)
        sub["_patient"] = p
        frames.append(sub)
        print(f"  {p}: {keep.sum()} windows kept  "
              f"({int((y[keep] > 0).sum())} peri-ictal, "
              f"{int((y[keep] == 0).sum())} interictal)")

    df = pd.concat(frames, ignore_index=True)
    return df


# ── main ─────────────────────────────────────────────────────────────────────

print("Loading data...")
df = pool_data(patients)

y_all = df["_y"].values
feat_cols = [c for c in df.columns if not c.startswith("_")]
X_all = df[feat_cols].fillna(0).values

print(f"\nTotal: {len(df)} windows | {y_all.sum()} peri-ictal, {(y_all==0).sum()} interictal")
print(f"Features: {len(feat_cols)}")

# ── 1. Mutual Information ─────────────────────────────────────────────────────
print("\nComputing Mutual Information...")
mi = mutual_info_classif(X_all, y_all, random_state=42)

# ── 2. Per-feature AUC (logistic regression, per-patient normalised) ──────────
print("Computing per-feature AUC...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

auc_scores = []
for i in range(X_scaled.shape[1]):
    xi = X_scaled[:, i].reshape(-1, 1)
    try:
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(xi, y_all)
        proba = clf.predict_proba(xi)[:, 1]
        auc = roc_auc_score(y_all, proba)
        auc = max(auc, 1 - auc)   # direction-agnostic
    except Exception:
        auc = 0.5
    auc_scores.append(auc)

# ── 3. LightGBM gain importance ───────────────────────────────────────────────
print("Computing LightGBM importance...")
n_neg = (y_all == 0).sum()
n_pos = (y_all == 1).sum()
model = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    scale_pos_weight=n_neg / max(n_pos, 1),
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X_scaled, y_all)
lgb_imp = model.feature_importances_.astype(float)
lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-12)   # normalise to [0,1]

# ── Combined score ────────────────────────────────────────────────────────────
mi_norm  = mi / (mi.max() + 1e-12)
auc_norm = (np.array(auc_scores) - 0.5) / 0.5   # map [0.5,1] → [0,1]
auc_norm = np.clip(auc_norm, 0, 1)

combined = (mi_norm + auc_norm + lgb_imp) / 3.0

scores_df = pd.DataFrame({
    "feature":  feat_cols,
    "MI":       mi,
    "AUC":      auc_scores,
    "LGB_gain": lgb_imp,
    "combined": combined,
}).sort_values("combined", ascending=False).reset_index(drop=True)

out_csv = "feature_importance_scores_with_maacdfeat.csv"
scores_df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv} ({len(scores_df)} features)")

# ── Feature family grouping ───────────────────────────────────────────────────
def family(name):
    for prefix in ["MAACD_FEAT", "MAACD_WPLI", "MAACD_PLI", "WPLI", "PLI",
                   "DELTA_RBP", "THETA_RBP", "ALPHA_RBP", "BETA_RBP", "GAMMA_RBP",
                   "DELTA_BP",  "THETA_BP",  "ALPHA_BP",  "BETA_BP",  "GAMMA_BP",
                   "HJORTH_ACT", "HJORTH_MOB", "HJORTH_COMP",
                   "SPEC_ENTROPY", "PERM_ENTROPY", "FRACTAL_DIM",
                   "MEAN", "STD"]:
        if name.startswith(prefix):
            return prefix
    return "OTHER"

scores_df["family"] = scores_df["feature"].apply(family)

family_mean = (
    scores_df.groupby("family")["combined"]
    .mean()
    .sort_values(ascending=False)
)

# ── Plot 1: Top 40 features ───────────────────────────────────────────────────
top40 = scores_df.head(40)

fig, ax = plt.subplots(figsize=(14, 10))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top40)))[::-1]
bars = ax.barh(top40["feature"][::-1], top40["combined"][::-1], color=colors[::-1])
ax.set_xlabel("Combined importance score (MI + AUC + LGB gain, normalised)", fontsize=11)
ax.set_title("Top 40 most discriminative features\n(interictal vs peri-ictal)", fontsize=13)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance_top40_with_maacdfeat.png", dpi=150)
plt.close()
print("Saved feature_importance_top40.png")

# ── Plot 2: By feature family ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: mean combined score per family
family_colors = plt.cm.tab20(np.linspace(0, 1, len(family_mean)))
axes[0].barh(family_mean.index[::-1], family_mean.values[::-1], color=family_colors)
axes[0].set_xlabel("Mean combined importance", fontsize=11)
axes[0].set_title("Mean importance by feature family", fontsize=12)
axes[0].grid(axis="x", alpha=0.3)

# Right: AUC vs MI scatter, coloured by LGB gain, labelled by family
fam_auc = scores_df.groupby("family")["AUC"].mean()
fam_mi  = scores_df.groupby("family")["MI"].mean()
fam_lgb = scores_df.groupby("family")["LGB_gain"].mean()

sc = axes[1].scatter(
    fam_mi.values, fam_auc.values,
    c=fam_lgb.values, cmap="RdYlGn", s=150, zorder=3,
    vmin=0, vmax=fam_lgb.max()
)
for fam in fam_mi.index:
    axes[1].annotate(
        fam, (fam_mi[fam], fam_auc[fam]),
        fontsize=7, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points"
    )
plt.colorbar(sc, ax=axes[1], label="LGB gain")
axes[1].set_xlabel("Mean Mutual Information", fontsize=11)
axes[1].set_ylabel("Mean AUC", fontsize=11)
axes[1].set_title("MI vs AUC per family\n(colour = LGB gain)", fontsize=12)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("feature_importance_by_type_with_maacdfeat.png", dpi=150)
plt.close()
print("Saved feature_importance_by_type.png")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("TOP 20 FEATURES")
print(f"{'Rank':<5} {'Feature':<35} {'AUC':>6} {'MI':>8} {'LGB':>8} {'Score':>8}")
print("-"*65)
for i, row in scores_df.head(20).iterrows():
    print(f"{i+1:<5} {row['feature']:<35} {row['AUC']:>6.3f} "
          f"{row['MI']:>8.4f} {row['LGB_gain']:>8.4f} {row['combined']:>8.4f}")

print("\n" + "="*45)
print("MEAN SCORE BY FEATURE FAMILY")
print(f"{'Family':<20} {'Mean score':>12}")
print("-"*35)
for fam, score in family_mean.items():
    print(f"{fam:<20} {score:>12.4f}")
