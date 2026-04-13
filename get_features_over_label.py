import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

dataset_path = "./dataset/"
patients = sorted([p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))])

# 20 feature families — one subplot each
EEG_FEATURE_FAMILIES = [
    "DELTA_BP", "THETA_BP", "ALPHA_BP", "BETA_BP", "GAMMA_BP",
    "DELTA_RBP", "THETA_RBP", "ALPHA_RBP", "BETA_RBP", "GAMMA_RBP",
    "HJORTH_ACT", "HJORTH_MOB", "HJORTH_COMP",
    "SPEC_ENTROPY", "MEAN", "STD",
    "PERM_ENTROPY", "FRACTAL_DIM",
    "PLI", "WPLI",
]


def _sort_key(name):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name)]


def load_patient(patient):
    """
    Read per-recording CSVs and concatenate with column intersection.
    Avoids the ParserError caused by patient-level CSVs written from
    recordings with different channel counts.
    """
    csv_path = os.path.join(dataset_path, patient, "csv")
    rec_dirs = sorted(
        [d for d in os.listdir(csv_path)
         if os.path.isdir(os.path.join(csv_path, d)) and d.startswith(patient)],
        key=_sort_key,
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
        y   = pd.read_csv(y_f).values.flatten()
        if len(eeg) != len(y):
            print(f"  [warn] length mismatch in {rec}, skipping")
            continue
        eeg_parts.append(eeg)
        y_parts.append(y)

    if not eeg_parts:
        return None, None

    # Align columns (intersection) across recordings
    common_cols = eeg_parts[0].columns
    for df in eeg_parts[1:]:
        common_cols = common_cols.intersection(df.columns)

    features = pd.concat([df[common_cols] for df in eeg_parts], ignore_index=True)
    labels   = np.concatenate(y_parts)
    return features, labels


def getFeaturesOverLabel():
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2a2a2a",
        "axes.edgecolor": "#4f4f4f",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 8,
    })

    os.makedirs("figures", exist_ok=True)

    for patient in patients:
        print("Processing", patient)

        features, labels = load_patient(patient)
        if features is None:
            print("  No data for", patient)
            continue

        # z-score normalisation
        features = (features - features.mean()) / (features.std() + 1e-10)

        # Average across channels for each feature family
        meaned = {}
        for family in EEG_FEATURE_FAMILIES:
            cols = [c for c in features.columns if c.startswith(family + "_CH")]
            if cols:
                meaned[family] = features[cols].mean(axis=1).values

        if not meaned:
            print("  No matching family columns for", patient)
            continue

        # 4 rows × 5 cols = 20 subplots
        n = len(meaned)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        colors = plt.cm.tab20(np.linspace(0, 1, n))

        fig, axs = plt.subplots(nrows, ncols, figsize=(22, nrows * 3), squeeze=False)
        axs = axs.flatten()

        x = np.arange(len(labels))

        for k, (family, values) in enumerate(meaned.items()):
            ax = axs[k]

            # Shade seizure regions as background
            ax.fill_between(x, values.min(), values.max(),
                            where=(labels == 2),
                            color="#ff9900", alpha=0.25, linewidth=0)
            ax.fill_between(x, values.min(), values.max(),
                            where=(labels == 1),
                            color="#ff3333", alpha=0.35, linewidth=0)

            ax.plot(values, color=colors[k], linewidth=0.6)

            ax2 = ax.twinx()
            ax2.plot(labels, color="white", linewidth=0.8, alpha=0.7)
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(["inter", "ictal", "pre"], fontsize=5, color="white")
            ax2.set_ylim(-0.5, 4)

            ax.set_title(family, fontsize=9, pad=3)
            ax.set_xlabel("window", fontsize=7)
            ax.grid(True, alpha=0.15)
            ax.tick_params(labelsize=6)

        # Remove unused axes
        for k in range(n, len(axs)):
            fig.delaxes(axs[k])

        # Legend
        legend_patches = [
            mpatches.Patch(color="#ff9900", alpha=0.6, label="preictal (label=2)"),
            mpatches.Patch(color="#ff3333", alpha=0.6, label="ictal (label=1)"),
        ]
        fig.legend(handles=legend_patches, loc="lower right",
                   framealpha=0.3, fontsize=8)

        fig.suptitle(
            f"{patient} — mean feature value per family (z-scored, averaged across channels)",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()

        save_path = os.path.join("figures", f"{patient}_features_by_family.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved {save_path}")


getFeaturesOverLabel()
