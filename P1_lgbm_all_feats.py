"""
Patient-Specific Seizure Prediction Pipeline
=============================================
Evaluation: Leave-One-Seizure-Out (LOSO) per patient
  - For each patient with N seizures:
      - For k in 1..N: train on seizures 1..k-1, k+1..N (+ their interictal context),
        test on seizure k and its pre/post context.
  - Aggregate sensitivity and false-alarm rate across folds and patients.

This is the standard evaluation protocol in the seizure prediction literature
because EEG is highly subject-specific: cross-patient models have near-zero
sensitivity on unseen patients (the LOPO experiment confirmed this).

Key design choices:
  1. Interictal selection: 2h before each seizure's preictal onset (to avoid
     very long interictal that drift the distribution).
  2. Post-ictal exclusion: 1h after ictal end is masked (transitional activity).
  3. Feature normalisation: per-patient StandardScaler fitted only on train fold.
  4. Class weighting: inverse class frequency.
  5. Post-processing: majority-vote smoothing over 7 windows.
  6. Alarm rule: fire alarm if >=4 consecutive positives within the preictal window.
  7. Summary per patient: mean sensitivity, mean FA/h across LOSO folds.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier  # works without lgb installed
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
DATASET_PATH = "./dataset/"
STEP_SEC     = 1.0         # 6s window, 1s step (Detti et al. 2019)

# ── Parameters matched to Detti et al. (2019) ──
PREICTAL_SEC         = 300   # prediction interval: 5 min (paper uses 150/200/300s; we use max)
POSTICTAL_BUFFER_SEC = 1000  # discard 1000s after ictal end (paper value)
MAX_INTERICTAL_SEC   = 3600  # keep at most 1h of interictal before each seizure (paper value)
MIN_INTER_SEIZURE_SEC = 1300 # ignore seizures separated by less than this (paper value)

SMOOTHING_K  = 10          # majority-vote window in windows
ALARM_CONSEC = 6           # consecutive positives to fire alarm (~18s at 3s step)
REFRACTORY_WINDOWS = int(10 * 60 / STEP_SEC)  # 10-min refractory period after alarm

# Feature mode:
#   "all"        — use all available features
#   "maacd_only" — MAACD of PLI/WPLI only (paper's approach; PLI/WPLI are now alpha-band diff)
FEATURE_MODE = "all"

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])


# ──────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────

def compute_maacd_features(eeg: pd.DataFrame, w: int = 7, p: int = 27) -> pd.DataFrame:
    """
    Implements the MAACD feature from Detti et al. (2019) on PLI and WPLI columns.

    For each PLI_CHx / WPLI_CHx time series (across windows, temporal order):
      T(t) = EMA(span=w, adjust=False) of the sync measure   [trend]
      L(t) = rolling min of T over the past p windows        [lower limit]
      M(t) = T(t) - L(t)                                     [MAACD = elevation]

    Parameters (from paper): w=7, p=27.
    Uses pandas ewm (adjust=False) which matches the paper's recursive EMA:
      T(t) = alpha*f(t) + (1-alpha)*T(t-1),  alpha = 2/(w+1) = 0.25
    """
    sync_cols = [c for c in eeg.columns if
                 c.startswith("PLI_CH") or c.startswith("WPLI_CH") or
                 c.startswith("DIFF_PLI_CH") or c.startswith("DIFF_WPLI_CH")]
    new_cols = {}
    for col in sync_cols:
        vals  = eeg[col]
        ema   = vals.ewm(span=w, adjust=False).mean()
        lower = ema.rolling(window=p, min_periods=1).min()
        new_cols[f"MAACD_{col}"] = (ema - lower).values
    return eeg.assign(**new_cols)


def _recording_sort_key(name: str) -> list:
    """Natural numeric sort so PN10-2 < PN10-10."""
    import re
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name)]


def load_patient(patient: str):
    """
    Load EEG features + labels for a patient by reading each per-recording CSV
    individually, computing MAACD on each recording separately (correct temporal
    context), then concatenating in chronological order.
    Returns (features_df, labels_array) or (None, None).
    """
    csv_path = os.path.join(DATASET_PATH, patient, "csv")

    # Find per-recording subdirectories and sort chronologically
    rec_dirs = sorted(
        [d for d in os.listdir(csv_path)
         if os.path.isdir(os.path.join(csv_path, d)) and d.startswith(patient)],
        key=_recording_sort_key,
    )

    if not rec_dirs:
        return None, None

    eeg_parts = []
    y_parts   = []

    for rec in rec_dirs:
        rec_path = os.path.join(csv_path, rec)
        eeg_f = os.path.join(rec_path, f"{rec}_EEG_X.csv")
        y_f   = os.path.join(rec_path, f"{rec}_Y.csv")

        if not os.path.exists(eeg_f) or not os.path.exists(y_f):
            continue

        eeg = pd.read_csv(eeg_f)
        eeg = eeg.loc[:, eeg.std() > 0]
        y   = pd.read_csv(y_f).values.flatten()

        if len(eeg) != len(y):
            print(f"  [warn] length mismatch in {rec}, skipping")
            continue

        # Compute MAACD per recording — correct: EMA stays within one continuous session
        eeg = compute_maacd_features(eeg)

        eeg_parts.append(eeg)
        y_parts.append(y)

    if not eeg_parts:
        return None, None

    # Align columns across recordings (take intersection)
    common_cols = eeg_parts[0].columns
    for df in eeg_parts[1:]:
        common_cols = common_cols.intersection(df.columns)

    eeg_all = pd.concat([df[common_cols] for df in eeg_parts], ignore_index=True)
    y_all   = np.concatenate(y_parts)

    return eeg_all, y_all


# ──────────────────────────────────────────
# SEIZURE SEGMENTATION
# ──────────────────────────────────────────

def find_seizure_episodes(labels):
    """
    Identify continuous blocks of label==1 (ictal) or label==2 (preictal).
    Applies Detti et al. filters:
      - preictal trimmed to last PREICTAL_SEC seconds before ictal onset
      - episodes closer than MIN_INTER_SEIZURE_SEC to the previous one are dropped
    Returns list of dicts with keys: preictal_start, ictal_start, ictal_end.
    """
    n = len(labels)
    raw_episodes = []
    i = 0
    while i < n:
        if labels[i] == 2:
            pre_start = i
            while i < n and labels[i] == 2:
                i += 1
            ictal_start = i
            while i < n and labels[i] == 1:
                i += 1
            ictal_end = i
            raw_episodes.append({
                "preictal_start": pre_start,
                "ictal_start":    ictal_start,
                "ictal_end":      ictal_end,
            })
        else:
            i += 1

    # Filter: drop episodes too close to the previous one (< MIN_INTER_SEIZURE_SEC)
    min_gap_windows = int(MIN_INTER_SEIZURE_SEC / STEP_SEC)
    filtered = []
    for ep in raw_episodes:
        if filtered:
            gap = ep["preictal_start"] - filtered[-1]["ictal_end"]
            if gap < min_gap_windows:
                continue
        filtered.append(ep)

    # Trim preictal to last PREICTAL_SEC seconds before ictal onset
    preictal_windows = int(PREICTAL_SEC / STEP_SEC)
    episodes = []
    for ep in filtered:
        trimmed_pre = max(ep["preictal_start"], ep["ictal_start"] - preictal_windows)
        episodes.append({
            "preictal_start": trimmed_pre,
            "ictal_start":    ep["ictal_start"],
            "ictal_end":      ep["ictal_end"],
        })

    return episodes


def build_fold_datasets(eeg: pd.DataFrame, labels: np.ndarray, episodes: list, k: int):
    """
    For fold k (0-indexed):
      - test = episode k's preictal (trimmed to PREICTAL_SEC) + ictal + interictal context
      - train = all other episodes + their surrounding interictal

    Returns X_train, y_train, X_test, y_test (binary: 0=interictal, 1=peri-ictal)
    """
    n = len(labels)
    max_inter_windows  = int(MAX_INTERICTAL_SEC   / STEP_SEC)
    postictal_windows  = int(POSTICTAL_BUFFER_SEC / STEP_SEC)

    def get_episode_mask(ep_idx):
        """Boolean mask covering the peri-ictal + nearby interictal of one episode."""
        ep    = episodes[ep_idx]
        pre   = ep["preictal_start"]
        i_end = ep["ictal_end"]
        inter_start = max(0, pre - max_inter_windows)
        post_end    = min(n, i_end + postictal_windows)
        mask = np.zeros(n, dtype=bool)
        mask[inter_start:post_end] = True
        return mask

    test_mask  = get_episode_mask(k)
    train_mask = np.zeros(n, dtype=bool)
    for j, ep in enumerate(episodes):
        if j != k:
            train_mask |= get_episode_mask(j)

    # Remove overlap
    train_mask &= ~test_mask

    def make_Xy(mask):
        X = eeg.values[mask]
        y_raw = labels[mask]
        y_bin = (y_raw > 0).astype(int)
        return X, y_bin

    X_tr, y_tr = make_Xy(train_mask)
    X_te, y_te = make_Xy(test_mask)
    return X_tr, y_tr, X_te, y_te


# ──────────────────────────────────────────
# CLASSIFIER
# ──────────────────────────────────────────

def build_model(pos_weight: float):
    """
    Gradient Boosting works well for tabular EEG features.
    LightGBM is preferred if available; this falls back to sklearn.
    """
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        # sklearn GradientBoosting does not support scale_pos_weight natively;
        # compensate via sample_weight in fit()
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


# ──────────────────────────────────────────
# POST-PROCESSING
# ──────────────────────────────────────────

def smooth_predictions(proba: np.ndarray, k: int = SMOOTHING_K, threshold: float = 0.55) -> np.ndarray:
    smoothed = np.convolve((proba > threshold).astype(float), np.ones(k) / k, mode="same")
    return (smoothed > 0.5).astype(int)


# ──────────────────────────────────────────
# SEIZURE-LEVEL EVALUATION
# ──────────────────────────────────────────

def seizure_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels_raw: np.ndarray):
    """
    Compute seizure-level sensitivity and false-alarm rate per hour.
    Uses the raw 3-class labels to locate true preictal windows precisely.

    A seizure is "detected" if >= ALARM_CONSEC consecutive positive predictions
    occur within the preictal window (label==2).

    False alarms are counted in strictly interictal windows (label==0).
    """
    preictal_mask  = (labels_raw == 2)
    interictal_mask = (labels_raw == 0)

    # Identify preictal blocks
    blocks = []
    in_block = False
    for i in range(len(labels_raw)):
        if preictal_mask[i] and not in_block:
            start = i
            in_block = True
        elif not preictal_mask[i] and in_block:
            blocks.append((start, i))
            in_block = False
    if in_block:
        blocks.append((start, len(labels_raw)))

    detected = 0
    for (bs, be) in blocks:
        window_preds = y_pred[bs:be]
        consec = 0
        for val in window_preds:
            consec = consec + 1 if val == 1 else 0
            if consec >= ALARM_CONSEC:
                detected += 1
                break

    n_seizures = len(blocks)
    sensitivity = detected / max(n_seizures, 1)

    # False alarms in interictal with refractory period
    false_alarms = 0
    consec = 0
    refractory_remaining = 0
    for i in range(len(labels_raw)):
        if refractory_remaining > 0:
            refractory_remaining -= 1
            consec = 0
            continue
        if interictal_mask[i] and y_pred[i] == 1:
            consec += 1
            if consec == ALARM_CONSEC:
                false_alarms += 1
                consec = 0
                refractory_remaining = REFRACTORY_WINDOWS
        else:
            consec = 0

    interictal_hours = interictal_mask.sum() * STEP_SEC / 3600
    fpr_per_hour = false_alarms / max(interictal_hours, 1e-3)

    return sensitivity, fpr_per_hour, n_seizures


# ──────────────────────────────────────────
# LOSO CV FOR ONE PATIENT
# ──────────────────────────────────────────

def select_features(eeg: pd.DataFrame) -> pd.DataFrame:
    """Filter columns according to FEATURE_MODE."""
    if FEATURE_MODE == "maacd_only":
        cols = [c for c in eeg.columns if c.startswith("MAACD_")]
        if not cols:
            raise ValueError("FEATURE_MODE='maacd_only' but no MAACD_* columns found.")
        return eeg[cols]
    if FEATURE_MODE == "maacd_diff":
        cols = [c for c in eeg.columns
                if c.startswith("MAACD_DIFF_PLI_CH") or c.startswith("MAACD_DIFF_WPLI_CH")]
        if not cols:
            raise ValueError("FEATURE_MODE='maacd_diff' but no MAACD_DIFF_* columns found.")
        return eeg[cols]
    return eeg  # "all" mode


def loso_patient(patient: str, verbose: bool = True):
    eeg, labels = load_patient(patient)
    if eeg is None:
        return None

    eeg = select_features(eeg)
    if verbose:
        print(f"  Features used: {len(eeg.columns)} ({FEATURE_MODE} mode)")

    episodes = find_seizure_episodes(labels)
    n_seizures = len(episodes)

    if n_seizures < 2:
        if verbose:
            print(f"  [skip] {patient}: only {n_seizures} seizure(s) — need >=2 for LOSO")
        return None

    if verbose:
        counts = {v: int((labels == v).sum()) for v in [0, 1, 2]}
        print(f"\n  {patient}: {n_seizures} seizures | "
              f"interictal={counts[0]}, preictal={counts[2]}, ictal={counts[1]}")

    fold_results = []
    for k in range(n_seizures):
        X_tr, y_tr, X_te, y_te = build_fold_datasets(eeg, labels, episodes, k)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        n_neg = (y_tr == 0).sum()
        n_pos = (y_tr == 1).sum()
        pos_w = n_neg / max(n_pos, 1)

        model = build_model(pos_w)

        # For sklearn GBT: pass sample_weight manually
        import inspect
        fit_params = {}
        if "sample_weight" in inspect.signature(model.fit).parameters:
            sw = np.where(y_tr == 1, pos_w, 1.0)
            fit_params["sample_weight"] = sw

        model.fit(X_tr, y_tr, **fit_params)

        proba  = model.predict_proba(X_te)[:, 1]
        y_pred = smooth_predictions(proba)

        # Get raw labels for the test fold (for seizure-level eval)
        test_ep = episodes[k]
        pre_s   = test_ep["preictal_start"]
        i_end   = test_ep["ictal_end"]
        inter_s = max(0, pre_s - int(MAX_INTERICTAL_SEC / STEP_SEC))
        post_e  = min(len(labels), i_end + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
        labels_test_raw = labels[inter_s:post_e]

        sensitivity, fpr_per_hour, _ = seizure_level_metrics(y_te, y_pred, labels_test_raw)

        if len(np.unique(y_te)) > 1:
            auc = roc_auc_score(y_te, proba)
        else:
            auc = float("nan")

        bal_acc = balanced_accuracy_score(y_te, y_pred)

        if verbose:
            print(f"    Fold {k+1}/{n_seizures}: AUC={auc:.3f}  "
                  f"BalAcc={bal_acc:.3f}  Sens={sensitivity:.2f}  FA/h={fpr_per_hour:.2f}  "
                  f"[train: {len(X_tr)} | test: {len(X_te)}]")

        fold_results.append({
            "fold": k,
            "auc": auc,
            "bal_acc": bal_acc,
            "sensitivity": sensitivity,
            "fpr_per_hour": fpr_per_hour,
        })

    if not fold_results:
        return None

    df_r = pd.DataFrame(fold_results)
    mean_r = {
        "patient": patient,
        "n_seizures": n_seizures,
        "mean_auc": df_r["auc"].mean(),
        "mean_bal_acc": df_r["bal_acc"].mean(),
        "mean_sensitivity": df_r["sensitivity"].mean(),
        "mean_fpr_per_hour": df_r["fpr_per_hour"].mean(),
    }
    if verbose:
        print(f"  ── Patient mean: AUC={mean_r['mean_auc']:.3f}  "
              f"BalAcc={mean_r['mean_bal_acc']:.3f}  "
              f"Sens={mean_r['mean_sensitivity']:.3f}  "
              f"FA/h={mean_r['mean_fpr_per_hour']:.2f}")
    return mean_r


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running patient-specific LOSO-CV on {len(patients)} patients\n")
    all_results = []

    for p in patients:
        r = loso_patient(p, verbose=True)
        if r is not None:
            all_results.append(r)

    if not all_results:
        print("No results — check dataset path.")
    else:
        df = pd.DataFrame(all_results)
        print("\n" + "="*70)
        print("FINAL SUMMARY — Patient-specific LOSO-CV")
        print("="*70)
        fmt = f"{'Patient':<10} {'#Seiz':>6} {'AUC':>7} {'BalAcc':>8} {'Sens':>7} {'FA/h':>7}"
        print(fmt)
        print("-"*50)
        for _, row in df.iterrows():
            print(f"{row['patient']:<10} {int(row['n_seizures']):>6} "
                  f"{row['mean_auc']:>7.3f} {row['mean_bal_acc']:>8.3f} "
                  f"{row['mean_sensitivity']:>7.3f} {row['mean_fpr_per_hour']:>7.2f}")
        print("-"*50)
        print(f"{'MEAN':<10} {'':>6} "
              f"{df['mean_auc'].mean():>7.3f} {df['mean_bal_acc'].mean():>8.3f} "
              f"{df['mean_sensitivity'].mean():>7.3f} {df['mean_fpr_per_hour'].mean():>7.2f}")
        print()
        print("Note: FA/h = false alarms per hour during interictal periods.")
        print("A clinically useful predictor targets: sensitivity > 0.70, FA/h < 0.15")
