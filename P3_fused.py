"""
Fused Seizure Prediction: LightGBM + MAACD Threshold
======================================================
Runs three strategies per fold and compares them:
  A) Pure LightGBM   — static feature classification only
  B) Pure MAACD      — threshold on normalised MAACD time series only
  C) Fused (α)       — combined_score = α*P_lgbm + (1-α)*MAACD_norm
                        α is grid-searched on training folds

For each strategy the alarm rule is identical:
  - smooth with majority-vote window (SMOOTHING_K)
  - alarm fires when ALARM_CONSEC consecutive positives occur
  - 10-min refractory period after each alarm

Saves results to results_FUSED_2026-04-09.txt
"""

import os
import re
import inspect
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_PATH          = "./dataset/"
STEP_SEC              = 1.0
PREICTAL_SEC          = 300
POSTICTAL_BUFFER_SEC  = 1000
MAX_INTERICTAL_SEC    = 3600
MIN_INTER_SEIZURE_SEC = 1300
SMOOTHING_K           = 10
ALARM_CONSEC          = 6
REFRACTORY_WINDOWS    = int(10 * 60 / STEP_SEC)
MAACD_W               = 7
MAACD_P               = 27
ALPHA_GRID            = np.linspace(0, 1, 11)   # α candidates for grid search

FEATURE_MODE = "all"

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])


# ── DATA LOADING (per-recording MAACD) ───────────────────────────────────────

def _sort_key(name):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name)]


def compute_maacd(eeg: pd.DataFrame, w=MAACD_W, p=MAACD_P) -> pd.DataFrame:
    sync_cols = [c for c in eeg.columns if
                 c.startswith("PLI_CH") or c.startswith("WPLI_CH")]
    new_cols = {}
    for col in sync_cols:
        vals  = eeg[col]
        ema   = vals.ewm(span=w, adjust=False).mean()
        lower = ema.rolling(window=p, min_periods=1).min()
        new_cols[f"MAACD_{col}"] = (ema - lower).values
    return eeg.assign(**new_cols)


def load_patient(patient: str):
    csv_path = os.path.join(DATASET_PATH, patient, "csv")
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
        eeg = eeg.loc[:, eeg.std() > 0]
        y   = pd.read_csv(y_f).values.flatten()
        if len(eeg) != len(y):
            continue
        eeg = compute_maacd(eeg)
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


# ── SEIZURE SEGMENTATION ─────────────────────────────────────────────────────

def find_seizure_episodes(labels):
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
            raw_episodes.append({
                "preictal_start": pre_start,
                "ictal_start":    ictal_start,
                "ictal_end":      i,
            })
        else:
            i += 1

    min_gap = int(MIN_INTER_SEIZURE_SEC / STEP_SEC)
    filtered = []
    for ep in raw_episodes:
        if filtered and ep["preictal_start"] - filtered[-1]["ictal_end"] < min_gap:
            continue
        filtered.append(ep)

    preictal_win = int(PREICTAL_SEC / STEP_SEC)
    episodes = []
    for ep in filtered:
        trimmed = max(ep["preictal_start"], ep["ictal_start"] - preictal_win)
        episodes.append({
            "preictal_start": trimmed,
            "ictal_start":    ep["ictal_start"],
            "ictal_end":      ep["ictal_end"],
        })
    return episodes


def build_fold_masks(labels, episodes, k):
    """Return (train_mask, test_mask) boolean arrays for fold k."""
    n = len(labels)
    max_inter = int(MAX_INTERICTAL_SEC   / STEP_SEC)
    post_buf  = int(POSTICTAL_BUFFER_SEC / STEP_SEC)

    def ep_mask(ep_idx):
        ep = episodes[ep_idx]
        s  = max(0, ep["preictal_start"] - max_inter)
        e  = min(n, ep["ictal_end"] + post_buf)
        m  = np.zeros(n, dtype=bool)
        m[s:e] = True
        return m

    test_mask  = ep_mask(k)
    train_mask = np.zeros(n, dtype=bool)
    for j in range(len(episodes)):
        if j != k:
            train_mask |= ep_mask(j)
    train_mask &= ~test_mask
    return train_mask, test_mask


# ── CLASSIFIER ───────────────────────────────────────────────────────────────

def build_lgbm(pos_weight):
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


# ── POST-PROCESSING & EVALUATION ─────────────────────────────────────────────

def smooth(proba, k=SMOOTHING_K, thr=0.55):
    return (np.convolve((proba > thr).astype(float),
                        np.ones(k) / k, mode="same") > 0.5).astype(int)


def seizure_metrics(y_true, y_pred, labels_raw):
    preictal_mask   = (labels_raw == 2)
    interictal_mask = (labels_raw == 0)

    blocks, in_block = [], False
    for i in range(len(labels_raw)):
        if preictal_mask[i] and not in_block:
            start, in_block = i, True
        elif not preictal_mask[i] and in_block:
            blocks.append((start, i))
            in_block = False
    if in_block:
        blocks.append((start, len(labels_raw)))

    detected = 0
    for bs, be in blocks:
        consec = 0
        for val in y_pred[bs:be]:
            consec = consec + 1 if val == 1 else 0
            if consec >= ALARM_CONSEC:
                detected += 1
                break

    n_seiz = len(blocks)
    sens   = detected / max(n_seiz, 1)

    false_alarms, consec, refrac = 0, 0, 0
    for i in range(len(labels_raw)):
        if refrac > 0:
            refrac -= 1
            consec = 0
            continue
        if interictal_mask[i] and y_pred[i] == 1:
            consec += 1
            if consec == ALARM_CONSEC:
                false_alarms += 1
                consec = 0
                refrac = REFRACTORY_WINDOWS
        else:
            consec = 0

    inter_hours = interictal_mask.sum() * STEP_SEC / 3600
    fpr = false_alarms / max(inter_hours, 1e-3)
    return sens, fpr, n_seiz


# ── MAACD NORMALISATION ───────────────────────────────────────────────────────

def maacd_signal(eeg: pd.DataFrame) -> np.ndarray:
    """Mean of all MAACD_PLI_CH* and MAACD_WPLI_CH* columns → (n,) array."""
    cols = [c for c in eeg.columns if
            c.startswith("MAACD_PLI_CH") or c.startswith("MAACD_WPLI_CH")]
    if not cols:
        return np.zeros(len(eeg))
    return eeg[cols].mean(axis=1).values


def fit_maacd_normaliser(maacd_tr: np.ndarray, y_tr: np.ndarray):
    """Fit μ, σ on TRAINING interictal windows."""
    inter = maacd_tr[y_tr == 0]
    mu    = inter.mean()
    sigma = inter.std() + 1e-12
    return mu, sigma


def normalise_maacd(maacd: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.clip((maacd - mu) / sigma, 0, 1)


# ── ALPHA GRID SEARCH ─────────────────────────────────────────────────────────

def search_alpha(p_lgbm_tr, maacd_norm_tr, y_tr):
    """
    Find α that maximises AUC on the training set.
    Using training AUC is a rough proxy; a proper nested-CV would be
    cleaner but prohibitively slow here.
    """
    best_alpha, best_auc = 0.5, -1.0
    if len(np.unique(y_tr)) < 2:
        return best_alpha
    for alpha in ALPHA_GRID:
        combined = alpha * p_lgbm_tr + (1 - alpha) * maacd_norm_tr
        try:
            auc = roc_auc_score(y_tr, combined)
            if auc > best_auc:
                best_auc   = auc
                best_alpha = alpha
        except Exception:
            pass
    return best_alpha


# ── LOSO FOR ONE PATIENT ──────────────────────────────────────────────────────

def loso_patient(patient: str, verbose: bool = True):
    eeg, labels = load_patient(patient)
    if eeg is None:
        return None

    # Separate MAACD signal (raw, not scaled) from full feature matrix
    maacd_full = maacd_signal(eeg)                     # (n,)
    feat_cols  = eeg.columns.tolist()
    eeg_vals   = eeg.values                            # (n, n_feats)

    episodes  = find_seizure_episodes(labels)
    n_seizures = len(episodes)

    if n_seizures < 2:
        if verbose:
            print(f"  [skip] {patient}: only {n_seizures} seizure(s)")
        return None

    if verbose:
        counts = {v: int((labels == v).sum()) for v in [0, 1, 2]}
        print(f"\n  {patient}: {n_seizures} seizures | "
              f"interictal={counts[0]}, preictal={counts[2]}, ictal={counts[1]}")

    res_lgbm, res_maacd, res_fused = [], [], []

    for k in range(n_seizures):
        train_mask, test_mask = build_fold_masks(labels, episodes, k)

        X_tr_raw = eeg_vals[train_mask]
        y_tr     = (labels[train_mask] > 0).astype(int)
        X_te_raw = eeg_vals[test_mask]
        y_te     = (labels[test_mask] > 0).astype(int)

        if len(X_tr_raw) == 0 or len(X_te_raw) == 0:
            continue
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        maacd_tr_raw = maacd_full[train_mask]
        maacd_te_raw = maacd_full[test_mask]

        # ── scale features ──
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)

        # ── LightGBM ──
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = build_lgbm(pos_w)
        model.fit(X_tr, y_tr)

        p_lgbm_tr = model.predict_proba(X_tr)[:, 1]
        p_lgbm_te = model.predict_proba(X_te)[:, 1]

        # ── MAACD normalisation ──
        mu, sigma        = fit_maacd_normaliser(maacd_tr_raw, y_tr)
        maacd_norm_tr    = normalise_maacd(maacd_tr_raw, mu, sigma)
        maacd_norm_te    = normalise_maacd(maacd_te_raw, mu, sigma)

        # ── Alpha grid search on training set ──
        best_alpha = search_alpha(p_lgbm_tr, maacd_norm_tr, y_tr)

        # ── Raw labels for seizure-level eval ──
        test_ep = episodes[k]
        inter_s = max(0, test_ep["preictal_start"] - int(MAX_INTERICTAL_SEC / STEP_SEC))
        post_e  = min(len(labels), test_ep["ictal_end"] + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
        labels_raw_te = labels[inter_s:post_e]

        def eval_strategy(score_te, name):
            y_pred = smooth(score_te)
            sens, fpr, _ = seizure_metrics(y_te, y_pred, labels_raw_te)
            auc = roc_auc_score(y_te, score_te) if len(np.unique(y_te)) > 1 else float("nan")
            bal = balanced_accuracy_score(y_te, y_pred)
            return {"fold": k, "auc": auc, "bal_acc": bal,
                    "sensitivity": sens, "fpr_per_hour": fpr}

        r_lgbm  = eval_strategy(p_lgbm_te,                                          "lgbm")
        r_maacd = eval_strategy(maacd_norm_te,                                       "maacd")
        r_fused = eval_strategy(best_alpha * p_lgbm_te + (1-best_alpha)*maacd_norm_te, "fused")

        res_lgbm.append(r_lgbm)
        res_maacd.append(r_maacd)
        res_fused.append(r_fused)

        if verbose:
            print(f"    Fold {k+1}/{n_seizures} (α={best_alpha:.2f}): "
                  f"LGBM  sens={r_lgbm['sensitivity']:.2f} FA/h={r_lgbm['fpr_per_hour']:.2f} | "
                  f"MAACD sens={r_maacd['sensitivity']:.2f} FA/h={r_maacd['fpr_per_hour']:.2f} | "
                  f"FUSED sens={r_fused['sensitivity']:.2f} FA/h={r_fused['fpr_per_hour']:.2f}")

    if not res_lgbm:
        return None

    def mean_results(res, tag):
        df = pd.DataFrame(res)
        return {
            "patient":    patient,
            "n_seizures": n_seizures,
            "strategy":   tag,
            "mean_auc":         df["auc"].mean(),
            "mean_bal_acc":     df["bal_acc"].mean(),
            "mean_sensitivity": df["sensitivity"].mean(),
            "mean_fpr_per_hour":df["fpr_per_hour"].mean(),
        }

    r = {
        "lgbm":  mean_results(res_lgbm,  "lgbm"),
        "maacd": mean_results(res_maacd, "maacd"),
        "fused": mean_results(res_fused, "fused"),
    }

    if verbose:
        for tag, v in r.items():
            print(f"  ── {tag.upper():<6}: "
                  f"AUC={v['mean_auc']:.3f}  "
                  f"BalAcc={v['mean_bal_acc']:.3f}  "
                  f"Sens={v['mean_sensitivity']:.3f}  "
                  f"FA/h={v['mean_fpr_per_hour']:.2f}")
    return r


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(all_results):
    strategies = ["lgbm", "maacd", "fused"]
    labels_str = {"lgbm": "LightGBM", "maacd": "MAACD thr.", "fused": "Fused"}

    print("\n" + "="*75)
    print("FINAL SUMMARY — Patient-specific LOSO-CV (3 strategies)")
    print("="*75)

    for strat in strategies:
        rows = [r[strat] for r in all_results if strat in r]
        print(f"\n{'─'*75}")
        print(f"  {labels_str[strat]}")
        print(f"  {'Patient':<10} {'#Seiz':>6} {'AUC':>7} {'BalAcc':>8} {'Sens':>7} {'FA/h':>7}")
        print(f"  {'-'*50}")
        aucs, senss, fprs, bals = [], [], [], []
        for v in rows:
            print(f"  {v['patient']:<10} {int(v['n_seizures']):>6} "
                  f"{v['mean_auc']:>7.3f} {v['mean_bal_acc']:>8.3f} "
                  f"{v['mean_sensitivity']:>7.3f} {v['mean_fpr_per_hour']:>7.2f}")
            aucs.append(v["mean_auc"])
            senss.append(v["mean_sensitivity"])
            fprs.append(v["mean_fpr_per_hour"])
            bals.append(v["mean_bal_acc"])
        print(f"  {'-'*50}")
        print(f"  {'MEAN':<10} {'':>6} "
              f"{np.nanmean(aucs):>7.3f} {np.mean(bals):>8.3f} "
              f"{np.mean(senss):>7.3f} {np.mean(fprs):>7.2f}")

    # ── delta table ──
    print(f"\n{'─'*75}")
    print("  DELTA vs pure LightGBM")
    print(f"  {'Patient':<10} {'Sens MAACD':>12} {'Sens FUSED':>12} {'FA/h MAACD':>12} {'FA/h FUSED':>12}")
    print(f"  {'-'*60}")
    for r in all_results:
        p = r["lgbm"]["patient"]
        ds_m = r["maacd"]["mean_sensitivity"] - r["lgbm"]["mean_sensitivity"]
        ds_f = r["fused"]["mean_sensitivity"] - r["lgbm"]["mean_sensitivity"]
        df_m = r["maacd"]["mean_fpr_per_hour"] - r["lgbm"]["mean_fpr_per_hour"]
        df_f = r["fused"]["mean_fpr_per_hour"] - r["lgbm"]["mean_fpr_per_hour"]
        print(f"  {p:<10} {ds_m:>+12.3f} {ds_f:>+12.3f} {df_m:>+12.2f} {df_f:>+12.2f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running fused LOSO-CV on {len(patients)} patients\n")
    all_results = []

    for p in patients:
        r = loso_patient(p, verbose=True)
        if r is not None:
            all_results.append(r)

    if all_results:
        print_summary(all_results)
