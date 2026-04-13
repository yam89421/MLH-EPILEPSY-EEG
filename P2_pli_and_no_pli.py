"""
LightGBM LOSO — no PLI/WPLI features
======================================
Compares two strategies on the same LOSO scheme as seizure_prediction_focal_pairs.py:
  - lgbm_all   : LightGBM on ALL features (baseline, thr=0.55)
  - lgbm_nopli : LightGBM with ALL PLI/WPLI columns removed (raw + MAACD)
Both use per-patient threshold tuning targeting FA/h <= 1.0 on training data.

Goal: test whether PLI/WPLI features help or hurt LightGBM.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH          = "./dataset/"
STEP_SEC              = 1.0
PREICTAL_SEC          = 300
POSTICTAL_BUFFER_SEC  = 1000
MAX_INTERICTAL_SEC    = 3600
MIN_INTER_SEIZURE_SEC = 1300
SMOOTHING_K           = 10
ALARM_CONSEC          = 6
REFRACTORY_WINDOWS    = int(10 * 60 / STEP_SEC)
MAACD_W, MAACD_P      = 7, 27
FAH_TARGET            = 1.0

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])


# ── DATA LOADING ──────────────────────────────────────────────────────────────

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
    return eeg_all, np.concatenate(y_parts)


# ── SEIZURE SEGMENTATION ──────────────────────────────────────────────────────

def find_seizure_episodes(labels):
    n, raw = len(labels), []
    i = 0
    while i < n:
        if labels[i] == 2:
            pre = i
            while i < n and labels[i] == 2: i += 1
            ics = i
            while i < n and labels[i] == 1: i += 1
            raw.append({"preictal_start": pre, "ictal_start": ics, "ictal_end": i})
        else:
            i += 1

    min_gap = int(MIN_INTER_SEIZURE_SEC / STEP_SEC)
    filtered = []
    for ep in raw:
        if filtered and ep["preictal_start"] - filtered[-1]["ictal_end"] < min_gap:
            continue
        filtered.append(ep)

    pw = int(PREICTAL_SEC / STEP_SEC)
    return [{
        "preictal_start": max(ep["preictal_start"], ep["ictal_start"] - pw),
        "ictal_start":    ep["ictal_start"],
        "ictal_end":      ep["ictal_end"],
    } for ep in filtered]


def build_fold_masks(labels, episodes, k):
    n         = len(labels)
    max_inter = int(MAX_INTERICTAL_SEC   / STEP_SEC)
    post_buf  = int(POSTICTAL_BUFFER_SEC / STEP_SEC)

    def ep_mask(idx):
        ep = episodes[idx]
        m  = np.zeros(n, dtype=bool)
        m[max(0, ep["preictal_start"] - max_inter) : min(n, ep["ictal_end"] + post_buf)] = True
        return m

    test_mask  = ep_mask(k)
    train_mask = np.zeros(n, dtype=bool)
    for j in range(len(episodes)):
        if j != k:
            train_mask |= ep_mask(j)
    train_mask &= ~test_mask
    return train_mask, test_mask


# ── POST-PROCESSING & METRICS ─────────────────────────────────────────────────

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
            blocks.append((start, i)); in_block = False
    if in_block:
        blocks.append((start, len(labels_raw)))

    detected = 0
    for bs, be in blocks:
        consec = 0
        for val in y_pred[bs:be]:
            consec = consec + 1 if val == 1 else 0
            if consec >= ALARM_CONSEC:
                detected += 1; break

    sens = detected / max(len(blocks), 1)

    fa, consec, refrac = 0, 0, 0
    for i in range(len(labels_raw)):
        if refrac > 0:
            refrac -= 1; consec = 0; continue
        if interictal_mask[i] and y_pred[i] == 1:
            consec += 1
            if consec == ALARM_CONSEC:
                fa += 1; consec = 0; refrac = REFRACTORY_WINDOWS
        else:
            consec = 0

    fpr = fa / max(interictal_mask.sum() * STEP_SEC / 3600, 1e-3)
    return sens, fpr, len(blocks)


def tune_threshold_for_fah(p_scores_tr, labels_tr_raw, target_fah=FAH_TARGET):
    y_tr = (labels_tr_raw > 0).astype(int)
    if len(np.unique(y_tr)) < 2:
        return 0.55
    best_thr, best_sens = 0.90, -1.0
    for thr in np.arange(0.25, 0.96, 0.025):
        y_pred = smooth(p_scores_tr, thr=thr)
        sens, fah, _ = seizure_metrics(y_tr, y_pred, labels_tr_raw)
        if fah <= target_fah and sens > best_sens:
            best_sens = sens
            best_thr  = thr
    return best_thr


# ── LOSO FOR ONE PATIENT ──────────────────────────────────────────────────────

def loso_patient(patient: str, verbose: bool = True):
    eeg, labels = load_patient(patient)
    if eeg is None:
        return None

    col_names = list(eeg.columns)
    eeg_vals  = eeg.values.astype(np.float32)

    # ── Feature masks ──────────────────────────────────────────────────────
    pli_mask = np.array([
        c.startswith("PLI_CH") or c.startswith("WPLI_CH") or
        c.startswith("MAACD_PLI_CH") or c.startswith("MAACD_WPLI_CH")
        for c in col_names
    ])
    nopli_idx = np.where(~pli_mask)[0]
    n_all   = len(col_names)
    n_nopli = len(nopli_idx)

    episodes   = find_seizure_episodes(labels)
    n_seizures = len(episodes)

    if n_seizures < 2:
        if verbose:
            print(f"  [skip] {patient}: only {n_seizures} seizure(s)")
        return None

    if verbose:
        counts = {v: int((labels == v).sum()) for v in [0, 1, 2]}
        print(f"\n  {patient}: {n_seizures} seizures | "
              f"inter={counts[0]} pre={counts[2]} ictal={counts[1]} | "
              f"all_feats={n_all}  nopli_feats={n_nopli}")

    res = {s: [] for s in ["lgbm_all_fah1", "lgbm_nopli_fah1"]}

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

        labels_tr_raw = labels[train_mask]

        ep      = episodes[k]
        inter_s = max(0, ep["preictal_start"] - int(MAX_INTERICTAL_SEC / STEP_SEC))
        post_e  = min(len(labels), ep["ictal_end"] + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
        raw_te  = labels[inter_s:post_e]

        def train_eval(X_tr, X_te, tag):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_tr)
            Xte = scaler.transform(X_te)
            pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model = lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=7,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_w, random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(Xtr, y_tr)
            p_tr = model.predict_proba(Xtr)[:, 1]
            p_te = model.predict_proba(Xte)[:, 1]
            thr  = tune_threshold_for_fah(p_tr, labels_tr_raw)
            y_pred = smooth(p_te, thr=thr)
            sens, fpr, _ = seizure_metrics(y_te, y_pred, raw_te)
            auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")
            if verbose:
                print(f"    Fold {k+1}/{n_seizures} [{tag}]: "
                      f"thr={thr:.3f}  sens={sens:.2f}  FA/h={fpr:.2f}  AUC={auc:.3f}")
            return {"fold": k, "auc": auc,
                    "bal_acc": balanced_accuracy_score(y_te, y_pred),
                    "sensitivity": sens, "fpr_per_hour": fpr}

        res["lgbm_all_fah1"].append(
            train_eval(X_tr_raw, X_te_raw, "all_feats"))
        res["lgbm_nopli_fah1"].append(
            train_eval(X_tr_raw[:, nopli_idx], X_te_raw[:, nopli_idx], "nopli"))

    if not res["lgbm_all_fah1"]:
        return None

    def mean_r(s):
        df = pd.DataFrame(res[s])
        return {
            "patient": patient, "n_seizures": n_seizures, "strategy": s,
            "mean_auc":          df["auc"].mean(),
            "mean_bal_acc":      df["bal_acc"].mean(),
            "mean_sensitivity":  df["sensitivity"].mean(),
            "mean_fpr_per_hour": df["fpr_per_hour"].mean(),
        }

    return {s: mean_r(s) for s in res} | {"patient": patient}


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(all_results):
    strategies = ["lgbm_all_fah1", "lgbm_nopli_fah1"]
    labels_str = {
        "lgbm_all_fah1":   "LightGBM ALL features  (tuned FA/h≤1)",
        "lgbm_nopli_fah1": "LightGBM NO PLI/WPLI   (tuned FA/h≤1)",
    }
    print("\n" + "="*65)
    print("FINAL SUMMARY — ALL features vs NO PLI/WPLI  (lgbm_fah1)")
    print("="*65)
    for strat in strategies:
        rows = [r[strat] for r in all_results if strat in r]
        print(f"\n{'─'*65}")
        print(f"  {labels_str[strat]}")
        print(f"  {'Patient':<10} {'#Seiz':>6} {'AUC':>7} {'BalAcc':>8} {'Sens':>7} {'FA/h':>7}")
        print(f"  {'-'*48}")
        aucs, senss, fprs, bals = [], [], [], []
        for v in rows:
            print(f"  {v['patient']:<10} {int(v['n_seizures']):>6} "
                  f"{v['mean_auc']:>7.3f} {v['mean_bal_acc']:>8.3f} "
                  f"{v['mean_sensitivity']:>7.3f} {v['mean_fpr_per_hour']:>7.2f}")
            aucs.append(v["mean_auc"]); senss.append(v["mean_sensitivity"])
            fprs.append(v["mean_fpr_per_hour"]); bals.append(v["mean_bal_acc"])
        print(f"  {'-'*48}")
        print(f"  {'MEAN':<10} {'':>6} "
              f"{np.nanmean(aucs):>7.3f} {np.mean(bals):>8.3f} "
              f"{np.mean(senss):>7.3f} {np.mean(fprs):>7.2f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running LOSO: ALL features vs NO PLI/WPLI — lgbm_fah1 (FA/h≤{FAH_TARGET})")
    print(f"Preictal={PREICTAL_SEC}s | ALARM_CONSEC={ALARM_CONSEC} | REFRACTORY={REFRACTORY_WINDOWS//60}min\n")

    all_results = []
    for p in patients:
        r = loso_patient(p, verbose=True)
        if r is not None:
            all_results.append(r)

    if all_results:
        print_summary(all_results)
