"""
LightGBM no-sync  OR  focal MAACD alarm  (patient-specific OR gate)
====================================================================
Evaluation window: preictal (300s) + ictal = "seizure-related".

Two independent arms with shared refractory:
  1. LightGBM on all non-PLI/WPLI features  (fixed threshold grid)
  2. Focal MAACD arm: MAACD(f_S1) > thr_maacd for each patient
       thr_maacd calibrated per fold on training interictal (FA/h ≤ 1)

An alarm fires when EITHER arm reaches alarm_consec consecutive hits.
Shared 10-min refractory period suppresses both arms after any alarm.

Fixed threshold grid:
  threshold    : 0.60, 0.70, 0.80
  alarm_consec : 6, 7, 8, 10
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
REFRACTORY_WINDOWS    = int(10 * 60 / STEP_SEC)
MAACD_W, MAACD_P      = 7, 27

CONSEC_SWEEP = [6, 7, 8, 10]
THR_SWEEP    = [0.60, 0.70, 0.80]

# Only the raw PLI/WPLI columns are excluded from LightGBM.
# MAACD_ columns are now only the 2 focal ones (computed by compute_focal_maacd)
# and are intentionally kept out of LightGBM (used in OR-gate arm only).
sync_prefixes = ("PLI_CH", "WPLI_CH", "MAACD_")

# ── Focal pairs per patient (ThAlgo: f_S1=best preictal rise, f_S2=best interictal silence)
FOCAL_PAIRS = {
    "PN00": ["WPLI_CH11_CH24", "WPLI_CH14_CH24"],
    "PN01": ["WPLI_CH1_CH22",  "WPLI_CH2_CH19"],
    "PN03": ["WPLI_CH3_CH12",  "PLI_CH26_CH29"],
    "PN05": ["WPLI_CH15_CH27", "PLI_CH19_CH26"],
    "PN06": ["WPLI_CH19_CH22", "PLI_CH4_CH18"],
    "PN09": ["WPLI_CH8_CH16",  "PLI_CH5_CH19"],
    "PN10": ["WPLI_CH1_CH7",   "PLI_CH13_CH28"],
    "PN12": ["WPLI_CH23_CH29", "PLI_CH23_CH24"],
    "PN13": ["WPLI_CH23_CH24", "PLI_CH10_CH15"],
    "PN14": ["WPLI_CH21_CH23", "PLI_CH1_CH5"],
    "PN16": ["WPLI_CH14_CH21", "PLI_CH1_CH28"],
    "PN17": ["WPLI_CH11_CH12", "PLI_CH9_CH14"],
}

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def _sort_key(name):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name)]


def compute_focal_maacd(eeg: pd.DataFrame, patient: str, w=MAACD_W, p=MAACD_P) -> pd.DataFrame:
    """Compute MAACD only for the 2 focal pairs of this patient (keeps DataFrame small)."""
    new_cols = {}
    for pair in FOCAL_PAIRS.get(patient, []):
        if pair not in eeg.columns:
            continue
        vals  = eeg[pair]
        ema   = vals.ewm(span=w, adjust=False).mean()
        lower = ema.rolling(window=p, min_periods=1).min()
        new_cols[f"MAACD_{pair}"] = (ema - lower).values
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
        eeg = compute_focal_maacd(eeg, patient)
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


# ── METRICS ───────────────────────────────────────────────────────────────────

def smooth(proba, k=SMOOTHING_K, thr=0.5):
    return (np.convolve((proba > thr).astype(float),
                        np.ones(k) / k, mode="same") > 0.5).astype(int)


def calibrate_maacd_threshold(maacd_inter_vals, alarm_consec, fah_target=1.0):
    """
    Find lowest percentile threshold of training interictal MAACD such that
    FA/h <= fah_target (treating training interictal as one contiguous block).
    Returns np.inf if no threshold achieves the target (arm disabled).
    """
    total_hours = len(maacd_inter_vals) * STEP_SEC / 3600
    if total_hours < 0.1:
        return np.inf
    for pct in [50, 60, 70, 75, 80, 85, 90, 92, 95, 97, 99]:
        thr = np.percentile(maacd_inter_vals, pct)
        fa, c, refrac = 0, 0, 0
        for v in maacd_inter_vals:
            if refrac > 0:
                refrac -= 1; c = 0; continue
            if v > thr:
                c += 1
                if c >= alarm_consec:
                    fa += 1; c = 0; refrac = REFRACTORY_WINDOWS
            else:
                c = 0
        if fa / max(total_hours, 1e-3) <= fah_target:
            return thr
    return np.inf


def or_gate_metrics(y_lgbm, maacd_vals, maacd_thr, labels_raw, alarm_consec):
    """
    OR gate alarm: fires when LightGBM arm OR MAACD arm reaches alarm_consec.
    Both arms share a single refractory counter.
    """
    seizure_mask    = (labels_raw >= 1)
    interictal_mask = (labels_raw == 0)
    use_maacd = (maacd_vals is not None) and (not np.isinf(maacd_thr))

    # ── sensitivity: detect seizure blocks ───────────────────────────────────
    blocks, in_block = [], False
    for i in range(len(labels_raw)):
        if seizure_mask[i] and not in_block:
            start, in_block = i, True
        elif not seizure_mask[i] and in_block:
            blocks.append((start, i)); in_block = False
    if in_block:
        blocks.append((start, len(labels_raw)))

    detected = 0
    for bs, be in blocks:
        cl, cm = 0, 0
        for i in range(bs, be):
            cl = cl + 1 if y_lgbm[i] == 1 else 0
            if cl >= alarm_consec:
                detected += 1; break
            if use_maacd:
                cm = cm + 1 if maacd_vals[i] > maacd_thr else 0
                if cm >= alarm_consec:
                    detected += 1; break
    sens = detected / max(len(blocks), 1)

    # ── FA/h: interictal only, shared refractory ─────────────────────────────
    fa, cl, cm, refrac = 0, 0, 0, 0
    for i in range(len(labels_raw)):
        if refrac > 0:
            refrac -= 1; cl = 0; cm = 0; continue
        if interictal_mask[i]:
            cl = cl + 1 if y_lgbm[i] == 1 else 0
            if cl == alarm_consec:
                fa += 1; cl = 0; cm = 0; refrac = REFRACTORY_WINDOWS; continue
            if use_maacd:
                cm = cm + 1 if maacd_vals[i] > maacd_thr else 0
                if cm == alarm_consec:
                    fa += 1; cl = 0; cm = 0; refrac = REFRACTORY_WINDOWS
        else:
            cl = 0; cm = 0

    fpr = fa / max(interictal_mask.sum() * STEP_SEC / 3600, 1e-3)
    return sens, fpr, len(blocks)


# ── LOSO FOR ONE PATIENT ──────────────────────────────────────────────────────

def loso_patient(patient: str, verbose: bool = True):
    eeg, labels = load_patient(patient)
    if eeg is None:
        return None

    col_names = list(eeg.columns)
    eeg_vals  = eeg.values.astype(np.float32)

    # Non-sync indices for LightGBM (no PLI/WPLI or their MAACs)
    nopli_idx = np.array([
        i for i, c in enumerate(col_names)
        if not any(c.startswith(p) for p in sync_prefixes)
    ])

    # Focal MAACD column index for OR-gate arm (f_S1 = best preictal rise)
    focal_s1      = FOCAL_PAIRS.get(patient, [None])[0]
    focal_col     = f"MAACD_{focal_s1}" if focal_s1 else None
    focal_col_idx = col_names.index(focal_col) if focal_col and focal_col in col_names else None

    episodes   = find_seizure_episodes(labels)
    n_seizures = len(episodes)

    if n_seizures < 2:
        if verbose:
            print(f"  [skip] {patient}: only {n_seizures} seizure(s)")
        return None

    if verbose:
        counts = {v: int((labels == v).sum()) for v in [0, 1, 2]}
        arm_str = focal_col if focal_col_idx is not None else "none"
        print(f"\n  {patient}: {n_seizures} seizures | "
              f"inter={counts[0]} pre={counts[2]} ictal={counts[1]} | "
              f"lgbm_feats={len(nopli_idx)}  focal_arm={arm_str}")

    strat_keys = [(c, t) for c in CONSEC_SWEEP for t in THR_SWEEP]
    res = {k: [] for k in strat_keys}

    for fold in range(n_seizures):
        train_mask, test_mask = build_fold_masks(labels, episodes, fold)

        y_tr = (labels[train_mask] > 0).astype(int)
        y_te = (labels[test_mask]  > 0).astype(int)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        ep      = episodes[fold]
        inter_s = max(0, ep["preictal_start"] - int(MAX_INTERICTAL_SEC / STEP_SEC))
        post_e  = min(len(labels), ep["ictal_end"] + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
        raw_te  = labels[inter_s:post_e]

        # ── LightGBM arm ─────────────────────────────────────────────────────
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(eeg_vals[train_mask][:, nopli_idx])
        Xte = scaler.transform(eeg_vals[test_mask][:,   nopli_idx])
        pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=7,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(Xtr, y_tr)
        p_te = model.predict_proba(Xte)[:, 1]
        auc  = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")

        # ── Focal MAACD arm: calibrate threshold on training interictal ───────
        if focal_col_idx is not None:
            maacd_tr_inter = eeg_vals[train_mask][y_tr == 0, focal_col_idx]
            maacd_te_vals  = eeg_vals[test_mask, focal_col_idx]
        else:
            maacd_tr_inter = None
            maacd_te_vals  = None

        # Calibrate MAACD threshold once per unique consec value
        maacd_thrs = {}
        for consec_val in set(c for c, _ in strat_keys):
            if maacd_tr_inter is not None:
                maacd_thrs[consec_val] = calibrate_maacd_threshold(maacd_tr_inter, consec_val)
            else:
                maacd_thrs[consec_val] = np.inf

        for (consec, thr) in strat_keys:
            y_pred = smooth(p_te, thr=thr)
            sens, fpr, _ = or_gate_metrics(y_pred, maacd_te_vals, maacd_thrs[consec], raw_te, consec)
            res[(consec, thr)].append({
                "fold": fold, "auc": auc,
                "bal_acc": balanced_accuracy_score(y_te, y_pred),
                "sensitivity": sens, "fpr_per_hour": fpr,
            })

        if verbose:
            r = res[(6, 0.60)][-1]
            print(f"    Fold {fold+1}/{n_seizures} [consec=6, thr=0.60] "
                  f"sens={r['sensitivity']:.2f}  FA/h={r['fpr_per_hour']:.2f}  AUC={auc:.3f}")

    if not any(res[k] for k in strat_keys):
        return None

    def mean_r(key):
        df = pd.DataFrame(res[key])
        return {
            "patient": patient, "n_seizures": n_seizures,
            "consec": key[0], "thr": key[1],
            "mean_auc":          df["auc"].mean(),
            "mean_bal_acc":      df["bal_acc"].mean(),
            "mean_sensitivity":  df["sensitivity"].mean(),
            "mean_fpr_per_hour": df["fpr_per_hour"].mean(),
        }

    return {k: mean_r(k) for k in strat_keys if res[k]} | {"patient": patient}


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(all_results):
    strat_keys = [(c, t) for c in CONSEC_SWEEP for t in THR_SWEEP]

    print("\n" + "="*72)
    print(f"FINAL SUMMARY — lgbm_nopli OR focal_MAACD (calibrated, shared refrac) | PREICTAL={PREICTAL_SEC}s")
    print(f"Grid: ALARM_CONSEC={CONSEC_SWEEP}  THR={THR_SWEEP}")
    print("="*72)

    print(f"\n  {'consec':>7} {'thr':>6} {'mean_AUC':>9} {'mean_Sens':>10} {'mean_FA/h':>10}")
    print(f"  {'-'*48}")
    for (consec, thr) in strat_keys:
        rows = [r[(consec, thr)] for r in all_results if (consec, thr) in r]
        if not rows:
            continue
        aucs  = [v["mean_auc"]          for v in rows]
        senss = [v["mean_sensitivity"]  for v in rows]
        fprs  = [v["mean_fpr_per_hour"] for v in rows]
        print(f"  {consec:>7}   {thr:>4.2f}   "
              f"{np.nanmean(aucs):>8.3f}   {np.mean(senss):>9.3f}   {np.mean(fprs):>9.2f}")

    for (consec, thr) in strat_keys:
        rows = [r[(consec, thr)] for r in all_results if (consec, thr) in r]
        if not rows:
            continue
        print(f"\n{'─'*72}")
        print(f"  ALARM_CONSEC={consec}, thr={thr}")
        print(f"  {'Patient':<10} {'#Seiz':>6} {'AUC':>7} {'BalAcc':>8} {'Sens':>7} {'FA/h':>7}")
        print(f"  {'-'*50}")
        aucs, senss, fprs, bals = [], [], [], []
        for v in rows:
            print(f"  {v['patient']:<10} {int(v['n_seizures']):>6} "
                  f"{v['mean_auc']:>7.3f} {v['mean_bal_acc']:>8.3f} "
                  f"{v['mean_sensitivity']:>7.3f} {v['mean_fpr_per_hour']:>7.2f}")
            aucs.append(v["mean_auc"]); senss.append(v["mean_sensitivity"])
            fprs.append(v["mean_fpr_per_hour"]); bals.append(v["mean_bal_acc"])
        print(f"  {'-'*50}")
        print(f"  {'MEAN':<10} {'':>6} "
              f"{np.nanmean(aucs):>7.3f} {np.mean(bals):>8.3f} "
              f"{np.mean(senss):>7.3f} {np.mean(fprs):>7.2f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"LightGBM no-sync  OR  focal MAACD (calibrated FA/h≤1) | PREICTAL={PREICTAL_SEC}s")
    print(f"CONSEC: {CONSEC_SWEEP}  THR: {THR_SWEEP}")
    print(f"REFRACTORY={REFRACTORY_WINDOWS//60}min | SMOOTHING_K={SMOOTHING_K}\n")

    all_results = []
    for p in patients:
        r = loso_patient(p, verbose=True)
        if r is not None:
            all_results.append(r)

    if all_results:
        print_summary(all_results)
