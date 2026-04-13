"""
ThAlgo — Detti et al. (2019) Threshold-Based Seizure Prediction
=================================================================
Exact implementation of ThAlgo from:
  "A Patient-Specific Approach for Short-Term Epileptic Seizures Prediction
   Through the Analysis of EEG Synchronization", IEEE TBME, Vol. 66 No. 6, 2019.

CV scheme matches the paper (Section VI):
  - q rounds (q=5 if ms≥4, q=3 if ms=3)
  - Each round: train on ⌈0.5·ms⌉ seizures + ALL pure interictal data
                test  on remaining seizures
  - Every seizure appears in the test set at least once across q rounds

ThAlgo per round:
  1. MAACD(EMA w=7, rolling-min p=27) on all pairwise PLI/WPLI columns
     → feature matrix F  (2 × n*(n-1)/2 features, "arc" feature set A)
  2. θ_f = mean(F_f) over training preictal windows   ← round+patient-specific
  3. S1_f = fraction of training preictal   windows where F_f > θ_f
     S2_f = fraction of training interictal windows where F_f < θ_f
     *** interictal = ALL non-seizure data (far from any episode) ***
  4. f_S1 = argmax(S1),  f_S2 = argmax(S2)
  5. AND gate: ŷ(t) = 1 if F_{f_S1}(t) ≥ θ_S1 AND F_{f_S2}(t) ≥ θ_S2

Metrics (both reported for transparency):
  - Paper  metric: sens = fraction of preictal intervals with ≥1 positive record
                   FA/h = total FP records / total interictal hours
  - Alarm  metric: ALARM_CONSEC consecutive positives needed (same as other scripts)

Comparison: LightGBM trained on the same MAACD arc features only (no BP/entropy).
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from datetime import date
from collections import Counter

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
MAACD_W, MAACD_P      = 7, 27       # EMA span, rolling-min window (paper values)
ALARM_CONSEC          = 6           # consecutive positives to fire alarm
SMOOTHING_K           = 10
REFRACTORY_WINDOWS    = int(10 * 60 / STEP_SEC)

patients = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def _sort_key(name):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name)]


def _maacd(series: pd.Series, w=MAACD_W, p=MAACD_P) -> np.ndarray:
    """MAACD = EMA(w) − rolling_min(EMA(w), p).  Eq. (4) of Detti et al."""
    ema   = series.ewm(span=w, adjust=False).mean()
    lower = ema.rolling(window=p, min_periods=1).min()
    return (ema - lower).values


def load_patient(patient: str):
    """
    Load per-recording CSVs, compute MAACD on pairwise PLI/WPLI columns
    (feature set A of the paper) per recording, then concatenate.
    Returns (feat_df, labels) or (None, None).
    """
    csv_path = os.path.join(DATASET_PATH, patient, "csv")
    rec_dirs = sorted(
        [d for d in os.listdir(csv_path)
         if os.path.isdir(os.path.join(csv_path, d)) and d.startswith(patient)],
        key=_sort_key,
    )
    if not rec_dirs:
        return None, None

    feat_parts, y_parts = [], []
    for rec in rec_dirs:
        eeg_f = os.path.join(csv_path, rec, f"{rec}_EEG_X.csv")
        y_f   = os.path.join(csv_path, rec, f"{rec}_Y.csv")
        if not os.path.exists(eeg_f) or not os.path.exists(y_f):
            continue
        eeg = pd.read_csv(eeg_f)
        y   = pd.read_csv(y_f).values.flatten()
        if len(eeg) != len(y):
            continue

        # Pairwise PLI/WPLI columns (arc feature set A)
        pair_cols = [c for c in eeg.columns
                     if re.match(r'(PLI|WPLI)_CH\d+_CH\d+$', c)]
        if not pair_cols:
            continue

        feat_dict = {f"MAACD_{c}": _maacd(eeg[c]) for c in pair_cols}
        feat_parts.append(pd.DataFrame(feat_dict))
        y_parts.append(y)

    if not feat_parts:
        return None, None

    # Keep only columns present in every recording (handles channel-count differences)
    common = feat_parts[0].columns
    for df in feat_parts[1:]:
        common = common.intersection(df.columns)

    feat_all = pd.concat([df[common] for df in feat_parts], ignore_index=True)
    return feat_all, np.concatenate(y_parts)


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
    n        = len(labels)
    max_i    = int(MAX_INTERICTAL_SEC   / STEP_SEC)
    post_buf = int(POSTICTAL_BUFFER_SEC / STEP_SEC)

    def ep_mask(idx):
        ep = episodes[idx]
        m  = np.zeros(n, dtype=bool)
        m[max(0, ep["preictal_start"] - max_i) : min(n, ep["ictal_end"] + post_buf)] = True
        return m

    test_mask  = ep_mask(k)
    train_mask = np.zeros(n, dtype=bool)
    for j in range(len(episodes)):
        if j != k:
            train_mask |= ep_mask(j)
    train_mask &= ~test_mask
    return train_mask, test_mask


# ── THALGO SELECTION (Section V-C of the paper) ───────────────────────────────

def thalgo_select(feat_tr: np.ndarray, labels_tr: np.ndarray):
    """
    θ_f  = mean(F_f) over ALL training preictal windows
    S1_f = fraction of training preictal   windows where F_f > θ_f
    S2_f = fraction of training interictal windows where F_f < θ_f
           *** labels_tr should include ALL interictal (pure + near-seizure) ***

    Returns (idx_S1, idx_S2, theta_S1, theta_S2).
    """
    pre_mask   = (labels_tr == 2)
    inter_mask = (labels_tr == 0)

    if pre_mask.sum() == 0 or inter_mask.sum() == 0:
        return 0, 0, 0.0, 0.0

    theta = feat_tr[pre_mask].mean(axis=0)
    s1    = (feat_tr[pre_mask]   > theta).mean(axis=0)
    s2    = (feat_tr[inter_mask] < theta).mean(axis=0)

    idx_S1 = int(np.argmax(s1))
    idx_S2 = int(np.argmax(s2))

    return idx_S1, idx_S2, float(theta[idx_S1]), float(theta[idx_S2])


# ── METRICS ───────────────────────────────────────────────────────────────────

def _preictal_blocks(labels_raw):
    blocks, in_block = [], False
    for i, lbl in enumerate(labels_raw):
        if lbl == 2 and not in_block:
            start, in_block = i, True
        elif lbl != 2 and in_block:
            blocks.append((start, i)); in_block = False
    if in_block:
        blocks.append((start, len(labels_raw)))
    return blocks


def paper_metrics(y_pred: np.ndarray, labels_raw: np.ndarray):
    """
    Detti et al. metric:
      sensitivity = fraction of preictal intervals with ≥1 positive record
      FA/h        = total FP records / total interictal hours
    """
    pre_mask   = (labels_raw == 2)
    inter_mask = (labels_raw == 0)
    blocks     = _preictal_blocks(labels_raw)

    detected   = sum(1 for bs, be in blocks if y_pred[bs:be].any())
    sens       = detected / max(len(blocks), 1)

    fp         = int((y_pred[inter_mask] == 1).sum())
    fah        = fp / max(inter_mask.sum() * STEP_SEC / 3600, 1e-3)
    return sens, fah, len(blocks)


def alarm_metrics(y_pred: np.ndarray, labels_raw: np.ndarray):
    """
    Alarm-based metric (requires ALARM_CONSEC consecutive positives).
    Same logic as all other scripts in this repo.
    """
    pre_mask   = (labels_raw == 2)
    inter_mask = (labels_raw == 0)
    blocks     = _preictal_blocks(labels_raw)

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
        if inter_mask[i] and y_pred[i] == 1:
            consec += 1
            if consec == ALARM_CONSEC:
                fa += 1; consec = 0; refrac = REFRACTORY_WINDOWS
        else:
            consec = 0

    fah = fa / max(inter_mask.sum() * STEP_SEC / 3600, 1e-3)
    return sens, fah, len(blocks)


def smooth_proba(proba, k=SMOOTHING_K, thr=0.5):
    return (np.convolve((proba > thr).astype(float),
                        np.ones(k) / k, mode="same") > 0.5).astype(int)


# ── PAPER k-FOLD CV ───────────────────────────────────────────────────────────

import math

def kfold_patient(patient: str, q: int = None, seed: int = 42, verbose: bool = True):
    """
    Paper's CV scheme (Section VI of Detti et al. 2019).

    q rounds, each round:
      - Training seizures : ⌈0.5·ms⌉ randomly selected seizure episodes
      - Training interictal: ALL windows not within any episode context
                             PLUS interictal windows near training seizures
      - Test seizures     : remaining episodes (each tested at least once)

    The key difference from LOSO: S2 is scored on ALL interictal data
    (potentially many hours), not just ±1h around training seizures.
    """
    feat, labels = load_patient(patient)
    if feat is None:
        return None

    col_names = list(feat.columns)
    F         = feat.values.astype(np.float32)
    n         = len(labels)

    episodes   = find_seizure_episodes(labels)
    ms         = len(episodes)

    if ms < 2:
        if verbose:
            print(f"  [skip] {patient}: only {ms} seizure(s)")
        return None

    if q is None:
        q = 5 if ms >= 4 else 3

    n_train_seiz = math.ceil(0.5 * ms)
    n_test_seiz  = ms - n_train_seiz

    if verbose:
        counts = {v: int((labels == v).sum()) for v in [0, 1, 2]}
        print(f"\n  {patient}: {ms} seizures | "
              f"inter={counts[0]} pre={counts[2]} ictal={counts[1]} | "
              f"arc features={F.shape[1]} | q={q} rounds "
              f"(train {n_train_seiz} / test {n_test_seiz} seizures)")

    # Per-episode context mask (±MAX_INTERICTAL + postictal buffer)
    ep_ctx = []
    for ep in episodes:
        m = np.zeros(n, dtype=bool)
        s = max(0, ep["preictal_start"] - int(MAX_INTERICTAL_SEC / STEP_SEC))
        e = min(n, ep["ictal_end"] + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
        m[s:e] = True
        ep_ctx.append(m)

    # Pure interictal: interictal windows not within ANY episode context
    any_ep = np.zeros(n, dtype=bool)
    for m in ep_ctx:
        any_ep |= m
    pure_inter_mask = (~any_ep) & (labels == 0)

    # Generate q rounds ensuring every seizure appears in test at least once
    rng      = np.random.default_rng(seed)
    untested = set(range(ms))
    rounds   = []

    for _ in range(q):
        if untested:
            n_from_new  = min(len(untested), n_test_seiz)
            forced      = list(rng.choice(sorted(untested), n_from_new, replace=False))
            remaining   = n_test_seiz - n_from_new
            if remaining > 0:
                pool  = [i for i in range(ms) if i not in forced]
                extra = list(rng.choice(pool, remaining, replace=False))
                test_idx = sorted(forced + extra)
            else:
                test_idx = sorted(forced)
        else:
            test_idx = sorted(rng.choice(ms, n_test_seiz, replace=False).tolist())
        train_idx = [i for i in range(ms) if i not in test_idx]
        rounds.append((train_idx, test_idx))
        untested -= set(test_idx)

    rounds_thalgo, rounds_lgbm, pairs_selected = [], [], []

    for r_idx, (train_seiz, test_seiz) in enumerate(rounds):

        # ── Build training mask ────────────────────────────────────────────────
        # Windows near training seizures (preictal + interictal context)
        train_ep_union = np.zeros(n, dtype=bool)
        for i in train_seiz:
            train_ep_union |= ep_ctx[i]

        # Exclude test episode contexts from training
        test_ep_union = np.zeros(n, dtype=bool)
        for i in test_seiz:
            test_ep_union |= ep_ctx[i]

        # Train = (training episode windows  ∪  ALL pure interictal) \ test contexts
        train_mask = (train_ep_union | pure_inter_mask) & ~test_ep_union

        F_tr      = F[train_mask]
        labels_tr = labels[train_mask]

        if (labels_tr == 2).sum() == 0 or (labels_tr == 0).sum() == 0:
            continue

        # ── ThAlgo selection ───────────────────────────────────────────────────
        idx_S1, idx_S2, theta_S1, theta_S2 = thalgo_select(F_tr, labels_tr)
        f_S1_name = col_names[idx_S1].replace("MAACD_", "")
        f_S2_name = col_names[idx_S2].replace("MAACD_", "")
        pairs_selected.append((f_S1_name, f_S2_name))

        # AND gate over full timeline
        y_th_full = ((F[:, idx_S1] >= theta_S1) &
                     (F[:, idx_S2] >= theta_S2)).astype(int)

        # ── LightGBM training ──────────────────────────────────────────────────
        y_tr_bin = (labels_tr > 0).astype(int)
        do_lgbm  = len(np.unique(y_tr_bin)) == 2
        if do_lgbm:
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(F_tr)
            pos_w  = (y_tr_bin == 0).sum() / max((y_tr_bin == 1).sum(), 1)
            model  = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_w, random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(X_tr, y_tr_bin)

        # ── Evaluate on each test seizure ──────────────────────────────────────
        r_th, r_lgbm = [], []
        for k in test_seiz:
            ep    = episodes[k]
            ctx_s = max(0, ep["preictal_start"] - int(MAX_INTERICTAL_SEC / STEP_SEC))
            ctx_e = min(n, ep["ictal_end"] + int(POSTICTAL_BUFFER_SEC / STEP_SEC))
            ctx_lbl = labels[ctx_s:ctx_e]

            if (ctx_lbl == 2).sum() == 0:
                continue

            # ThAlgo
            y_ctx    = y_th_full[ctx_s:ctx_e]
            sens_p, fah_p, _ = paper_metrics(y_ctx, ctx_lbl)
            sens_a, fah_a, _ = alarm_metrics(y_ctx, ctx_lbl)

            # AUC (continuous bottleneck score)
            ctx_mask  = np.zeros(n, dtype=bool); ctx_mask[ctx_s:ctx_e] = True
            test_mask = ctx_mask & ~train_mask
            y_bin     = (labels[test_mask] > 0).astype(int)
            score     = np.minimum(F[test_mask, idx_S1] / (theta_S1 + 1e-12),
                                   F[test_mask, idx_S2] / (theta_S2 + 1e-12))
            try:
                auc = roc_auc_score(y_bin, score) if len(np.unique(y_bin)) > 1 else float("nan")
            except Exception:
                auc = float("nan")

            r_th.append({"auc": auc, "sens_paper": sens_p, "fah_paper": fah_p,
                         "sens_alarm": sens_a, "fah_alarm": fah_a})

            # LightGBM
            if do_lgbm:
                F_te     = F[test_mask]
                X_te     = scaler.transform(F_te)
                proba_te = model.predict_proba(X_te)[:, 1]
                proba_ctx        = np.zeros(ctx_e - ctx_s)
                te_in_ctx        = test_mask[ctx_s:ctx_e]
                proba_ctx[te_in_ctx] = proba_te
                y_lgbm_ctx = smooth_proba(proba_ctx)
                sens_l, fah_l, _ = alarm_metrics(y_lgbm_ctx, ctx_lbl)
                try:
                    auc_l = roc_auc_score(y_bin, proba_te) if len(np.unique(y_bin)) > 1 else float("nan")
                except Exception:
                    auc_l = float("nan")
                r_lgbm.append({"auc": auc_l, "sens_alarm": sens_l, "fah_alarm": fah_l})

        if not r_th:
            continue

        def _avg(lst, key):
            return float(np.nanmean([x[key] for x in lst]))

        th_row = {
            "round": r_idx,
            "f_S1": f_S1_name, "theta_S1": theta_S1,
            "f_S2": f_S2_name, "theta_S2": theta_S2,
            "auc":        _avg(r_th, "auc"),
            "sens_paper": _avg(r_th, "sens_paper"),
            "fah_paper":  _avg(r_th, "fah_paper"),
            "sens_alarm": _avg(r_th, "sens_alarm"),
            "fah_alarm":  _avg(r_th, "fah_alarm"),
            "n_inter_tr": int(pure_inter_mask.sum()),
        }
        rounds_thalgo.append(th_row)

        if r_lgbm:
            rounds_lgbm.append({
                "round": r_idx,
                "auc":        _avg(r_lgbm, "auc"),
                "sens_alarm": _avg(r_lgbm, "sens_alarm"),
                "fah_alarm":  _avg(r_lgbm, "fah_alarm"),
            })

        if verbose:
            print(f"    Round {r_idx+1}/{q} "
                  f"(train seiz={train_seiz} test seiz={test_seiz} "
                  f"inter_tr={th_row['n_inter_tr']}): "
                  f"ThAlgo paper sens={th_row['sens_paper']:.2f} FA/h={th_row['fah_paper']:.2f} | "
                  f"ThAlgo alarm sens={th_row['sens_alarm']:.2f} FA/h={th_row['fah_alarm']:.2f} | "
                  f"S1={f_S1_name}  S2={f_S2_name}")

    if not rounds_thalgo:
        return None

    if verbose:
        def _m(lst, key): return np.nanmean([x[key] for x in lst])
        print(f"  ── ThAlgo (paper): AUC={_m(rounds_thalgo,'auc'):.3f}  "
              f"Sens={_m(rounds_thalgo,'sens_paper'):.3f}  "
              f"FA/h={_m(rounds_thalgo,'fah_paper'):.2f}")
        print(f"  ── ThAlgo (alarm): AUC={_m(rounds_thalgo,'auc'):.3f}  "
              f"Sens={_m(rounds_thalgo,'sens_alarm'):.3f}  "
              f"FA/h={_m(rounds_thalgo,'fah_alarm'):.2f}")
        if rounds_lgbm:
            print(f"  ── LightGBM MAACD: AUC={_m(rounds_lgbm,'auc'):.3f}  "
                  f"Sens={_m(rounds_lgbm,'sens_alarm'):.3f}  "
                  f"FA/h={_m(rounds_lgbm,'fah_alarm'):.2f}")

    return {
        "patient":        patient,
        "n_seizures":     ms,
        "thalgo":         rounds_thalgo,
        "lgbm":           rounds_lgbm,
        "pairs_selected": pairs_selected,
    }


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(all_results):

    def _m(lst, key):
        return np.nanmean([x[key] for x in lst])

    print("\n" + "=" * 75)
    print(f"FINAL SUMMARY — ThAlgo paper k-fold CV  (preictal={PREICTAL_SEC}s)")
    print("=" * 75)

    header = f"  {'Patient':<10} {'#Seiz':>6} {'q':>3} {'AUC':>7} {'Sens':>7} {'FA/h':>7}"
    sep    = "  " + "-" * 46

    for strategy, label in [
        ("thalgo_paper", "ThAlgo  (paper metric — ≥1 positive in preictal)"),
        ("thalgo_alarm", f"ThAlgo  (alarm metric — {ALARM_CONSEC} consec. positives)"),
        ("lgbm_alarm",   "LightGBM MAACD-only (alarm metric)"),
    ]:
        print(f"\n{'─'*75}")
        print(f"  {label}")
        print(header); print(sep)
        aucs, senss, fahs = [], [], []
        for r in all_results:
            src   = r["thalgo"] if "thalgo" in strategy else r["lgbm"]
            if not src:
                continue
            s_key = "sens_paper" if "paper" in strategy else "sens_alarm"
            f_key = "fah_paper"  if "paper" in strategy else "fah_alarm"
            auc   = _m(src, "auc")
            sens  = _m(src, s_key)
            fah   = _m(src, f_key)
            q_val = len(src)
            print(f"  {r['patient']:<10} {int(r['n_seizures']):>6} {q_val:>3} "
                  f"{auc:>7.3f} {sens:>7.3f} {fah:>7.2f}")
            aucs.append(auc); senss.append(sens); fahs.append(fah)
        print(sep)
        print(f"  {'MEAN':<10} {'':>6} {'':>3} "
              f"{np.nanmean(aucs):>7.3f} {np.mean(senss):>7.3f} {np.mean(fahs):>7.2f}")

    print(f"\n{'─'*75}")
    print("  SELECTED PAIRS (f_S1 | f_S2) — most frequent across rounds")
    print(f"  {'Patient':<10}  f_S1 (best preictal rise)       f_S2 (best interictal silence)")
    print(f"  {'-'*70}")
    for r in all_results:
        s1_c = Counter(p[0] for p in r["pairs_selected"])
        s2_c = Counter(p[1] for p in r["pairs_selected"])
        print(f"  {r['patient']:<10}  "
              f"{s1_c.most_common(1)[0][0]:<33}  "
              f"{s2_c.most_common(1)[0][0]}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    today    = date.today().strftime("%Y-%m-%d")
    out_file = f"results_THALGO_{today}.txt"

    header_line = (f"ThAlgo paper k-fold CV on {len(patients)} patients\n"
                   f"MAACD w={MAACD_W} p={MAACD_P} | preictal={PREICTAL_SEC}s | "
                   f"alarm={ALARM_CONSEC} consec.\n")
    print(header_line)

    all_results = []
    for p in patients:
        r = kfold_patient(p, verbose=True)
        if r is not None:
            all_results.append(r)

    if all_results:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(header_line)
            print_summary(all_results)
        output = buf.getvalue()
        print(output)
        with open(out_file, "w") as f:
            f.write(output)
        print(f"\nResults saved to {out_file}")
