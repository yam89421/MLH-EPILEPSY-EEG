"""
Append pairwise PLI/WPLI columns to existing per-recording EEG CSVs
====================================================================
For each recording, computes PLI and WPLI for ALL channel pairs (upper
triangle: n_ch*(n_ch-1)/2 pairs) using the correct continuous-signal
pipeline (filter → diff → Hilbert on the full chunk, then slice per window).

New columns appended: PLI_CH{i}_CH{j} and WPLI_CH{i}_CH{j} for all i < j
(1-indexed, matching the channel order in the existing CHx naming convention).

Existing columns (per-channel mean PLI_CH*, WPLI_CH*) are left untouched.
After all recordings are updated, patient-level CSVs are rebuilt via mergeData.

29 channels → 406 pairs × 2 = 812 new columns per recording.
"""

import mne
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, hilbert

from label_extraction import records, dataset_path

# ── Parameters (must match preprocessing.py) ────────────────────────────────
WINDOW         = 6
SFREQ          = 128
CHUNK_DURATION = 60 * 60
window_size    = int(WINDOW * SFREQ)
step           = SFREQ

EEG_CHANNELS_REF = [
    'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5',
    'Fc1', 'Fc5', 'Cp1', 'Cp5', 'F9',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6',
    'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10',
]

MONTAGE = mne.channels.make_standard_montage("standard_1020")


# ── Connectivity ─────────────────────────────────────────────────────────────

def _pli(phase_x, phase_y):
    return np.abs(np.mean(np.sign(np.sin(phase_x - phase_y))))


def _wpli(hx, hy):
    im  = np.imag(hx * np.conj(hy))
    return np.abs(np.mean(im)) / (np.mean(np.abs(im)) + 1e-12)


# ── Core: compute all pairs on continuous chunk ───────────────────────────────

def compute_pairwise_continuous(eeg_uv, sfreq, step, window_size):
    """
    eeg_uv : (n_channels, n_times) µV, full continuous chunk.
    Returns:
      pli_all  : (n_windows, n_pairs)
      wpli_all : (n_windows, n_pairs)
      pair_labels : list of (i, j) tuples (0-indexed channel pairs)
    """
    n_channels, n_times = eeg_uv.shape

    nyq  = sfreq / 2.0
    b, a = butter(4, [8 / nyq, 13 / nyq], btype='band')
    cont_alpha    = filtfilt(b, a, eeg_uv, axis=1)
    cont_diff     = np.diff(cont_alpha, axis=1)
    cont_analytic = hilbert(cont_diff, axis=1)          # (n_ch, T-1)

    n_positions   = n_times - window_size + 1
    if n_positions <= 0:
        return None, None, None

    window_starts = np.arange(0, n_positions, step)
    n_windows     = len(window_starts)

    # pre-compute all pairs
    pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    n_pairs = len(pairs)

    pli_all  = np.zeros((n_windows, n_pairs))
    wpli_all = np.zeros((n_windows, n_pairs))

    for w_idx, t0 in enumerate(window_starts):
        t1           = t0 + window_size - 1
        analytic_win = cont_analytic[:, t0:t1]          # (n_ch, window_size-1)
        phases_win   = np.angle(analytic_win)

        for p_idx, (i, j) in enumerate(pairs):
            pli_all[w_idx, p_idx]  = _pli(phases_win[i],    phases_win[j])
            wpli_all[w_idx, p_idx] = _wpli(analytic_win[i], analytic_win[j])

    return pli_all, wpli_all, pairs


# ── Per-recording update ─────────────────────────────────────────────────────

def update_record(r):
    edf_file = r.split("/")[-1].strip()
    patient  = r.split("/")[0].strip()

    csv_dir = os.path.join(dataset_path, patient, "csv", edf_file.split(".")[0])
    eeg_csv = os.path.join(csv_dir, edf_file.split(".")[0] + "_EEG_X.csv")

    if not os.path.exists(eeg_csv):
        print(f"    CSV not found, skipping: {eeg_csv}")
        return

    # Load EDF
    raw = mne.io.read_raw_edf(
        os.path.join(dataset_path, r.strip()), preload=False, verbose=False
    )
    raw.rename_channels(lambda x: x.replace("EEG ", ""))
    raw.resample(SFREQ, npad="auto")

    eeg_ch  = [ch for ch in raw.ch_names if ch in EEG_CHANNELS_REF]
    raw_eeg = raw.copy().pick(eeg_ch)
    raw_eeg.filter(0.5, 40, verbose=False)
    raw_eeg.set_montage(MONTAGE, on_missing="ignore")

    # Channel indices are 1-based and follow EEG_CHANNELS_REF order
    ch_indices = {ch: EEG_CHANNELS_REF.index(ch) + 1 for ch in eeg_ch}

    n_channels    = len(eeg_ch)
    samples_chunk = int(CHUNK_DURATION * SFREQ)

    pli_chunks, wpli_chunks, pairs_ref = [], [], None

    for start in range(0, raw_eeg.n_times, samples_chunk):
        stop     = min(start + samples_chunk, raw_eeg.n_times)
        eeg_data, _ = raw_eeg[:, start:stop]

        if np.var(eeg_data) < 1e-12 or eeg_data.shape[1] < window_size:
            continue

        pli_c, wpli_c, pairs = compute_pairwise_continuous(
            eeg_data * 1e6, SFREQ, step, window_size
        )
        if pli_c is None:
            continue

        pli_chunks.append(pli_c)
        wpli_chunks.append(wpli_c)
        if pairs_ref is None:
            pairs_ref = pairs

    if not pli_chunks:
        print(f"    No valid chunks for {edf_file}, skipping.")
        return

    pli_new  = np.concatenate(pli_chunks,  axis=0)
    wpli_new = np.concatenate(wpli_chunks, axis=0)

    df = pd.read_csv(eeg_csv)

    if len(df) != len(pli_new):
        print(f"    Row mismatch for {edf_file}: "
              f"CSV={len(df)}, recomputed={len(pli_new)}. Skipping.")
        return

    # Build column names using global channel indices (1-based, EEG_CHANNELS_REF order)
    local_to_global = [EEG_CHANNELS_REF.index(ch) + 1 for ch in eeg_ch]

    new_cols = {}
    for p_idx, (li, lj) in enumerate(pairs_ref):
        gi = local_to_global[li]
        gj = local_to_global[lj]
        new_cols[f"PLI_CH{gi}_CH{gj}"]  = pli_new[:, p_idx]
        new_cols[f"WPLI_CH{gi}_CH{gj}"] = wpli_new[:, p_idx]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df.to_csv(eeg_csv, index=False)
    print(f"    Appended {len(new_cols)} pair columns → {eeg_csv}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    for r in records:
        edf_file = r.split("/")[-1].strip()
        patient  = r.split("/")[0].strip()
        print(f"[{patient}] {edf_file}")
        update_record(r)

    print("\nRe-merging patient-level CSVs...")
    from features_discrimination import mergeData
    mergeData()
    print("Done.")


if __name__ == "__main__":
    main()
