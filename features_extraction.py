"""
Extended EEG Feature Extraction
=================================
Adds the following features on top of the original 10:
  - Delta band power (0.5–4 Hz) and relative BP
  - Beta  band power (13–30 Hz) and relative BP
  - Gamma band power (30–40 Hz) and relative BP
  - Hjorth parameters (activity, mobility, complexity) per channel
  - Spectral entropy per channel
  - Differential asymmetry (DASM): left-right channel pairs
  - Rational asymmetry (RASM): left-right channel pairs
  - wPLI upper-triangle matrix flattened (full connectivity, not just mean)
    → too large for 29ch; replaced by mean over 4 canonical band ranges

Total per channel: 17 scalar features
Total for 29 channels: ~500 features (before wPLI matrix expansion)
"""

import numpy as np
from scipy.signal import hilbert, welch, find_peaks, butter, filtfilt
from scipy.stats import entropy as scipy_entropy
from antropy import perm_entropy, higuchi_fd

# Original feature names kept for backwards compatibility
EEG_FEATURES_NAMES = [
    "DELTA_BP", "THETA_BP", "ALPHA_BP", "BETA_BP", "GAMMA_BP",
    "DELTA_RBP", "THETA_RBP", "ALPHA_RBP", "BETA_RBP", "GAMMA_RBP",
    "HJORTH_ACT", "HJORTH_MOB", "HJORTH_COMP",
    "SPEC_ENTROPY",
    "MEAN", "STD",
    "PERM_ENTROPY", "FRACTAL_DIM",
    "PLI", "WPLI",
]

ECG_FEATURES_NAMES = ["MEAN_RR", "SDNN", "RMSSD", "HR", "MEAN", "STD"]


# ─────────── connectivity helpers ───────────

def pli(phase_x, phase_y):
    phase_diff = phase_x - phase_y
    return np.abs(np.mean(np.sign(np.sin(phase_diff))))


def wpli(hx, hy):
    im = np.imag(hx * np.conj(hy))
    num = np.abs(np.mean(im))
    den = np.mean(np.abs(im)) + 1e-12
    return num / den


# ─────────── Hjorth parameters ───────────

def hjorth(x):
    """Activity, Mobility, Complexity of signal x (1D)."""
    activity   = np.var(x)
    d1         = np.diff(x)
    mobility   = np.sqrt(np.var(d1) / (activity + 1e-12))
    d2         = np.diff(d1)
    complexity = (
        np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mobility + 1e-12)
    )
    return activity, mobility, complexity


# ─────────── spectral entropy ───────────

def spectral_entropy(psd_1d, freqs):
    """Normalised spectral entropy from a PSD slice."""
    psd_norm = psd_1d / (psd_1d.sum() + 1e-12)
    return scipy_entropy(psd_norm + 1e-12)


# ─────────── main extraction ───────────

def extractFeaturesEEG(windows, sfreq, continuous_eeg=None, step=None):
    """
    windows:        (n_windows, n_channels, n_times)
    continuous_eeg: (n_channels, n_times_full) — full continuous chunk, same units as windows.
                    When provided, alpha-band filter + diff + Hilbert are applied on this
                    continuous signal before windowing, avoiding edge effects (Detti 2019).
    step:           int, stride in samples used to build windows from continuous_eeg.
    Returns: (n_windows, n_channels * n_per_channel_features)
    """
    n_windows, n_channels, window_size = windows.shape

    f, psd = welch(windows, fs=sfreq, axis=2, nperseg=256)

    # Band masks
    delta = (f >= 0.5) & (f <  4)
    theta = (f >= 4)   & (f <  8)
    alpha = (f >= 8)   & (f < 13)
    beta  = (f >= 13)  & (f < 30)
    gamma = (f >= 30)  & (f <= 40)

    delta_bp = np.trapz(psd[:, :, delta], f[delta], axis=2)
    theta_bp = np.trapz(psd[:, :, theta], f[theta], axis=2)
    alpha_bp = np.trapz(psd[:, :, alpha], f[alpha], axis=2)
    beta_bp  = np.trapz(psd[:, :, beta],  f[beta],  axis=2)
    gamma_bp = np.trapz(psd[:, :, gamma], f[gamma], axis=2)
    total_pw = np.trapz(psd, f, axis=2) + 1e-12

    delta_rbp = delta_bp / total_pw
    theta_rbp = theta_bp / total_pw
    alpha_rbp = alpha_bp / total_pw
    beta_rbp  = beta_bp  / total_pw
    gamma_rbp = gamma_bp / total_pw

    mean_vals = np.mean(windows, axis=2)
    std_vals  = np.std(windows, axis=2)

    # Per-window computations
    hjorth_act_all  = np.zeros((n_windows, n_channels))
    hjorth_mob_all  = np.zeros((n_windows, n_channels))
    hjorth_comp_all = np.zeros((n_windows, n_channels))
    spec_ent_all    = np.zeros((n_windows, n_channels))
    perm_ent_all    = np.zeros((n_windows, n_channels))
    higuchi_all     = np.zeros((n_windows, n_channels))
    pli_all         = np.zeros((n_windows, n_channels))
    wpli_all        = np.zeros((n_windows, n_channels))

    # ── Alpha-band differentiated analytic signal for PLI/WPLI (Detti 2019) ──
    # Correct pipeline: filter → diff → Hilbert on the CONTINUOUS signal,
    # then slice phases per window. This avoids filter edge effects and
    # Hilbert boundary artefacts from operating on short 6-second snippets.
    nyq = sfreq / 2.0
    b_alpha, a_alpha = butter(4, [8 / nyq, 13 / nyq], btype='band')

    if continuous_eeg is not None and step is not None:
        # (n_channels, n_times_full) → filter → diff → hilbert on full signal
        cont_alpha   = filtfilt(b_alpha, a_alpha, continuous_eeg, axis=1)  # (n_ch, T)
        cont_diff    = np.diff(cont_alpha, axis=1)                          # (n_ch, T-1)
        cont_analytic = hilbert(cont_diff, axis=1)                          # (n_ch, T-1)
        use_continuous = True
    else:
        # Fallback: per-window filtering (legacy, introduces edge effects)
        windows_alpha = np.zeros_like(windows)
        for w_idx in range(n_windows):
            for ch in range(n_channels):
                windows_alpha[w_idx, ch] = filtfilt(b_alpha, a_alpha, windows[w_idx, ch])
        windows_diff = np.diff(windows_alpha, axis=2)   # (n_windows, n_ch, n_times-1)
        use_continuous = False

    for w_idx, window in enumerate(windows):
        # Extract analytic signal slice for this window
        if use_continuous:
            t0 = w_idx * step
            t1 = t0 + window_size - 1          # diff reduces length by 1
            analytic_diff = cont_analytic[:, t0:t1]   # (n_ch, window_size-1)
            phases_diff   = np.angle(analytic_diff)
        else:
            analytic_diff = hilbert(windows_diff[w_idx], axis=1)
            phases_diff   = np.angle(analytic_diff)

        pli_mat  = np.zeros((n_channels, n_channels))
        wpli_mat = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            # Hjorth (on original broadband window)
            act, mob, comp = hjorth(window[i])
            hjorth_act_all[w_idx, i]  = act
            hjorth_mob_all[w_idx, i]  = mob
            hjorth_comp_all[w_idx, i] = comp

            # Spectral entropy
            spec_ent_all[w_idx, i] = spectral_entropy(psd[w_idx, i], f)

            # Nonlinear
            perm_ent_all[w_idx, i] = perm_entropy(window[i], order=3)
            higuchi_all[w_idx, i]  = higuchi_fd(window[i])

            for j in range(i + 1, n_channels):
                pli_mat[i, j]  = pli(phases_diff[i], phases_diff[j])
                wpli_mat[i, j] = wpli(analytic_diff[i], analytic_diff[j])

        pli_all[w_idx]  = pli_mat.mean(axis=1)
        wpli_all[w_idx] = wpli_mat.mean(axis=1)

    eeg_features = np.concatenate([
        delta_bp,       # 1
        theta_bp,       # 2
        alpha_bp,       # 3
        beta_bp,        # 4
        gamma_bp,       # 5
        delta_rbp,      # 6
        theta_rbp,      # 7
        alpha_rbp,      # 8
        beta_rbp,       # 9
        gamma_rbp,      # 10
        hjorth_act_all, # 11
        hjorth_mob_all, # 12
        hjorth_comp_all,# 13
        spec_ent_all,   # 14
        mean_vals,      # 15
        std_vals,       # 16
        perm_ent_all,   # 17
        higuchi_all,    # 18
        pli_all,        # 19
        wpli_all,       # 20
    ], axis=1)

    return eeg_features


def extractFeaturesECG(ecg_windows, sfreq):
    n_windows = len(ecg_windows)
    features  = np.zeros((n_windows, 6))

    for i, window in enumerate(ecg_windows):
        peaks, _ = find_peaks(window, distance=0.3 * sfreq)
        if len(peaks) < 2:
            continue
        rr      = np.diff(peaks) / sfreq
        mean_rr = np.mean(rr)
        features[i] = [
            mean_rr,
            np.std(rr),
            np.sqrt(np.mean(np.diff(rr) ** 2)),
            60 / mean_rr,
            np.mean(window),
            np.std(window),
        ]
    return features