import numpy as np 
from scipy.signal import hilbert, welch, find_peaks
from antropy import perm_entropy, higuchi_fd


EEG_FEATURES_NAMES = ["BP_DELTA", "BP_THETA", "BP_ALPHA", "BP_BETA", "STD BP_DELTA", "STD BP_THETA", "STD BP_ALPHA", "STD BP_BETA", "RBP_DELTA", "RBP_THETA", "RBP_ALPHA", "RBP_BETA", "TKEO MEAN", "TKEO STD", "HJORTH ACTIVITY", "HJORTH MOBILITY", "HJORTH COMPLEXITY", "SPECTRAL ENTROPY", "PERMUTATION ENTROPY", "STD PERMUTATION ENTROPY", "FRACTAL DIMENSION", "STD FRACTAL DIMENSION", "KURTOSIS", "PLI", "DPLI", "WPLI", "PLI EMA", "WPLI EMA", "SPECTRAL ENTROPY EMA"]
ECG_FEATURES_NAMES = ["MEAN RR", "SDNN", "RMSSD", "HR", "MEAN EKG", "STD EKG"]

def pli(phase_x, phase_y):

	phase_diff = phase_x - phase_y

	pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))

	return pli

def wpli(hx, hy):

	im = np.imag(hx * np.conj(hy))

	num = np.abs(np.mean(im))
	den = np.mean(np.abs(im))

	return num / den

def dpli(phase_x, phase_y):

	phase_diff = phase_x - phase_y

	return np.mean(phase_diff > 0)

def ema(x, alpha):
    ema_vals = np.zeros_like(x)
    ema_vals[0] = x[0]
    
    for i in range(1, len(x)):
        ema_vals[i] = alpha * x[i] + (1 - alpha) * ema_vals[i-1]
        
    return ema_vals

def extractFeaturesEEG(windows, sfreq):

	plis_mean = []
	wplis_mean = []
	dplis_mean = []
	print(f"size windows:{len(windows)}")
	print(windows.shape)
	print("flat channels:", np.sum(np.var(windows, axis=2) == 0))

	perm_enthropies_mean = []
	perm_entropies_std = []

	higuchi_fd_mean = []
	higuchi_fd_std = []

	dx = np.diff(windows, axis=2)
	ddx = np.diff(dx, axis=2)
	
	activities = np.var(windows, axis=2) + 1e-10
	mobilities = np.sqrt(np.var(dx, axis=2)/activities)
	complexities = np.sqrt(np.var(ddx, axis=2) / np.var(dx, axis=2) + 1e-10)  
	print(f"var(diff): {np.var(dx, axis=2) + 1e-10}")
	f, psd = welch(windows, fs=sfreq, axis=2, nperseg=256)
	print("psd min:", psd.min())
	print("psd max:", psd.max())
	print("windows var min:", np.min(np.var(windows, axis=2)))

	delta = (f >= 0.5) & (f <= 4)
	theta = (f >= 4) & (f <= 8)
	alpha = (f >= 8) & (f <= 13)
	beta  = (f >= 13) & (f <= 30)

	delta_bps = np.trapezoid(psd[:, :, delta], f[delta], axis=2)
	theta_bps = np.trapezoid(psd[:, :, theta], f[theta], axis=2)
	alpha_bps = np.trapezoid(psd[:, :, alpha], f[alpha], axis=2)
	beta_bps = np.trapezoid(psd[:, :, beta],  f[beta], axis=2)

	total_power = np.trapezoid(psd, f, axis=2)
	
	delta_rbps = delta_bps / total_power
	theta_rbps = theta_bps / total_power
	alpha_rbps = alpha_bps / total_power
	beta_rbps  = beta_bps  / total_power

	print("zeros total_power:", np.sum(total_power == 0))
	print("nan total_power:", np.isnan(total_power).sum())
	print("inf total_power:", np.isinf(total_power).sum())

	print("zeros delta_bps:", np.sum(delta_bps == 0))
	print("nan delta_bps:", np.isnan(delta_bps).sum())
	print("nan psd:", np.isnan(psd).sum())
	print("inf psd:", np.isinf(psd).sum())
	print("")
	p = psd / np.sum(psd, axis=2, keepdims=True)
	spectral_entropies = -np.sum(p * np.log2(p + 1e-12), axis=2)
	spectral_entropies /= np.log2(p.shape[2])

	mean = np.mean(windows, axis=2, keepdims=True)
	std = np.std(windows, axis=2, keepdims=True) + 1e-8

	kurtosis = np.mean(((windows - mean)/std)**4, axis=2) - 3	

	psi = windows[:,:,1:-1]**2 - windows[:,:,:-2]*windows[:,:,2:]

	tkeo_mean = np.mean(psi, axis=2)
	tkeo_std = np.std(psi, axis=2)

	#synchronization measures
	for window in windows:

		plis = []
		wplis = []
		dplis = []

		perm_enthropies = []

		higuchi_fds = []

		analytic = hilbert(window, axis=1)
		phases = np.angle(analytic)

		for i in range(len(window)):

			for j in range(i+1, len(window)):

				x = analytic[i]
				y = analytic[j]
				phase_x = phases[i]
				phase_y = phases[j]

				plis.append(pli(phase_x, phase_y))
				dplis.append(dpli(phase_x, phase_y))
				wplis.append(wpli(x, y))

			perm_enthropies.append(perm_entropy(window[i], order=3))
			higuchi_fds.append(higuchi_fd(window[i]))

		plis_mean.append(np.mean(plis))
		wplis_mean.append(np.mean(wplis))
		dplis_mean.append(np.mean(dplis))

		perm_enthropies_mean.append(np.mean(perm_enthropies))
		perm_entropies_std.append(np.std(perm_enthropies))

		higuchi_fd_mean.append(np.mean(higuchi_fds))
		higuchi_fd_std.append(np.std(higuchi_fds))

		w = 7
		alpha = 2 / (w + 1)

		pli_ema = ema(np.array(plis_mean), alpha)
		wpli_ema = ema(np.array(wplis_mean), alpha)
		entropy_ema = ema(np.array(spectral_entropies.mean(axis=1)), alpha)



	eeg_features = [delta_bps.mean(axis=1), 
			theta_bps.mean(axis=1), 
			alpha_bps.mean(axis=1), 
			beta_bps.mean(axis=1),

			delta_bps.std(axis=1), 
			theta_bps.std(axis=1), 
			alpha_bps.std(axis=1), 
			beta_bps.std(axis=1), 

			delta_rbps.mean(axis=1), 
			theta_rbps.mean(axis=1), 
			alpha_rbps.mean(axis=1), 
			beta_rbps.mean(axis=1), 

			tkeo_mean.mean(axis=1),
			tkeo_std.mean(axis=1),

			activities.mean(axis=1), 
			mobilities.mean(axis=1), 
			complexities.mean(axis=1), 

			spectral_entropies.mean(axis=1), 
			perm_enthropies_mean,
			perm_entropies_std,

			higuchi_fd_mean,
			higuchi_fd_std,

			kurtosis.mean(axis=1),

			plis_mean, 
			dplis_mean, 
			wplis_mean,

			pli_ema,
			wpli_ema,
			entropy_ema
			]

	return np.array(eeg_features).T


def extractFeaturesECG(ecg_windows, sfreq):

    n_windows = len(ecg_windows)
    features = np.zeros((n_windows, 6))

    for i, window in enumerate(ecg_windows):

        peaks, _ = find_peaks(window, distance=0.3*sfreq)

        if len(peaks) < 2:
            continue

        rr = np.diff(peaks) / sfreq

        mean_rr = np.mean(rr)
        sdnn = np.std(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr)**2))
        hr = 60 / mean_rr

        features[i] = [
            mean_rr,
            sdnn,
            rmssd,
            hr,
            np.mean(window),
            np.std(window),
        ]

    return features


