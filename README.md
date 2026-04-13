# Epileptic Seizure Recognition from EEG Signals Using Classical Machine Learning

*A Study on the SIENA Scalp EEG — Yanis AMARA, ESIEE Paris, 2026*

> The full methodology is described in `seizure_recognition_paper-1.pdf`.

---

## Overview

Epilepsy affects ~50 million people worldwide. This project implements a patient-specific seizure prediction pipeline on the **SIENA Scalp EEG** database, combining spectro-temporal feature extraction, phase synchrony connectivity (PLI/wPLI), and a novel **OR-Gate** architecture that decouples a LightGBM classifier from a focal MAACD alarm.

**The repository includes pre-extracted features** in `final_assignment/dataset/` — no raw EDF files or preprocessing step is needed to run the pipelines.

---

## Dataset

**SIENA Scalp EEG Database** — Università di Siena, available on [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/).

| Property | Value |
|---|---|
| Patients total | 14 (9M / 5F, age 20–71) |
| Patients used | **12** (PN07 and PN11 excluded: only 1 seizure each) |
| Seizures used | **43** |
| Total recording | ~128 hours |
| Sampling rate | 512 Hz (resampled to 128 Hz) |
| Channels | Up to 32 (10–20 system) |
| Seizure type | Focal epilepsy |

**Patient summary:**

| Patient | Sex | Seizures | Used |
|---|---|---|---|
| PN00 | M | 5 | Yes |
| PN01 | F | 2 | Yes |
| PN03 | M | 2 | Yes |
| PN05 | M | 3 | Yes |
| PN06 | F | 2 | Yes |
| PN07 | M | 1 | No (< 2 seizures) |
| PN09 | M | 3 | Yes |
| PN10 | F | 10 | Yes |
| PN11 | M | 1 | No (< 2 seizures) |
| PN12 | M | 3 | Yes |
| PN13 | M | 3 | Yes |
| PN14 | F | 4 | Yes |
| PN16 | M | 2 | Yes |
| PN17 | F | 2 | Yes |

The average peri-ictal/total window ratio is ~9.9%, highlighting the strong **class imbalance** that shapes every design choice.

---

## Repository Structure

```
final_assignment/                   ← Run all pipelines from here
│
├── dataset/                        ← Pre-extracted features (no EDF needed)
│   └── PNxx/csv/
│       ├── PNxx-k/                 ← Per-recording features
│       │   ├── PNxx-k_EEG_X.csv   ← EEG feature matrix (one row = one 6s window)
│       │   ├── PNxx-k_ECG_X.csv   ← ECG features (when available)
│       │   └── PNxx-k_Y.csv       ← Labels (0=interictal, 1=ictal, 2=preictal)
│       └── PNxx_EEG_X.csv         ← Patient-level concatenated features
│
├── seizure_prediction_thalgo.py    ← Baseline: ThAlgo (Detti et al. 2019)
├── P1_lgbm_all_feats.py            ← P1: LightGBM on all features
├── P2_pli_and_no_pli.py            ← P2: LightGBM without connectivity features
├── P3_fused.py                     ← P3: Fused LightGBM + MAACD score
├── P4_or_gate_final.py             ← P4: OR-Gate (proposed method)
├── feature_importance_analysis.py  ← Feature ranking (MI, AUC, LightGBM gain)
│
├── features_extraction.py          ← Feature computation code
├── label_extraction.py             ← Label extraction from annotation files
└── preprocessing.py                ← EDF → CSV pipeline (reference only)
```

---

## Preprocessing (already done)

Raw EDF files were processed with `preprocessing.py` using [MNE-Python](https://mne.tools/):

- Butterworth bandpass filter: **0.5–40 Hz**
- Resampled to **128 Hz** (Nyquist = 64 Hz, sufficient for all 5 EEG bands)
- Sliding windows: **6 seconds**, step **1 second** (≈83% overlap)
- Processed in 1-hour chunks to manage memory

This produces the CSVs already present in `final_assignment/dataset/`.

---

## Feature Extraction

For each 6-second window, **18 features per channel** are computed, plus full pairwise PLI/wPLI connectivity and their MAACD transforms.

| Group | Features | Count/channel |
|---|---|---|
| **Spectral** | Band power: δ (0.5–4), θ (4–8), α (8–13), β (13–30), γ (30–40) Hz | 5 |
| **Spectral** | Relative band power (RBP): Bk / Σ Bk | 5 |
| **Temporal** | Hjorth: Activity, Mobility, Complexity | 3 |
| **Temporal** | Permutation entropy (order m=3) | 1 |
| **Temporal** | Higuchi fractal dimension (HFD) | 1 |
| **Temporal** | Spectral entropy | 1 |
| **Temporal** | Mean, Std | 2 |

**Band power** uses Welch PSD (256-sample Hann window):

$$B_k = \int_{f_k^{\text{low}}}^{f_k^{\text{high}}} S(f)\,df, \qquad \text{RBP}_k = \frac{B_k}{\sum_k B_k + \varepsilon}$$

**Hjorth parameters:**

$$\text{Act.} = \text{Var}(x), \quad \text{Mob.} = \sqrt{\frac{\text{Var}(x')}{\text{Var}(x)+\varepsilon}}, \quad \text{Comp.} = \frac{\sqrt{\text{Var}(x'')/(\text{Var}(x')+\varepsilon)}}{\text{Mob.}+\varepsilon}$$

### Connectivity — PLI and wPLI

Phase synchrony is computed for **all Nc(Nc−1)/2 channel pairs** in the alpha band (8–13 Hz). To avoid filter edge effects, the pipeline operates on the **continuous signal** (up to 1 hour) before windowing:

1. Alpha bandpass (Butterworth 4th order, `filtfilt`)
2. Differentiation
3. Hilbert transform → analytic signal

$$\text{PLI}_{ij}[w] = \left|\frac{1}{W-1}\sum_t \text{sgn}(\sin(\phi_i[t] - \phi_j[t]))\right|$$

$$\text{wPLI}_{ij}[w] = \frac{|\mathbb{E}[\text{Im}(\tilde{x}_i \tilde{x}_j^*)]|}{\mathbb{E}[|\text{Im}(\tilde{x}_i \tilde{x}_j^*)|] + \varepsilon}$$

### MAACD

The **MAACD** (Moving Average Acceleration-Deceleration, Detti et al. 2019) transforms each PLI/wPLI time series into a trend-elevation measure inspired by financial analysis:

$$\text{MAACD}[t] = \text{EMA}(\text{PLI}[t],\ w=7) - \min_{s \in [t-26,\,t]} \text{EMA}(\text{PLI}[s],\ w=7)$$

with EMA span w=7 (α = 2/(w+1) = 0.25) and rolling-min window p=27. A rising MAACD captures progressive synchrony increases that precede seizure onset.

### Total feature dimensions

| Patients | Channels (Nc) | Temp.+Spectr. | PLI+wPLI pairs | MAACD | **Total** |
|---|---|---|---|---|---|
| PN00, PN01 | 29 | 522 | 812 | 812 | **2 146** |
| PN03–PN13 | 28 | 504 | 756 | 756 | **2 016** |
| PN14, PN16, PN17 | 27 | 486 | 702 | 702 | **1 890** |

---

## Labeling

Three-class labeling per window:

| Label | Class | Definition |
|---|---|---|
| **0** | Interictal | Background — excluding 1000 s post-ictal; capped at 3600 s before each seizure |
| **1** | Ictal | Within the annotated seizure interval |
| **2** | Preictal | 5 min (300 s) before ictal onset (Detti et al. 2019) |

Total across 12 patients: 132 221 interictal + 14 526 peri-ictal = **146 747 windows**.

---

## Experimental Protocol

### LOSO cross-validation

**Leave-One-Seizure-Out (LOSO)**: for each patient with N seizures, N folds are run — each fold leaves one seizure episode out as the test set. This is preferred over LOPO (Leave-One-Patient-Out) because brain signatures are highly patient-specific (cross-patient models achieve near-zero sensitivity).

Per fold:
- `StandardScaler` fitted on training data only (no data leakage)
- Class-imbalance handled by inverse-frequency weighting: w⁺ = N⁻/N⁺

### Alarm post-processing (shared across all pipelines)

1. **Majority-vote smoothing** over 10 windows
2. **Alarm trigger**: ≥ 6 consecutive positive classifications
3. **Refractory period**: 10 minutes after each alarm (no new alarm)

### Evaluation metrics

$$\text{Sensitivity} = \frac{TP}{TP+FN}, \qquad \text{FA/h} = \frac{\text{false alarms}}{\text{interictal hours}}, \qquad \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t))\,dt$$

**Clinical target** (Winterhalder et al. 2003): Sensitivity > 0.70 and FA/h < 0.15.

---

## Pipelines and Classifiers

Five pipelines are compared under the LOSO protocol:

| ID | Name | Features | Classifier |
|---|---|---|---|
| Baseline | **ThAlgo** | MAACD of 2 PLI+wPLI pairs (702–812) | Rule-based thresholds |
| P1 | **LGBM-All** | All features (~1900–2200) | LightGBM |
| P2 | **LGBM-NoPLI** | Spectro-temporal only (486–522) | LightGBM |
| P3 | **Fused** | All + scalar MAACD score | LGBM + α-weighted mix |
| **P4** | **OR-Gate** | Non-PLI + 1 focal wPLI pair | **LightGBM ∨ MAACD** |

### Baseline — ThAlgo (`seizure_prediction_thalgo.py`)

Reproduces Detti et al. (2019). Selects two electrode pairs per patient per fold:
- **S₁** (*signal pair*): the pair with highest mean MAACD during peri-ictal training windows
- **S₂** (*silence pair*): the pair with lowest MAACD during interictal training windows

Alarm rule:

$$\text{Alarm}[t] = 1 \iff \text{MAACD}_{S_1}[t] > \theta_1 \ \wedge \ \text{MAACD}_{S_2}[t] < \theta_2$$

Thresholds θ₁ and θ₂ are calibrated on the training fold. All 12 patients select a wPLI pair for S₁.

### P1 — LightGBM: Full Feature Set (`P1_lgbm_all_feats.py`)

LightGBM trained naively on all ~1900–2200 features.

Hyperparameters: n_est=600, η=0.03, max_depth=7, 31 leaves, subsample=0.8, colsample=0.8, fixed decision threshold=0.55.

### P2 — LightGBM: No Connectivity (`P2_pli_and_no_pli.py`)

Same LightGBM architecture as P1 but all PLI, wPLI, MAACD-PLI, and MAACD-wPLI columns are removed (486–522 features). Decision threshold tuned per fold to target FA/h ≤ 1.

Goal: quantify the contribution of connectivity features to LightGBM.

### P3 — Fused LightGBM + MAACD Score (`P3_fused.py`)

Combines P1's probability output with a scalar MAACD score (mean of all normalised MAACD-PLI and MAACD-wPLI values):

$$s_{\text{fused}}[w] = \alpha \cdot P_{\text{LGB}}[w] + (1-\alpha) \cdot s_{\text{MAACD}}[w]$$

α ∈ {0.0, 0.1, …, 1.0} is optimised by grid-search on training folds (targeting sensitivity with FA/h ≤ 2). The decision threshold is calibrated on the fused score.

### P4 — OR-Gate: LightGBM ∪ Focal MAACD (`P4_or_gate_final.py`) — *Proposed method*

The proposed architecture **decouples** spectro-temporal classification from connectivity in two independent arms:

**Arm 1 — LightGBM (no connectivity):** same as P2 (486–522 features), threshold calibrated per fold for FA/h ≤ 1.

**Arm 2 — Focal MAACD:** a single patient-specific wPLI pair is selected per fold (the pair with highest mean MAACD on training peri-ictal windows). A threshold τ̂ is calibrated on training interictal data for FA/h ≤ 1. Alarm fires when k consecutive MAACD values exceed τ̂.

**Fusion rule (OR gate):**

$$\text{Alarm}[t] = \text{Alarm}_{\text{LGB}}[t] \vee \text{Alarm}_{\text{MAACD}}[t]$$

A shared refractory period suppresses both arms after any alarm. Hyperparameter grid: ALARM_CONSEC ∈ {6, 7, 8, 10}, threshold ∈ {0.60, 0.70, 0.80}.

The decoupling is critical: any direct combination (P3) or inclusion of MAACD as input to LightGBM (P1) degrades performance because global MAACD noise contaminates the classifier.

---

## Results

### Feature importance (combined MI + AUC + LightGBM gain score)

Top families by combined score: **θ-BP (0.261) > HFD (0.206) > α-BP (0.194) > Hjorth Act. (0.189) > STD (0.188)**. Connectivity features rank low (PLI: 0.131, wPLI: 0.061, MAACD-PLI: 0.059), explaining why including them in LightGBM does not help.

### Cross-pipeline comparison (12 patients, 43 seizures)

| Pipeline | AUC | Sensitivity | FA/h |
|---|---|---|---|
| Baseline (ThAlgo) | 0.513 | 0.764 | 5.32 |
| P1 — LGBM-All | 0.679 | 0.374 | 1.30 |
| P2 — LGBM-NoPLI | 0.677 | 0.432 | 1.52 |
| P3 — Fused | 0.642 | 0.318 | 0.82 |
| **P4 — OR-Gate** | **0.677** | **0.878** | **1.35** |

*Clinical target: Sensitivity > 0.70, FA/h < 0.15*

**P4 is the only pipeline exceeding the clinical sensitivity target**, with Sensitivity = 1.000 for 8 out of 12 patients. FA/h = 1.35 remains above the clinical threshold of 0.15, but Sensitivity = 87.8% is clinically significant in a supervised hospital monitoring context.

**Key observations:**
1. The OR-gate architecture is essential — direct fusion (P3) or feeding MAACD into LightGBM (P1) degrades sensitivity because MAACD noise corrupts the classifier
2. Connectivity features (PLI/wPLI) don't help LightGBM: P2 (no PLI) ≥ P1 (all features) on all metrics
3. The focal MAACD arm recovers seizures invisible to LightGBM (e.g. PN01: 0.00→1.00, PN03: 0.00→1.00), where the peri-ictal change is concentrated on a single focal wPLI pair

---

## Quick Start

All scripts must be run from the `final_assignment/` directory (the `./dataset/` path is relative).

```bash
cd final_assignment/

# Install dependencies
pip install -r ../requirements.txt

# Baseline: ThAlgo (Detti et al. 2019 reproduction)
python seizure_prediction_thalgo.py

# P1: LightGBM on all features
python P1_lgbm_all_feats.py

# P2: LightGBM without connectivity features
python P2_pli_and_no_pli.py

# P3: Fused LightGBM + MAACD score
python P3_fused.py

# P4: OR-Gate (proposed method) — best results
python P4_or_gate_final.py

# Feature importance analysis
python feature_importance_analysis.py
```

No raw EDF files are needed. The `dataset/` folder already contains the pre-extracted feature CSVs for all 12 patients.

---

## References

1. P. Detti, G. Vatti, G. de Lara Matiasek, "EEG synchronization analysis for seizure prediction," *Processes*, vol. 8, no. 7, p. 846, 2020.
2. B. Hjorth, "EEG analysis based on time domain properties," *Electroencephalogr. Clin. Neurophysiol.*, vol. 29, 1970.
3. T. Higuchi, "Approach to an irregular time series on the basis of the fractal theory," *Physica D*, vol. 31, 1988.
4. M. Vinck et al., "An improved index of phase-synchronization for electrophysiological data," *NeuroImage*, vol. 55, no. 4, 2011.
5. M. Winterhalder et al., "The seizure prediction characteristic," *Epilepsy & Behavior*, vol. 4, no. 3, 2003.
6. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, 2000. https://physionet.org/content/siena-scalp-eeg/1.0.0/

---

*By Yanis AMARA — ESIEE Paris, 2026*
