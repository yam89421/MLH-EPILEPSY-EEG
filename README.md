# Epileptic Seizure Prediction using EEG Signals

## Project Overview

Epilepsy is a neurological disorder characterized by recurrent and unpredictable seizures caused by abnormal electrical activity in the brain.

The objective of this project is to analyze clinical EEG data and develop a machine learning model capable of detecting or predicting epileptic seizures using signal processing and statistical feature extraction techniques.

The project is based on the **Siena Scalp EEG Dataset**.

---

## Objectives

The main goals of this project are:

- Analyze multi-channel EEG recordings from epileptic patients
- Extract relevant signal features from EEG time series
- Identify patterns associated with epileptic seizures
- Train and evaluate machine learning models for seizure detection or prediction

---

## Dataset

This project uses the **Siena Scalp EEG Database** available on PhysioNet.

Dataset characteristics:

- EEG recordings from epileptic patients
- Multi-channel scalp EEG signals
- Annotated seizure events
- Long continuous recordings

EEG signals are recorded using the **10–20 electrode placement system**, which captures brain electrical activity from multiple regions simultaneously.

Each recording contains:

- Multiple EEG channels (electrodes)
- High-frequency temporal sampling
- Seizure onset and offset annotations

Dataset source:

https://physionet.org/content/siena-scalp-eeg/1.0.0/

---

## EEG Signal Characteristics

EEG signals represent the electrical activity produced by populations of neurons in the brain.

These signals contain oscillatory components known as **brain rhythms**, typically categorized into frequency bands:

| Band | Frequency Range | Typical Meaning |
|-----|-----|-----|
| Delta | 0.5 – 4 Hz | Deep sleep |
| Theta | 4 – 8 Hz | Drowsiness / memory |
| Alpha | 8 – 13 Hz | Relaxation |
| Beta | 13 – 30 Hz | Active thinking |
| Gamma | >30 Hz | High-level processing |

During epileptic seizures, abnormal synchronization and spikes appear in these signals.

---

## Project Pipeline

The project follows a typical EEG machine learning pipeline:


EEG raw signals
↓
Signal preprocessing
↓
Signal segmentation
↓
Feature extraction
↓
Dataset construction
↓
Machine learning model
↓
Evaluation


---

## Preprocessing

The EEG signals are first cleaned to remove noise and artifacts.

Typical preprocessing steps include:

- Bandpass filtering (0.5 – 40 Hz)
- Artifact removal
- Signal normalization

---

## Signal Segmentation

EEG recordings are divided into fixed-length time windows.

Example:
Window length: 5–10 seconds


Each window becomes one training sample.

---

## Feature Extraction

Several types of features are extracted from each EEG window.
This article : https://www.mdpi.com/2227-9717/8/7/846/
has shown relevance in synchronization measures, such as PLI and WPLI, Hjorth parameters and entropy for epileptic seizure recognition and prediction.

These features are so our main targeted variables for this project 
Here is below others features on which our model may rely on for seizure recognition: 

### Time-domain features

- Mean
- Variance
- Skewness
- Kurtosis
- Zero-crossing rate

These describe the statistical properties of the signal.

### Frequency-domain features

Using spectral analysis (FFT / PSD):

- Delta band power
- Theta band power
- Alpha band power
- Beta band power

These capture the energy distribution across brain rhythms.

### Complexity features

- Approximate entropy
- Sample entropy
- Fractal dimension

These measure the irregularity and complexity of EEG signals.

### Hjorth parameters

- Activity
- Mobility
- Complexity

These are commonly used EEG descriptors.

---

## Machine Learning Models

Several machine learning models can be explored:

- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Neural Networks

These models classify EEG segments into different brain states.

Possible classification tasks:

- Seizure vs Non-seizure detection
- Preictal vs Interictal prediction

---

## Evaluation Metrics

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- False alarm rate

Below is the work from EpilepsyBenchmarks, which evaluates several of these metrics using different types of models trained on this dataset
        https://epilepsybenchmarks.com/datasets/siena/

---

## State of the Art

 A few studies relying on the same dataset for epilepsy seizures recognition
  - https://www.mdpi.com/2227-9717/8/7/846
  - https://www.researchgate.net/publication/370951763_Patient-specific_approach_using_data_fusion_and_adversarial_training_for_epileptic_seizure_prediction 
  - https://www.kaggle.com/datasets/abhishekinnvonix/epilepsy-seizure-dataset-seina-scalp-complete
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10869477/
  - https://www.nature.com/articles/s41598-025-90164-3
  - https://epilepsybenchmarks.com/datasets/siena/


 By Yanis AMARA
