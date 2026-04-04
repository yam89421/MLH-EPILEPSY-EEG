import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from features_extraction import extractFeaturesEEG, extractFeaturesECG, EEG_FEATURES_NAMES, ECG_FEATURES_NAMES
from label_extraction import extractLabels, records, dataset_path

def save(data, filepath, column_names):
    df = pd.DataFrame(data)
    df.columns = column_names
    df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)


def process():

    WINDOW = 10  # segmentation window in sec
    chunk_duration = 60 * 60  # 1h chunks
    sfreq = 128

    window_size = int(WINDOW * sfreq)
    step = window_size // 2

    patient = ""

    montage = mne.channels.make_standard_montage("standard_1020")

    physio_channels = ['SPO2', 'HR', '1', '2', 'MK', 'EKG EKG']

    eeg_channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'F9', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Fc2', 'Fc6', 'Cp2', 'Cp6', 'F10']


    for r in records[26:27]:
        
        edf_file = r.split("/")[-1].strip()

        if patient != r.split("/")[0].strip():
            patient = r.split("/")[0].strip()
            print(f"Extracting features and labels for {patient}")

        print(f"   ==> Processing {edf_file}")

        csv_dirpath = os.path.join(dataset_path, patient, "csv", edf_file.split(".")[0])

        os.makedirs(csv_dirpath, exist_ok=True)

        ecg_features_fullpath = os.path.join(csv_dirpath, edf_file.split(".")[0] + "_ECG_X.csv")
        eeg_features_fullpath = os.path.join(csv_dirpath, edf_file.split(".")[0] + "_EEG_X.csv")
        labels_fullpath = os.path.join(csv_dirpath, edf_file.split(".")[0] + "_Y.csv")

        raw = mne.io.read_raw_edf(os.path.join(dataset_path, r.strip()), preload=False)

        raw.rename_channels(lambda x: x.replace("EEG ", ""))

        raw.resample(sfreq, npad="auto")

        eeg_channels = [ch for ch in raw.ch_names if ch in eeg_channels]
        print("channels:", raw.ch_names)
        print("n_channels:", len(raw.ch_names))
        print("EEG picked:", eeg_channels)
        print("len:", len(eeg_channels))
        raw_eeg = raw.copy().pick(eeg_channels)
        print("EEG channels:", len(raw_eeg.ch_names))
        print("channels:", raw.ch_names)
        raw_eeg.filter(0.5, 40)
        
        if "EKG EKG" in raw.ch_names:
            raw_ecg = raw.copy().pick(["EKG EKG"])
        elif "ECG ECG" in raw.ch_names:
            raw_ecg = raw.copy().pick(["ECG ECG"])
        else:
            print(f"No ECG channel in {edf_file}, skipping ECG features")
            raw_ecg = None

        raw_eeg.set_montage(montage, on_missing="ignore")
        samples_chunk = int(chunk_duration * sfreq)

        for start in range(0, raw_eeg.n_times, samples_chunk):

            stop = min(start + samples_chunk, raw_eeg.n_times)

            if raw_ecg is not None:
                raw_ecg.filter(0.5, 40)
                ecg_data, _ = raw_ecg[:, start:stop]
                ecg_windows = np.lib.stride_tricks.sliding_window_view(ecg_data, window_shape=window_size, axis=1)
                ecg_windows = ecg_windows[:, ::step, :]
                ecg_windows = ecg_windows.transpose(1, 0, 2)
                
                # garder seulement le canal ECG -> 1D
                ecg_windows = ecg_windows[:, 0, :]


                ecg_features = extractFeaturesECG(ecg_windows*1e6, sfreq)
                save(ecg_features, ecg_features_fullpath, ECG_FEATURES_NAMES)

                del ecg_windows
                del ecg_data

            eeg_data, _ = raw_eeg[:, start:stop]
            if np.var(eeg_data) < 1e-12:
                print("EMPTY CHUNK DETECTED")
                continue
            print("chunk var:", np.var(eeg_data))
            print("chunk min:", np.min(eeg_data))
            print("chunk max:", np.max(eeg_data))
            print("std:", np.std(eeg_data))
            print("var:", np.var(eeg_data))
            print(np.std(eeg_data, axis=1))
            print("zero ratio:", np.mean(eeg_data == 0))


            if eeg_data.shape[1] < window_size:
                continue


            eeg_windows = np.lib.stride_tricks.sliding_window_view(
                eeg_data,
                window_shape=window_size,
                axis=1
            )

            eeg_windows = eeg_windows[:, ::step, :]
            eeg_windows = eeg_windows.transpose(1, 0, 2)

            



            eeg_features = extractFeaturesEEG(eeg_windows*1e6, sfreq)
            save(eeg_features, eeg_features_fullpath, EEG_FEATURES_NAMES)


            labels = extractLabels(
                edf_file,
                len(eeg_windows),
                window_size,
                step,
                sfreq,
                start_sample=start
            )

            save(labels, labels_fullpath, ["Y"])

            del eeg_windows
            del eeg_data

