import os
import pandas as pd
import numpy as np 

dataset_path = "./dataset/"
patients = [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]

def getMeanFeaturesEEG(patients):
	global dataset_path
	ecg_data = None
	eeg_data = pd.concat([pd.read_csv("./dataset/"+patient+"/csv/"+patient+"_EEG_X.csv") for patient in patients])
	labels = pd.concat([pd.read_csv("./dataset/"+patient+"/csv/"+patient+"_Y.csv") for patient in patients])
	try:
		ecg_data = pd.concat([pd.read_csv("./dataset/"+patient+"/csv/"+patient+"_ECG_X.csv") for patient in patients if patient not in ["PN14", "PN16", "PN17"]])
	except Exception as e:
		print(e)

	return eeg_data, ecg_data, labels

#enlever 1h après crise 
#labelise 30min avant crise
#enlever les interictal > 3600

def removeBufferArea(raw_dataset, overlap, window_size):
	
	df = raw_dataset.copy()
	labels = df["Seizure"].values

	buffer = 60 * 60  # 1h en sec
	step = window_size * (1 - overlap)
	buffer_size = int(buffer / step)

	mask = np.ones(len(df), dtype=bool)

	start_inter = None
	inter_count = 0

	for i in range(1, len(labels)):

	    # interictal start
	    if labels[i] == 0:
	        if start_inter is None:
	            start_inter = i
	            inter_count = 0
	        inter_count += 1

	    # interictal ends
	    if labels[i] != 0 and start_inter is not None:
	        
	        # cropping interictal if > 1h
	        if inter_count > buffer_size:
	            mask[start_inter + buffer_size : i] = False
	        
	        start_inter = None
	        inter_count = 0

	    # delete postictal > 1000sec 
	    if labels[i] == 0 and labels[i-1] == 1:
	        end = min(len(labels), i + 1000)
	        mask[i:end] = False

	df_filtered = df[mask]

	return df_filtered



