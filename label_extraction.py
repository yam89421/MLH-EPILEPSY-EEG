import re
import os 
import numpy as np 

records_filename = "RECORDS"
dataset_path = "./dataset"

records_file = open(os.path.join(dataset_path, records_filename), "r")
records = records_file.readlines()
records_file.close()

patients = np.unique([r.split("/")[0] for r in records])

def timeToSec(t):
    h, m, s = map(int, t.split("."))
    return h*3600 + m*60 + s


def extract_time(line):

    electric = re.search(r'(\d{2}\.\d{2}\.\d{2}) \(ELECTRIC', line)
    clinical = re.search(r'(\d{2}\.\d{2}\.\d{2}) \(CLINICAL', line)

    if electric:
        return timeToSec(electric.group(1))

    if clinical:
        return timeToSec(clinical.group(1))

    times = re.findall(r"\d{2}\.\d{2}\.\d{2}", line)

    if len(times) == 1:
    	return timeToSec(times[0])

    #if several times => take the maximum
    return max(timeToSec(t) for t in times)

def getSeizure(edf_file):

	reg_start_time = 0
	reg_end_time = 0
	seizure_start_time = 0
	seizure_end_time = 0

	seizures = []
	patient = edf_file.split("-")[0]
	seizures_filepath = os.path.join(dataset_path, patient, "Seizures-list-"+patient+".txt")

	with open(seizures_filepath, "r") as f:
		lines = f.readlines()

	for line_index in range(len(lines)):

		lines[line_index] = lines[line_index].strip()
		if lines[line_index].startswith("File name"):
			
			if lines[line_index].split(":")[1].strip() == edf_file:
				
				reg_start_time = extract_time(lines[line_index+1])
				reg_end_time = extract_time(lines[line_index+2])
				seizure_start_time = extract_time(lines[line_index+3])
				seizure_end_time = extract_time(lines[line_index+4])

				if reg_end_time < reg_start_time:
					reg_end_time += 24*3600

					if seizure_start_time < reg_start_time:
						seizure_start_time += 24*3600

					if seizure_end_time < seizure_start_time:
						seizure_end_time += 24*3600 

				seizures.append({"file":edf_file, "reg_start":reg_start_time, "reg_end": reg_end_time, "start":seizure_start_time, "end":seizure_end_time})

	return seizures

	

def extractLabels(edf_file, n_windows, window_size, step, sfreq, start_sample):

	step_sec = step / sfreq
	offset_sec = start_sample / sfreq

	preictal_range_min = 30
	preictal_range = int(preictal_range_min * 60 / step_sec)

	labels = np.zeros(n_windows)

	seizures_info = getSeizure(edf_file)

	for seizure_info in seizures_info:

	    seizure_start = seizure_info["start"] - seizure_info["reg_start"]
	    seizure_end = seizure_info["end"] - seizure_info["reg_start"]

	    chunk_end = offset_sec + n_windows * step_sec

	    if seizure_end < offset_sec or seizure_start > chunk_end:
	        continue

	    start_window = int(np.floor((seizure_start - offset_sec) / step_sec))
	    end_window = int(np.ceil((seizure_end - offset_sec) / step_sec))

	    start_window = max(0, start_window)
	    end_window = min(n_windows, end_window)

	    if end_window <= start_window:
	    	end_window = start_window + 1

	    pre_start = max(0, start_window - preictal_range)

	    labels[pre_start:start_window] = 2
	    labels[start_window:end_window] = 1

	    print("chunk start sec:", offset_sec)
	    print("seizure start global:", seizure_start)
	    print("seizure end global:", seizure_end)
	    print("start_window:", start_window)
	    print("end_window:", end_window)

	return labels


