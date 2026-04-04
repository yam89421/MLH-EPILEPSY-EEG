import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from features_extraction import EEG_FEATURES_NAMES, ECG_FEATURES_NAMES
import os
from preprocessing import save
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import random
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb 
from sklearn.svm import SVC
import data_preprocessing
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
from sklearn.utils.class_weight import compute_sample_weight

dataset_path = "./dataset/"
patients = [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]

def getFeaturesOverLabel():
    global dataset_path
    global patients
   
    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2a2a2a",
        "axes.edgecolor": "#4f4f4f",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 8
    })


    for patient in patients:

        print("Processing", patient)

        patient_path = os.path.join(dataset_path, patient, "csv")

        eeg_feat_file = os.path.join(patient_path, patient + "_EEG_X.csv")
        ecg_feat_file = os.path.join(patient_path, patient + "_ECG_X.csv")
        labels_file   = os.path.join(patient_path, patient + "_Y.csv")

        if not os.path.exists(labels_file):
            print("No labels for", patient)
            continue

        labels = pd.read_csv(labels_file).values.flatten()

        def plot_features(feat_file, save_path):

            if not os.path.exists(feat_file):
                print("Missing file:", feat_file)
                return

            features = pd.read_csv(feat_file)

            # normalisation
            features = (features - features.mean()) / (features.std() + 1e-10)

            n_features = len(features.columns)
            cols = 6
            rows = int(np.ceil(n_features / cols))

            fig, axs = plt.subplots(rows, cols, figsize=(22, rows * 2))
            axs = axs.flatten()

            colors = plt.cm.tab20(np.linspace(0, 1, n_features))

            for k, col in enumerate(features.columns):

                axs[k].plot(features[col], color=colors[k])
                axs[k].plot(labels, color="white")
                axs[k].set_title(col)
                axs[k].grid(True, alpha=0.2)

            # supprimer axes vides
            for k in range(n_features, len(axs)):
                fig.delaxes(axs[k])

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        # EEG
        plot_features(
            eeg_feat_file,
            os.path.join(dataset_path, patient, "features_eeg.png")
        )

        # ECG (optionnel)
        plot_features(
            ecg_feat_file,
            os.path.join(dataset_path, patient, "features_ecg.png")
        )


def mergeData():

	global dataset_path
	global patients

	for patient in patients:

		try:
			os.system("rm "+csvPath+patient+"_EEG_X.csv "+csvPath+patient+"_ECG_X.csv " +csvPath+patient+"_Y.csv")
		except Exception as e:
			print(e)
		
		patient_eeg_files = [csvPath+f for f in os.listdir(csvPath) if os.path.isdir(csvPath+f)]
		
		for file in patient_eeg_files:
			
			try:
				sample = file.split("/")[-1]
				eeg_data = pd.read_csv(file+"/"+sample+"_EEG_X.csv")
				labels = pd.read_csv(file+"/"+sample+"_Y.csv")
				try:
					ecg_data = pd.read_csv(file+"/"+sample+"_ECG_X.csv")
					save(ecg_data, csvPath+patient+"_ECG_X.csv", ECG_FEATURES_NAMES)

				except Exception as e:
					print(e)

				save(eeg_data, csvPath+patient+"_EEG_X.csv", EEG_FEATURES_NAMES)
				save(labels, csvPath+patient+"_Y.csv", ["Y"])

			except Exception as e:
				print(e)


def getMeanFeaturesEEG(patients, undersample=False):
	global dataset_path
	ecg_data = None
	eeg_data = []
	for patient in patients:
		eeg= pd.read_csv("./dataset/"+patient+"/csv/"+patient+"_EEG_X.csv")
		labels = pd.read_csv("./dataset/"+patient+"/csv/"+patient+"_Y.csv") 
		df = eeg.copy()
		df["Seizure"] = labels
		if undersample:
			df = data_preprocessing.removeBufferArea(df, 0.5, 10)
		eeg_data.append(df)

	return pd.concat(eeg_data)

"""
eeg_data = getMeanFeaturesEEG(patients)

df = eeg_data.copy()
print(df["Seizure"].value_counts())



#melange
df_balanced = df.sample(frac=1, random_state=42).reset_index(drop=True)


X = df_balanced.drop(columns="Seizure")
y = df_balanced["Seizure"]

X = (X - X.mean()) / X.std()

auc_scores = []
for col in X.columns:
    clf = LogisticRegression(solver="newton-cholesky").fit(X[col].to_numpy().reshape(-1, 1), y.values)
    auc = roc_auc_score(y, clf.predict_proba(X[col].to_numpy().reshape(-1, 1)), multi_class="ovo")
    auc = max(auc, 1 - auc)
    auc_scores.append(auc)

auc_scores = pd.Series(auc_scores, index=X.columns)


mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=X.columns)
print(f"{'FEATURE':<30} {'AUC':<10} {'MI':<10} {'RF_IMPORTANCE':<15}")

sum_scores = []
for feat in EEG_FEATURES_NAMES:
    sum_scores.append((feat, (auc_scores[feat] + mi_scores[feat] + importance[feat])))
    print(f"{feat:<30} {auc_scores[feat]:<10.3f} {mi_scores[feat]:<10.3f} {importance[feat]:<15.3f}")

def sortKey(e):
	return e[1]



# compare with statsmodels
X = sm.add_constant(X)
print(y.value_counts())
sm_model = sm.MNLogit(y, X)

result = sm_model.fit()
print(result.summary())


corr = df_balanced.corr()

sum_scores.sort(key=sortKey, reverse=True)
for i in range(len(sum_scores)):
	print(f"{i+1} ---> {sum_scores[i][0]}")
ordered_features = pd.Series([feat for feat, score in sum_scores])
"""

train_ratio = 0.8

random.shuffle(patients)
train_patients = patients[:int(np.floor(train_ratio*len(patients)))]
test_patients = patients[len(train_patients):]

eeg_train = getMeanFeaturesEEG(train_patients, undersample=True)
eeg_test = getMeanFeaturesEEG(test_patients, undersample=True)
y_test = (eeg_test["Seizure"] > 0).astype(int)
X_test = eeg_test.drop(columns="Seizure")

print(f"\nTrain set: {train_patients} | n_windows: {len(eeg_train)}")
print(f"Test set: {test_patients} | n_windows: {len(eeg_test)}")


"""
# MANUAL STRATIFYING
---------------------------------------------------------------------------------------------------------------------
#stratifying train set 60% y = 0 & 40% y = {1,2}

zeros_ratio = 1

df = eeg_train.copy()
labels_train = (labels_train > 0).astype(int)

df["Seizure"] = labels_train
df0 = df[df["Seizure"] == 0]
df12 = df[df["Seizure"].isin([1, 2])]

n12 = len(df12)
n0 = int((zeros_ratio/(1 - zeros_ratio))*n12) #stratifying

df_test = eeg_test.copy()

df_test["Seizure"] = labels_test
n0_test = len(df_test[df_test["Seizure"] == 0])
n12_test = len(df_test[df_test["Seizure"].isin([1, 2])])

print("\nSampling ...\n")

print("Before SMOTE: ")
print(f"Final train set: {n0} 0 and {n12} y label = 1 or 2")
print(f"Final test set: {n0_test} 0 and {n12_test} y label = 1 or 2\n")
#echantillonage
df0_sample = df0.sample(n=n0, replace=len(df0) < n0, random_state=42)
df12_sample = df12.sample(n=n12, replace=len(df12) < n12, random_state=42)
df_balanced = pd.concat([df0_sample, df12_sample])

#shuffling
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
---------------------------------------------------------------------------------------------------------------------
"""



result_file = open("results_LogitClassifier_undersampled_data_train=70%_seiz=40%.txt", "w")

df = eeg_train.copy()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = df.drop(columns="Seizure")
y_train = (df["Seizure"] > 0).astype(int)

before_smote_counts = pd.Series(y_train.value_counts())
labels_count = pd.Series(y_test.value_counts())

print(f"Final train set: {before_smote_counts[0]} \"0\", {before_smote_counts[1]} \"1\"")
print(f"Final test set: {labels_count[0]} \"0\", {labels_count[1]} \"1\"")

"""
model = model = lgb.LGBMClassifier(
class_weight="balanced",
n_estimators=800,
learning_rate=0.03,
max_depth=10,
num_leaves=20,
min_child_samples=20,
subsample=0.8,
colsample_bytree=0.8,
random_state=42
)
"""


model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sample_weights = compute_sample_weight(
    class_weight={0:1, 1:15},  # ← tu boost la classe 1
    y=y_train
)

model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("AUC:", auc)
print("\nAccuracy: ", accuracy)
print("Balanced accuracy: ", balanced_accuracy)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

