import librosa
import soundfile as sf
import os
import numpy as np
import csv

feature_names = [
"MFCC1","MFCC2","MFCC3","MFCC4","MFCC5","MFCC6","MFCC7",
"MFCC8","MFCC9","MFCC10","MFCC11","MFCC12","MFCC13",
"Spectral_Centroid",
"Spectral_Bandwidth",
"Spectral_Rolloff",
"Zero_Crossing_Rate",
"RMS_Energy",
"Spectral_Contrast1","Spectral_Contrast2","Spectral_Contrast3",
"Spectral_Contrast4","Spectral_Contrast5","Spectral_Contrast6","Spectral_Contrast7",
"Instrument"
]

def make_feature(path):
    y, sr = sf.read(path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)


    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = centroid.mean()


    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = bandwidth.mean()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()


    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()


    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = contrast.mean(axis=1)

    features = np.hstack([
        mfcc_mean,
        centroid_mean,
        bandwidth_mean,
        rolloff_mean,
        zcr_mean,
        rms_mean,
        contrast_mean
    ])

    return features

dataset_path = r"C:\Users\LENOVO\Documents\ML Project(Instrument Classifier)"

f=open("Music Dataset.csv","w",newline="")
cw=csv.writer(f)
cw.writerow(feature_names)

for instrument in os.listdir(dataset_path):
    fol_path=os.path.join(dataset_path,instrument)
    print("Processing: ",instrument)
    if not os.path.isdir(fol_path):
        continue
    for file in os.listdir(fol_path):
        if(file.endswith(".wav")):
            path=os.path.join(fol_path,file)
            l=list(make_feature(path))+[instrument]
            cw.writerow(l)
