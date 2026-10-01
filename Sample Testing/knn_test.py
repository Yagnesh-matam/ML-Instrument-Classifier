import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

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

p=r"C:\course\sem 4\ml Project\IRMAS-TrainingData\voi\171__[voi][dru][pop_roc]2315__1.wav"
x=make_feature(p)
x=x.reshape(1,-1)
# 1. Load the dataset
data=pd.read_csv(r"C:\course\sem 4\ml Project\Music Dataset.csv")
X = data.drop("Instrument",axis=1)
y = data["Instrument"]


# 2. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

x=scaler.transform(X)
# 3. Initialize the k-NN classifier (using k=3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 4. Train the model
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(x)

print(y_pred[0])
probs = knn.predict_proba(x)
print("Confidence:", max(probs[0])*100, "%")