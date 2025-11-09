import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========== CONFIG ==========
DATASET_PATH = "dataset"
MODEL_PATH = "models"

ALLOWED_CLASSES = ["buka_orang_1", "buka_orang_2", "tutup_orang_1", "tutup_orang_2"]
SR = 16000
DURATION = 1
SAMPLES = SR * DURATION

os.makedirs(MODEL_PATH, exist_ok=True)

# ========== AUDIO PREPROCESS ==========
def preprocess_audio(y):
    y, _ = librosa.effects.trim(y, top_db=30)  # Hapus silence di awal/akhir
    y = librosa.effects.preemphasis(y)         # Clear audio
    return y

# ========== AUGMENTATION ==========
def augment(y):
    aug_data = [y]
    aug_data.append(y + np.random.normal(0, 0.01, len(y)))       # noise
    aug_data.append(librosa.effects.time_stretch(y, rate=1.1))    # cepat
    aug_data.append(librosa.effects.time_stretch(y, rate=0.9))    # lambat
    return aug_data

# ========== EXTRACT MFCC ==========
def extract_mfcc(y):
    y = preprocess_audio(y)

    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, max(0, SAMPLES - len(y))))

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ========== LOAD DATASET ==========
X, y = [], []

for label in ALLOWED_CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    print(f"Processing: {label}")

    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            y_raw, _ = librosa.load(os.path.join(folder, file), sr=SR)

            for audio_aug in augment(y_raw):
                X.append(extract_mfcc(audio_aug))
                y.append(label)

X = np.array(X)
y = np.array(y)

# ========== ENCODE & SCALE ==========
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ========== TRAIN ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# ========== EVALUATION ==========
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.show()

# ========== SAVE MODEL ==========
joblib.dump(model,  f"{MODEL_PATH}/voice_model.pkl")
joblib.dump(scaler, f"{MODEL_PATH}/voice_scaler.pkl")
joblib.dump(le,     f"{MODEL_PATH}/label_encoder.pkl")

print("\n✅ Model saved inside /models folder.")
