import streamlit as st
import numpy as np
import librosa
import joblib
from audiorecorder import audiorecorder

# ================= MODEL LOAD =================
model = joblib.load("models/voice_model.pkl")
scaler = joblib.load("models/voice_scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

SR = 16000
SAMPLES = SR * 1
THRESHOLD = 60  # lebih fleksibel

st.set_page_config(page_title="Voice Recognition", layout="centered")
st.title("🎤 Voice Recognition (Buka / Tutup - 2 Orang)")

st.write("Klik tombol untuk mulai merekam suara:")

# ================= RECORDING UI =================
audio = audiorecorder("🎤 Rekam Suara", "⏹ Stop Rekaman")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")

    with open("mic_temp.wav", "wb") as f:
        f.write(audio.export().read())

    # Extract MFCC
    y, _ = librosa.load("mic_temp.wav", sr=SR)

    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0)
    feat_scaled = scaler.transform([feat])

    prob = model.predict_proba(feat_scaled)[0]
    pred = model.predict(feat_scaled)[0]

    label = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100

    st.subheader("🔍 Hasil Prediksi")
    if confidence >= THRESHOLD:
        st.success(f"✅ {label} ({confidence:.2f}%)")
    else:
        st.error(f"⛔ Suara tidak dikenali ({confidence:.2f}%)")
