import streamlit as st
import numpy as np
import librosa
import joblib
from streamlit_mic_recorder import mic_recorder
import soundfile as sf
import io

# ===== LOAD MODEL =====
model = joblib.load("models/voice_model.pkl")
scaler = joblib.load("models/voice_scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

SR = 16000
SAMPLES = SR * 1
THRESHOLD = 60  # confidence minimal diterima

st.set_page_config(page_title="Voice Recognition", layout="centered")
st.title("🎤 Voice Recognition (Buka / Tutup - 2 Orang)")

st.write("Klik tombol mic di bawah untuk merekam suara:")

audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)

if audio and "bytes" in audio:
    # 1. Simpan bytes ke file wav
    with open("temp.wav", "wb") as f:
        f.write(audio["bytes"])

    # 2. Load pakai librosa
    y, sr = librosa.load("temp.wav", sr=16000)

    # 3. Pastikan panjang 1 detik
    SAMPLES = 16000
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    # 4. Tampilkan audio di UI
    st.audio("temp.wav", format="audio/wav")

    # 5. Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0)
    feat = scaler.transform([feat])

    # 6. Prediksi
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100

    # 7. Output hasil
    st.subheader("🔍 Hasil Prediksi")
    if confidence >= THRESHOLD:
        st.success(f"✅ {label} ({confidence:.2f}%)")
    else:
        st.error(f"⛔ Suara tidak dikenali ({confidence:.2f}%)")
