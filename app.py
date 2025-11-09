import streamlit as st
import numpy as np
import librosa
import joblib
from streamlit_mic_recorder import mic_recorder
import soundfile as sf

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

if audio:
    # Convert bytes menjadi file WAV
    audio_bytes = audio["bytes"]
    audio_io = io.BytesIO(audio_bytes)

    # Baca langsung pakai soundfile
    y, sr = sf.read(audio_io)

    # Jika stereo, convert ke mono
    if len(y.shape) > 1:
        y = y[:, 0]

    # Resample jika perlu
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Normalisasi panjang
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    # Simpan sementara agar bisa dimainkan di UI
    sf.write("temp.wav", y, SR)
    st.audio("temp.wav", format="audio/wav")

    # ===== EKSTRAKSI FITUR & PREDIKSI =====
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    feat = np.mean(mfcc.T, axis=0)
    feat = scaler.transform([feat])

    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100

    st.subheader("🔍 Hasil Prediksi")
    if confidence >= THRESHOLD:
        st.success(f"✅ {label} ({confidence:.2f}%)")
    else:
        st.error(f"⛔ Suara tidak dikenali ({confidence:.2f}%)")
