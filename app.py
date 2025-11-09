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

# ===== Recorder dari browser (Cloud-safe) =====
audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)

if audio:
    # simpan file sementara
    sf.write("temp.wav", audio['bytes'], samplerate=audio['sample_rate'])

    st.audio("temp.wav", format="audio/wav")

    # load audio
    y, _ = librosa.load("temp.wav", sr=SR)
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

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
