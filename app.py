import os
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc
from scipy.signal import resample
import librosa

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "models/voice_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models/voice_scaler.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_model()

st.title("🎤 Voice Command Recognition")
st.write("Ucapkan atau upload suara **buka** atau **tutup**")

mode = st.radio("Pilih Mode Input Suara:", ["Rekam Langsung 🎙", "Upload File 📁"])
audio_ready = False  # penanda lanjut proses

# ================== AUDIO INPUT ===================
if mode == "Rekam Langsung 🎙":
    audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)
    if audio and "bytes" in audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio["bytes"])
        audio_ready = True

else:
    uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        audio_ready = True

# ================== PROSES AUDIO ===================
if audio_ready:
    st.audio("temp_audio.wav", format="audio/wav")

    # Load audio dengan librosa
    y, sr = librosa.load("temp_audio.wav", sr=None, mono=True)
    y = y.astype(np.float32)

    # Cek jika audio kosong
    if len(y) == 0:
        st.error("⚠ Audio tidak terbaca! Coba ulangi.")
        st.stop()

    # Normalize RMS volume
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Resample ke 16kHz jika perlu
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))

    # Trim silence (buang depan & belakang)
    idx = np.where(np.abs(y) > 0.02)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # Kalau audio kependekan → warning
    if len(y) < 3000:
        st.warning("⚠ Suara terlalu pendek, coba bicara lebih jelas.")
        st.stop()

    # Fix 1 detik (16000 samples)
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    # Ekstraksi MFCC 13
    mfcc_feat = mfcc(y, 16000, numcep=13)
    feat = np.mean(mfcc_feat, axis=0).reshape(1, -1)

    # Scaling sesuai model
    feat = scaler.transform(feat)

    # Prediksi model
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    # ================== OUTPUT ===================
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"🧾 Label  : **{label}**")
    st.write(f"📊 Confidence : **{conf:.2f}%**")

    if conf > 55:
        if "buka" in label.lower():
            st.success(f"✅ BUKA — {conf:.2f}%")
        elif "tutup" in label.lower():
            st.error(f"🔒 TUTUP — {conf:.2f}%")
        else:
            st.info(f"🎯 {label} — {conf:.2f}%")
    else:
        st.warning("⚠ Model kurang yakin (confidence rendah)")
