import os
import numpy as np
import joblib
import streamlit as st
from python_speech_features import mfcc
from scipy.signal import resample
from scipy.io import wavfile

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
audio_ready = False

# ==================== REKAM MIC (WAV PCM ASLI) ====================
if mode == "Rekam Langsung 🎙":
    st.write("🎙 Klik lalu bicara (output WAV PCM):")
    recorded_audio = st.experimental_audio_input(" ")

    if recorded_audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(recorded_audio.getbuffer())
        audio_ready = True

# ==================== UPLOAD FILE WAV ====================
else:
    uploaded_file = st.file_uploader("Upload file audio", type=["wav"])
    if uploaded_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        audio_ready = True

# ==================== PROSES & PREDIKSI ====================
if audio_ready:

    st.audio("temp_audio.wav", format="audio/wav")

    # ----- Baca audio WAV PCM -----
    try:
        sr, y = wavfile.read("temp_audio.wav")
    except:
        st.error("⚠ Format audio bukan WAV PCM! Ulangi rekam.")
        st.stop()

    # Jika stereo → convert ke mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    # Normanisasi volume (RMS)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Resample ke 16kHz (jika perlu)
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))

    # Buang silent di awal/akhir
    idx = np.where(np.abs(y) > 0.02)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # Minimal panjang bicara
    if len(y) < 3000:
        st.warning("⚠ Suara terlalu pendek, bicara lebih jelas!")
        st.stop()

    # Paksa 1 detik (16000 sample)
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    # ---- Ekstraksi Fitur MFCC 13 (sesuai training) ----
    mf = mfcc(y, 16000, numcep=13)
    feat = np.mean(mf, axis=0).reshape(1, -1)

    # Scaling
    feat = scaler.transform(feat)

    # Prediksi
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    # Output UI
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"🧾 Label      : **{label}**")
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
