import os
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc, delta
from scipy.signal import resample
import soundfile as sf

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

# ================= PILIH MODE INPUT =================
mode = st.radio("Pilih Mode Input Suara:", ["Rekam Langsung 🎙", "Upload File 📁"])

audio_bytes = None

# =============== MODE 1: REKAM LANGSUNG ===============
if mode == "Rekam Langsung 🎙":
    audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)
    if audio and "bytes" in audio:
        audio_bytes = audio["bytes"]

# =============== MODE 2: UPLOAD FILE ===============
else:
    uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()

# =============== PROSES PREDIKSI ===============
if audio_bytes is not None:

    # Simpan sementara agar bisa diputar
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    st.audio("temp_audio.wav", format="audio/wav")

    # Load audio (support wav/mp3)
    y, sr = sf.read("temp_audio.wav")

    # Convert stereo → mono jika perlu
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    # Normalize volume (RMS)
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Resample ke 16kHz
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))
        sr = 16000

    # Trim silence
    idx = np.where(np.abs(y) > 0.02)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # Fix 1 detik
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    # MFCC + Delta + Delta2
    m = mfcc(y, 16000, numcep=13)
    d1 = delta(m, 2)
    d2 = delta(d1, 2)
    feat = np.hstack([m, d1, d2])
    feat = np.mean(feat, axis=0).reshape(1, -1)
    feat = scaler.transform(feat)

    # Prediksi
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    # Output UI
    st.subheader("🔍 Hasil Prediksi")

    if conf > 55:
        if "buka" in label.lower():
            st.success(f"✅ BUKA — {conf:.2f}%")
        elif "tutup" in label.lower():
            st.error(f"🔒 TUTUP — {conf:.2f}%")
        else:
            st.info(f"🎯 {label} — {conf:.2f}%")
    else:
        st.warning(f"⚠ Tidak dikenali ({conf:.2f}%)")
