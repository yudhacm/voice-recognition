import os
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc
from scipy.signal import resample
from scipy.io import wavfile  # ✅ gantikan librosa

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

# ========== INPUT AUDIO ==========
if mode == "Rekam Langsung 🎙":
    audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)
    if audio and "bytes" in audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio["bytes"])
        audio_ready = True

else:
    uploaded_file = st.file_uploader("Upload file audio", type=["wav"])
    if uploaded_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        audio_ready = True

# ========== PROSES & PREDIKSI ==========
if audio_ready:

    st.audio("temp_audio.wav", format="audio/wav")

    # ✅ Baca audio pakai scipy (tanpa librosa!)
    try:
        sr, y = wavfile.read("temp_audio.wav")
    except:
        st.error("⚠ Gagal membaca audio. Pastikan format WAV.")
        st.stop()

    # Jika stereo → convert mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    # Normalisasi RMS volume
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Resample 16 kHz
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))

    # Trim silence (hapus bagian sangat kecil)
    idx = np.where(np.abs(y) > 0.02)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # Jika terlalu pendek
    if len(y) < 3000:
        st.warning("⚠ Suara terlalu pendek, coba bicara lebih jelas.")
        st.stop()

    # Paksa 1 detik (16000 sampel)
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    # Ekstraksi MFCC 13
    mf = mfcc(y, 16000, numcep=13)
    feat = np.mean(mf, axis=0).reshape(1, -1)

    # Scaling
    feat = scaler.transform(feat)

    # Prediksi
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    # Output
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
