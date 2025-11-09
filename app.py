import os
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc, delta
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
audio_bytes = None

if mode == "Rekam Langsung 🎙":
    audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)
    if audio and "bytes" in audio:
        audio_bytes = audio["bytes"]

else:
    uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()

if audio_bytes:

    # ========== SIMPAN SEBAGAI WAV VALID ==========
    try:
        # decode ke int16 untuk mic input
        samples = np.frombuffer(audio_bytes, dtype=np.int16)

        # stereo → mono
        if len(samples) % 2 == 0:
            samples = samples.reshape(-1, 2).mean(axis=1)

        # simpan sebagai wav 44.1kHz dulu
        from scipy.io.wavfile import write
        write("temp_audio.wav", 44100, samples.astype(np.int16))
    except:
        # jika upload file (sudah valid audio)
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

    # Putar audio
    st.audio("temp_audio.wav", format="audio/wav")

    # ========== LOAD AUDIO ==========
    y, sr = librosa.load("temp_audio.wav", sr=None)
    y = y.astype(np.float32)

    # jika kosong
    if len(y) == 0:
        st.error("⚠ Audio tidak terbaca! Coba rekam ulang.")
        st.stop()

    # Normalize volume RMS
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms * 0.1

    # Resample ke 16kHz
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))

    # Trim silence
    idx = np.where(np.abs(y) > 0.02)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # Fix 1 detik
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    m = mfcc(y, 16000, numcep=13)
    feat = np.mean(m, axis=0).reshape(1, -1)

    feat = scaler.transform(feat)

    # ========== PREDIKSI ==========
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    # ========== TAMPILKAN HASIL (PASTI MUNCUL) ==========
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"🧾 Label Terbaca : **{label}**")
    st.write(f"📊 Confidence   : **{conf:.2f}%**")

    # Warna indikator
    if conf > 55:
        if "buka" in label.lower():
            st.success(f"✅ BUKA — {conf:.2f}%")
        elif "tutup" in label.lower():
            st.error(f"🔒 TUTUP — {conf:.2f}%")
        else:
            st.info(f"🎯 {label} — {conf:.2f}%")
    else:
        st.warning("⚠ Model kurang yakin (confidence rendah)")
