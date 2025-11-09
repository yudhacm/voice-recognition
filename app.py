import os
import io
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc
from scipy.io.wavfile import write
from scipy.signal import resample

BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "models/voice_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models/voice_scaler.pkl"))
    le = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_model()

st.title("🎤 Voice Command Recognition")
st.write("Ucapkan **buka** atau **tutup**")

audio = mic_recorder(start_prompt="🎙 Mulai Rekam", stop_prompt="⏹ Stop Rekam", just_once=True)

if audio and "bytes" in audio:

    # === Decode Audio Byte ke int16 ===
    raw = np.frombuffer(audio["bytes"], dtype=np.int16)

    # === Konversi ke mono jika stereo ===
    if len(raw) % 2 == 0:
        raw = raw.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # === Simpan sementara 44.1 kHz (biar audio bisa diputar jelas) ===
    write("temp.wav", 44100, raw)
    st.audio("temp.wav", format="audio/wav")

    # === Resample ke 16kHz untuk model ===
    target_len = int(len(raw) * 16000 / 44100)
    y = resample(raw, target_len).astype(np.float32)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Force 1 detik (16000 sample)
    if len(y) > 16000:
        y = y[:16000]
    else:
        y = np.pad(y, (0, 16000 - len(y)))

    # === Ekstraksi MFCC ===
    feat = mfcc(y, samplerate=16000, numcep=13)
    feat = np.mean(feat, axis=0).reshape(1, -1)
    feat = scaler.transform(feat)

    # === Prediksi ===
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100

    # === Output UI ===
    st.subheader("🔍 Hasil Prediksi")

    if confidence > 70:
        if "buka" in label.lower():
            st.success(f"✅ BUKA — Keyakinan {confidence:.2f}%")
        elif "tutup" in label.lower():
            st.error(f"🔒 TUTUP — Keyakinan {confidence:.2f}%")
        else:
            st.info(f"🔎 {label} — {confidence:.2f}%")
    else:
        st.warning(f"⚠ Tidak dikenali ({confidence:.2f}%)")
