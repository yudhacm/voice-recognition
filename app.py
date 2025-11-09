import os
import numpy as np
import joblib
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from python_speech_features import mfcc
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

    raw_audio = audio["bytes"]

    # === 1. Simpan sebagai WAV valid (dengan header) ===
    with open("recorded.wav", "wb") as f:
        f.write(raw_audio)

    # === 2. Putar audio (sekarang pasti jalan) ===
    st.audio("recorded.wav", format="audio/wav")

    # === 3. Convert byte ke numpy int16 ===
    samples = np.frombuffer(raw_audio, dtype=np.int16)

    # === 4. Jika stereo → convert ke mono ===
    if samples.ndim > 1 or len(samples.shape) == 1 and len(samples) % 2 == 0:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # === 5. Normalisasi ===
    samples = samples.astype(np.float32)
    samples /= np.max(np.abs(samples)) + 1e-9

    # === 6. Resample 44.1kHz → 16kHz (model kamu pakai 16kHz) ===
    target_len = int(len(samples) * 16000 / 44100)
    samples = resample(samples, target_len)

    # === 7. Pastikan 1 detik (16000 sample) ===
    if len(samples) > 16000:
        samples = samples[:16000]
    else:
        samples = np.pad(samples, (0, 16000 - len(samples)))

    # === 8. Ekstraksi MFCC ===
    feat = mfcc(samples, samplerate=16000, numcep=13)
    feat = np.mean(feat, axis=0).reshape(1, -1)
    feat = scaler.transform(feat)

    # === 9. Prediksi ===
    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    confidence = max(prob) * 100

    # === 10. Output ke UI ===
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
