import os  # ✅ harus ada
BASE_DIR = os.path.dirname(__file__)  # ✅ harus ada

import streamlit as st
import joblib
import numpy as np
import wave
from streamlit_mic_recorder import mic_recorder
from scipy.io import wavfile
from python_speech_features import mfcc

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

    with wave.open("temp.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio["bytes"])

    sr, y = wavfile.read("temp.wav")
    y = y.astype(np.float32)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    SAMPLES = 16000
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, SAMPLES - len(y)))

    st.audio("temp.wav", format="audio/wav")

    feat = mfcc(y, samplerate=16000, numcep=13)
    feat = np.mean(feat, axis=0).reshape(1, -1)
    feat = scaler.transform(feat)

    prob = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    label = le.inverse_transform([pred])[0]
    conf = max(prob) * 100

    st.subheader("🔍 Hasil Prediksi")

    if conf >= 70:
        if "buka" in label.lower():
            st.success(f"✅ Terdeteksi **BUKA** ({conf:.2f}%)")
        elif "tutup" in label.lower():
            st.warning(f"🔒 Terdeteksi **TUTUP** ({conf:.2f}%)")
        else:
            st.success(f"✅ {label} ({conf:.2f}%)")
    else:
        st.error(f"⛔ Suara tidak dikenali ({conf:.2f}%)")
