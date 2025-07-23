import streamlit as st
import whisper
import tempfile
import torchaudio
import soundfile as sf
import os

st.title("🌟 Telugu ASR Studio 🌟")
st.caption("Simple and fast Telugu ASR using Whisper (tiny)")

@st.cache_resource(show_spinner=True)
def load_model():
    return whisper.load_model("tiny")

model = load_model()

audio_file = st.file_uploader("🎤 Upload Audio File (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    st.info("Transcribing...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    waveform, sr = torchaudio.load(tmp_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    sf.write(tmp_path, waveform.T.numpy(), 16000)

    result = model.transcribe(tmp_path, language="te")
    st.subheader("📄 Telugu Transcript")
    st.text(result["text"])

    os.remove(tmp_path)