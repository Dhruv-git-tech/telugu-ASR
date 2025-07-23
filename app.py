import streamlit as st
import whisper
import os
from datetime import datetime
import tempfile

# Title
st.set_page_config(page_title="Telugu ASR - Speech to Text", layout="centered")
st.title("🗣️ Telugu Audio Transcriber")
st.write("Upload an audio or video file in Telugu and get the transcribed text.")

# Load model (only once)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Upload audio/video
uploaded_file = st.file_uploader("Upload Audio or Video File", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    st.audio(uploaded_file, format='audio/wav')
    st.info("Transcribing... Please wait.")

    result = model.transcribe(temp_path, language="te")

    st.success("✅ Transcription Complete:")
    st.text_area("Telugu Transcription", result["text"], height=200)

    # Download option
    file_name = f"telugu_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("📄 Download Transcript", result["text"], file_name=file_name)

    os.remove(temp_path)
