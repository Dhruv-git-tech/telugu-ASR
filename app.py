import streamlit as st
import whisper
import os
import tempfile
import torchaudio
import soundfile as sf
from deep_translator import GoogleTranslator
import datetime
import srt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from moviepy.editor import VideoFileClip
from streamlit_player import st_player

# Title
st.title("🌟 Telugu ASR Studio 🌟")
st.caption("Real-time transcription, translation & subtitle generation for Telugu")

# Load Whisper model
@st.cache_resource(show_spinner=True)
def load_model():
    return whisper.load_model("medium")

model = load_model()

# Audio or Video Upload
st.subheader("🎤 Upload Audio or Video")
audio_file = st.file_uploader("Upload an audio/video file (WAV, MP3, MP4, MKV)", type=["wav", "mp3", "ogg", "mp4", "mkv"])

temp_audio_from_video = None

# Handle video to audio extraction
if audio_file is not None and audio_file.name.endswith((".mp4", ".mkv")):
    st.info("Extracting audio from video...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(audio_file.read())
        tmp_video_path = tmp_video.name
        video_clip = VideoFileClip(tmp_video_path)
        audio_path = tmp_video_path.replace(".mp4", ".wav").replace(".mkv", ".wav")
        video_clip.audio.write_audiofile(audio_path)
        audio_file = open(audio_path, "rb")
        temp_audio_from_video = audio_path

        # Optional: Preview video
        st_player(f"file://{tmp_video_path}")

        # Optional: Chunk long videos
        if video_clip.duration > 300:
            st.warning("Video longer than 5 minutes — will transcribe in chunks.")

# Mic Recording with WebRTC
st.markdown("### 🎙️ Or Record Audio Directly:")
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.int16).tobytes()
        self.audio_chunks.append(pcm)
        return frame

audio_ctx = webrtc_streamer(
    key="telugu-asr-mic",
    mode="SENDONLY",
    audio_receiver_size=256,
    client_settings=ClientSettings(
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

recorded_wav_path = None
if audio_ctx and audio_ctx.state.playing:
    st.info("Recording... Speak now")

if audio_ctx and not audio_ctx.state.playing and hasattr(audio_ctx.audio_processor, "audio_chunks"):
    pcm_data = b"".join(audio_ctx.audio_processor.audio_chunks)
    recorded_wav_path = "recorded_audio.wav"
    with sf.SoundFile(recorded_wav_path, mode="w", samplerate=48000, channels=1, subtype='PCM_16') as f:
        audio_np = np.frombuffer(pcm_data, dtype=np.int16)
        f.write(audio_np)
    st.success("Recording saved!")
    audio_file = open(recorded_wav_path, "rb")

# Translation Option
translate = st.checkbox("Translate Telugu → English")

# Dialect Tagging
dialect = st.selectbox("Select Dialect (optional)", ["", "Telangana", "Rayalaseema", "Coastal Andhra"])

# Transcribe
if st.button("🔊 Transcribe") and audio_file:
    with st.spinner("Processing audio..."):

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Load and convert to 16kHz mono if needed
        audio_data, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            audio_data = torchaudio.functional.resample(audio_data, sr, 16000)
        sf.write(tmp_path, audio_data.T.numpy(), 16000)

        # Visualize waveform
        st.subheader("📈 Audio Waveform")
        y, sr = librosa.load(tmp_path, sr=16000)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

        # Speech speed & pitch
        st.subheader("🎚️ Audio Features")
        duration = librosa.get_duration(y=y, sr=sr)
        pitch = librosa.yin(y, fmin=50, fmax=500)
        avg_pitch = np.mean(pitch)
        st.write(f"**Duration:** {duration:.2f} seconds")
        st.write(f"**Average Pitch:** {avg_pitch:.2f} Hz")

        # Transcribe
        result = model.transcribe(tmp_path, language="te")

        st.subheader("📄 Transcript (Telugu)")
        st.text(result["text"])

        if translate:
            st.subheader("English Translation")
            try:
                translated = GoogleTranslator(source='auto', target='en').translate(result["text"])
                st.text(translated)
            except Exception as e:
                st.warning("Translation failed: " + str(e))

        # Subtitle generation
        segments = result.get("segments")
        if segments:
            subtitles = []
            for i, seg in enumerate(segments):
                start = datetime.timedelta(seconds=seg["start"])
                end = datetime.timedelta(seconds=seg["end"])
                content = seg["text"]
                sub = srt.Subtitle(index=i+1, start=start, end=end, content=content)
                subtitles.append(sub)
            srt_text = srt.compose(subtitles)
            st.download_button("Download Subtitles (.srt)", srt_text, file_name="telugu_output.srt")

        # Download transcript
        st.download_button("Download Transcript (.txt)", result["text"], file_name="telugu_transcript.txt")

        # Optionally save for dataset
        if st.checkbox("Contribute this audio + transcript to Telugu dataset"):
            with open("telugu_dataset.txt", "a", encoding="utf-8") as f:
                f.write(f"
---
Dialect: {dialect}
Text: {result['text']}
")
            st.success("Saved to dataset!")

        os.remove(tmp_path)
        if recorded_wav_path:
            os.remove(recorded_wav_path)
        if temp_audio_from_video:
            os.remove(temp_audio_from_video)

elif audio_file is None:
    st.info("Please upload, record, or extract from a video first.")

# Footer
st.markdown("""
---
Made with ❤️ for Telugu preservation | Whisper + Streamlit + HuggingFace
""")
