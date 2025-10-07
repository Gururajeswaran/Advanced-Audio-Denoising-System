import streamlit as st
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment

st.title("Audio Uploader and Noise Mixer")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

def add_noise(audio, noise_level=0.02):
    noise = np.random.normal(0, noise_level, audio.shape)
    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)

if uploaded_file is not None:
    # Read audio (handle mp3 or wav)
    if uploaded_file.name.endswith(".mp3"):
        audio_segment = AudioSegment.from_file(uploaded_file, format="mp3")
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2**15)
        samplerate = audio_segment.frame_rate
        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)
        audio_data = samples
    else:
        audio_data, samplerate = sf.read(uploaded_file)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

    noisy_audio = add_noise(audio_data)

    # Save noisy audio as WAV
    buf_wav = io.BytesIO()
    sf.write(buf_wav, noisy_audio, samplerate, format='WAV')
    st.audio(buf_wav.getvalue(), format='audio/wav')

    # Save noisy audio as MP3
    buf_mp3 = io.BytesIO()
    temp_wav = io.BytesIO()
    sf.write(temp_wav, noisy_audio, samplerate, format='WAV')
    temp_wav.seek(0)
    audio_seg = AudioSegment.from_file(temp_wav, format="wav")
    audio_seg.export(buf_mp3, format="mp3")
    buf_mp3.seek(0)

    # Download buttons
    st.download_button("Download Noisy Audio (WAV)", buf_wav.getvalue(), file_name="noisy_audio.wav")
    st.download_button("Download Noisy Audio (MP3)", buf_mp3.getvalue(), file_name="noisy_audio.mp3")
