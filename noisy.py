import streamlit as st
import numpy as np
import soundfile as sf
import io

st.title("Audio Uploader and Noise Mixer")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

def add_noise(audio, noise_level=0.02):
    noise = np.random.normal(0, noise_level, audio.shape)
    noisy_audio = audio + noise
    # Clip to valid range
    return np.clip(noisy_audio, -1.0, 1.0)

if uploaded_file is not None:
    audio_data, samplerate = sf.read(uploaded_file)
    # If stereo, convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    noisy_audio = add_noise(audio_data)
    # Save to buffer
    buf = io.BytesIO()
    sf.write(buf, noisy_audio, samplerate, format='WAV')
    st.audio(buf.getvalue(), format='audio/wav')
    st.download_button("Download Noisy Audio", buf.getvalue(), file_name="noisy_audio.wav")