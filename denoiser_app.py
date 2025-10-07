import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
import pywt
from scipy.fft import fft, ifft, fftfreq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import tempfile
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from pystoi import stoi

    STOI_AVAILABLE = True
except:
    STOI_AVAILABLE = False

try:
    import noisereduce as nr

    NOISEREDUCE_AVAILABLE = True
except:
    NOISEREDUCE_AVAILABLE = False

try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    DEMUCS_AVAILABLE = True
except:
    DEMUCS_AVAILABLE = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_audio(file_path, sr=22050):
    """Load audio file and return audio array and sample rate"""
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error loading audio: {str(e)}")
        return None, None


import soundfile as sf
import numpy as np
import tempfile

def save_audio(audio, sr):
    # --- FIX START ---
    if audio is None or len(audio) == 0:
        raise ValueError("Audio array is empty, cannot save.")

    # Ensure correct dtype
    audio = np.asarray(audio, dtype=np.float32)

    # Ensure correct shape: (samples,) or (samples, channels)
    if audio.ndim == 1:
        pass  # mono OK
    elif audio.ndim == 2:
        audio = audio.T if audio.shape[0] < audio.shape[1] else audio  # make (samples, channels)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    # --- FIX END ---

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio, sr, format="WAV", subtype="PCM_16")
    return temp_file.name



def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def estimate_noise_profile(audio, sr, noise_duration=0.5):
    """Estimate noise profile from quietest portion of audio"""
    frame_length = int(noise_duration * sr)
    hop_length = frame_length // 4

    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Find quietest section
    quietest_idx = np.argmin(rms)
    start_sample = quietest_idx * hop_length
    end_sample = start_sample + frame_length

    # Ensure we don't go out of bounds
    end_sample = min(end_sample, len(audio))

    noise_sample = audio[start_sample:end_sample]
    return noise_sample


# ============================================================================
# DSP PROCESSING METHODS
# ============================================================================

def spectral_subtraction(audio, sr, noise_sample, alpha=2.0, beta=0.1):
    """
    Spectral subtraction for noise reduction

    Parameters:
    - alpha: Over-subtraction factor (higher = more aggressive)
    - beta: Spectral floor (prevents over-subtraction)
    """
    n_fft = 2048
    hop_length = n_fft // 4

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise magnitude spectrum
    noise_stft = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)
    noise_magnitude = np.abs(noise_stft)
    noise_avg = np.mean(noise_magnitude, axis=1, keepdims=True)

    # Spectral subtraction
    magnitude_cleaned = magnitude - alpha * noise_avg

    # Apply spectral floor
    magnitude_cleaned = np.maximum(magnitude_cleaned, beta * magnitude)

    # Reconstruct signal
    stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
    audio_cleaned = librosa.istft(stft_cleaned, hop_length=hop_length, length=len(audio))

    return audio_cleaned


def wiener_filter(audio, sr, noise_sample, noise_power_scale=1.0):
    """
    Wiener filtering for noise reduction

    Parameters:
    - noise_power_scale: Scaling factor for noise power estimation
    """
    n_fft = 2048
    hop_length = n_fft // 4

    # STFT of signal
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(stft) ** 2

    # Estimate noise power
    noise_stft = librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True) * noise_power_scale

    # Wiener filter
    wiener_gain = np.maximum(power - noise_power, 0) / (power + 1e-10)

    # Apply filter
    stft_filtered = stft * wiener_gain
    audio_filtered = librosa.istft(stft_filtered, hop_length=hop_length, length=len(audio))

    return audio_filtered


def wavelet_denoising(audio, wavelet='db8', level=6, threshold_scale=1.0):
    """
    Wavelet-based denoising using soft thresholding

    Parameters:
    - wavelet: Wavelet family to use
    - level: Decomposition level
    - threshold_scale: Threshold scaling factor
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(audio, wavelet, level=level)

    # Calculate threshold using universal threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(audio))) * threshold_scale

    # Apply soft thresholding to detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # Reconstruct signal
    audio_denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Handle length mismatch
    if len(audio_denoised) > len(audio):
        audio_denoised = audio_denoised[:len(audio)]
    elif len(audio_denoised) < len(audio):
        audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))

    return audio_denoised


def butterworth_filter(audio, sr, lowcut=80, highcut=8000, order=5):
    """
    Apply Butterworth band-pass filter

    Parameters:
    - lowcut: Low frequency cutoff (Hz)
    - highcut: High frequency cutoff (Hz)
    - order: Filter order
    """
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1.0:
        high = 0.99

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)

    return filtered_audio


def noise_gate(audio, threshold_db=-40, attack=0.01, release=0.1, sr=22050):
    """
    Apply noise gate to suppress signals below threshold

    Parameters:
    - threshold_db: Gate threshold in dB
    - attack: Attack time in seconds
    - release: Release time in seconds
    """
    # Convert threshold to linear
    threshold_linear = 10 ** (threshold_db / 20)

    # Calculate envelope
    envelope = np.abs(audio)

    # Smooth envelope
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)

    gain = np.zeros_like(envelope)
    for i in range(len(envelope)):
        if envelope[i] > threshold_linear:
            # Attack
            if i > 0:
                gain[i] = min(1.0, gain[i - 1] + 1.0 / max(attack_samples, 1))
            else:
                gain[i] = 1.0
        else:
            # Release
            if i > 0:
                gain[i] = max(0.0, gain[i - 1] - 1.0 / max(release_samples, 1))
            else:
                gain[i] = 0.0

    return audio * gain


def dsp_denoise_pipeline(audio, sr, noise_sample, params):
    """
    Complete DSP denoising pipeline
    """
    audio_clean = audio.copy()

    # Stage 1: Spectral Subtraction
    if params['use_spectral_subtraction']:
        audio_clean = spectral_subtraction(
            audio_clean, sr, noise_sample,
            alpha=params['spectral_alpha'],
            beta=params['spectral_beta']
        )

    # Stage 2: Wiener Filter
    if params['use_wiener']:
        audio_clean = wiener_filter(
            audio_clean, sr, noise_sample,
            noise_power_scale=params['wiener_scale']
        )

    # Stage 3: Wavelet Denoising
    if params['use_wavelet']:
        audio_clean = wavelet_denoising(
            audio_clean,
            wavelet=params['wavelet_type'],
            level=params['wavelet_level'],
            threshold_scale=params['wavelet_threshold']
        )

    # Stage 4: Band-pass Filter
    if params['use_bandpass']:
        audio_clean = butterworth_filter(
            audio_clean, sr,
            lowcut=params['lowcut'],
            highcut=params['highcut'],
            order=params['filter_order']
        )

    # Stage 5: Noise Gate
    if params['use_noise_gate']:
        audio_clean = noise_gate(
            audio_clean,
            threshold_db=params['gate_threshold'],
            attack=params['gate_attack'],
            release=params['gate_release'],
            sr=sr
        )

    # Normalize
    audio_clean = normalize_audio(audio_clean)

    return audio_clean



# ENHANCEMENT METHODS
#optional - heavy on gpu

def noisereduce_denoise(audio, sr, noise_sample, stationary=True, prop_decrease=1.0):
    """
    Apply noisereduce library (spectral gating) - CPU friendly!

    Parameters:
    - stationary: Whether noise is stationary
    - prop_decrease: Proportion of noise to reduce (0-1)
    """
    if not NOISEREDUCE_AVAILABLE:
        st.warning("NoiseReduce not available. Install with: pip install noisereduce")
        return audio

    try:
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_sample,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        return normalize_audio(reduced)
    except Exception as e:
        st.warning(f"NoiseReduce failed: {str(e)}")
        return audio


def demucs_denoise(audio, sr, model_name='htdemucs'):
    """
    Apply Demucs model for noise reduction - FORCES execution regardless of GPU
    """
    if not DEMUCS_AVAILABLE:
        st.error(" Demucs not available. Install with: pip install torch torchaudio demucs")
        return audio

    try:
        # Load model - will use whatever device is available
        with st.spinner("Loading Demucs model (first time may take a while)..."):
            model = get_model(model_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            if device == 'cpu':
                st.warning("Running on CPU - this will be slow!")

        # Prepare audio
        # Prepare audio
        audio = np.array(audio)

        # Ensure stereo for Demucs
        if audio.ndim == 1:  # mono
            audio = np.stack([audio, audio], axis=0)  # [2, samples]
        elif audio.shape[0] == 1:  # single-channel
            audio = np.repeat(audio, 2, axis=0)  # [2, samples]

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, 2, samples]

        # Resample if needed (Demucs expects 44100 Hz)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(device)
            audio_tensor = resampler(audio_tensor)

        # Apply model
        with st.spinner("Processing with Demucs... (this may take a while)"):
            with torch.no_grad():
                sources = apply_model(model, audio_tensor, device=device)

        # Extract vocals/main source
        # Demucs outputs: [drums, bass, other, vocals]
        denoised = sources[0, 3, :].cpu().numpy()  # vocals

        # Resample back if needed
        if sr != 44100:
            denoised = librosa.resample(denoised, orig_sr=44100, target_sr=sr)

        # Match length
        if len(denoised) > len(audio):
            denoised = denoised[:len(audio)]
        elif len(denoised) < len(audio):
            denoised = np.pad(denoised, (0, len(audio) - len(denoised)))

        return normalize_audio(denoised)

    except Exception as e:
        st.error(f"Demucs processing failed: {str(e)}")
        return audio


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0 or noise_power < 1e-10:
        return 100.0  # Very high SNR

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_metrics(original, denoised, sr):
    """Calculate comprehensive audio quality metrics"""
    metrics = {}

    # --- FIX: Ensure both arrays are 1D and same length ---
    # Convert stereo → mono if needed
    if denoised.ndim > 1:
        denoised = np.mean(denoised, axis=0)
    if original.ndim > 1:
        original = np.mean(original, axis=0)

    # Match lengths
    min_len = min(len(original), len(denoised))
    original = original[:min_len]
    denoised = denoised[:min_len]
    # -------------------------------------------------------

    # SNR improvement
    noise_estimate = original - denoised
    original_snr = calculate_snr(original, noise_estimate)
    metrics['SNR_improvement'] = original_snr

    # RMS Energy
    metrics['RMS_original'] = np.sqrt(np.mean(original ** 2))
    metrics['RMS_denoised'] = np.sqrt(np.mean(denoised ** 2))
    metrics['RMS_reduction'] = (1 - metrics['RMS_denoised'] / (metrics['RMS_original'] + 1e-10)) * 100

    # Spectral Centroid
    centroid_orig = librosa.feature.spectral_centroid(y=original, sr=sr)[0]
    centroid_den = librosa.feature.spectral_centroid(y=denoised, sr=sr)[0]
    metrics['Spectral_centroid_shift'] = np.mean(centroid_den) - np.mean(centroid_orig)

    # Zero Crossing Rate
    zcr_orig = librosa.feature.zero_crossing_rate(original)[0]
    zcr_den = librosa.feature.zero_crossing_rate(denoised)[0]
    metrics['ZCR_change'] = np.mean(zcr_den) - np.mean(zcr_orig)

    # STOI (if available)
    if STOI_AVAILABLE:
        try:
            if sr != 10000:
                orig_10k = librosa.resample(original, orig_sr=sr, target_sr=10000)
                den_10k = librosa.resample(denoised, orig_sr=sr, target_sr=10000)
            else:
                orig_10k = original
                den_10k = denoised

            stoi_score = stoi(orig_10k, den_10k, 10000, extended=False)
            metrics['STOI'] = stoi_score
        except Exception as e:
            st.warning(f"STOI computation failed: {e}")
            metrics['STOI'] = None
    else:
        metrics['STOI'] = None

    return metrics


def plot_waveforms(original, denoised, sr):
    """Create interactive waveform comparison"""
    time_orig = np.linspace(0, len(original) / sr, len(original))
    time_den = np.linspace(0, len(denoised) / sr, len(denoised))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Original Audio', 'Denoised Audio'),
        vertical_spacing=0.1
    )

    # Sample data for performance (plot every 10th point for long audio)
    step = max(1, len(original) // 10000)

    fig.add_trace(
        go.Scatter(x=time_orig[::step], y=original[::step], mode='lines', name='Original',
                   line=dict(color='#FF6B6B', width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time_den[::step], y=denoised[::step], mode='lines', name='Denoised',
                   line=dict(color='#4ECDC4', width=1)),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_dark',
        title_text="Waveform Comparison"
    )

    return fig


def plot_spectrograms(original, denoised, sr):
    """Create interactive spectrogram comparison"""
    # Compute spectrograms
    D_orig = librosa.amplitude_to_db(
        np.abs(librosa.stft(original, n_fft=2048, hop_length=512)),
        ref=np.max
    )
    D_den = librosa.amplitude_to_db(
        np.abs(librosa.stft(denoised, n_fft=2048, hop_length=512)),
        ref=np.max
    )

    # Time and frequency axes
    times = librosa.frames_to_time(
        np.arange(D_orig.shape[1]),
        sr=sr, hop_length=512
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original Spectrogram', 'Denoised Spectrogram'),
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Heatmap(
            z=D_orig, x=times, y=freqs,
            colorscale='Viridis',
            colorbar=dict(x=0.45, len=0.9),
            name='Original'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=D_den, x=times, y=freqs,
            colorscale='Viridis',
            colorbar=dict(x=1.02, len=0.9),
            name='Denoised'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)

    fig.update_layout(
        height=500,
        template='plotly_dark',
        title_text="Spectrogram Comparison (dB)"
    )

    return fig


def plot_frequency_spectrum(original, denoised, sr):
    """Plot frequency spectrum comparison"""
    # Compute FFT
    fft_orig = np.abs(fft(original))
    fft_den = np.abs(fft(denoised))

    freqs = fftfreq(len(original), 1 / sr)

    # Only positive frequencies
    positive_freqs = freqs[:len(freqs) // 2]
    fft_orig = fft_orig[:len(fft_orig) // 2]
    fft_den = fft_den[:len(fft_den) // 2]

    # Convert to dB
    fft_orig_db = 20 * np.log10(fft_orig + 1e-10)
    fft_den_db = 20 * np.log10(fft_den + 1e-10)

    fig = go.Figure()

    # Sample for performance
    step = max(1, len(positive_freqs) // 5000)

    fig.add_trace(go.Scatter(
        x=positive_freqs[::step], y=fft_orig_db[::step],
        mode='lines', name='Original',
        line=dict(color='#FF6B6B', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=positive_freqs[::step], y=fft_den_db[::step],
        mode='lines', name='Denoised',
        line=dict(color='#4ECDC4', width=2)
    ))

    fig.update_layout(
        title="Frequency Spectrum Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template='plotly_dark',
        height=400,
        xaxis_type='log'
    )

    return fig


def plot_psd(original, denoised, sr):
    """Plot Power Spectral Density"""
    freqs_orig, psd_orig = signal.welch(original, sr, nperseg=2048)
    freqs_den, psd_den = signal.welch(denoised, sr, nperseg=2048)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs_orig, y=10 * np.log10(psd_orig + 1e-10),
        mode='lines', name='Original',
        line=dict(color='#FF6B6B', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=freqs_den, y=10 * np.log10(psd_den + 1e-10),
        mode='lines', name='Denoised',
        line=dict(color='#4ECDC4', width=2)
    ))

    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power/Frequency (dB/Hz)",
        template='plotly_dark',
        height=400
    )

    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Audio Denoising System",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
        }
        .stAlert {
            background-color: #1E1E1E;
        }
        h1 {
            color: #4ECDC4;
            font-weight: bold;
        }
        h2, h3 {
            color: #FF6B6B;
        }
        .metric-card {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #4ECDC4;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Advanced Audio Denoising System")
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Processing mode
        st.subheader("Processing Mode")
        processing_mode = st.radio(
            "Select method:",
            ["Classical Filtering (Fast ⚡)", "🌟 Classical Filtering + Light Enhancement (using NoiseReduce)", " 🔥 Classical Filtering + Heavy Enhancement (using Demucs (DL model by Meta)"],
            index=1,
            help="Classical Filtering Only: Pure signal processing\nLight: Classical Filtering + NoiseReduce (fast) does Adaptive Noise Suppression (ANS) which is basically Dynamic noise estimation & reduction\nHeavy: Classical Filtering  + Demucs (slow but best quality) uses Neural Channel Restoration (NCR) for Deep learning-based reconstruction"
        )

        if "Heavy" in processing_mode:
            st.info(" Demucs will run regardless of hardware. May be slow on CPU!")

        st.markdown("---")

        # DSP Parameters
        st.subheader("🔧 Classical Filtering Parameters")

        # Quick presets
        preset = st.selectbox(
            "Preset",
            ["Custom", "Light Cleaning", "Medium Denoising", "Aggressive Removal"],
            index=2
        )

        # Set defaults based on preset
        if preset == "Light Cleaning":
            default_alpha, default_beta, default_wiener = 1.5, 0.2, 0.8
        elif preset == "Medium Denoising":
            default_alpha, default_beta, default_wiener = 2.0, 0.1, 1.0
        elif preset == "Aggressive Removal":
            default_alpha, default_beta, default_wiener = 3.0, 0.05, 1.2
        else:
            default_alpha, default_beta, default_wiener = 2.0, 0.1, 1.0

        with st.expander("Spectral Subtraction", expanded=False):
            use_spectral = st.checkbox("Enable", value=True, key='spectral')
            spectral_alpha = st.slider("Alpha (aggressiveness)", 0.5, 5.0, default_alpha, 0.1)
            spectral_beta = st.slider("Beta (floor)", 0.0, 0.5, default_beta, 0.01)

        with st.expander("Wiener Filter", expanded=False):
            use_wiener = st.checkbox("Enable", value=True, key='wiener')
            wiener_scale = st.slider("Noise scale", 0.5, 2.0, default_wiener, 0.1)

        with st.expander("Wavelet Denoising", expanded=False):
            use_wavelet = st.checkbox("Enable", value=True, key='wavelet')
            wavelet_type = st.selectbox("Wavelet", ['db8', 'sym8', 'coif5', 'bior3.9'])
            wavelet_level = st.slider("Decomposition level", 3, 10, 6)
            wavelet_threshold = st.slider("Threshold scale", 0.5, 3.0, 1.0, 0.1)

        with st.expander("Band-pass Filter", expanded=False):
            use_bandpass = st.checkbox("Enable", value=True, key='bandpass')
            lowcut = st.slider("Low cutoff (Hz)", 20, 500, 80, 10)
            highcut = st.slider("High cutoff (Hz)", 4000, 16000, 8000, 100)
            filter_order = st.slider("Filter order", 2, 10, 5)

        with st.expander("Noise Gate", expanded=False):
            use_gate = st.checkbox("Enable", value=False, key='gate')
            gate_threshold = st.slider("Threshold (dB)", -60, -20, -40, 5)
            gate_attack = st.slider("Attack (s)", 0.001, 0.1, 0.01, 0.001)
            gate_release = st.slider("Release (s)", 0.01, 0.5, 0.1, 0.01)

        st.markdown("---")

        # Enhancement Parameters
        if "Light" in processing_mode:
            st.subheader(" Light Enhancement")
            if NOISEREDUCE_AVAILABLE:
                nr_stationary = st.checkbox("Stationary Noise", value=True,
                                            help="Check if noise is consistent throughout")
                nr_strength = st.slider("Enhancement Strength", 0.5, 1.0, 0.9, 0.05)
                st.success("✅ NoiseReduce available")
            else:
                st.error("❌ NoiseReduce not available")
                st.code("pip install noisereduce")

        if "Heavy" in processing_mode:
            st.subheader(" Heavy Enhancement (Demucs)")
            if DEMUCS_AVAILABLE:
                demucs_model = st.selectbox("Model", ["htdemucs", "mdx_extra"])
                st.success("Demucs available")
                st.warning(" First run will download model (~300MB)")
            else:
                st.error("❌ Demucs not available")
                st.code("pip install torch torchaudio demucs")

        st.markdown("---")

        # Audio parameters
        st.subheader("🎚️ Audio Settings")
        target_sr = st.selectbox("Sample Rate (Hz)", [16000, 22050, 44100], index=1)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file (MP3 or WAV)",
            type=['mp3', 'wav', 'ogg', 'flac'],
            help="Upload the audio file you want to denoise"
        )

    if uploaded_file is not None:
        # Save uploaded file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        temp_input.write(uploaded_file.read())
        temp_input.close()

        # Load audio
        with st.spinner("Loading audio..."):
            audio, sr = load_audio(temp_input.name, sr=target_sr)

        if audio is not None:
            duration = len(audio) / sr
            st.success(f"✅ Audio loaded: {duration:.2f}s @ {sr}Hz")

            # Display original audio
            with col2:
                st.header("🎧 Original Audio")
                st.audio(temp_input.name)

            # Estimate noise profile
            with st.spinner("Analyzing noise profile..."):
                noise_sample = estimate_noise_profile(audio, sr)
                st.info(f"📊 Noise profile extracted from quietest {len(noise_sample) / sr:.2f}s segment")

            # Process button
            st.markdown("---")
            if st.button(" Process Audio", type="primary", use_container_width=True):

                # Prepare parameters
                dsp_params = {
                    'use_spectral_subtraction': use_spectral,
                    'spectral_alpha': spectral_alpha,
                    'spectral_beta': spectral_beta,
                    'use_wiener': use_wiener,
                    'wiener_scale': wiener_scale,
                    'use_wavelet': use_wavelet,
                    'wavelet_type': wavelet_type,
                    'wavelet_level': wavelet_level,
                    'wavelet_threshold': wavelet_threshold,
                    'use_bandpass': use_bandpass,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'filter_order': filter_order,
                    'use_noise_gate': use_gate,
                    'gate_threshold': gate_threshold,
                    'gate_attack': gate_attack,
                    'gate_release': gate_release,
                }

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Stage 1: DSP Processing
                status_text.text(" Applying Filtering methods...")
                progress_bar.progress(20)

                audio_denoised = dsp_denoise_pipeline(audio, sr, noise_sample, dsp_params)

                progress_bar.progress(50)

                # Stage 2: Enhancement
                if "Light" in processing_mode and NOISEREDUCE_AVAILABLE:
                    status_text.text("✨ Applying light enhancement (NoiseReduce)...")
                    progress_bar.progress(60)

                    audio_denoised = noisereduce_denoise(
                        audio_denoised, sr, noise_sample,
                        stationary=nr_stationary,
                        prop_decrease=nr_strength
                    )

                    progress_bar.progress(80)

                elif "Heavy" in processing_mode:
                    if DEMUCS_AVAILABLE:
                        status_text.text(" Applying heavy enhancement (Demucs)...")
                        progress_bar.progress(60)

                        audio_denoised = demucs_denoise(audio_denoised, sr, model_name=demucs_model)

                        progress_bar.progress(80)
                    else:
                        st.error("❌ Demucs not installed. Using DSP only.")
                        progress_bar.progress(80)

                # Final normalization
                audio_denoised = normalize_audio(audio_denoised)

                status_text.text("📊 Calculating metrics...")
                progress_bar.progress(90)

                # Calculate metrics
                metrics = calculate_metrics(audio, audio_denoised, sr)

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Save denoised audio
                output_path = save_audio(audio_denoised, sr)

                # Display results
                st.markdown("---")
                st.header("📊 Results")

                # Audio comparison
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Original")
                    st.audio(temp_input.name)

                with col_b:
                    st.subheader("Denoised")
                    st.audio(output_path)

                # Metrics display
                st.markdown("---")
                st.subheader("📈 Quality Metrics")

                metric_cols = st.columns(3)

                with metric_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "SNR Improvement",
                        f"{metrics['SNR_improvement']:.2f} dB",
                        delta=f"+{metrics['SNR_improvement']:.1f} dB" if metrics[
                                                                             'SNR_improvement'] > 0 else f"{metrics['SNR_improvement']:.1f} dB"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with metric_cols[1]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Noise Reduction",
                        f"{metrics['RMS_reduction']:.2f}%",
                        delta=f"-{metrics['RMS_reduction']:.1f}%" if metrics['RMS_reduction'] > 0 else "No change"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with metric_cols[2]:
                    if metrics['STOI'] is not None:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "STOI Score",
                            f"{metrics['STOI']:.3f}",
                            help="Short-Time Objective Intelligibility (0.0-1.0, higher is better, >0.8 is good)"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("STOI: Not available\n(pip install pystoi)")

                # Detailed metrics
                with st.expander("Detailed Metrics"):
                    detail_cols = st.columns(2)

                    with detail_cols[0]:
                        st.write("**Energy Analysis:**")
                        st.write(f"- Original RMS: {metrics['RMS_original']:.6f}")
                        st.write(f"- Denoised RMS: {metrics['RMS_denoised']:.6f}")
                        st.write(f"- Energy Reduction: {metrics['RMS_reduction']:.2f}%")

                    with detail_cols[1]:
                        st.write("**Spectral Analysis:**")
                        st.write(f"- Centroid Shift: {metrics['Spectral_centroid_shift']:.2f} Hz")
                        st.write(f"- ZCR Change: {metrics['ZCR_change']:.6f}")
                        if metrics['STOI'] is not None:
                            st.write(f"- STOI Score: {metrics['STOI']:.3f}")

                # Visualizations
                st.markdown("---")
                st.subheader("📊 Visual Analysis")

                # Waveforms
                with st.spinner("Generating waveform comparison..."):
                    fig_wave = plot_waveforms(audio, audio_denoised, sr)
                    st.plotly_chart(fig_wave, use_container_width=True)

                # Spectrograms
                with st.spinner("Generating spectrograms..."):
                    fig_spec = plot_spectrograms(audio, audio_denoised, sr)
                    st.plotly_chart(fig_spec, use_container_width=True)

                # Frequency analysis
                col_freq1, col_freq2 = st.columns(2)

                with col_freq1:
                    with st.spinner("Generating frequency spectrum..."):
                        fig_freq = plot_frequency_spectrum(audio, audio_denoised, sr)
                        st.plotly_chart(fig_freq, use_container_width=True)

                with col_freq2:
                    with st.spinner("Generating power spectral density..."):
                        fig_psd = plot_psd(audio, audio_denoised, sr)
                        st.plotly_chart(fig_psd, use_container_width=True)

                # Download section
                st.markdown("---")
                st.subheader(" Download")

                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Denoised Audio",
                        data=f,
                        file_name=f"denoised_{uploaded_file.name}",
                        mime="audio/wav",
                        type="primary",
                        use_container_width=True
                    )

                # Processing summary
                with st.expander("📝 Processing Summary"):
                    st.write("**Mode:**")
                    st.write(f"- {processing_mode}")

                    st.write("\n**DSP Methods Applied:**")
                    if use_spectral:
                        st.write(f"- ✅ Spectral Subtraction (α={spectral_alpha}, β={spectral_beta})")
                    if use_wiener:
                        st.write(f"- ✅ Wiener Filter (scale={wiener_scale})")
                    if use_wavelet:
                        st.write(f"- ✅ Wavelet Denoising ({wavelet_type}, level={wavelet_level})")
                    if use_bandpass:
                        st.write(f"- ✅ Band-pass Filter ({lowcut}-{highcut} Hz)")
                    if use_gate:
                        st.write(f"- ✅ Noise Gate ({gate_threshold} dB)")

                    if "Light" in processing_mode and NOISEREDUCE_AVAILABLE:
                        st.write("\n**Enhancement:**")
                        st.write(f"- ✅ NoiseReduce (strength={nr_strength})")

                    if "Heavy" in processing_mode and DEMUCS_AVAILABLE:
                        st.write("\n**Enhancement:**")
                        st.write(f"- ✅ Demucs ({demucs_model})")

                    st.write("\n**Processing Info:**")
                    st.write(f"- Audio Duration: {duration:.2f}s")
                    st.write(f"- Sample Rate: {sr}Hz")
                    st.write(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

                # Cleanup temp files
                try:
                    os.unlink(temp_input.name)
                except:
                    pass

    else:
        # Instructions when no file uploaded
        st.info("👆 Please upload an audio file to begin")

        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            ### Quick Start Guide:

            1. **Upload Audio**: Click the upload button and select your MP3 or WAV file
            2. **Select Mode**: 
               - **DSP Only**: Fast, pure signal processing
               - **Light Enhancement**: DSP + NoiseReduce (recommended)
               - **Heavy Enhancement**: DSP + Demucs (best quality, slowest)
            3. **Choose Preset** (or customize):
               - **Light Cleaning**: Gentle noise reduction
               - **Medium Denoising**: Balanced approach (default)
               - **Aggressive Removal**: Maximum noise reduction
            4. **Process**: Click "Process Audio" and wait
            5. **Analyze**: Review metrics, waveforms, and spectrograms
            6. **Download**: Save your cleaned audio

            ### Processing Modes:

            ** Classical Filtering Only (Fast ⚡)**
            - Spectral Subtraction, Wiener Filter, Wavelets
            - Very fast, works on any hardware
            - Good for most use cases

            **Classical Filtering + Light Enhancement (NoiseReduce )**
            - Classical Filtering + spectral gating
            - Adds 2-5 seconds processing time
            - Best balance of speed and quality

            ** Heavy Enhancement (Demucs)**
            - Classical Filtering + state-of-the-art deep learning
            - Slow on CPU, fast on GPU
            - Best possible quality
            - First run downloads ~300MB model

            ### Metrics Explained:

            - **SNR Improvement**: Signal clarity increase (higher = better)
            - **Noise Reduction %**: Amount of noise removed
            - **STOI Score**: Speech intelligibility (0-1, >0.8 is good)
            """)

        with st.expander("Technical Information"):
            st.markdown("""
            ### DSP Methods:

            1. **Spectral Subtraction**: Frequency-domain noise removal
            2. **Wiener Filter**: Optimal MMSE filtering
            3. **Wavelet Denoising**: Multi-resolution thresholding
            4. **Butterworth Filter**: Band-pass frequency selection
            5. **Noise Gate**: Amplitude-based suppression

            ### Deep Learning:

            **NoiseReduce**: Fast spectral gating algorithm
            **Demucs**: Meta's hybrid spectrogram/waveform model

            ### Quality Metrics:

            **STOI**: Short-Time Objective Intelligibility
            - Measures speech clarity (0.0-1.0 scale)
            - >0.8 = good intelligibility
            - Standard metric in hearing aid research

            **SNR**: Signal-to-Noise Ratio
            - Mathematical measure of signal quality
            - Expressed in decibels (dB)

            ### References:

            - Boll (1979): Spectral Subtraction
            - Ephraim & Malah (1984): Wiener Filtering
            - Donoho (1995): Wavelet Denoising
            - Taal et al. (2011): STOI metric
            - Défossez et al. (2021): Demucs architecture
            """)
# Run the app
if __name__ == "__main__":
    main()