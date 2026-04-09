import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write
from scipy.signal import wiener, medfilt
import librosa
import librosa.display
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import io
from scipy.io import wavfile
import pandas as pd


# Load trained model
MODEL_PATH = "data/denoising_model.h5"


def load_model_compat(path):
    """Try to load a Keras HDF5 model, with a fallback that patches
    'batch_shape' -> 'batch_input_shape' in the saved JSON config when
    deserialization raises a TypeError due to version mismatch.
    """
    # Prefer reconstructing the original architecture and loading weights
    # directly from the HDF5 file (avoids full-model deserialization issues).
    try:
        with h5py.File(path, "r") as f:
            if "model_config" in f.attrs:
                model_config = f.attrs["model_config"]
            elif "model_config" in f:
                model_config = f["model_config"][()]
            else:
                model_config = None

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        seq_len = None
        if model_config is not None:
            try:
                cfg = json.loads(model_config)
            except Exception:
                cfg = None

            def find_batch_shape(obj):
                if isinstance(obj, dict):
                    if 'batch_shape' in obj:
                        return obj['batch_shape']
                    for v in obj.values():
                        res = find_batch_shape(v)
                        if res is not None:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_batch_shape(item)
                        if res is not None:
                            return res
                return None

            if cfg is not None:
                bs = find_batch_shape(cfg)
                if bs and isinstance(bs, list) and len(bs) >= 2 and isinstance(bs[1], int):
                    seq_len = int(bs[1])

        if seq_len is None:
            seq_len = 1000

        # Try to recreate model using the training builder
        try:
            from train_model import build_model
            model = build_model((seq_len, 1))
            model.load_weights(path)
            return model
        except Exception:
            # Last-resort: try a local reproduction of the training architecture
            from tensorflow.keras import layers, models

            def _build_local_model(input_shape):
                m = models.Sequential()
                m.add(layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape))
                m.add(layers.MaxPooling1D(2, padding='same'))
                m.add(layers.Conv1D(128, 3, activation='relu', padding='same'))
                m.add(layers.MaxPooling1D(2, padding='same'))
                m.add(layers.Conv1D(256, 3, activation='relu', padding='same'))
                m.add(layers.GlobalAveragePooling1D())
                m.add(layers.Dense(input_shape[-1], activation='linear'))
                m.compile(optimizer='adam', loss='mean_squared_error')
                return m

            model = _build_local_model((seq_len, 1))
            model.load_weights(path)
            return model
    except Exception as e:
        # Fall back to standard load_model if reconstruction fails
        try:
            return load_model(path)
        except Exception:
            raise e


try:
    model = load_model_compat(MODEL_PATH)
except Exception as e:
    model = None
    print("Could not load AI model; continuing without it.")
    print(repr(e))

# Infer the fixed time dimension the model was trained on (if any)
MODEL_INPUT_LEN = None
try:
    # Typical Conv1D shape: (None, T, C)
    if isinstance(model.input_shape, (list, tuple)) and len(model.input_shape) >= 2:
        MODEL_INPUT_LEN = model.input_shape[1]
except Exception:
    MODEL_INPUT_LEN = None


# ---------- Utility Functions ----------

# Function to calculate SNR (Signal-to-Noise Ratio)
def calculate_snr(clean_signal, noisy_signal):
    """Calculate SNR in dB"""
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum((clean_signal - noisy_signal) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr


# Function to calculate MSE (Mean Squared Error)
def calculate_mse(original_signal, processed_signal):
    """Calculate Mean Squared Error"""
    return np.mean((original_signal - processed_signal) ** 2)


# Function to calculate RMSE (Root Mean Squared Error)
def calculate_rmse(original_signal, processed_signal):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(original_signal, processed_signal))


# Function to boost audio volume
def boost_audio(signal, boost_db=6.0):
    """
    Boost audio signal by specified dB amount.
    Applies soft clipping to prevent harsh distortion.
    """
    # Convert dB to linear gain
    gain = 10 ** (boost_db / 20.0)
    
    # Apply gain
    boosted = signal * gain
    
    # Soft clipping using tanh for smoother saturation
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        normalized = boosted / (max_val * gain)
        # Soft clip
        clipped = np.tanh(normalized * 1.5) * 0.95
        # Scale back
        boosted = clipped * max_val * gain
    
    # Hard limit to prevent overflow
    boosted = np.clip(boosted, -1.0, 1.0)
    
    return boosted


# Function to extract and visualize noise
def extract_noise(noisy_signal, denoised_signal):
    """
    Extract the noise component by subtracting denoised from noisy.
    """
    return noisy_signal - denoised_signal


def plot_noise_analysis(noisy_signal, denoised_signal, title_suffix=""):
    """
    Create comprehensive noise analysis visualization.
    Shows: 1) Original noisy signal, 2) Extracted noise, 3) Denoised signal
    """
    noise = extract_noise(noisy_signal, denoised_signal)
    
    st.write(f"### 🔬 Noise Analysis {title_suffix}")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    # Plot 1: Noisy Signal
    axes[0].plot(noisy_signal, color='orange', alpha=0.8, linewidth=0.8)
    axes[0].set_title('Input: Noisy Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, len(noisy_signal))
    
    # Plot 2: Extracted Noise
    axes[1].plot(noise, color='red', alpha=0.7, linewidth=0.8)
    axes[1].set_title('Extracted Noise Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, len(noise))
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 3: Denoised Signal
    axes[2].plot(denoised_signal, color='blue', linewidth=1)
    axes[2].set_title('Output: Denoised Signal', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (samples)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, len(denoised_signal))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Noise statistics
    noise_power = np.mean(noise ** 2)
    noise_std = np.std(noise)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Noise Power", f"{noise_power:.6f}")
    with col2:
        st.metric("Noise Std Dev", f"{noise_std:.4f}")
    with col3:
        st.metric("Noise RMS", f"{np.sqrt(noise_power):.4f}")
    with col4:
        noise_reduction_pct = (1 - noise_std / (np.std(noisy_signal) + 1e-8)) * 100
        st.metric("Noise Reduction", f"{noise_reduction_pct:.1f}%")


def plot_before_after_comparison(noisy_signal, denoised_signal, title="Signal Comparison"):
    """
    Side-by-side comparison of noisy and denoised signals.
    """
    st.write(f"### 📊 {title}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Noisy signal
    axes[0].plot(noisy_signal, color='orange', alpha=0.8, linewidth=0.8)
    axes[0].set_title('Before: Noisy Signal', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time (samples)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Denoised signal
    axes[1].plot(denoised_signal, color='blue', linewidth=0.8)
    axes[1].set_title('After: Denoised Signal', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time (samples)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def plot_overlay_comparison(noisy_signal, denoised_signal):
    """
    Overlay comparison showing both signals on same plot.
    """
    st.write("### 🎯 Overlay Comparison")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(noisy_signal, label="Noisy Input", color="orange", alpha=0.6, linewidth=1)
    ax.plot(denoised_signal, label="Denoised Output", color="blue", linewidth=1.2)
    ax.set_xlabel("Time (samples)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title("Noisy vs Denoised Signal Overlay", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def plot_frequency_comparison(noisy_signal, denoised_signal, sr=16000):
    """
    Frequency domain comparison using FFT.
    """
    st.write("### 📈 Frequency Domain Analysis")
    
    # Compute FFT
    noisy_fft = np.fft.fft(noisy_signal)
    denoised_fft = np.fft.fft(denoised_signal)
    
    # Frequency bins
    freqs = np.fft.fftfreq(len(noisy_signal), 1/sr)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    noisy_magnitude = np.abs(noisy_fft)[:len(freqs)//2]
    denoised_magnitude = np.abs(denoised_fft)[:len(freqs)//2]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(positive_freqs, 20 * np.log10(noisy_magnitude + 1e-10), 
            label="Noisy", color="orange", alpha=0.7, linewidth=1)
    ax.plot(positive_freqs, 20 * np.log10(denoised_magnitude + 1e-10), 
            label="Denoised", color="blue", linewidth=1.2)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Magnitude (dB)", fontsize=11)
    ax.set_title("Frequency Spectrum Comparison", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sr/2)
    
    plt.tight_layout()
    st.pyplot(fig)


# Function to create metrics dataframe
def create_metrics_table(noisy_signal, denoised_signal, original_signal=None):
    """
    Create a metrics table comparing noisy and denoised signals.
    If original_signal is provided, calculate additional accuracy metrics.
    """
    metrics = {}
    
    # Signal statistics
    metrics['Metric'] = ['Mean', 'Std Dev', 'Min', 'Max', 'RMS']
    
    noisy_rms = np.sqrt(np.mean(noisy_signal ** 2))
    denoised_rms = np.sqrt(np.mean(denoised_signal ** 2))
    
    metrics['Noisy'] = [
        f"{np.mean(noisy_signal):.4f}",
        f"{np.std(noisy_signal):.4f}",
        f"{np.min(noisy_signal):.4f}",
        f"{np.max(noisy_signal):.4f}",
        f"{noisy_rms:.4f}"
    ]
    
    metrics['Denoised'] = [
        f"{np.mean(denoised_signal):.4f}",
        f"{np.std(denoised_signal):.4f}",
        f"{np.min(denoised_signal):.4f}",
        f"{np.max(denoised_signal):.4f}",
        f"{denoised_rms:.4f}"
    ]
    
    # Calculate improvement percentages
    improvements = []
    for i in range(len(metrics['Metric'])):
        try:
            noisy_val = float(metrics['Noisy'][i])
            denoised_val = float(metrics['Denoised'][i])
            if noisy_val != 0:
                improvement = ((denoised_val - noisy_val) / abs(noisy_val)) * 100
                improvements.append(f"{improvement:+.2f}%")
            else:
                improvements.append("N/A")
        except:
            improvements.append("N/A")
    
    metrics['Change'] = improvements
    
    df = pd.DataFrame(metrics)
    return df


def create_quality_metrics_table(noisy_signal, denoised_signal, method_name):
    """
    Create a quality metrics table for denoising performance.
    """
    # Estimate "clean" signal by assuming denoised is closer to truth
    # This is an approximation since we don't have ground truth
    noise_estimate = noisy_signal - denoised_signal
    
    # Calculate metrics
    snr_before = calculate_snr(denoised_signal, noisy_signal)
    noise_reduction = np.std(noise_estimate) / (np.std(noisy_signal) + 1e-8) * 100
    mse = calculate_mse(noisy_signal, denoised_signal)
    rmse = calculate_rmse(noisy_signal, denoised_signal)
    
    metrics = {
        'Metric': ['SNR Improvement (dB)', 'Noise Reduction (%)', 'MSE', 'RMSE'],
        'Value': [
            f"{snr_before:.2f}",
            f"{noise_reduction:.2f}",
            f"{mse:.6f}",
            f"{rmse:.6f}"
        ]
    }
    
    df = pd.DataFrame(metrics)
    return df


# Function to denoise using AI model
def denoise_signal(noisy_signal):
    noisy_signal = np.asarray(noisy_signal).astype(np.float32)
    if noisy_signal.ndim == 2:
        noisy_signal = noisy_signal[:, 0]
    if noisy_signal.ndim != 1:
        st.error("❌ Error: Invalid signal shape for AI model. The signal must be 1D.")
        st.stop()

    original_len = noisy_signal.shape[0]
    target_len = MODEL_INPUT_LEN or original_len

    # Resize (pad / truncate) to match the model's expected time dimension
    if target_len != original_len:
        proc = np.resize(noisy_signal, target_len)
    else:
        proc = noisy_signal

    if model is None:
        st.error("❌ AI model is not available (failed to load). Use a classical filter instead.")
        st.stop()

    noisy_signal_reshaped = proc.reshape(1, -1, 1)

    denoised = model.predict(noisy_signal_reshaped).squeeze()

    # Resize back to original length so plots and downloads match the input
    if denoised.ndim == 0:
        # Extremely defensive: treat scalar output as constant signal
        denoised = np.full(original_len, float(denoised), dtype=np.float32)
    elif denoised.shape[0] != original_len:
        denoised = np.resize(denoised, original_len)

    return denoised.astype(np.float32)


# Denoising with Wiener Filter
def apply_wiener(noisy_signal):
    noisy_signal = np.asarray(noisy_signal).astype(np.float32)
    return wiener(noisy_signal)


# Denoising with Median Filter
def apply_median(noisy_signal, kernel_size=5):
    noisy_signal = np.asarray(noisy_signal).astype(np.float32)
    return medfilt(noisy_signal, kernel_size=kernel_size)


# Save denoised signal as a WAV file
def save_wav(signal, output_path, sr=16000):
    signal = np.asarray(signal).astype(np.float32)
    signal_reshaped = signal.reshape(-1, 1)
    write(output_path, sr, signal_reshaped)
    return output_path


def plot_time_series(noisy_signal, denoised_signal, title_suffix=""):
    st.write(f"### 📊 Comparison of Noisy and Denoised Signals {title_suffix}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(noisy_signal, label="Noisy Signal", color="orange", alpha=0.7)
    ax.plot(denoised_signal, label="Denoised Signal", color="blue", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)


def plot_spectrograms(noisy_signal, denoised_signal, sr, is_audio=True):
    if not is_audio:
        return
    st.write("### 🎨 Spectrogram Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_signal)), ref=np.max)
    D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_signal)), ref=np.max)

    librosa.display.specshow(D_noisy, sr=sr, x_axis="time", y_axis="log", ax=ax[0])
    ax[0].set_title("Noisy Signal Spectrogram")

    librosa.display.specshow(D_denoised, sr=sr, x_axis="time", y_axis="log", ax=ax[1])
    ax[1].set_title("Denoised Signal Spectrogram")
    plt.colorbar(format="%+2.0f dB", ax=ax[1])
    st.pyplot(fig)


def _ensure_1d(signal):
    signal = np.asarray(signal)
    # Handle scalar / 0D edge case (e.g., file with a single value)
    if signal.ndim == 0:
        signal = signal.reshape(1)
    signal = np.squeeze(signal)
    if signal.ndim != 1:
        raise ValueError("Expected a 1D signal after loading.")
    return signal


def load_uploaded_1d_signal(uploaded_file, kind="audio", csv_sr=16000):
    """
    Generic loader for 1D signals from .npy, .wav, or .csv files.
    Returns (signal, sample_rate_or_1_for_non_audio)
    """
    name = uploaded_file.name.lower()
    if name.endswith(".npy"):
        signal = np.load(uploaded_file)
        sr = 16000 if kind == "audio" else 1
    elif name.endswith(".wav"):
        signal, sr = librosa.load(uploaded_file, sr=None)
    elif name.endswith(".csv") or name.endswith(".txt"):
        # Accept both comma-separated and newline-separated single-column files
        try:
            signal = np.loadtxt(uploaded_file, delimiter=",")
        except Exception:
            uploaded_file.seek(0)
            signal = np.loadtxt(uploaded_file)
        sr = int(csv_sr) if kind == "audio" else 1
    else:
        st.error("❌ Unsupported file format. Please upload a .npy, .wav, .csv, or .txt file.")
        st.stop()

    try:
        signal = _ensure_1d(signal).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

    return signal, sr


def wav_bytes_from_float(signal, sr):
    """
    Convert float32 waveform (mono) into WAV bytes for st.audio without touching disk.
    """
    buf = io.BytesIO()
    sf.write(buf, np.asarray(signal, dtype=np.float32), int(sr), format="WAV")
    return buf.getvalue()


def compare_all_methods(noisy_signal, sr=None, is_audio=False):
    """
    Compare all three denoising methods side by side with metrics.
    """
    st.write("## 🔬 Method Comparison")
    
    # Apply all three methods
    methods = {
        "AI Model": denoise_signal(noisy_signal) if model is not None else None,
        "Wiener Filter": apply_wiener(noisy_signal),
        "Median Filter": apply_median(noisy_signal)
    }
    
    # Create tabs for each method
    tabs = st.tabs(["AI Model", "Wiener Filter", "Median Filter", "📊 Comparison Table"])
    
    comparison_data = []
    
    for idx, (method_name, denoised) in enumerate(methods.items()):
        if denoised is None:
            continue
            
        with tabs[idx]:
            st.write(f"### {method_name}")
            
            # Show metrics
            metrics_df = create_quality_metrics_table(noisy_signal, denoised, method_name)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Audio playback for speech
            if is_audio and sr:
                st.write("#### 🔊 Audio Preview")
                
                # Apply boost FIRST, then denoise AGAIN
                enable_boost = st.checkbox(f"Enable Audio Boost (+6dB) with Re-Denoising", key=f"boost_{method_name}")
                
                if enable_boost:
                    # Boost the denoised signal
                    boosted_signal = boost_audio(denoised, boost_db=6.0)
                    
                    # Apply denoising again on the boosted signal
                    if method_name == "AI Model" and model is not None:
                        final_signal = denoise_signal(boosted_signal)
                    elif method_name == "Wiener Filter":
                        final_signal = apply_wiener(boosted_signal)
                    else:
                        final_signal = apply_median(boosted_signal)
                    
                    st.audio(wav_bytes_from_float(final_signal, sr), format="audio/wav")
                    st.caption("✅ Boosted (+6dB) → Re-Denoised")
                else:
                    st.audio(wav_bytes_from_float(denoised, sr), format="audio/wav")
                    st.caption("Standard denoised output")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(noisy_signal, label="Noisy", alpha=0.5, color="orange")
            ax.plot(denoised, label=f"{method_name}", color="blue")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.set_title(f"{method_name} Results")
            st.pyplot(fig)
            
        # Collect comparison data
        noise_estimate = noisy_signal - denoised
        snr = calculate_snr(denoised, noisy_signal)
        mse = calculate_mse(noisy_signal, denoised)
        noise_reduction = np.std(noise_estimate) / (np.std(noisy_signal) + 1e-8) * 100
        
        comparison_data.append({
            'Method': method_name,
            'SNR (dB)': f"{snr:.2f}",
            'MSE': f"{mse:.6f}",
            'Noise Reduction (%)': f"{noise_reduction:.2f}",
            'Processing': 'Neural Network' if method_name == 'AI Model' else 'Classical Filter'
        })
    
    # Comparison table
    with tabs[-1]:
        st.write("### 📊 Performance Comparison")
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        method_names = [d['Method'] for d in comparison_data]
        snr_values = [float(d['SNR (dB)']) for d in comparison_data]
        mse_values = [float(d['MSE']) for d in comparison_data]
        nr_values = [float(d['Noise Reduction (%)']) for d in comparison_data]
        
        axes[0].bar(method_names, snr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_ylabel('SNR (dB)')
        axes[0].set_title('Signal-to-Noise Ratio')
        axes[0].tick_params(axis='x', rotation=15)
        
        axes[1].bar(method_names, mse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Mean Squared Error')
        axes[1].tick_params(axis='x', rotation=15)
        
        axes[2].bar(method_names, nr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[2].set_ylabel('Noise Reduction (%)')
        axes[2].set_title('Noise Reduction')
        axes[2].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        st.pyplot(fig)


# ---------- Real-time Microphone Denoising (Speech) ----------

class RealTimeAudioDenoiser(AudioProcessorBase):
    """
    Real-time audio processor using a classical Wiener filter for low-latency denoising.
    """

    def recv_audio(self, frames):
        output_frames = []
        for frame in frames:
            # Convert incoming frame to mono float32 in [-1, 1]
            samples = frame.to_ndarray()
            if samples.ndim > 1:
                samples = samples.mean(axis=0)
            samples = samples.astype(np.float32) / 32768.0

            # Apply fast classical denoising
            denoised = apply_wiener(samples)

            # Back to int16 for WebRTC
            denoised_int16 = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16)
            denoised_int16 = denoised_int16.reshape(1, -1)

            out_frame = av.AudioFrame.from_ndarray(denoised_int16, layout="mono")
            out_frame.sample_rate = frame.sample_rate
            output_frames.append(out_frame)

        return output_frames


# ---------- Streamlit Layout ----------

st.title("🎧 AI-Powered Multi-Domain Signal Denoising System")
st.write(
    "Real-time speech denoising and file-based denoising for wireless and ECG signals "
    "using AI and classical filters with advanced noise visualization."
)

mode = st.sidebar.selectbox(
    "Select Signal Type",
    ["Speech (Audio)", "Wireless Signal", "ECG Signal"],
)

# Sidebar options
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Advanced Options")
show_comparison = st.sidebar.checkbox("Show Method Comparison", value=False)
show_noise_analysis = st.sidebar.checkbox("Show Detailed Noise Analysis", value=True)
show_frequency_analysis = st.sidebar.checkbox("Show Frequency Analysis", value=False)
audio_boost_default = st.sidebar.slider("Default Audio Boost (dB)", 0.0, 12.0, 6.0, 0.5)


# ---------- Speech (Audio) Denoising ----------
if mode == "Speech (Audio)":
    st.header("🎙️ Speech Denoising")
    tab_upload, tab_live = st.tabs(["Upload Audio File", "Real-time Microphone (Beta)"])

    with tab_upload:
        csv_sr = st.number_input(
            "Sampling rate for uploaded .csv/.txt (only used for playback)",
            min_value=8000,
            max_value=48000,
            value=16000,
            step=1000,
        )
        uploaded_file = st.file_uploader(
            "📂 Upload a Noisy Speech Signal (.npy, .wav, .csv, .txt)",
            type=["npy", "wav", "csv", "txt"],
        )

        if uploaded_file is not None:
            try:
                noisy_signal, sr = load_uploaded_1d_signal(uploaded_file, kind="audio", csv_sr=csv_sr)

                st.success("✅ Noisy Speech Signal Loaded Successfully!")
                st.write(
                    f"🎵 Audio Duration: {len(noisy_signal) / max(sr, 1):.2f} seconds | "
                    f"Sampling Rate: {sr} Hz"
                )

                if show_comparison:
                    # Show all methods comparison
                    compare_all_methods(noisy_signal, sr=sr, is_audio=True)
                else:
                    # Original single-method workflow
                    st.write("### 🔊 Audio Comparison (Noisy vs Denoised)")
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.caption("🔴 Noisy Input")
                        st.audio(wav_bytes_from_float(noisy_signal, sr), format="audio/wav")

                    st.write("### ⚡ Select Denoising Method")
                    method = st.radio(
                        "Choose a denoising method:",
                        ["AI Model (Autoencoder)", "Wiener Filter", "Median Filter"],
                        key="speech_method",
                    )

                    if method == "AI Model (Autoencoder)":
                        denoised_signal = denoise_signal(noisy_signal)
                    elif method == "Wiener Filter":
                        denoised_signal = apply_wiener(noisy_signal)
                    else:
                        denoised_signal = apply_median(noisy_signal)

                    with c2:
                        st.caption("🟢 Denoised")
                        st.audio(wav_bytes_from_float(denoised_signal, sr), format="audio/wav")

                    # Audio boost and re-denoise option
                    st.write("### 🔊 Volume Boost & Re-Denoising")
                    enable_boost = st.checkbox("Enable Audio Boost with Re-Denoising", value=True)
                    
                    if enable_boost:
                        boost_amount = st.slider("Boost Amount (dB)", 0.0, 12.0, audio_boost_default, 0.5)
                        
                        # Step 1: Boost
                        boosted_signal = boost_audio(denoised_signal, boost_amount)
                        
                        # Step 2: Denoise the boosted signal
                        if method == "AI Model (Autoencoder)":
                            final_signal = denoise_signal(boosted_signal)
                        elif method == "Wiener Filter":
                            final_signal = apply_wiener(boosted_signal)
                        else:
                            final_signal = apply_median(boosted_signal)
                        
                        with c3:
                            st.caption(f"🟦 Boosted (+{boost_amount:.1f}dB) → Re-Denoised")
                            st.audio(wav_bytes_from_float(final_signal, sr), format="audio/wav")
                        
                        download_signal = final_signal
                    else:
                        download_signal = denoised_signal

                    # Show metrics table
                    st.write("### 📈 Signal Quality Metrics")
                    if enable_boost:
                        st.info("📊 Metrics show comparison between original noisy and final boosted+re-denoised signal")
                        metrics_df = create_quality_metrics_table(noisy_signal, final_signal, method)
                    else:
                        metrics_df = create_quality_metrics_table(noisy_signal, denoised_signal, method)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.write("### 📊 Signal Statistics")
                    if enable_boost:
                        stats_df = create_metrics_table(noisy_signal, final_signal)
                    else:
                        stats_df = create_metrics_table(noisy_signal, denoised_signal)
                    st.dataframe(stats_df, use_container_width=True)

                    # Visual comparisons
                    if show_noise_analysis:
                        if enable_boost:
                            plot_noise_analysis(noisy_signal, final_signal, title_suffix="(Boosted + Re-Denoised)")
                        else:
                            plot_noise_analysis(noisy_signal, denoised_signal, title_suffix="(Speech)")
                    
                    if enable_boost:
                        plot_before_after_comparison(noisy_signal, final_signal, "Before vs After (with Boost)")
                        plot_overlay_comparison(noisy_signal, final_signal)
                    else:
                        plot_before_after_comparison(noisy_signal, denoised_signal, "Before vs After")
                        plot_overlay_comparison(noisy_signal, denoised_signal)
                    
                    if show_frequency_analysis:
                        if enable_boost:
                            plot_frequency_comparison(noisy_signal, final_signal, sr)
                        else:
                            plot_frequency_comparison(noisy_signal, denoised_signal, sr)

                    plot_spectrograms(noisy_signal, 
                                    final_signal if enable_boost else denoised_signal, 
                                    sr, is_audio=True)

                    # Save and allow download of denoised signal
                    output_path = "denoised_speech.wav"
                    save_wav(download_signal, output_path, sr)
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Final Denoised Speech as WAV",
                            f,
                            file_name="denoised_speech.wav",
                        )

            except Exception as e:
                st.error(f"❌ Error while processing speech file: {str(e)}")

    with tab_live:
        st.write(
            "Record from your microphone, then the app will **denoise** and show a **Noisy vs Denoised** comparison."
        )

        recorded = st.audio_input("🎙️ Record audio")
        if recorded is not None:
            try:
                raw_bytes = recorded.read()
                sr, data = wavfile.read(io.BytesIO(raw_bytes))

                # Convert to mono float32 in [-1, 1]
                data = np.asarray(data)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                if data.dtype.kind in {"i", "u"}:
                    # Assume PCM int
                    maxv = np.iinfo(data.dtype).max
                    noisy_signal = (data.astype(np.float32) / float(maxv)).astype(np.float32)
                else:
                    noisy_signal = data.astype(np.float32)

                st.write("### 🔊 Audio Comparison")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.caption("🔴 Noisy (Recorded)")
                    st.audio(raw_bytes, format="audio/wav")

                # Denoise (fast default for mic recordings)
                denoised_signal = apply_wiener(noisy_signal)
                
                with c2:
                    st.caption("🟢 Denoised")
                    st.audio(wav_bytes_from_float(denoised_signal, sr), format="audio/wav")
                
                # Audio boost for recording
                enable_boost = st.checkbox("🔊 Enable Audio Boost with Re-Denoising", value=True, key="boost_rec")
                
                if enable_boost:
                    boost_amount = st.slider("Boost (dB)", 0.0, 12.0, audio_boost_default, 0.5, key="boost_rec_slider")
                    
                    # Boost then re-denoise
                    boosted = boost_audio(denoised_signal, boost_amount)
                    final_signal = apply_wiener(boosted)
                    
                    with c3:
                        st.caption(f"🟦 Boosted (+{boost_amount:.1f}dB) → Re-Denoised")
                        st.audio(wav_bytes_from_float(final_signal, sr), format="audio/wav")
                    
                    download_signal = final_signal
                else:
                    download_signal = denoised_signal

                # Metrics for recording
                st.write("### 📈 Recording Quality Metrics")
                metrics_df = create_quality_metrics_table(noisy_signal, download_signal, "Wiener Filter")
                st.dataframe(metrics_df, use_container_width=True)
                
                # Noise analysis
                if show_noise_analysis:
                    plot_noise_analysis(noisy_signal, download_signal, title_suffix="(Recording)")

                # Download denoised
                st.download_button(
                    "⬇️ Download Final Denoised Recording (WAV)",
                    data=wav_bytes_from_float(download_signal, sr),
                    file_name="denoised_recording.wav",
                    mime="audio/wav",
                )

            except Exception as e:
                st.error(f"❌ Error while processing microphone recording: {str(e)}")

        with st.expander("Live microphone denoising (optional, Beta)"):
            st.write(
                "This streams audio through the browser and applies low-latency denoising. "
                "Use it if you want to **hear** denoising live; recording+comparison is above."
            )
            webrtc_streamer(
                key="speech-webrtc",
                mode=WebRtcMode.SENDRECV,
                audio_receiver_size=256,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
                audio_processor_factory=RealTimeAudioDenoiser,
            )


# ---------- Wireless Signal Denoising ----------
elif mode == "Wireless Signal":
    st.header("📡 Wireless Signal Denoising")

    uploaded_file = st.file_uploader(
        "📂 Upload a Noisy Wireless Signal (.npy, .csv, .txt)",
        type=["npy", "csv", "txt"],
    )

    if uploaded_file is not None:
        try:
            noisy_signal, _ = load_uploaded_1d_signal(uploaded_file, kind="wireless")

            st.success("✅ Wireless Signal Loaded Successfully!")
            st.write(f"📏 Signal Length: {len(noisy_signal)} samples")

            if show_comparison:
                # Show all methods comparison
                compare_all_methods(noisy_signal, is_audio=False)
            else:
                # Original single-method workflow
                st.write("### ⚡ Select Denoising Method")
                method = st.radio(
                    "Choose a denoising method:",
                    ["AI Model (Autoencoder)", "Wiener Filter", "Median Filter"],
                    key="wireless_method",
                )

                if method == "AI Model (Autoencoder)":
                    denoised_signal = denoise_signal(noisy_signal)
                elif method == "Wiener Filter":
                    denoised_signal = apply_wiener(noisy_signal)
                else:
                    denoised_signal = apply_median(noisy_signal)

                # Show metrics table
                st.write("### 📈 Signal Quality Metrics")
                metrics_df = create_quality_metrics_table(noisy_signal, denoised_signal, method)
                st.dataframe(metrics_df, use_container_width=True)
                
                st.write("### 📊 Signal Statistics")
                stats_df = create_metrics_table(noisy_signal, denoised_signal)
                st.dataframe(stats_df, use_container_width=True)

                # Visual analysis
                if show_noise_analysis:
                    plot_noise_analysis(noisy_signal, denoised_signal, title_suffix="(Wireless)")
                
                plot_before_after_comparison(noisy_signal, denoised_signal, "Wireless Signal Comparison")
                plot_overlay_comparison(noisy_signal, denoised_signal)

                # Save denoised wireless signal as text
                output_path = "denoised_wireless_signal.txt"
                np.savetxt(output_path, denoised_signal, delimiter=",")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Denoised Wireless Signal (.txt)",
                        f,
                        file_name="denoised_wireless_signal.txt",
                    )

        except Exception as e:
            st.error(f"❌ Error while processing wireless signal: {str(e)}")


# ---------- ECG Signal Denoising ----------
elif mode == "ECG Signal":
    st.header("🫀 ECG Signal Denoising")

    uploaded_file = st.file_uploader(
        "📂 Upload a Noisy ECG Signal (.npy, .csv, .txt)",
        type=["npy", "csv", "txt"],
    )

    if uploaded_file is not None:
        try:
            noisy_signal, _ = load_uploaded_1d_signal(uploaded_file, kind="ecg")

            st.success("✅ ECG Signal Loaded Successfully!")
            st.write(f"📏 Signal Length: {len(noisy_signal)} samples")

            if show_comparison:
                # Show all methods comparison
                compare_all_methods(noisy_signal, is_audio=False)
            else:
                # Original single-method workflow
                st.write("### ⚡ Select Denoising Method")
                method = st.radio(
                    "Choose a denoising method:",
                    ["AI Model (Autoencoder)", "Wiener Filter", "Median Filter"],
                    key="ecg_method",
                )

                if method == "AI Model (Autoencoder)":
                    denoised_signal = denoise_signal(noisy_signal)
                elif method == "Wiener Filter":
                    denoised_signal = apply_wiener(noisy_signal)
                else:
                    denoised_signal = apply_median(noisy_signal)

                # Show metrics table
                st.write("### 📈 Signal Quality Metrics")
                metrics_df = create_quality_metrics_table(noisy_signal, denoised_signal, method)
                st.dataframe(metrics_df, use_container_width=True)
                
                st.write("### 📊 Signal Statistics")
                stats_df = create_metrics_table(noisy_signal, denoised_signal)
                st.dataframe(stats_df, use_container_width=True)

                # Visual analysis
                if show_noise_analysis:
                    plot_noise_analysis(noisy_signal, denoised_signal, title_suffix="(ECG)")
                
                plot_before_after_comparison(noisy_signal, denoised_signal, "ECG Signal Comparison")
                plot_overlay_comparison(noisy_signal, denoised_signal)

                # Save denoised ECG signal as text
                output_path = "denoised_ecg_signal.txt"
                np.savetxt(output_path, denoised_signal, delimiter=",")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Denoised ECG Signal (.txt)",
                        f,
                        file_name="denoised_ecg_signal.txt",
                    )

        except Exception as e:
            st.error(f"❌ Error while processing ECG signal: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎯 Multi-Domain Signal Denoising | Powered by AI & Classical Filters</p>
        <p style='font-size: 0.9em;'>✨ Now with Advanced Noise Visualization & Boost+Re-Denoise Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True
)
