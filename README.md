
## 🔊 Advanced-Audio-Denoising-System

### Denoiser app hosted on : https://advanced-audio-denoising-system.streamlit.app/
### Noise mixer hosted on : https://noise-mixer.streamlit.app/
## 📖 Overview

This project focuses on **speech enhancement and noise reduction** to improve the clarity and intelligibility of audio signals in noisy environments.
It demonstrates how a combination of traditional signal processing, adaptive noise estimation, and deep learning can effectively suppress noise while preserving speech quality.

The system implements a **multi-mode audio enhancement pipeline** offering three operational modes:

1. **Baseband Filtering (Fast)** – Basic filtering and noise gating for quick, low-latency cleanup.
2. **Adaptive Equalization (Enhanced)** – Dynamic noise estimation and frequency shaping for improved clarity.
3. **Deep Learning Post-Processing (DL)** – Neural network–based restoration (Demucs) for advanced noise suppression.

This multi-level design allows users to balance **speed**, **processing complexity**, and **output quality** depending on their needs.

---

## ⚙️ Features

* Three distinct processing modes with adjustable complexity
* Built-in metrics for enhancement analysis:

  * **Signal-to-Noise Ratio (SNR)**
  * **Short-Time Objective Intelligibility (STOI)**
  * **Zero-Crossing Rate (ZCR) Change**
  * **Spectral Centroid Shift**
* Interactive visualization of waveform and spectrogram (before & after)
* User-friendly **Streamlit interface** for testing and demonstration

---

## 🧠 Technical Overview

### 🔹 Signal Enhancement Methods

* **Baseband Filtering:** Removes stationary noise using frequency-domain filtering and simple gating.
* **Adaptive Equalization:** Performs adaptive spectral subtraction using the `noisereduce` library for dynamic noise control.
* **Deep Learning Restoration:** Utilizes the **Demucs** neural model (`torch` + `torchaudio`) for high-quality speech restoration.
* **Optional Noise Gate:** Suppresses background noise during silent intervals.

### 🔹 Evaluation Metrics

* **SNR Improvement** – Quantifies background noise reduction.
* **STOI (Short-Time Objective Intelligibility)** – Measures speech intelligibility compared to a clean reference.
* **ZCR Change** – Indicates smoothing and reduction in high-frequency artifacts.
* **Spectral Centroid Shift** – Reflects tonal balance and clarity enhancement.

### 🔹 Visualization

* Waveform and spectrogram plots before and after enhancement.
* Real-time metric computation and comparison.

---

## 🧰 Technologies and Libraries Used

| Category                          | Tools / Libraries                                            |
| --------------------------------- | ------------------------------------------------------------ |
| **Language**                      | Python 3.x                                                   |
| **Interface**                     | Streamlit                                                    |
| **Audio Processing**              | `librosa`, `soundfile`, `noisereduce`, `torchaudio`, `pydub` |
| **Deep Learning Model**           | `Demucs` via `torch` and `torchaudio`                        |
| **Numerical & Signal Processing** | `numpy`, `scipy`                                             |
| **Metrics & Evaluation**          | `pystoi`, `torchmetrics.audio`, custom SNR/ZCR calculations  |
| **Visualization**                 | `matplotlib`, `librosa.display`, `plotly`                    |
| **File Handling**                 | `os`, `io`, `tempfile`                                       |

---

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

### 3. Use the Interface

* Upload a noisy **speech audio file** (using **noisy.py**)
* Choose an enhancement **mode** (Fast / Enhanced / Deep Learning)
* Click **Process** to apply noise reduction
* View the waveform, spectrogram, and metric results
* Listen to and download the enhanced output

---

## 🧩 Workflow

1. **Input Acquisition:** User uploads or records an audio sample.
2. **Mode Selection:** Choose between baseband filtering, adaptive equalization, or deep learning restoration.
3. **Processing:** Selected algorithm suppresses noise and enhances clarity.
4. **Metric Evaluation:** Compute SNR, STOI, ZCR, and spectral centroid shift.
5. **Visualization & Output:** Streamlit displays the enhancement results interactively.

---

## 🎯 Purpose

The project demonstrates how combining **classical filtering**, **adaptive noise estimation**, and **deep learning models** can significantly improve **speech clarity and intelligibility** across different noise conditions.
It provides a practical foundation for applications in **voice communication**, **assistive hearing**, **teleconferencing**, and **audio preprocessing**.

---

## 🧾 Future Enhancements

* Real-time microphone input support
* Integration of advanced deep learning models (Conv-TasNet, SEGAN, DCCRN)
* Additional objective metrics like **PESQ** and **CSIG/CBAK/COVL**
* Batch processing and automatic noise profile detection

---

