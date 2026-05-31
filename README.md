<div align="center">
  <h1>🎧 SonicSplit AI Pro</h1>
  <p><strong>An advanced AI-powered audio separation engine and DSP platform.</strong></p>
  
  <a href="https://sonicsplit.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Play_Now-00ffcc?style=for-the-badge&logo=streamlit&logoColor=black" alt="Live Demo" />
  </a>
  <br/><br/>
</div>

**SonicSplit AI Pro** is a cloud-native, web-based audio processing application that empowers users to **decompose musical tracks into individual stems** (Vocals, Drums, Bass, Instrumentals, etc.), apply **digital signal processing (DSP) effects**, and visualize audio through **interactive real-time spectrograms** — all via a sleek, cyberpunk-inspired UI built with Streamlit.

By combining **pre-trained deep neural networks**, **DSP**, and **low-latency visualization**, SonicSplit provides an accessible yet powerful environment for audio engineering and analysis.

---

## 🚀 Live Deployment
Experience the application live without installing anything:
👉 **[Launch SonicSplit AI Pro](https://sonicsplit.streamlit.app/)**

---

## ✨ Key Features

### 🎵 AI-Powered Stem Separation
- **Dual Neural Network Modes:** Uses **Spleeter** models to handle:
  - **2-Stem Extraction:** Isolates Vocals vs. Accompaniment for instant karaoke generation.
  - **4-Stem Extraction:** Decomposes audio into Vocals, Drums, Bass, and Other instruments.
- **Adaptive Resource Management:** Intelligently downsamples audio (e.g., 44.1kHz to 16kHz) to execute complex separation tasks within strict cloud memory constraints.

### 🎚 DSP Effects Engine
- **Pitch Shifting:** Adjust track keys from **−12 to +12 semitones** on the fly.
- **Time Stretching:** Alter tempo/speed smoothly from **0.5× to 2.0×** without affecting pitch.
- Audio manipulation is powered by the highly optimized `librosa` library.

### 🌈 Interactive Spectrograms & Analysis
- **Live Visualizations:** Generates frequency-vs-time heatmaps using Librosa STFT and Plotly.
- **Musical Metrics:** Automatically analyzes and displays **BPM (Tempo)** and **Musical Key** for any uploaded track.

### 🎨 Immersive UI/UX
- **Cyberpunk Aesthetics:** Built with a dark, neon, glass-morphic design.
- **Responsive Layout:** Clean sidebar controls and interactive progress indicators.

---

## 🏗️ Architecture & Technologies

| Layer | Technologies Used |
|--------|------|
| **Frontend / UI** | Streamlit, HTML/CSS |
| **AI / Machine Learning** | TensorFlow, Spleeter |
| **Audio Processing (DSP)** | Librosa, SoundFile |
| **Data Visualization** | Plotly |
| **Numerical Computing** | NumPy |
| **Language** | Python 3.11 |

---

## ⚙️ Local Setup Instructions

If you prefer to run the application locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krishaank/SonicSplit.git
   cd SonicSplit
   ```

2. **Install dependencies:**
   Ensure you have Python 3.10+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

> **Note:** The first time you process audio, the system will automatically download the required Spleeter pre-trained neural network models.

---

## 🧪 Supported Audio Formats
* **MP3, WAV, M4A, FLAC, OGG**

---

## 🛡️ Performance Optimizations
To ensure stable execution on cloud environments with strict limits (like Streamlit Community Cloud):
- **Smart Sample-Rate Reduction** for heavy AI models.
- **Audio Duration Limiting** to prevent OOM (Out of Memory) crashes.
- **Resource Caching** using `@st.cache_resource` to keep neural networks loaded in RAM.
- **Explicit Garbage Collection** to flush memory safely after processing.

---

## 🎯 Primary Use Cases
* **Music Producers & DJs:** Instantly isolate vocal acapellas or drum loops for remixing.
* **Vocalists:** Create high-quality karaoke backing tracks.
* **Audio Engineers:** Visualize frequency distribution and detect exact track keys.

---

### 👨‍💻 Developed By
**Krishank Dubey** & **Anjali Sevkani**

*Disclaimer: This project uses pre-trained AI models and is intended strictly for educational and demonstration purposes. Output quality depends on the input audio, processing limitations, and model constraints.*
