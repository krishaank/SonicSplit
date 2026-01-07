# 🎧 SonicSplit AI Pro

**SonicSplit AI Pro** is a web-based audio processing application that allows users to **separate music tracks into individual stems** (Vocals, Drums, Bass, Instrumentals, etc.), apply **audio effects**, and visualize the audio using **interactive spectrograms** — all through an intuitive and futuristic UI built with **Streamlit**.

This project combines **AI-powered source separation**, **digital signal processing (DSP)**, and **audio visualization** into a single, easy-to-use application.

---

## 🚀 Features

### 🎵 AI Audio Stem Separation
- Uses **Spleeter** (pre-trained deep learning models) for:
  - **2-stem separation** (Vocals + Accompaniment)
  - **4-stem separation** (Vocals, Drums, Bass, Other)
- Smart logic automatically selects:
  - **High-quality mode** for vocals/karaoke
  - **Safe low-memory mode** for drums/bass/other

---

### 🎚 Audio Effects (DSP)
- **Pitch shifting** (Key control: −12 to +12 semitones)
- **Tempo control** (Speed: 0.5× to 2.0×)
- Effects are applied **after stem extraction** using Librosa

---

### 📊 Audio Analysis
- **Tempo (BPM) detection**
- **Musical key detection**
- Results displayed as clean metric cards in the UI

---

### 🌈 Interactive Spectrogram Visualization
- Real-time **frequency vs time heatmap**
- Built using **Librosa STFT + Plotly**
- Optimized for low memory usage

---

### 🎨 Professional Cyberpunk UI
- Glass-morphism design
- Dark neon cyber-theme
- Responsive layout with sidebar controls
- Interactive progress indicators & animations

---

## 🧠 How the System Works

1. User uploads an audio file  
2. The system:
   - Analyzes BPM & key
   - Chooses 2-stem or 4-stem AI model automatically  
3. **Spleeter separates the audio**  
4. Optional **DSP effects** (pitch & speed) are applied  
5. Output stem is:
   - Previewed as audio  
   - Visualized as a spectrogram  
   - Available for download  

---

## 🏗️ Project Architecture

SonicSplit/
│
├── app.py # Main Streamlit application
├── output_stems/ # Temporary AI-generated stems
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🧰 Technologies Used

| Category | Tools |
|--------|------|
| Frontend | Streamlit |
| AI Model | Spleeter |
| Audio Processing | Librosa, SoundFile |
| Visualization | Plotly |
| Numerical Computing | NumPy |
| Language | Python 3.11 |

---

## ⚙️ Setup (Optional)

This project requires Python 3.10+ and the following libraries:
- streamlit
- librosa
- soundfile
- numpy
- plotly
- spleeter

The application is intended for academic demonstration purposes.

## 🧪 Supported Audio Formats

- MP3  
- WAV  
- M4A  
- FLAC  
- OGG  

---

## 🛡️ Performance & Stability Optimizations

- Smart sample-rate reduction for heavy AI models  
- Audio duration limiting for safe and stable processing  
- Cached AI models using `st.cache_resource` to reduce reload time  
- Explicit garbage collection for improved memory safety  

---

## 🎯 Use Cases

- Music producers & DJs  
- Karaoke track creation  
- Audio analysis demonstrations  
- AI & DSP academic projects  
- Portfolio and hackathon submissions  

---

## ⚠️ Disclaimer

This project uses pre-trained AI models and is intended strictly for educational and demonstration purposes.  
Output quality depends on the input audio, processing limitations, and model constraints.

---

## ⭐ Future Enhancements

- Full-length track processing (removal of duration limits)  
- Custom deep learning model training  
- Batch audio processing support  
- Cloud-based deployment  
- User authentication and profile system  

---

## 📜 License

This project is open-source and intended for academic use only.

---

### 👩‍💻👨‍💻 Developed By

**Krishank Dubey**  
**Anjali Sevkani**

