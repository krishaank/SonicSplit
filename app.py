import os
import sys
import time
import shutil
import gc  # NEW: Garbage Collection to free memory
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import plotly.graph_objects as go 

# --- 1. WINDOWS FFMPEG FORCE FIX ---
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

# --- 2. IMPORT CHECK ---
try:
    from spleeter.separator import Separator
except ImportError:
    st.error("CRITICAL ERROR: Spleeter not found!")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SonicSplit AI Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&family=Exo+2:wght@300;600;800&display=swap');
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        font-family: 'Inter', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 12, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none;
        border-radius: 50px;
        height: 60px;
        font-size: 18px;
        font-weight: 900; 
        letter-spacing: 1.5px;
        font-family: 'Exo 2', sans-serif;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.6);
        color: #FFFFFF !important;
    }
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        transition: all 0.3s ease;
    }
    [data-testid='stFileUploader']:hover {
        border-color: #00d2ff;
        background-color: rgba(0, 210, 255, 0.05);
    }
    h1, h2, h3 { color: white !important; font-family: 'Orbitron', sans-serif; letter-spacing: 1px; }
    p, label, small { color: #a0a0a0 !important; }
    .metric-box {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-label { font-size: 0.85rem; color: #00d2ff; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }
    .metric-value { font-size: 2.5rem; font-family: 'Exo 2', sans-serif; font-weight: 800; color: white; text-shadow: 0 0 20px rgba(0, 210, 255, 0.8); }
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        background: -webkit-linear-gradient(0deg, #00d2ff, #928DAB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        filter: drop-shadow(0 0 10px rgba(0,210,255,0.3));
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER 1: CACHED MODEL LOADING (CRITICAL FIX) ---
# Using cache_resource ensures the heavy model is loaded only once and stays in memory
# preventing crashes on repeated clicks.
@st.cache_resource
def get_separator(stem_count):
    return Separator(f'spleeter:{stem_count}stems', multiprocess=False)

def split_audio(file_path, stem_count):
    # Retrieve cached model
    separator = get_separator(stem_count)
    
    output_dir = "output_stems"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Force garbage collection before heavy lift
    gc.collect()
    
    separator.separate_to_file(file_path, output_dir)
    
    # Force garbage collection after heavy lift
    gc.collect()
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    base_path = os.path.join(output_dir, filename)
    return {
        "vocals": os.path.join(base_path, "vocals.wav"),
        "accompaniment": os.path.join(base_path, "accompaniment.wav"), 
        "drums": os.path.join(base_path, "drums.wav"),
        "bass": os.path.join(base_path, "bass.wav"),
        "other": os.path.join(base_path, "other.wav")
    }

# --- HELPER 2: AUDIO EFFECTS ---
def apply_audio_effects(input_path, output_path, pitch_steps, speed_rate):
    y, sr = librosa.load(input_path, sr=None)
    if pitch_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
    if speed_rate != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_rate)
    sf.write(output_path, y, sr)
    return output_path

# --- HELPER 3: MUSIC ANALYSIS ---
def analyze_track(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(tempo)) if np.isscalar(tempo) else int(round(tempo[0]))
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    best_score = -1
    best_key = "Unknown"
    for i in range(12):
        major_shifted = np.roll(major_profile, i)
        minor_shifted = np.roll(minor_profile, i)
        maj_corr = np.corrcoef(major_shifted, chroma_vals)[0, 1]
        min_corr = np.corrcoef(minor_shifted, chroma_vals)[0, 1]
        if maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{pitches[i]} Major"
        if min_corr > best_score:
            best_score = min_corr
            best_key = f"{pitches[i]} Minor"
    return bpm, best_key

# --- HELPER 4: OPTIMIZED INTERACTIVE VISUALIZER ---
def plot_interactive_spectrogram(file_path, title="Audio Analysis"):
    # Clear memory before plotting
    gc.collect()
    
    y, sr = librosa.load(file_path, sr=None)
    stft_matrix = librosa.stft(y, hop_length=1024)
    D = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    
    # Aggressive downsampling for Cloud stability
    max_width = 1500 # Reduced from 3000 to save RAM
    if D.shape[1] > max_width:
        step = int(np.ceil(D.shape[1] / max_width))
        D = D[:, ::step]
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis', 
        colorbar=dict(title='Intensity (dB)'),
        hoverongaps=False
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time Frame",
        yaxis_title="Frequency (Hz)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 4rem; filter: drop-shadow(0 0 15px rgba(0,210,255,0.6));">⚡</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ STUDIO CONTROLS")
    mode = st.radio("Target Stem:", ["🎤 Vocals Only", "🎹 Karaoke (No Vocals)", "🥁 Drums Only", "🎸 Bass Only", "🎹 Other Instruments"])
    st.markdown("---")
    st.markdown("### 🎚️ MASTER FX")
    pitch = st.slider("Key / Pitch", -12, 12, 0, 1)
    speed = st.slider("Tempo / Speed", 0.5, 2.0, 1.0, 0.1)
    st.markdown("---")

# --- MAIN LOGIC ---
top_section = st.container()
st.markdown("<br>", unsafe_allow_html=True) 
bottom_section = st.container()

with bottom_section:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_file = st.file_uploader("📂 Drop your track here", type=["mp3", "wav"])

if uploaded_file is None:
    with top_section:
        st.markdown('<div class="main-title">SONIC SPLIT</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center; color: #a0a0a0; font-size: 1.2rem; margin-bottom: 30px;">
                The Next-Gen AI Audio Separation Engine.
            </div>
        """, unsafe_allow_html=True)
else:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_filename = f"temp_input{file_extension}"
    
    st.markdown("""<style>[data-testid='stFileUploader'] { padding: 15px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; background: transparent !important; } [data-testid='stFileUploader'] section > button { display: none; }</style>""", unsafe_allow_html=True)
    
    with top_section:
        col1, col2 = st.columns(2, gap="large")

        # --- LEFT: ORIGINAL ---
        with col1:
            st.markdown("### 🎧 INPUT SOURCE")
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("Analyzing Waveform..."):
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                time.sleep(0.5)
                # Plot Spectrogram (try/except to prevent crash on graph)
                try:
                    fig = plot_interactive_spectrogram(temp_filename, "SOURCE SPECTRUM")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.warning("Visualizer disabled to save memory.")

        # --- RIGHT: OUTPUT ---
        with col2:
            if "Karaoke" in mode: display_label = "INSTRUMENTAL"
            elif "Vocals" in mode: display_label = "VOCALS"
            elif "Drums" in mode: display_label = "DRUMS"
            elif "Bass" in mode: display_label = "BASS"
            else: display_label = "OTHER"
                
            st.markdown(f"### ✨ AI OUTPUT: <span style='color:#00d2ff'>{display_label}</span>", unsafe_allow_html=True)
            
            process_btn = st.button("INITIALIZE SEPARATION", use_container_width=True)
            
            if process_btn:
                progress_text = st.empty()
                bar = st.progress(0)
                
                try:
                    progress_text.text("⚙️ CALCULATING BPM & KEY...")
                    bpm_val, key_val = analyze_track(temp_filename)
                    bar.progress(15)
                    
                    if "Drums" in mode or "Bass" in mode or "Other" in mode:
                        stems_needed = 4
                        progress_text.text("🧠 ACTIVATING 4-STEM NEURAL NET...")
                    else:
                        stems_needed = 2
                        progress_text.text("🧠 ACTIVATING 2-STEM NEURAL NET...")
                    
                    # Run Separation
                    stems = split_audio(temp_filename, stems_needed)
                    bar.progress(60)
                    
                    if "Karaoke" in mode: raw_target = stems["accompaniment"]
                    elif "Vocals" in mode: raw_target = stems["vocals"]
                    elif "Drums" in mode: raw_target = stems["drums"]
                    elif "Bass" in mode: raw_target = stems["bass"]
                    else: raw_target = stems["other"]

                    final_file = raw_target
                    if pitch != 0 or speed != 1.0:
                        progress_text.text(f"🎚️ APPLYING DSP EFFECTS...")
                        fx_filename = raw_target.replace(".wav", "_modified.wav")
                        final_file = apply_audio_effects(raw_target, fx_filename, pitch, speed)
                    
                    bar.progress(100)
                    time.sleep(0.5)
                    progress_text.empty()
                    bar.empty()
                    
                    st.success("PROCESS COMPLETE")
                    
                    m1, m2 = st.columns(2)
                    with m1: st.markdown(f'<div class="metric-box"><div class="metric-label">TEMPO</div><div class="metric-value">{bpm_val}</div></div>', unsafe_allow_html=True)
                    with m2: st.markdown(f'<div class="metric-box"><div class="metric-label">KEY</div><div class="metric-value">{key_val}</div></div>', unsafe_allow_html=True)
                    
                    st.audio(final_file, format='audio/wav')
                    
                    with st.expander("👁️ OUTPUT SPECTRUM", expanded=True):
                        try:
                            fig_out = plot_interactive_spectrogram(final_file, "OUTPUT ANALYSIS")
                            st.plotly_chart(fig_out, use_container_width=True)
                        except Exception:
                            st.warning("Visualizer disabled to save memory.")
                    
                    with open(final_file, "rb") as f:
                        st.download_button(label="⬇️ EXPORT STEM", data=f, file_name=f"{display_label}_Processed.wav", mime="audio/wav", use_container_width=True)
                
                except MemoryError:
                    st.error("⚠️ OUT OF MEMORY: The song is too long for the free server. Try a shorter clip (under 3 mins).")
                except Exception as e:
                    st.error(f"SYSTEM ERROR: {e}")