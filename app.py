import os
import sys
import time
import shutil
import gc
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import plotly.graph_objects as go 

# --- 1. MEMORY OPTIMIZATION FLAGS ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

# --- 2. IMPORT CHECK ---
try:
    from spleeter.separator import Separator
except ImportError:
    st.error("CRITICAL ERROR: Spleeter not found!")
    st.stop()

# --- PAGE CONFIGURATION (RESTORED ORIGINAL UI) ---
st.set_page_config(
    page_title="SonicSplit AI Pro",
    page_icon="🎵",
    layout="wide", # RESTORED: Wide layout looks better on desktop
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (RESTORED CYBERPUNK GLASS CARDS) ---
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

    /* RESTORED: Glass Card Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
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

# --- HELPER 1: CACHED MODEL ---
@st.cache_resource(max_entries=1) 
def get_separator(stem_count):
    gc.collect()
    return Separator(f'spleeter:{stem_count}stems', multiprocess=False)

# --- HELPER 2: SMART SPLITTER (CONDITIONAL LOGIC) ---
def split_audio(file_path, stem_count):
    gc.collect()
    
    # --- SMART LOGIC STARTS HERE ---
    if stem_count == 4:
        # 4-STEM MODE: Use "Nuclear" settings to prevent crash
        target_sr = 16000  # Low Res
        target_duration = 30 # Short Clip
    else:
        # 2-STEM MODE: Use "High Quality" settings
        target_sr = None   # Native Quality
        target_duration = 60 # Longer Clip allowed
    # -------------------------------

    # Load and Trim based on logic
    y, sr = librosa.load(file_path, sr=target_sr, duration=target_duration)
    
    short_filename = "temp_snippet.wav"
    sf.write(short_filename, y, sr)
    del y
    gc.collect()
    
    # Separate
    separator = get_separator(stem_count)
    
    output_dir = "output_stems"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    separator.separate_to_file(short_filename, output_dir)
    gc.collect()
    
    base_name = os.path.splitext(short_filename)[0]
    base_path = os.path.join(output_dir, base_name)
    
    return {
        "vocals": os.path.join(base_path, "vocals.wav"),
        "accompaniment": os.path.join(base_path, "accompaniment.wav"), 
        "drums": os.path.join(base_path, "drums.wav"),
        "bass": os.path.join(base_path, "bass.wav"),
        "other": os.path.join(base_path, "other.wav")
    }

# --- HELPER 3: AUDIO EFFECTS ---
def apply_audio_effects(input_path, output_path, pitch_steps, speed_rate):
    gc.collect()
    y, sr = librosa.load(input_path, sr=None)
    if pitch_steps != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
    if speed_rate != 1.0:
        y = librosa.effects.time_stretch(y, rate=speed_rate)
    sf.write(output_path, y, sr)
    del y
    gc.collect()
    return output_path

# --- HELPER 4: ANALYSIS ---
def analyze_track(file_path):
    y, sr = librosa.load(file_path, sr=None, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(tempo)) if np.isscalar(tempo) else int(round(tempo[0]))
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_key = "C Major"
    max_val = -1
    for i, p in enumerate(pitches):
        if chroma_vals[i] > max_val:
            max_val = chroma_vals[i]
            best_key = f"{p} Major"
            
    return bpm, best_key

# --- HELPER 5: VISUALIZER (RESTORED) ---
def plot_interactive_spectrogram(file_path, title="Audio Analysis"):
    gc.collect()
    # Always load low-res for graph to save RAM, regardless of mode
    y, sr = librosa.load(file_path, sr=None, duration=30) 
    stft_matrix = librosa.stft(y, hop_length=1024)
    D = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    
    max_width = 1000
    if D.shape[1] > max_width:
        step = int(np.ceil(D.shape[1] / max_width))
        D = D[:, ::step]
    
    fig = go.Figure(data=go.Heatmap(
        z=D, colorscale='Viridis', colorbar=dict(title='Intensity (dB)'), hoverongaps=False
    ))
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Hz",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"), margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-bottom: 2rem;"><div style="font-size: 4rem; filter: drop-shadow(0 0 15px rgba(0,210,255,0.6));">⚡</div></div>""", unsafe_allow_html=True)
    st.markdown("### 🎛️ STUDIO CONTROLS")
    mode = st.radio("Target Stem:", ["🎤 Vocals Only", "🎹 Karaoke (No Vocals)", "🥁 Drums Only", "🎸 Bass Only", "🎹 Other Instruments"])
    st.markdown("---")
    st.markdown("### 🎚️ MASTER FX")
    pitch = st.slider("Key / Pitch", -12, 12, 0, 1)
    speed = st.slider("Tempo / Speed", 0.5, 2.0, 1.0, 0.1)
    st.markdown("---")

# --- MAIN UI (RESTORED WIDE LAYOUT) ---
top_section = st.container()
st.markdown("<br>", unsafe_allow_html=True) 
bottom_section = st.container()

with bottom_section:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_file = st.file_uploader("📂 Drop your track here", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is None:
    with top_section:
        st.markdown('<div class="main-title">SONIC SPLIT</div>', unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; color: #a0a0a0; font-size: 1.2rem; margin-bottom: 30px;">The Next-Gen AI Audio Separation Engine.</div>""", unsafe_allow_html=True)
else:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_filename = f"temp_input{file_extension}"
    
    st.markdown("""<style>[data-testid='stFileUploader'] { padding: 15px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; background: transparent !important; } [data-testid='stFileUploader'] section > button { display: none; }</style>""", unsafe_allow_html=True)
    
    with top_section:
        col1, col2 = st.columns(2, gap="large")

        # --- LEFT: ORIGINAL ---
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True) # RESTORED
            st.markdown("### 🎧 INPUT SOURCE")
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("Analyzing Waveform..."):
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                time.sleep(0.5)
                try:
                    fig = plot_interactive_spectrogram(temp_filename, "SOURCE SPECTRUM")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Visualizer disabled.")
            st.markdown('</div>', unsafe_allow_html=True) # RESTORED

        # --- RIGHT: OUTPUT ---
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True) # RESTORED
            
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
                    
                    # --- SMART LOGIC ---
                    if "Drums" in mode or "Bass" in mode or "Other" in mode:
                        stems_needed = 4
                        progress_text.text("🧠 4-STEM NET (30s Safe Mode)...")
                    else:
                        stems_needed = 2
                        progress_text.text("🧠 2-STEM NET (High Quality)...")
                    
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
                    
                    with open(final_file, "rb") as f:
                        st.download_button(label="⬇️ EXPORT STEM", data=f, file_name=f"{display_label}_Processed.wav", mime="audio/wav", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"SYSTEM ERROR: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True) # RESTORED