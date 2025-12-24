import os
import sys
import time
import shutil
import gc
import streamlit as st
import librosa
import soundfile as sf
import numpy as np

# --- 1. MEMORY OPTIMIZATION FLAGS ---
# Turn off heavy TensorFlow optimizations to save RAM
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

# --- 2. IMPORT CHECK ---
try:
    from spleeter.separator import Separator
except ImportError:
    st.error("CRITICAL ERROR: Spleeter not found!")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SonicSplit AI Mobile",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
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
    .stButton>button {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none;
        border-radius: 50px;
        height: 70px;
        font-size: 20px;
        font-weight: 900; 
        width: 100%;
        font-family: 'Exo 2', sans-serif;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(0, 210, 255, 0.6);
        color: #FFFFFF !important;
    }
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    h1, h2, h3 { color: white !important; font-family: 'Orbitron', sans-serif; letter-spacing: 1px; }
    p, label, small { color: #a0a0a0 !important; }
    .metric-box {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.8rem; color: #00d2ff; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; font-weight: 600; }
    .metric-value { font-size: 2rem; font-family: 'Exo 2', sans-serif; font-weight: 800; color: white; }
    .main-title {
        font-size: 3rem;
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

# --- HELPER 1: SINGLE MODEL CACHE ---
@st.cache_resource(max_entries=1) 
def get_separator(stem_count):
    gc.collect()
    # Explicitly force CPU mode to be safe
    return Separator(f'spleeter:{stem_count}stems', multiprocess=False)

# --- HELPER 2: LOW-RES AUDIO SPLITTER (CRITICAL FIX) ---
def split_audio(file_path, stem_count):
    # 1. CLEANUP
    gc.collect()
    
    # 2. DOWNSAMPLE & TRIM
    # sr=16000 cuts memory usage by 65%. 
    # duration=20 keeps the array small.
    y, sr = librosa.load(file_path, sr=16000, duration=20)
    
    short_filename = "temp_lowres_snippet.wav"
    sf.write(short_filename, y, sr)
    del y # Delete RAM immediately
    gc.collect()
    
    # 3. LOAD & SEPARATE
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
    # Analyze just 20s
    y, sr = librosa.load(file_path, sr=None, duration=20)
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="text-align: center;"><div style="font-size: 3rem;">⚡</div></div>""", unsafe_allow_html=True)
    st.markdown("### 🎛️ CONTROLS")
    mode = st.radio("Target:", ["🎤 Vocals", "🎹 Karaoke", "🥁 Drums", "🎸 Bass", "🎹 Other"])
    st.markdown("---")
    pitch = st.slider("Pitch", -12, 12, 0, 1)
    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)

# --- MAIN LOGIC ---
st.markdown('<div class="main-title">SONIC SPLIT</div>', unsafe_allow_html=True)
st.markdown("""<div style="text-align: center; color: #a0a0a0; font-size: 1rem; margin-bottom: 20px;">AI Audio Engine (Mobile Edition)</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Tap to Upload Audio", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_filename = f"temp_input{file_extension}"
    
    st.audio(uploaded_file, format='audio/wav')
    
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if "Karaoke" in mode: display_label = "INSTRUMENTAL"
    elif "Vocals" in mode: display_label = "VOCALS"
    elif "Drums" in mode: display_label = "DRUMS"
    elif "Bass" in mode: display_label = "BASS"
    else: display_label = "OTHER"
        
    st.markdown(f"### ✨ OUTPUT: <span style='color:#00d2ff'>{display_label}</span>", unsafe_allow_html=True)
    
    process_btn = st.button("🚀 START SEPARATION (Safe Mode)", use_container_width=True)
    
    if process_btn:
        progress_text = st.empty()
        bar = st.progress(0)
        
        try:
            progress_text.text("⚙️ ANALYZING...")
            bpm_val, key_val = analyze_track(temp_filename)
            bar.progress(20)
            
            if "Drums" in mode or "Bass" in mode or "Other" in mode:
                stems_needed = 4
                progress_text.text("🧠 4-STEM NET (Low Res Mode)...")
            else:
                stems_needed = 2
                progress_text.text("🧠 2-STEM NET (Low Res Mode)...")
            
            # Run Separation with RAM limits
            stems = split_audio(temp_filename, stems_needed)
            bar.progress(60)
            
            if "Karaoke" in mode: raw_target = stems["accompaniment"]
            elif "Vocals" in mode: raw_target = stems["vocals"]
            elif "Drums" in mode: raw_target = stems["drums"]
            elif "Bass" in mode: raw_target = stems["bass"]
            else: raw_target = stems["other"]

            final_file = raw_target
            
            if pitch != 0 or speed != 1.0:
                progress_text.text(f"🎚️ ADDING EFFECTS...")
                fx_filename = raw_target.replace(".wav", "_modified.wav")
                final_file = apply_audio_effects(raw_target, fx_filename, pitch, speed)
            
            bar.progress(100)
            time.sleep(0.5)
            progress_text.empty()
            bar.empty()
            
            st.success("DONE!")
            
            m1, m2 = st.columns(2)
            with m1: st.markdown(f'<div class="metric-box"><div class="metric-label">BPM</div><div class="metric-value">{bpm_val}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-box"><div class="metric-label">KEY</div><div class="metric-value">{key_val}</div></div>', unsafe_allow_html=True)
            
            st.audio(final_file, format='audio/wav')
            
            with open(final_file, "rb") as f:
                st.download_button(label="⬇️ DOWNLOAD", data=f, file_name=f"{display_label}_Mobile.wav", mime="audio/wav", use_container_width=True)
        
        except MemoryError:
            st.error("⚠️ Server RAM Full. Try refreshing the page.")
        except Exception as e:
            st.error(f"System Error: {e}")