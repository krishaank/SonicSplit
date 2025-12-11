import os
import sys
import time
import shutil
import streamlit as st

# --- 1. WINDOWS FFMPEG FORCE FIX ---
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

# --- 2. IMPORT CHECK ---
try:
    from spleeter.separator import Separator
except ImportError:
    st.error("CRITICAL ERROR: Spleeter not found!")
    st.stop()

import preprocess  # Your local helper script

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SonicSplit AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    .stApp { background: radial-gradient(circle at 10% 20%, rgb(18, 18, 25) 0%, rgb(5, 5, 10) 90%); font-family: 'Inter', sans-serif; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 15px; padding: 25px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); color: white; border: none; border-radius: 12px; height: 55px; font-size: 16px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4); }
    [data-testid='stFileUploader'] { background-color: rgba(255, 255, 255, 0.03); border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 50px; text-align: center; transition: all 0.5s ease; }
    [data-testid='stFileUploader']:hover { border-color: #00d2ff; background-color: rgba(255, 255, 255, 0.06); }
    section[data-testid="stSidebar"] { background-color: rgb(18, 18, 25); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    h1, h2, h3 { color: white !important; font-family: 'Orbitron', sans-serif; }
    p, label, small { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION: SMART AI SWITCHER ---
def split_audio(file_path, stem_count):
    """
    Uses Spleeter with dynamic model loading (2stems vs 4stems)
    """
    model_name = f'spleeter:{stem_count}stems'
    
    # Initialize Spleeter (Multiprocessing disabled for Windows safety)
    separator = Separator(model_name, multiprocess=False)
    
    # Output setup
    output_dir = "output_stems"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Run Separation
    separator.separate_to_file(file_path, output_dir)
    
    # Locate files
    filename = os.path.splitext(os.path.basename(file_path))[0]
    base_path = os.path.join(output_dir, filename)
    
    return {
        "vocals": os.path.join(base_path, "vocals.wav"),
        "accompaniment": os.path.join(base_path, "accompaniment.wav"), 
        "drums": os.path.join(base_path, "drums.wav"),
        "bass": os.path.join(base_path, "bass.wav"),
        "other": os.path.join(base_path, "other.wav")
    }

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-bottom: 2rem;"><div style="font-size: 3.5rem;">🎛️</div><h2 style="margin: 0;">SonicSplit</h2></div>""", unsafe_allow_html=True)
    st.markdown("### ⚙️ Studio Settings")
    
    mode = st.radio("Target Stem:", 
        ["🎤 Vocals Only", "🎹 Karaoke (No Vocals)", "🥁 Drums Only", "🎸 Bass Only", "🎹 Other Instruments"]
    )
    
    st.markdown("---")
    st.markdown("<small>Powered by TensorFlow & Spleeter</small>", unsafe_allow_html=True)

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
        st.markdown("""<div style="text-align: center; padding-top: 40px;"><h1 style="font-size: 4.5rem;">Unleash the Stems.</h1><p style="color: #888;">Isolate Vocals, Drums, Bass, and more.</p></div>""", unsafe_allow_html=True)
else:
    # Get Extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    temp_filename = f"temp_input{file_extension}"
    
    # Shrink Uploader
    st.markdown("""<style>[data-testid='stFileUploader'] { padding: 15px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; background: transparent !important; } [data-testid='stFileUploader'] section > button { display: none; }</style>""", unsafe_allow_html=True)
    
    with top_section:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎧 Original Mix")
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("Generating Spectrogram..."):
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                time.sleep(0.5)
                spec, phase, sr = preprocess.load_and_convert(temp_filename)
                fig = preprocess.generate_spectrogram_image(spec, sr)
                st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # --- FIXED LABEL DISPLAY LOGIC ---
            # We explicitly check for Karaoke FIRST to override the word "Vocals"
            if "Karaoke" in mode:
                display_label = "Instrumental (Karaoke)"
            elif "Vocals" in mode:
                display_label = "Vocals"
            elif "Drums" in mode:
                display_label = "Drums"
            elif "Bass" in mode:
                display_label = "Bass"
            else:
                display_label = "Other Instruments"
                
            st.markdown(f"### ✨ AI Output: <span style='color:#00d2ff'>{display_label}</span>", unsafe_allow_html=True)
            
            process_btn = st.button("🚀 Process Audio", use_container_width=True)
            
            if process_btn:
                progress_text = st.empty()
                bar = st.progress(0)
                
                try:
                    # 1. INTELLIGENT MODEL SELECTION
                    if "Drums" in mode or "Bass" in mode or "Other" in mode:
                        stems_needed = 4
                        progress_text.text("🧠 Loading 4-Stem Model (Vocals/Drums/Bass/Other)...")
                    else:
                        stems_needed = 2
                        progress_text.text("🧠 Loading 2-Stem Model (Vocals/Karaoke)...")
                    
                    bar.progress(20)
                    
                    # 2. RUN AI
                    progress_text.text("⚡ Separating Stems...")
                    stems = split_audio(temp_filename, stems_needed)
                    bar.progress(60)
                    
                    # 3. PICK THE CORRECT FILE
                    if "Karaoke" in mode:
                        target_file = stems["accompaniment"]
                    elif "Vocals" in mode:
                        target_file = stems["vocals"]
                    elif "Drums" in mode:
                        target_file = stems["drums"]
                    elif "Bass" in mode:
                        target_file = stems["bass"]
                    else:
                        target_file = stems["other"]

                    progress_text.text("📊 Generating Analysis...")
                    time.sleep(1.0)
                    
                    spec_out, phase_out, sr_out = preprocess.load_and_convert(target_file)
                    fig_out = preprocess.generate_spectrogram_image(spec_out, sr_out)
                    
                    bar.progress(100)
                    time.sleep(0.5)
                    progress_text.empty()
                    bar.empty()
                    
                    st.success("Separation Complete!")
                    st.audio(target_file, format='audio/wav')
                    
                    with st.expander("👁️ View Spectrogram Analysis", expanded=True):
                        st.pyplot(fig_out, use_container_width=True)
                    
                    with open(target_file, "rb") as f:
                        st.download_button(label="⬇️ Download Stem", data=f, file_name=f"{display_label}.wav", mime="audio/wav", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    st.warning("⚠️ Troubleshooting: If asking for Drums/Bass, ensure internet is connected to download the 4-stem model.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with bottom_section:
        st.markdown("""<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 30px; margin-bottom: 10px;">Process a different file? Drop it below.</div>""", unsafe_allow_html=True)