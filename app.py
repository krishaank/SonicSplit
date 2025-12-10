import streamlit as st
import time
import os
import shutil
from spleeter.separator import Separator
import preprocess  # Your local helper script for visualization

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SonicSplit AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (The "Pro" Look) ---
st.markdown("""
<style>
    /* 1. GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* 2. MAIN BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 18, 25) 0%, rgb(5, 5, 10) 90%);
        font-family: 'Inter', sans-serif;
    }

    /* 3. GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* 4. BUTTON STYLES */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        height: 55px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }

    /* 5. DEFAULT BIG UPLOADER STYLE (Hero State) */
    [data-testid='stFileUploader'] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 50px;
        text-align: center;
        transition: all 0.5s ease;
    }
    [data-testid='stFileUploader']:hover {
        border-color: #00d2ff;
        background-color: rgba(255, 255, 255, 0.06);
    }

    /* 6. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgb(18, 18, 25);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    h1, h2, h3 { color: white !important; font-family: 'Orbitron', sans-serif; }
    p, label, small { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTION: THE REAL AI ---
def split_audio(file_path):
    """
    Uses the Spleeter library to separate the audio.
    """
    # Initialize Spleeter with the '2stems' model (Vocals + Accompaniment)
    separator = Separator('spleeter:2stems')
    
    # Define output directory
    output_dir = "output_stems"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # Clean up previous run
    
    # Run Separation
    separator.separate_to_file(file_path, output_dir)
    
    # Locate the files (Spleeter creates a folder named after the filename)
    filename = os.path.splitext(os.path.basename(file_path))[0]
    vocals_path = os.path.join(output_dir, filename, "vocals.wav")
    music_path = os.path.join(output_dir, filename, "accompaniment.wav")
    
    return vocals_path, music_path

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3.5rem; filter: drop-shadow(0 0 10px rgba(0,210,255,0.5));">🎛️</div>
            <h2 style="margin: 0; font-size: 1.8rem; background: linear-gradient(to right, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">SonicSplit</h2>
            <div style="font-size: 0.8rem; letter-spacing: 2px; color: #666; margin-top: 5px;">AI STEM SEPARATOR</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Studio Settings")
    
    # Note: Spleeter 2stems only supports separating Vocals from Music.
    # To support Drums/Bass, you would need to change Separator('spleeter:2stems') to 'spleeter:4stems' above.
    mode = st.radio("Target Stem:", ["🎤 Vocals Only", "🎹 Karaoke (No Vocals)"])
    
    st.markdown("---")
    st.markdown("<small>Powered by TensorFlow & Spleeter</small>", unsafe_allow_html=True)

# --- MAIN LAYOUT LOGIC ---

# 1. Define Containers
top_section = st.container()
st.markdown("<br>", unsafe_allow_html=True) # Visual Spacer
bottom_section = st.container()

# 2. Render Uploader in Bottom Section
with bottom_section:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        uploaded_file = st.file_uploader("📂 Drop your track here", type=["mp3", "wav"])

# 3. Logic Controller
if uploaded_file is None:
    # --- SCENARIO A: NO FILE (Show Hero Text) ---
    with top_section:
        st.markdown("""
            <div style="text-align: center; padding-top: 40px; padding-bottom: 20px;">
                <h1 style="font-size: 4.5rem; margin-bottom: 10px; text-shadow: 0 0 30px rgba(0,210,255,0.3);">Unleash the Stems.</h1>
                <p style="font-size: 1.2rem; color: #888;">Upload your track below to isolate vocals instantly.</p>
            </div>
        """, unsafe_allow_html=True)

else:
    # --- SCENARIO B: FILE LOADED (Show Results) ---
    
    # A. Inject CSS to shrink the uploader to "Secondary" mode
    st.markdown("""
    <style>
        [data-testid='stFileUploader'] {
            padding: 15px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: transparent !important;
        }
        [data-testid='stFileUploader'] section > button {
            display: none; /* Hide 'Browse' button for cleaner look */
        }
    </style>
    """, unsafe_allow_html=True)
    
    with top_section:
        col1, col2 = st.columns(2, gap="large")

        # --- LEFT CARD: ORIGINAL MIX ---
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎧 Original Mix")
            st.markdown(f"**Track:** `{uploaded_file.name}`")
            
            # 1. Audio Player
            st.audio(uploaded_file, format='audio/wav')
            
            # 2. Spectrogram Visualization (Input)
            with st.spinner("Generating Spectrogram..."):
                # Save temp file for processing
                with open("temp_input.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Use preprocess.py to generate image
                spec, phase, sr = preprocess.load_and_convert("temp_input.wav")
                fig = preprocess.generate_spectrogram_image(spec, sr)
                st.pyplot(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # --- RIGHT CARD: AI OUTPUT ---
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### ✨ AI Output: <span style='color:#00d2ff'>{mode.split()[1]}</span>", unsafe_allow_html=True)
            
            st.write("") 
            process_btn = st.button("🚀 Process Audio", use_container_width=True)
            
            if process_btn:
                # --- AI PROCESSING ---
                progress_text = st.empty()
                bar = st.progress(0)
                
                try:
                    # Step 1: Loading
                    progress_text.text("🧠 Initializing Neural Network...")
                    bar.progress(10)
                    
                    # Step 2: Separation
                    progress_text.text("⚡ Separating Stems (This may take a moment)...")
                    vocals, music = split_audio("temp_input.wav")
                    bar.progress(60)
                    
                    # Step 3: Select Target
                    target_file = vocals if "Vocals" in mode else music
                    
                    # Step 4: Post-Processing Visualization
                    progress_text.text("📊 Generating Analysis...")
                    # Generate the "After" spectrogram to prove it worked
                    spec_out, phase_out, sr_out = preprocess.load_and_convert(target_file)
                    fig_out = preprocess.generate_spectrogram_image(spec_out, sr_out)
                    
                    bar.progress(100)
                    time.sleep(0.5)
                    progress_text.empty()
                    bar.empty()
                    
                    st.success("Separation Complete!")
                    
                    # --- SHOW RESULTS ---
                    st.audio(target_file, format='audio/wav')
                    
                    # Expandable Analysis Section
                    with st.expander("👁️ View Spectrogram Analysis (Before vs After)", expanded=True):
                        st.caption("Visual proof of separation:")
                        st.pyplot(fig_out, use_container_width=True)
                    
                    # Download Button
                    with open(target_file, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Stem",
                            data=f,
                            file_name=f"separated_stem.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    st.info("Tip: Ensure FFmpeg is installed and added to your system PATH.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Add hint text above the secondary uploader
    with bottom_section:
        st.markdown("""
            <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 30px; margin-bottom: 10px;">
                Process a different file? Drop it below.
            </div>
        """, unsafe_allow_html=True)