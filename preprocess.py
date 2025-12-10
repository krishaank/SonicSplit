import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# SR = Sampling Rate. Standard CD quality is 44,100Hz.
# We use 22,050Hz to save RAM (it's half the size but still sounds good).
SR = 22050 

# FFT_SIZE = How "detailed" the snapshot is. 2048 is standard for music.
FFT_SIZE = 2048

# HOP_LENGTH = How many samples we slide over. 512 is standard.
HOP_LENGTH = 512

def load_and_convert(file_path):
    """
    1. Loads audio file.
    2. Converts it to a Spectrogram (The 'Image').
    3. Returns the Log-Spectrogram (for the AI) and Phase (for reconstructing audio later).
    """
    
    # 1. Load Audio
    # duration=30 loads only the first 30s to keep it fast for testing
    y, sr = librosa.load(file_path, sr=SR, duration=30) 
    
    # 2. Perform STFT (Short-Time Fourier Transform)
    # This is the math that breaks sound into frequencies
    spectrogram_complex = librosa.stft(y, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
    
    # 3. Separate Magnitude (Loudness) and Phase (Timing)
    # We only show Magnitude to the AI. We keep Phase to rebuild the audio later.
    magnitude, phase = librosa.magphase(spectrogram_complex)
    
    # 4. Convert to Decibels (Log Scale)
    # Because valid pixel values are 0-255, but sound is exponential.
    log_spectrogram = librosa.amplitude_to_db(magnitude)
    
    return log_spectrogram, phase, sr

def generate_spectrogram_image(log_spectrogram, sr):
    """
    Generates a Matplotlib Figure to display in Streamlit.
    """
    plt.figure(figsize=(10, 4))
    
    # Draw the heatmap
    librosa.display.specshow(
        log_spectrogram, 
        sr=sr, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='log', # Log scale for Y-axis helps visualize low bass notes better
        cmap='magma' # 'magma' is a cool dark-mode friendly color map
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Audio Spectrogram (What the AI Sees)')
    plt.tight_layout()
    
    # Return the plot object so Streamlit can draw it
    return plt.gcf()