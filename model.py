from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Multiply
from tensorflow.keras.models import Model

def unet_model(input_shape=(512, 128, 1)):
    """
    Builds the U-Net Architecture.
    
    Args:
        input_shape: The size of the audio slice (Frequency, Time, Channels).
                     512 freq bins x 128 time steps x 1 channel (grayscale).
    """
    
    # 1. Input Layer
    inputs = Input(input_shape)

    # --- ENCODER (The "Contracting Path") ---
    # Compresses the image to find features
    
    # Block 1
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1) # Cut size in half
    
    # Block 2
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2) # Cut size in half again
    
    # Block 3
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # --- BOTTLENECK (The deepest thought) ---
    b = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    
    # --- DECODER (The "Expansive Path") ---
    # Resizes back up to generate the mask
    
    # Block 4 (Upscale + Skip Connection from Block 3)
    u1 = UpSampling2D((2, 2))(b)
    concat1 = Concatenate()([u1, c3]) # Skip Connection!
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    
    # Block 5 (Upscale + Skip Connection from Block 2)
    u2 = UpSampling2D((2, 2))(c4)
    concat2 = Concatenate()([u2, c2])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    
    # Block 6 (Upscale + Skip Connection from Block 1)
    u3 = UpSampling2D((2, 2))(c5)
    concat3 = Concatenate()([u3, c1])
    c6 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat3)
    
    # --- OUTPUT LAYER (The Mask) ---
    # Sigmoid activation outputs values between 0.0 and 1.0
    # 1.0 = Keep this sound. 0.0 = Delete this sound.
    mask = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    
    # Apply the mask to the input
    # (Original Audio * Mask) = Isolated Instrument
    outputs = Multiply()([inputs, mask])
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# --- TEST CODE (To check if it works) ---
if __name__ == "__main__":
    model = unet_model()
    model.summary()
    print("✅ Model created successfully!") 