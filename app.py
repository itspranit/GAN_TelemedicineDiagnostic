import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from skimage.exposure import match_histograms

# Import our unified settings
from config import *
from networks.generator import build_generator

# --- PAGE SETUP ---
st.set_page_config(page_title="Telemedicine SRGAN", layout="wide")
st.title("Diagnostic SRGAN: Biological Texture Reconstruction")
st.write("Upload a highly compressed, low-resolution field image to reconstruct diagnostic details.")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_sr_model():
    # Pass the dynamic shape from config
    g = build_generator(lr_shape=(LR_SIZE, LR_SIZE, CHANNELS))
    
    weights_path = f"{WEIGHTS_DIR}/gen_100.weights.h5"
    if os.path.exists(weights_path):
        g.load_weights(weights_path)
    else:
        st.error(f"Weights not found at {weights_path}. Did you finish training?")
    return g

generator = load_sr_model()

# --- UI & INFERENCE ---
uploaded_file = st.file_uploader("Upload a field image (lesion, bite, etc.)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open and preprocess the low-resolution image
    image = Image.open(uploaded_file).convert('RGB')
    lr_image = image.resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)
    lr_array = np.array(lr_image) / 255.0
    lr_tensor = np.expand_dims(lr_array, axis=0)
    
    # 2. Feed it to the Generator
    with st.spinner("AI is reconstructing high-resolution features..."):
        preds = generator(lr_tensor, training=False)
        
    # 3. Post-process the output
    # Remove the batch dimension and scale to 255
    sr_array = np.clip(preds[0].numpy(), 0.0, 1.0)
    sr_array = (sr_array * 255.0).astype(np.uint8)
    
    # --- COLOR CORRECTION ---
    # Resize the original image to HR size to use strictly as a color/lighting reference
    color_reference = np.array(image.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC))
    
    # Force the GAN output to adopt the exact brightness and color palette of the original
    corrected_sr_array = match_histograms(sr_array, color_reference, channel_axis=-1)
    
    # THE FIX: Clip the array to prevent integer overflow BEFORE converting to uint8
    corrected_sr_array = np.clip(corrected_sr_array, 0, 255)
    
    sr_image = Image.fromarray(corrected_sr_array.astype(np.uint8))
    
    # 4. Display Side-by-Side
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Low-Res Input ({LR_SIZE}x{LR_SIZE})")
        st.image(lr_image, use_container_width=True)
        
    with col2:
        st.subheader(f"SRGAN Output ({HR_SIZE}x{HR_SIZE})")
        st.image(sr_image, use_container_width=True)