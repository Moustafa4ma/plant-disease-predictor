import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os

# Page config
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱")

# Model details
MODEL_PATH = "model.h5"
# Replace with your Hugging Face repository and filename
REPO_ID = "moustafa4ma/plant-disease-predictor"  # e.g., "john/plant-disease-model"
FILENAME = "model.h5"

# Function to download model from Hugging Face if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face... please wait ⏳"):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=".",
                local_dir_use_symlinks=False
            )

# Load model once and cache it
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

# Class labels (adjust to match your training)
CLASS_NAMES = ["healthy", "powdery", "rust"]

# UI
st.title("🌱 Plant Disease Classifier")
st.write("Upload a plant image to get disease prediction")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Predict
    if st.button("Predict"):
        model = load_model()
        
        # Preprocess image (resize & normalize)
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Get prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Display result
        st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")