import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('saved_model/cifar10_model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Page config
st.set_page_config(page_title="🔍 CIFAR-10 Classifier", layout="centered")

# 💄 Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }
        h1, h3 {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🖼️ CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Upload a 32x32 or larger image, and I’ll guess the object!</p>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((32, 32))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown(f"<h3 style='color: green;'> Prediction: <b>{predicted_class}</b></h3>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        Built by Rushalle Diya Sureshbabu Poornima
    </div>
""", unsafe_allow_html=True)
