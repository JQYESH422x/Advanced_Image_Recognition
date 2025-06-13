import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 
from PIL import Image
import cv2

# Load the model
model = load_model(r"C:\Users\lenovo\Documents\Projects\Facial_Detection\models\model.keras")
class_names = ['Akshay Kumar', 'Amitabh Bachchan', 'Prabhas', 'Vijay']

def preprocess_image(image, size=(128, 128)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit UI
st.set_page_config(page_title="Visionary: Image Recognition", layout="centered")
st.title("Advanced Image Recognition🌟")
st.write("Upload an image of a Hero, and the model will predict who it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # Show the result
    st.success(f"**Prediction:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}%")
else:
    st.warning("Please upload an image to get a prediction.")
