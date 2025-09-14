import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("tb_xray_classifier.h5", compile=False)

# Streamlit UI
st.set_page_config(page_title="TB X-ray Detector", layout="centered")
st.title("🩻 Tuberculosis Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Show result
    if prediction > 0.5:
        st.error(f"❌ Tuberculosis Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No Tuberculosis Detected (Confidence: {1 - prediction:.2f})")
