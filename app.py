import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. UI Configuration
st.set_page_config(page_title="Pokemon Classifier", layout="centered")
st.title("🐾 Pokemon Classifier (HW6)")
st.write("Upload a Pokemon image to see the model prediction.")

# 2. Select Model from 4 experiments
model_option = st.selectbox(
    'Select Model:',
    ('Experiment 1 (MobileNetV2-Frozen)', 
     'Experiment 2 (MobileNetV2-FineTune)', 
     'Experiment 3 (ResNet50-Frozen)', 
     'Experiment 4 (ResNet50-FineTune)')
)

model_mapping = {
    'Experiment 1 (MobileNetV2-Frozen)': 'models/pokemon_exp_1.h5',
    'Experiment 2 (MobileNetV2-FineTune)': 'models/pokemon_exp_2.h5',
    'Experiment 3 (ResNet50-Frozen)': 'models/pokemon_exp_3.h5',
    'Experiment 4 (ResNet50-FineTune)': 'models/pokemon_exp_4.h5'
}

# 3. Load model with caching
@st.cache_resource
def load_pokemon_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

model = load_pokemon_model(model_mapping[model_option])

# 4. Get class labels from directory names
if os.path.exists("data/pokemon"):
    class_names = sorted(os.listdir("data/pokemon"))
else:
    class_names = [f"Class {i}" for i in range(150)]

# 5. Image upload and prediction logic
uploaded_file = st.file_uploader("Choose a Pokemon image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model is not None:
        st.write("🔍 Analyzing...")
        
        # Preprocessing input image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0 
        
        # Model inference
        predictions = model.predict(img_array)
        score = predictions[0]
        
        # Display Top-5 predictions
        top_5_indices = np.argsort(score)[-5:][::-1]
        
        st.subheader("Top-5 Predictions:")
        for i, idx in enumerate(top_5_indices):
            st.write(f"{i+1}. **{class_names[idx]}** ({score[idx]*100:.2f}%)")
    else:
        st.error("Model file not found. Please run train.py first.")