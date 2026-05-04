import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. UI Configuration
st.set_page_config(page_title="PokéVision Classifier", layout="centered")
st.title("🐾 PokéVision: Pokemon Classifier")
st.markdown("---")

# 2. Sidebar for Settings
st.sidebar.header("Settings")
# ဒီနေရာမှာ Experiment 2 ကို အဓိကထားပြီး load လုပ်မှာဖြစ်လို့ Model ရွေးတာကို ခဏထားခဲ့ပါမယ်
st.sidebar.info("Using Experiment 2 (MobileNetV2 Fine-Tuned) - Accuracy: 75.48%")

# 3. Model Loading (Optimized for Safari/Mac)
@st.cache_resource
def load_best_model():
    # သူဌေးရဲ့ terminal results အရ accuracy အကောင်းဆုံးဖြစ်တဲ့ exp_2 ကို သုံးပါမယ်
    model_path = 'models/pokemon_exp_2.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

# Load the model
with st.spinner('Loading Model... please wait.'):
    model = load_best_model()

# 4. Get Class Labels
# Pokemon 150 မျိုးရဲ့ နာမည်တွေကို data folder ထဲကနေ ဆွဲယူပါတယ်
DATA_PATH = "data/pokemon"
if os.path.exists(DATA_PATH):
    class_names = sorted(os.listdir(DATA_PATH))
else:
    # အကယ်၍ data path မတွေ့ရင် default list သုံးပါမယ်
    class_names = [f"Pokemon_{i}" for i in range(150)]

# 5. User Interface
st.write("Ready to classify! Upload a Pokemon image below.")
uploaded_file = st.file_uploader("Upload a Pokemon image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model is not None:
        # Preprocessing
        img = image.convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0 
        
        # Prediction
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            score = predictions[0]
            # Top-5 အများဆုံး ရလဒ်တွေကို ယူပါတယ်
            top_5_indices = np.argsort(score)[-5:][::-1]
            
        st.subheader("📊 Top-5 Predictions:")
        
        for i, idx in enumerate(top_5_indices):
            confidence = score[idx] * 100
            
            # Error မတက်အောင် float() ပြောင်းထားပါတယ်
            confidence_val = float(score[idx])
            
            # Progress bar နဲ့ နာမည်ကို ပြပါတယ်
            st.write(f"**{class_names[idx]}**: {confidence:.2f}%")
            st.progress(confidence_val)
    else:
        st.error("Model not found! Please check if 'models/pokemon_exp_2.h5' exists.")