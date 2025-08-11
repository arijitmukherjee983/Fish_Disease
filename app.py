import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model('fish_disease_improved.h5')

# Define your classes in the same order as your model output
class_names = [
    "Aeromoniasis",
    "Bacterial gill disease",
    "Bacterial Red disease",
    "Saprolegniasis (Fungal disease)",
    "Healthy Fish",
    "Parasitic disease",
    "White tail disease (Viral)"
]

st.title("Fish Disease Detection from Skin Image")

uploaded_file = st.file_uploader("Upload a fish skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image for your model
    # Adjust target_size based on what your model expects, e.g., (224,224)
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize if your model was trained with normalized images
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict button
    if st.button("Predict Disease"):
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100
        
        st.write(f"### Predicted : {predicted_class}")
        # st.write(f"Confidence: {confidence:.2f}%")
