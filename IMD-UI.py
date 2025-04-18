import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import time

#----Config------

IMG_H = 48
IMG_W = 48
IMG_CH = 3

CLASS_NAME = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']

MODEL_PATH_1 = 'best_emotion_detection_model.keras'
MODEL_NAME_1 = 'Best Model'
MODEL_PATH_2 = 'emotion_detection_model.keras'
MODEL_NAME_2 = "Final Epoch Model"

#----Helper functions------

def check_model_paths():
    paths_ok = True
    if not os.path.exists(MODEL_PATH_1):
        st.error(f"Error: Model file not found at {MODEL_PATH_1} for {MODEL_NAME_1}")
        paths_ok = False
    if not os.path.exists(MODEL_PATH_2):
        st.error(f"Error: Model file not found at {MODEL_PATH_2} FOR {MODEL_NAME_2}")
        paths_ok = False
    return paths_ok

#Cache the model loading
@st.cache_resource
def load_comparison_models(path1, path2):
    model1, model2 = None, None
    try:
        model1 = load_model(path1)
        print(f"Model 1 ({path1}) loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Model 1 ({MODEL_NAME_1}) from {path1}: {e}")
        print(f"Error loading Model 1 from {path1}: {e}")
    try:
        model2 = load_model(path2)
        print(f"Model 2 ({path2}) loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Model 2 ({MODEL_NAME_2}) from {path2}: {e}")
        print(f"Error loading Model 2 from {path2}: {e}")
    return model1, model2

# Preprocessing function (takes NumPy array)
def preprocess_image_array(img_array, target_size=(IMG_H, IMG_W)):
    try:
        if IMG_CH == 3 and img_array.shape[-1] == 3:
            img_processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif IMG_CH == 1:
            img_processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            img_processed = img_array

        img_resized = cv2.resize(img_processed, target_size)

        if IMG_CH == 1:
            img_reshaped = img_resized.reshape(target_size[0], target_size[1], 1)
        else:
            img_reshaped = img_resized

        img_batch = np.expand_dims(img_reshaped, axis=0)
        img_normalized = img_batch / 255.0
        return img_normalized

    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        print(f"Error during image preprocessing: {e}")
        return None

# Prediction function for comparison
def predict_emotions_comparison(image_tensor, model1, model2, CLASS_NAME):
    results = {}
    if model1 and image_tensor is not None:
        try:
            pred1 = model1.predict(image_tensor)[0]
            results[MODEL_NAME_1] = {CLASS_NAME[i]: pred1[i] for i in range(len(CLASS_NAME))}
        except Exception as e:
            st.warning(f"Could not get prediction from {MODEL_NAME_1}: {e}")
            results[MODEL_NAME_1] = None
    else:
        results[MODEL_NAME_1] = None

    if model2 and image_tensor is not None:
        try:
            pred2 = model2.predict(image_tensor)[0]
            results[MODEL_NAME_2] = {CLASS_NAME[i]: pred2[i] for i in range(len(CLASS_NAME))}
        except Exception as e:
            st.warning(f"Could not get prediction from {MODEL_NAME_2}: {e}")
            results[MODEL_NAME_2] = None
    else:
        results[MODEL_NAME_2] = None

    return results

# Function to display results side-by-side
def display_results(predictions):
    if not predictions or (predictions.get(MODEL_NAME_1) is None and predictions.get(MODEL_NAME_2) is None):
        st.write("No predictions available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Predictions ({MODEL_NAME_1})")
        preds1 = predictions.get(MODEL_NAME_1)
        if preds1:
            sorted_preds1 = dict(sorted(preds1.items(), key=lambda item: item[1], reverse=True))
            for emotion, probability in sorted_preds1.items():
                st.write(f"- {emotion}: {probability*100:.2f}%")
        else:
            st.write("N/A")

    with col2:
        st.subheader(f"Predictions ({MODEL_NAME_2})")
        preds2 = predictions.get(MODEL_NAME_2)
        if preds2:
            sorted_preds2 = dict(sorted(preds2.items(), key=lambda item: item[1], reverse=True))
            for emotion, probability in sorted_preds2.items():
                st.write(f"- {emotion}: {probability*100:.2f}%")
        else:
            st.write("N/A")

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Facial Emotion Recognition: Model Comparison")

if not check_model_paths():
    st.stop()

model1, model2 = load_comparison_models(MODEL_PATH_1, MODEL_PATH_2)

if model1 is None and model2 is None:
    st.error("Neither model could be loaded. Cannot proceed.")
    st.stop()
elif model1 is None:
    st.warning(f"Could not load {MODEL_NAME_1}. Only {MODEL_NAME_2} will be used.")
elif model2 is None:
    st.warning(f"Could not load {MODEL_NAME_2}. Only {MODEL_NAME_1} will be used.")
else:
    st.success(f"Models loaded: '{MODEL_NAME_1}' and '{MODEL_NAME_2}'")

# Input Selection
st.sidebar.header("Input Source")
input_source = st.sidebar.radio("Choose input type:", ("Upload Image", "Take a Photo"))

# --- Main Area ---
if input_source == "Upload Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        try:
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption='Uploaded Image', use_column_width=False, width=300)
        except Exception as e:
            st.error(f"Could not display image: {e}")

        st.write("Processing...")
        with st.spinner('Analyzing the image...'):
            nparr = np.frombuffer(image_bytes, np.uint8)
            if IMG_CH == 1:
                img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_np is None:
                st.error("Failed to decode uploaded image.")
            else:
                preprocessed_image = preprocess_image_array(img_np)
                if preprocessed_image is not None:
                    predictions = predict_emotions_comparison(preprocessed_image, model1, model2, CLASS_NAME)
                    st.subheader("Comparison Results")
                    display_results(predictions)
                else:
                    st.error("Image preprocessing failed.")
    else:
        st.info("Upload an image file (jpg, jpeg, png) to see predictions.")

elif input_source == "Take a Photo":
    st.header("Take a Photo")
    captured_image = st.camera_input("Capture an image")

    if captured_image is not None:
        # Process the captured image
        image_bytes = captured_image.getvalue()
        try:
            display_image = Image.open(captured_image)
            st.image(display_image, caption='Captured Image', use_column_width=False, width=300)
        except Exception as e:
            st.error(f"Could not display image: {e}")

        st.write("Processing...")
        with st.spinner('Analyzing the image...'):
            nparr = np.frombuffer(image_bytes, np.uint8)
            if IMG_CH == 1:
                img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_np is None:
                st.error("Failed to decode captured image.")
            else:
                preprocessed_image = preprocess_image_array(img_np)
                if preprocessed_image is not None:
                    predictions = predict_emotions_comparison(preprocessed_image, model1, model2, CLASS_NAME)
                    st.subheader("Comparison Results")
                    display_results(predictions)
                else:
                    st.error("Image preprocessing failed.")

# --- Footer Info ---
st.sidebar.header("About")
st.sidebar.info(
    f"**Models Compared:**\n"
    f"1. {MODEL_NAME_1} (`{os.path.basename(MODEL_PATH_1)}`)\n"
    f"2. {MODEL_NAME_2} (`{os.path.basename(MODEL_PATH_2)}`)\n\n"
    f"**Input Format:** `{IMG_W}x{IMG_H}x{IMG_CH}`\n\n"
    f"**Emotions:** {', '.join(CLASS_NAME)}\n\n"
    f"**Current Time (Kathmandu):** {time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}\n"
)