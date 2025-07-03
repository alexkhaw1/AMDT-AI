import streamlit as st
import cv2
import joblib
from PIL import Image
import numpy as np

model = joblib.load("trained_model_KNN.pkl")

st.title ('Dell Global Business Center')
st.text('Crack Detector Using KNN')

# Preprocessing and feature extraction
def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges
 
def extract_features(img):
    resized = cv2.resize(img, (64, 64))
    return resized.flatten().reshape(1, -1)

uploaded_file = st.file_uploader("Upload on image", type=["png","jpeg","jpg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_array = np.array(image)
    
    st.image(image, caption="You have uplaoded this image")
    
    processed_image = preprocess_image(image_array)
    features = extract_features(processed_image)

    prediction = model.predict(features)[0]
    label = 'Positive' if prediction == 1 else 'Negative'
    st.success(f"''Prediction:'' {label}")





#st.image('Screenshot 2024-07-16 094056.jpg')
st.button('Welcome')

st.date_input("Trasaction Date")
st.radio("your department name:",['DAA','NPI','GMOT'])
#st.camera_input("Case Reported")
