import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

model = tf.keras.models.load_model('trained_model.keras')

label_map = {
    'rice': 0,
    'maize': 1,
    'chickpea': 2,
    'kidneybeans': 3,
    'pigeonpeas': 4,
    'mothbeans': 5,
    'mungbean': 6,
    'blackgram': 7,
    'lentil': 8,
    'pomegranate': 9,
    'banana': 10,
    'mango': 11,
    'grapes': 12,
    'watermelon': 13,
    'muskmelon': 14,
    'apple': 15,
    'orange': 16,
    'papaya': 17,
    'coconut': 18,
    'cotton': 19,
    'jute': 20,
    'coffee': 21,
}

reverse_label_map = {v: k for k, v in label_map.items()}

st.title("Plant Classification Model")
st.write("Enter the features of the plant to predict its type:")

n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0)
p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0)
k = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0)

input_features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

if st.button("Predict Plant Type"):
    prediction = model.predict(input_features)
    
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    predicted_label = reverse_label_map[predicted_label_index]
    
    st.write(f"The predicted plant type is: **{predicted_label}**")
