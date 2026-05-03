import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load trained model and scaler
# -------------------------------
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# -------------------------------
# App UI
# -------------------------------
st.title("🚦 Traffic Congestion Prediction System")

st.write("Enter basic traffic conditions:")

# User inputs (simplified)
temp = st.number_input("Temperature", 250.0, 320.0, 290.0)
rain = st.number_input("Rain (last hour)", 0.0, 50.0, 0.0)
snow = st.number_input("Snow (last hour)", 0.0, 10.0, 0.0)
clouds = st.number_input("Cloud coverage (%)", 0, 100, 50)
hour = st.number_input("Hour of day", 0, 23, 12)

# -------------------------------
# Prediction Logic (FIXED)
# -------------------------------
if st.button("Predict"):

    n_features = scaler.n_features_in_
    data = np.zeros((1, n_features))

    # safer assignment (only if index exists)
    if n_features >= 5:
        data[0, 0] = temp
        data[0, 1] = rain
        data[0, 2] = snow
        data[0, 3] = clouds
        data[0, 4] = hour

    data = scaler.transform(data)
    prediction = model.predict(data)

    st.success(f"Predicted Traffic Level: {prediction[0]}")
    st.info("Demo mode: simplified inputs mapped to full feature space.")