import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model


import urllib.request

url = "https://drive.google.com/uc?id=1ZRLGREdsFMl3w-gB1hLN_VjLJ2HdLGHR&export=download"
filename = "house_price_model.pkl"
urllib.request.urlretrieve(url, filename)
model = joblib.load(filename)



st.title("🏡 California House Price Predictor")

# User input form
st.header("Enter house features:")

median_income = st.slider("Median Income (10k USD)", 0.0, 15.0, 3.0)
housing_median_age = st.slider("Housing Median Age", 1, 60, 20)
total_rooms = st.number_input("Total Rooms", 1, 50000, 2000)
total_bedrooms = st.number_input("Total Bedrooms", 1, 10000, 500)
population = st.number_input("Population", 1, 30000, 1000)
households = st.number_input("Households", 1, 10000, 400)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

if st.button("Predict Price"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "MedInc": [median_income],
        "HouseAge": [housing_median_age],
        "AveRooms": [total_rooms],
        "AveBedrms": [total_bedrooms],
        "Population": [population],
        "AveOccup": [households],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    # Scale input (optional – match training)
    # input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Value: ${prediction * 100000:.2f}")