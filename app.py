import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing pipeline
model = joblib.load("best_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

st.title("California Housing Price Predictor 🏡")

st.markdown("Fill in the feature values below:")

# Input form
MedInc = st.number_input("Median Income (in 10k USD)", min_value=0.0, max_value=20.0, value=3.0)
HouseAge = st.number_input("House Age", min_value=1, max_value=100, value=20)
AveRooms = st.number_input("Average Rooms", min_value=0.5, max_value=20.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.0)
Population = st.number_input("Population", min_value=1, max_value=5000, value=1000)
AveOccup = st.number_input("Average Occupants", min_value=0.5, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-118.0)

# Derived features
rooms_per_household = AveRooms / AveOccup
bedrooms_per_room = AveBedrms / AveRooms
population_per_household = Population / AveOccup

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
    }])
    
    # Transform and predict
    input_prepared = pipeline.transform(input_df)
    prediction = model.predict(input_prepared)[0]

    st.success(f"🏠 Predicted Median House Value: ${prediction * 100000:.2f}")

