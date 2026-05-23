import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Mushroom Yield Predictor",
    page_icon="🍄",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL + SCALER
# ---------------------------------------------------

model = joblib.load("src/random_forest_model.pkl")
scaler = joblib.load("src/scaler.pkl")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🍄 AI Mushroom Yield Predictor")

st.markdown("""
This application predicts mushroom yield based on environmental conditions.

Adjust the sliders below to simulate different growing conditions.
""")

st.divider()

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("Environmental Conditions")

temperature = st.slider(
    "🌡 Temperature (°C)",
    min_value=10.0,
    max_value=40.0,
    value=25.0
)

humidity = st.slider(
    "💧 Humidity (%)",
    min_value=30.0,
    max_value=100.0,
    value=70.0
)

co2 = st.slider(
    "🫧 CO2 Level (ppm)",
    min_value=200.0,
    max_value=2000.0,
    value=800.0
)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Yield"):

    # Create dataframe with feature names
    input_data = pd.DataFrame({
        "temperature_c": [temperature],
        "humidity_percent": [humidity],
        "co2_ppm": [co2]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict yield
    prediction = model.predict(input_scaled)

    # ---------------------------------------------------
    # OUTPUT SECTION
    # ---------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    st.success(f"🍄 Estimated Mushroom Yield: {float(prediction[0]):.2f} kg")

    # Simple interpretation
    if prediction[0] > 550:
        st.info("Excellent environmental conditions detected.")
    elif prediction[0] > 500:
        st.info("Moderate growing conditions detected.")
    else:
        st.warning("Yield may be lower under current conditions.")