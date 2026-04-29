import streamlit as st
import numpy as np
import pickle
import json

# 1. Load the trained model
with open('house_price_model.pickle', 'rb') as f:
    model = pickle.load(f)

# 2. Load the column names (Essential for the loc_index logic)
# Note: You should save your X.columns.tolist() as a JSON during training
# For this example, we assume 'columns.json' exists.
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # Assuming first 3 are sqft, bath, bhk

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location)
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# --- Streamlit UI ---
st.title("Real Estate Price Predictor")
st.write("Enter the details below to estimate the property price.")

# User Inputs
location = st.selectbox("Location", locations)
sqft = st.number_input("Total Square Feet", min_value=300, max_value=50000, value=1000)
bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    result = predict_price(location, sqft, bath, bhk)
    st.success(f"The estimated price is: ${result:,.2f}")