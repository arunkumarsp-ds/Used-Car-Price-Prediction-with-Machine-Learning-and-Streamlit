import streamlit as st
import pandas as pd
import joblib

# Loading the trained model and LabelEncoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("Used Car Price Prediction")

# Load dataset
cars_df = pd.read_csv("updated_cars_data.csv")

# User input form
st.header("Enter Car Details")

# Defining user input fields
car_name = st.selectbox('Car Name', cars_df['car_name'].unique(), index=list(cars_df['car_name'].unique()).index('Maruti Wagon R'))
vehicle_age = st.number_input('Vehicle Age', min_value=0, max_value=30, step=1, value=9)
km_driven = st.number_input('Kilometers Driven', min_value=100, max_value=4000000, step=100, value=58000)
seller_type = st.selectbox('Seller Type', cars_df['seller_type'].unique(), index=list(cars_df['seller_type'].unique()).index('Dealer'))
fuel_type = st.selectbox('Fuel Type', cars_df['fuel_type'].unique(), index=list(cars_df['fuel_type'].unique()).index('Petrol'))
transmission_type = st.selectbox('Transmission Type', cars_df['transmission_type'].unique(), index=list(cars_df['transmission_type'].unique()).index('Manual'))
mileage = st.number_input('Mileage (kmpl)', min_value=4.0, max_value=35.0, step=0.1, value=18.9)
engine = st.number_input('Engine Capacity (cc)', min_value=750, max_value=6600, step=50, value=998)
seats = st.number_input('Number of Seats', min_value=2, max_value=9, step=1, value=5)

# Creating input dataframe
input_data = pd.DataFrame({
    "car_name": [car_name],
    "vehicle_age": [vehicle_age],
    "km_driven": [km_driven],
    "seller_type": [seller_type],
    "fuel_type": [fuel_type],
    "transmission_type": [transmission_type],
    "mileage": [mileage],
    "engine": [engine],
    "seats": [seats]
})

# Label Encoder
input_data["car_name"] = label_encoder.transform(input_data["car_name"])


# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.subheader(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")