import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the pre-trained model

# Loading the model
model = joblib.load("model.pkl")

# App title
st.title("Used Car Price Prediction")

# User input form
st.header("Enter Car Details")

cars_df = pd.read_csv(r"updated_cars_data.csv")

# Define the input fields
brand = st.selectbox('brand', cars_df['brand'].unique())
filtered_names = cars_df[cars_df["name"].str.startswith(brand)]["name"].unique()
name = st.selectbox('name', filtered_names)
km_driven = st.number_input('km_driven', min_value=0,max_value=500000,step=100)
transmission = st.selectbox('transmission', cars_df['transmission'].unique())
year = st.selectbox('year', sorted(cars_df['year'].unique(), reverse=True))
seller_type = st.selectbox('seller_type', cars_df['seller_type'].unique())
owner = st.selectbox('owner', cars_df['owner'].unique())
fuel = st.selectbox('fuel', cars_df['fuel'].unique())

# features = ['name', 'year',  'km_driven', 'fuel', 'seller_type',
#              'transmission', 'owner', 'brand']


input_data = pd.DataFrame({
    "name": [name],
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type":[seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "brand": [brand]   
})

#  encoding the categorical features
input_train = cars_df[["year","km_driven","name","fuel", "seller_type",
                       "transmission","owner" ,"brand"]]

# Encoding the training data for matching the index with test data
input_train_encoded = pd.get_dummies(input_train)

# Extracting the feature names for matching the index with test data
feature_names = input_train_encoded.columns.tolist()

input_data_encoded = pd.get_dummies(input_data)

input_data_encoded = input_data_encoded.reindex(columns=feature_names, fill_value=0)


# Prediction
if st.button("Predict Price"):
    # Make prediction using the pipeline model 
    prediction = model.predict(input_data_encoded)
    
    # Display the result
    st.subheader(f"Predicted Selling Price: ₹{prediction[0]:,.2f}")

