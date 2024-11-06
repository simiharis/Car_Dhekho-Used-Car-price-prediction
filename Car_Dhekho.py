import streamlit as st
import pandas as pd
from datetime import datetime
import pickle

# Load the dataset to extract unique values 
car_data = pd.read_csv('all_city_cardata_outliers.csv')  

# Extract unique values from the dataframe for dropdown options
brands = car_data['Brand'].unique()
models = car_data.groupby('Brand')['model'].unique().to_dict()  # Dictionary of models for each brand
body_types = car_data['Body type'].unique()
insurance_options = car_data['Insurance Validity'].unique()
cities = car_data['city'].unique()

# Load your pre-trained model pipeline
with open('rf_model_pipeline.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Function to predict car price with car age calculation
def predict_car_price_with_age(input_data):
    current_year = datetime.now().year
    input_data['Car Age'] = current_year - input_data['modelYear']
    input_df = pd.DataFrame([input_data])
    predicted_price = model_pipeline.predict(input_df)
    return predicted_price[0], input_data['Car Age']

# Set page layout and add a header image
st.set_page_config(layout="wide")  # Set layout to wide


# Main title and description
st.title("Car Price Prediction App")
st.image("https://cdni.autocarindia.com/utils/ImageResizer.ashx?n=https://cms.haymarketindia.net/model/uploads/modelimages/Mini-3-Door-110720241620.png",width=300)  # Replace with your image file path

st.markdown("### Enter the details to predict its price")

# Sidebar for user inputs
with st.sidebar:
    st.header("Car Details")

    # User inputs in the sidebar
    brand = st.selectbox("Select Brand", brands)
    model = st.selectbox("Select Model", models.get(brand, []))  # Model dropdown updates based on selected brand
    body_type = st.selectbox("Select Body Type", body_types)
    owner = st.selectbox("Owner Type", [1, 2, 3, 4])
    model_year = st.number_input("Model Year", min_value=1990, max_value=2024, value=2015)
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, value=12000)
    mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=100.0, value=23.1)
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=998)
    fuel_type = st.selectbox("Fuel Type", car_data['Fuel Type'].unique())
    transmission = st.selectbox("Transmission", car_data['Transmission'].unique())
    insurance_validity = st.selectbox("Insurance Validity", insurance_options)
    city = st.selectbox("City", cities)

# Organize inputs into a dictionary
input_data = {
    'Brand': brand,
    'Model': model,
    'Body type': body_type,
    'Owner': owner,
    'modelYear': model_year,
    'Kms Driven': kms_driven,
    'Mileage': mileage,
    'Engine': engine,
    'Fuel Type': fuel_type,
    'Transmission': transmission,
    'Insurance Validity': insurance_validity,
    'city': city
}

# Center column for displaying predictions
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    
    # Predict button and display results
    if st.button("Predict Price"):
        predicted_price, car_age = predict_car_price_with_age(input_data)
        st.write(f"**Predicted Price:** ₹{predicted_price:,.2f}")
        st.write(f"**Car Age:** {car_age} years")

# Footer styling
st.markdown("---")
st.markdown("© 2024 Car Price Prediction. All rights reserved.")

# Additional styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
