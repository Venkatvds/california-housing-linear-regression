# ================= STEP 1: Imports & Page Setup =================
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STEP 2: Model Loading =================
# Use @st.cache_resource so the model is only loaded once, improving performance
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: Model file '{model_path}' not found. Please train and save the model first.")
        return None

# Load the model
model = load_model("linear_regression_model.pkl")

# ================= STEP 3: Sidebar & Extra Polish =================
# Add a professional sidebar with information about the app
st.sidebar.title("ℹ️ About the App")
st.sidebar.info(
    "This web application predicts the median house value for a given block group "
    "in California based on demographic and geographical data.\n\n"
    "**Model used:** Linear Regression\n"
    "**Dataset:** California Housing Dataset"
)
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developer:** Senior ML + UI Engineer")
st.sidebar.markdown("🚀 **Built with Streamlit**")

# ================= STEP 4: Main UI Setup =================
# Main title and description
st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Enter the characteristics of the neighborhood (block group) below to estimate the median house value. "
    "Adjust the parameters using the input fields."
)
st.markdown("---")

# ================= STEP 5: Input UI & Layout Design =================
st.subheader("📊 Neighborhood Characteristics")

# Use columns to create a clean, grid-like layout for inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    med_inc = st.number_input(
        "Median Income ($10k)", 
        min_value=0.0, max_value=15.0, value=3.5, step=0.1,
        help="Median income in block group (e.g., 3.5 = $35,000)"
    )
    population = st.number_input(
        "Population", 
        min_value=1.0, max_value=40000.0, value=1400.0, step=100.0,
        help="Block group population"
    )

with col2:
    house_age = st.number_input(
        "House Age (Years)", 
        min_value=1.0, max_value=100.0, value=28.0, step=1.0,
        help="Median house age in block group"
    )
    ave_occup = st.number_input(
        "Average Occupancy", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Average number of household members"
    )

with col3:
    ave_rooms = st.number_input(
        "Average Rooms", 
        min_value=1.0, max_value=20.0, value=5.0, step=0.5,
        help="Average number of rooms per household"
    )
    latitude = st.number_input(
        "Latitude", 
        min_value=32.0, max_value=42.0, value=34.0, step=0.1,
        help="Block group latitude"
    )

with col4:
    ave_bedrms = st.number_input(
        "Average Bedrooms", 
        min_value=0.5, max_value=5.0, value=1.0, step=0.1,
        help="Average number of bedrooms per household"
    )
    longitude = st.number_input(
        "Longitude", 
        min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1,
        help="Block group longitude"
    )

st.markdown("---")

# ================= STEP 6: Prediction Logic & Output Section =================
# Center the predict button and result using columns
col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])

with col_center:
    # Prediction Button
    if st.button("🔮 Predict Price", use_container_width=True):
        if model is not None:
            # Prepare the input data as a pandas DataFrame to match training features
            input_features = pd.DataFrame([{
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrms,
                "Population": population,
                "AveOccup": ave_occup,
                "Latitude": latitude,
                "Longitude": longitude
            }])
            
            # Predict using the loaded model
            prediction = model.predict(input_features)[0]
            
            # The target variable is in units of $100,000
            # We display the raw prediction formatted to 3 decimals, and the estimated dollar amount
            st.success(f"### Target Value: {prediction:.3f}")
            
            estimated_price = prediction * 100000
            st.info(f"💵 **Estimated Market Value:** ${estimated_price:,.2f}")
            st.caption("✨ Predicted Median House Value based on historical California data.")
            
            # Fun success balloon animation
            st.balloons()
        else:
            st.error("Model is not loaded. Cannot make predictions.")

# ================= STEP 7: Footer =================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: grey; font-size: small;'>"
    "Built with Streamlit 🎈</div>", 
    unsafe_allow_html=True
)