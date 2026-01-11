import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Car Price Predictor")

st.title("🚗 Car Valuation Dashboard")
st.markdown("---")

# Layout: Three main columns for the sections
col1, col2, col3 = st.columns(3, gap="large")

# --- SECTION 1: DESIGN & SAFETY ---
color_list = [
    'Black', 'Blue', 'Brown', 'Carnelian red', 'Golden', 
    'Green', 'Grey', 'Orange', 'Pink', 'Purple', 
    'Red', 'Silver', 'Sky blue', 'White', 'Yellow'
]
category_list = [
    "Coupe", "Goods wagon", "Hatchback", "Jeep", "Limousine", 
    "Microbus", "Minivan", "Pickup", "Sedan", "Universal"
]
with col1:
    with st.container(border=True):
        st.subheader("🎨 Design & Safety")
        leather = 1 if st.radio("Leather Interior", ["Yes", "No"], horizontal=True) == "Yes" else 0
        wheel = 1 if st.radio("Steering", ["Left wheel", "Right-hand drive"], horizontal=True) == "Left wheel" else 0
        airbags = st.slider("Airbags", 0, 16, 6)
        
        color_opt = st.selectbox(
            "Exterior Color", 
            color_list
        )
        
        category_opt = st.selectbox(
            "🚘 Select Vehicle Style",
            options=category_list,
            help="Slide to select the body category of the vehicle."
        )
    # Create a dictionary where the selected color is 1 and all others are 0
    color_data = {f"Color_{c}": (1 if color_opt == c else 0) for c in color_list}

    # Same for vehicle category
    category_data = {f"Category_{cat}": (1 if category_opt == cat else 0) for cat in category_list}


# --- SECTION 2: PERFORMANCE ---
with col2:
    with st.container(border=True):
        st.subheader("⚙️ Performance")
        engine_vol = st.number_input("Engine Volume (L)", 0.0, 10.0, 2.0, 0.1)
        cylinders = st.slider("Cylinders", 1, 16, 4)
        
        engine_type = 1 if st.radio(
            "Engine Type", ["Turbo", "Non-Turbo"], 
            horizontal=True
        ) == "Turbo" else 0
        
        drive_opt = st.radio("Drive Wheels", ["4WD", "FWD", "RWD"], horizontal=True)

        gear_box_options = ["Automatic", "Manual", "Tiptronic", "Variator"]
        gearbox_opt = st.selectbox("Gearbox Type", gear_box_options)
        
# --- SECTION 3: CORE SPECS ---
with col3:
    with st.container(border=True):
        st.subheader("📅 Core History")
        year = st.slider("Manufacturing Year", 1950, 2026, 2018)
        mileage = st.number_input(
            "Current Mileage", 
            min_value=0, max_value=1000000, value=50000, step=500,
            help="Total distance the car has traveled."
        )
        fuel_type = st.selectbox(
            "Fuel Type", 
            options=["Diesel", "Hybrid", "Hydrogen", "LPG", "Petrol", "Plug-in Hybrid"]
        )

# --- PREPARE DATA FOR MODEL ---
# Initialize the data dictionary with all required features
data = {
    "year": year,
    "Leather interior": leather,
    "Engine volume": engine_vol,
    "Mileage": mileage,
    "Cylinders": cylinders,
    "Wheel": wheel,
    "Airbags": airbags,
    "Engine Type": engine_type,
    "Gear box type_Automatic": int(gearbox_opt == "Automatic"),
    "Gear box type_Manual": int(gearbox_opt == "Manual"),
    "Gear box type_Tiptronic": int(gearbox_opt == "Tiptronic"),
    "Gear box type_Variator": int(gearbox_opt == "Variator"),
    "Drive wheels_4WD": int(drive_opt == "4WD"),
    "Drive wheels_FWD": int(drive_opt == "FWD"),
    "Drive wheels_RWD": int(drive_opt == "RWD"),
    "Fuel_Diesel": int(fuel_type == "Diesel"),
    "Fuel_Hybrid": int(fuel_type == "Hybrid"),
    "Fuel_Hydrogen": int(fuel_type == "Hydrogen"),
    "Fuel_LPG": int(fuel_type == "LPG"),
    "Fuel_Petrol": int(fuel_type == "Petrol"),
    "Fuel_Plug-in Hybrid": int(fuel_type == "Plug-in Hybrid"),
    **color_data,
    **category_data
}

# --- PREPARE DATA FOR MODEL ---
# --- SECTION 4: Load The Model and Predict Price ---
with col3:
    st.markdown("### ")
    st.write(" ")
    if st.button("Calculate Predicted Price", type="primary", use_container_width=True):
        try:      
            # 1. Make the prediction request to the backend      
            with st.spinner("Model is analyzing..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict", 
                    json=data  # Pass the raw dictionary here
                )
            
             # 2. Handle the response
            if response.status_code == 200:
                result = response.json().get("prediction")
                st.success(f"### Estimated Market Price: ${result:,.2f}")
            else:
                # This will now show you more specific validation errors if your columns don't match
                st.error(f"Backend Error: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the Backend. Is your FastAPI server running?")

