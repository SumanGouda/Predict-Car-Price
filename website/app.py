from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
price_model   = joblib.load(os.path.join(base_dir, 'Price', 'model', 'price_model.pkl'))
brand_mapping = joblib.load(os.path.join(base_dir, 'Price', 'model', 'brand_mapping.pkl'))
model_mapping = joblib.load(os.path.join(base_dir, 'Price', 'model', 'model_mapping.pkl'))
city_mapping  = joblib.load(os.path.join(base_dir, 'Price', 'model', 'city_mapping.pkl'))
power_model  = joblib.load(os.path.join(base_dir, 'Power', 'model', 'power_model.pkl'))
mileage_model = joblib.load(os.path.join(base_dir, 'Mileage', 'model', 'mileage_model.pkl'))

# Precompute global means once at startup
GLOBAL_BRAND_MEAN = sum(brand_mapping.values()) / len(brand_mapping)
GLOBAL_MODEL_MEAN = sum(model_mapping.values()) / len(model_mapping)

@app.route("/")
def home():
    return render_template("index.html")
 

@app.route("/predict_price", methods=["POST"])
def predict():
    # ── Numeric inputs ────────────────────────────────────────────
    reg_year   = int(request.form.get("reg_year", 2020))
    kms_driven = float(request.form.get("kms", 0))
    ownership  = int(request.form.get("ownership", 1))
    transmission = int(request.form.get("transmission", 0))
    engine     = float(request.form.get("engine", 1200))
    power      = float(request.form.get("power", 80))
    mileage    = float(request.form.get("mileage", 20))
    cylinders  = float(request.form.get("cylinders", 4))
    turbo      = int(request.form.get("turbo", 0))
    seats      = int(request.form.get("seats", 5))
    kerb       = float(request.form.get("kerb", 1000))
    ground     = float(request.form.get("ground", 170))

    # ── Fuel ──────────────────────────────────────────────────────
    selected_fuel    = request.form.get("fuel", "CNG")
    tank_size        = float(request.form.get("tank_size", 0) or 0)
    petrol_tank_cap  = tank_size if selected_fuel == "Petrol"  else 0
    diesel_tank_cap  = tank_size if selected_fuel == "Diesel"  else 0
    cng_tank_cap     = tank_size if selected_fuel == "CNG"     else 0
    fuel_diesel      = 1 if selected_fuel == "Diesel" else 0
    fuel_petrol      = 1 if selected_fuel == "Petrol" else 0
    fuel_lpg         = 1 if selected_fuel == "LPG"    else 0
    fuel_was_missing = 0

    # ── Drive Type ────────────────────────────────────────────────
    # AWD is the reference category (both FWD and RWD = 0)
    drive_type           = request.form.get("drive_type", "AWD")
    drive_type_was_missing = 0                                   # user always picks one
    drive_fwd            = 1 if drive_type == "FWD" else 0
    drive_rwd            = 1 if drive_type == "RWD" else 0

    # ── Target-encoded categoricals ───────────────────────────────
    city_name     = request.form.get("city", "Delhi").strip()
    brand_name    = request.form.get("brand", "").strip()
    model_name    = request.form.get("model", "").strip()

    brand_unknown = brand_name not in brand_mapping
    model_unknown = model_name not in model_mapping

    city_encoded  = city_mapping.get(city_name, 0.0)
    brand_encoded = brand_mapping.get(brand_name, GLOBAL_BRAND_MEAN)
    model_encoded = model_mapping.get(model_name, GLOBAL_MODEL_MEAN)

    # ── Missing-indicator flags ───────────────────────────────────
    mileage_was_missing       = 0
    turbo_charger_was_missing = 0

    # ── Assemble feature vector (exact column order) ──────────────
    features = [
        reg_year,          
        kms_driven,       
        ownership,       
        transmission,     
        engine,       
        power,             
        mileage,         
        cylinders,      
        turbo,              
        seats,            
        kerb,           
        ground,            
        petrol_tank_cap,   
        diesel_tank_cap,   
        cng_tank_cap,      
        city_encoded,       
        brand_encoded,     
        model_encoded,      
        fuel_was_missing, 
        fuel_diesel,      
        fuel_lpg,         
        fuel_petrol,       
        drive_type_was_missing,   
        drive_fwd,         
        drive_rwd,         
        mileage_was_missing,    
        turbo_charger_was_missing,  
    ]

    # ── Predict ───────────────────────────────────────────────────
    prediction       = price_model.predict([features])[0]
    prediction       = round(float(prediction), 2)
    prediction_formatted = f"₹ {prediction:,.0f}"

    return render_template(
        "index.html",
        price_prediction=prediction_formatted,
        brand_unknown=brand_unknown,
        model_unknown=model_unknown,
        brand_input=brand_name,
        model_input=model_name,
    )

@app.route("/predict_power", methods=["POST"])
def predict_power():
    engine              = float(request.form.get("engine", 1200))
    cylinders           = float(request.form.get("cylinders", 4))      
    transmission        = int(request.form.get("transmission", 0))
    valves_per_cylinder = float(request.form.get("valves_per_cylinder", 4))
    drive_type          = request.form.get("drive_type", "AWD")
    fuel                = request.form.get("fuel", "CNG")
    engine_type         = request.form.get("engine_type", "CRDi")   
    engine_density      = engine / cylinders

    EngineType_DDiS         = 1 if engine_type == "DDiS"          else 0
    EngineType_Diesel       = 1 if engine_type == "Diesel"         else 0
    EngineType_F8D          = 1 if engine_type == "F8D"            else 0
    EngineType_IRDE2        = 1 if engine_type == "IRDE2"          else 0
    EngineType_InLine       = 1 if engine_type == "In-Line"        else 0
    EngineType_KSeries      = 1 if engine_type == "K Series"       else 0
    EngineType_Kappa        = 1 if engine_type == "Kappa"          else 0
    EngineType_Kryotec      = 1 if engine_type == "Kryotec"        else 0
    EngineType_Other        = 1 if engine_type == "Other"          else 0
    EngineType_Petrol       = 1 if engine_type == "Petrol"         else 0
    EngineType_Revotorq     = 1 if engine_type == "Revotorq"       else 0
    EngineType_Revotron     = 1 if engine_type == "Revotron"       else 0
    EngineType_SmartStream  = 1 if engine_type == "SmartStream"    else 0
    EngineType_TDI          = 1 if engine_type == "TDI"            else 0
    EngineType_TSI          = 1 if engine_type == "TSI"            else 0
    EngineType_ToyotaDiesel = 1 if engine_type == "Toyota Diesel"  else 0
    EngineType_TwinPower    = 1 if engine_type == "TwinPower"      else 0
    EngineType_VVT          = 1 if engine_type == "VVT"            else 0
    EngineType_iDTEC        = 1 if engine_type == "i-DTEC"         else 0
    EngineType_iVTEC        = 1 if engine_type == "i-VTEC"         else 0
    EngineType_mHawk        = 1 if engine_type == "mHawk"          else 0
    EngineType_mStallion    = 1 if engine_type == "mStallion"      else 0
    
    DriveType_FWD = 1 if drive_type == "FWD" else 0
    DriveType_RWD = 1 if drive_type == "RWD" else 0

    Fuel_Diesel = 1 if fuel == "Diesel" else 0
    Fuel_LPG    = 1 if fuel == "LPG"    else 0
    Fuel_Petrol = 1 if fuel == "Petrol" else 0

     
    features = [
        engine, cylinders, transmission, valves_per_cylinder,
        DriveType_FWD, DriveType_RWD,
        Fuel_Diesel, Fuel_LPG, Fuel_Petrol,
        EngineType_DDiS, EngineType_Diesel, EngineType_F8D, EngineType_IRDE2,
        EngineType_InLine, EngineType_KSeries, EngineType_Kappa, EngineType_Kryotec,
        EngineType_Other, EngineType_Petrol, EngineType_Revotorq, EngineType_Revotron,
        EngineType_SmartStream, EngineType_TDI, EngineType_TSI, EngineType_ToyotaDiesel,
        EngineType_TwinPower, EngineType_VVT, EngineType_iDTEC, EngineType_iVTEC,
        EngineType_mHawk, EngineType_mStallion,
        engine_density
    ]

    prediction = power_model.predict([features])[0]
    prediction = round(float(prediction), 2)

    return render_template("index.html", power_prediction=prediction)

@app.route("/api/predict_mileage", methods=["POST"])
def predict_mileage():
    engine       = float(request.form.get("engine", 1200))
    kerb_weight  = float(request.form.get("kerb_weight", 1000))
    transmission = int(request.form.get("transmission_type", 0))
    power        = float(request.form.get("power", 80))
    cylinders    = float(request.form.get("cylinders", 4))
    year         = int(request.form.get("year", 2020))
    fuel         = request.form.get("fuel", "CNG")

    Fuel_Diesel = 1 if fuel == "Diesel" else 0
    Fuel_LPG    = 1 if fuel == "LPG"    else 0
    Fuel_Petrol = 1 if fuel == "Petrol" else 0

    # Exact column order from model
    features = [
        engine, kerb_weight, transmission, power,
        cylinders, year,
        0,           # Mileage_was_missing — always 0 (user provides real value)
        0,           # Fuel_was_missing    — always 0 (user provides real value)
        Fuel_Diesel, Fuel_LPG, Fuel_Petrol
    ]

    prediction = mileage_model.predict([features])[0]
    prediction = round(float(prediction), 2)

    return render_template("index.html", mileage_prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)