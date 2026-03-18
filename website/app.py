from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mileage_model = joblib.load(os.path.join(base_dir, 'mileage', 'model', 'mileage_model.pkl'))
power_model = joblib.load(os.path.join(base_dir, 'power', 'model', 'power_model.pkl'))
price_model   = joblib.load(os.path.join(base_dir, 'price', 'model', 'price_model.pkl'))
brand_mapping = joblib.load(os.path.join(base_dir, 'price', 'model', 'brand_mapping.pkl'))
model_mapping = joblib.load(os.path.join(base_dir, 'price', 'model', 'model_mapping.pkl'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict/mileage", methods=["POST"])
def predict_mileage():
    # ---------- Features -------------
    engine = float(request.form["engine"])
    kerb_weight = float(request.form["kerb_weight"])
    fuel_type = request.form["fuel_type"]
    transmission_type = request.form["transmission_type"]
    power = float(request.form['power'])
    cylinders = float(request.form['cylinders'])
    year = int(float(request.form['year']))

    transmission_Manual = 1 if transmission_type == "Manual" else 0  

    fuel_Diesel = 1 if fuel_type == "Diesel" else 0
    fuel_Petrol = 1 if fuel_type == "Petrol" else 0

    features = [
        engine,
        kerb_weight,
        transmission_Manual,    
        power,
        cylinders,
        year,
        fuel_Diesel,
        fuel_Petrol
    ]

    prediction = mileage_model.predict([features])[0]
    prediction = round(prediction, 2)

    return render_template("home.html", mileage_prediction=prediction)


@app.route("/predict/power", methods=["POST"])
def predict_power():
    # ---------- Numeric Features -------------
    engine = float(request.form["engine"])
    cylinders = float(request.form["cylinders"])
    valves_per_cylinder = float(request.form["valves_per_cylinder"])
    kerb_weight = float(request.form["kerb_weight"])
    transmission = int(request.form["transmission"])

    # ---------- Engineered Feature -------------
    engine_density = engine / cylinders

    # ---------- Categorical Features -------------
    fuel_supply = request.form["fuel_supply_system"]
    fuel_type = request.form["fuel_type"]
    engine_type = request.form["engine_type"]
    emission_norm = request.form["emission_norm"]
    drive_type = request.form["drive_type"]
    valve_config = request.form["valve_configuration"]

    # ---------- One Hot Encoding -------------
    FuelSys_EFI   = 1 if fuel_supply == "EFI" else 0
    FuelSys_GDI   = 1 if fuel_supply == "GDI" else 0
    FuelSys_MPFI  = 1 if fuel_supply == "MPFI" else 0
    FuelSys_Other = 1 if fuel_supply == "Other" else 0

    Fuel_Diesel = 1 if fuel_type == "Diesel" else 0
    Fuel_Petrol = 1 if fuel_type == "Petrol" else 0

    EngineType_DDiS         = 1 if engine_type == "DDiS" else 0
    EngineType_Diesel       = 1 if engine_type == "Diesel" else 0
    EngineType_F8D          = 1 if engine_type == "F8D" else 0
    EngineType_IRDE2        = 1 if engine_type == "IRDE2" else 0
    EngineType_InLine       = 1 if engine_type == "In-Line" else 0
    EngineType_KSeries      = 1 if engine_type == "K Series" else 0
    EngineType_Kappa        = 1 if engine_type == "Kappa" else 0
    EngineType_Kryotec      = 1 if engine_type == "Kryotec" else 0
    EngineType_Other        = 1 if engine_type == "Other" else 0
    EngineType_Petrol       = 1 if engine_type == "Petrol" else 0
    EngineType_Revotorq     = 1 if engine_type == "Revotorq" else 0
    EngineType_Revotron     = 1 if engine_type == "Revotron" else 0
    EngineType_SmartStream  = 1 if engine_type == "SmartStream" else 0
    EngineType_TDI          = 1 if engine_type == "TDI" else 0
    EngineType_TSI          = 1 if engine_type == "TSI" else 0
    EngineType_ToyotaDiesel = 1 if engine_type == "Toyota Diesel" else 0
    EngineType_TwinPower    = 1 if engine_type == "TwinPower" else 0
    EngineType_VVT          = 1 if engine_type == "VVT" else 0
    EngineType_iDTEC        = 1 if engine_type == "i-DTEC" else 0
    EngineType_iVTEC        = 1 if engine_type == "i-VTEC" else 0
    EngineType_mHawk        = 1 if engine_type == "mHawk" else 0
    EngineType_mStallion    = 1 if engine_type == "mStallion" else 0

    EmissionNorm_BS4  = 1 if emission_norm == "BS4" else 0
    EmissionNorm_BS6  = 1 if emission_norm == "BS6" else 0
    EmissionNorm_EURO = 1 if emission_norm == "EURO" else 0

    DriveType_FWD = 1 if drive_type == "FWD" else 0
    DriveType_RWD = 1 if drive_type == "RWD" else 0

    ValveConfig_IDSI = 1 if valve_config == "iDSI" else 0
    ValveConfig_SOHC = 1 if valve_config == "SOHC" else 0

    # ---------- Feature Array -------------
    features = [
        engine, cylinders, valves_per_cylinder, kerb_weight, transmission,
        FuelSys_EFI, FuelSys_GDI, FuelSys_MPFI, FuelSys_Other,
        Fuel_Diesel, Fuel_Petrol,
        EngineType_DDiS, EngineType_Diesel, EngineType_F8D, EngineType_IRDE2,
        EngineType_InLine, EngineType_KSeries, EngineType_Kappa, EngineType_Kryotec,
        EngineType_Other, EngineType_Petrol, EngineType_Revotorq, EngineType_Revotron,
        EngineType_SmartStream, EngineType_TDI, EngineType_TSI, EngineType_ToyotaDiesel,
        EngineType_TwinPower, EngineType_VVT, EngineType_iDTEC, EngineType_iVTEC,
        EngineType_mHawk, EngineType_mStallion,
        EmissionNorm_BS4, EmissionNorm_BS6, EmissionNorm_EURO,
        DriveType_FWD, DriveType_RWD,
        ValveConfig_IDSI, ValveConfig_SOHC,
        engine_density
    ]

    prediction = power_model.predict([features])[0]
    prediction = round(prediction, 2)

    return render_template("home.html", power_prediction=prediction)

@app.route("/predict/price", methods=["POST"])
def predict_price():
    # ---------- Numeric Features -------------
    year         = int(float(request.form["year"]))
    kms_driven   = float(request.form["kms_driven"])
    ownership    = int(request.form["ownership"])
    transmission = int(request.form["transmission"])
    engine       = float(request.form["engine"])
    power        = float(request.form["power"])
    mileage      = float(request.form["mileage"])
    cylinders    = float(request.form["cylinders"])
    turbo        = int(request.form["turbo_charger"])
    seats        = int(request.form["seats"])
    kerb_weight  = float(request.form["kerb_weight"])
    ground_clearance = float(request.form["ground_clearance"])

    # ---------- Fuel Tank -------------
    petrol_tank = float(request.form.get("petrol_tank") or 0)
    diesel_tank = float(request.form.get("diesel_tank") or 0)
    cng_tank    = float(request.form.get("cng_tank") or 0)

    # ---------- Brand & Model (Target Encoded) -------------
    brand_input = request.form["brand"].strip()
    model_input = request.form["model"].strip()

    brand_unknown = brand_input not in brand_mapping
    model_unknown = model_input not in model_mapping

    brand_encoded = brand_mapping.get(brand_input, sum(brand_mapping.values()) / len(brand_mapping))
    model_encoded = model_mapping.get(model_input, sum(model_mapping.values()) / len(model_mapping))
    
    # ---------- One Hot Encoding -------------
    # Fuel Type (dropped: CNG)
    fuel_type   = request.form["fuel_type"]
    Fuel_Diesel = 1 if fuel_type == "Diesel" else 0
    Fuel_Petrol = 1 if fuel_type == "Petrol" else 0

    # Drive Type (dropped: AWD)
    drive_type     = request.form["drive_type"]
    Drive_Type_FWD = 1 if drive_type == "FWD" else 0
    Drive_Type_RWD = 1 if drive_type == "RWD" else 0

    # ---------- Feature Array (exact order) -------------
    features = [
        year, kms_driven, ownership, transmission,
        engine, power, mileage, cylinders, turbo,
        seats, kerb_weight, ground_clearance,
        petrol_tank, diesel_tank, cng_tank,
        brand_encoded, model_encoded,
        Fuel_Diesel, Fuel_Petrol,
        Drive_Type_FWD, Drive_Type_RWD
    ]

    # ---------- Prediction -------------
    prediction = round(float(prediction), 2)
    prediction_formatted = f"₹ {prediction:,.0f}"

    return render_template("home.html", price_prediction=prediction_formatted)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)