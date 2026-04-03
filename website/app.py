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
    engine            = float(request.form["engine"])
    kerb_weight       = float(request.form["kerb_weight"])
    transmission_type = int(request.form["transmission_type"])
    power             = float(request.form["power"])
    cylinders         = float(request.form["cylinders"])
    year              = int(float(request.form["year"]))
    fuel              = request.form["fuel"]

    # missing indicators — always 0 since user provides real values
    mileage_was_missing = int(request.form.get("mileage_was_missing", 0))
    fuel_was_missing    = int(request.form.get("fuel_was_missing", 0))

    # one hot encoding
    Fuel_Diesel = 1 if fuel == "Diesel" else 0
    Fuel_LPG    = 1 if fuel == "LPG"    else 0
    Fuel_Petrol = 1 if fuel == "Petrol" else 0

    # exact column order from model
    features = [
        engine, kerb_weight, transmission_type, power,
        cylinders, year,
        mileage_was_missing, fuel_was_missing,
        Fuel_Diesel, Fuel_LPG, Fuel_Petrol
    ]

    prediction = mileage_model.predict([features])[0]
    prediction = round(prediction, 2)

    return render_template("home.html", mileage_prediction=prediction)

@app.route("/predict/power", methods=["POST"])
def predict_power():
    # ---------- Numeric Features -------------
    engine              = float(request.form["engine"])
    cylinders           = float(request.form["cylinders"])
    transmission        = int(request.form["transmission"])
    valves_per_cylinder = float(request.form["valves_per_cylinder"])
    drive_type          = request.form["drive_type"]
    fuel                = request.form["fuel"]
    engine_type         = request.form["engine_type"]

    # ---------- Engineered Feature -------------
    engine_density = engine / cylinders

    # ---------- One Hot Encoding -------------
    DriveType_FWD = 1 if drive_type == "FWD" else 0
    DriveType_RWD = 1 if drive_type == "RWD" else 0

    Fuel_Diesel = 1 if fuel == "Diesel" else 0
    Fuel_LPG    = 1 if fuel == "LPG"    else 0
    Fuel_Petrol = 1 if fuel == "Petrol" else 0

    EngineType_DDiS        = 1 if engine_type == "DDiS"          else 0
    EngineType_Diesel      = 1 if engine_type == "Diesel"         else 0
    EngineType_F8D         = 1 if engine_type == "F8D"            else 0
    EngineType_IRDE2       = 1 if engine_type == "IRDE2"          else 0
    EngineType_InLine      = 1 if engine_type == "In-Line"        else 0
    EngineType_KSeries     = 1 if engine_type == "K Series"       else 0
    EngineType_Kappa       = 1 if engine_type == "Kappa"          else 0
    EngineType_Kryotec     = 1 if engine_type == "Kryotec"        else 0
    EngineType_Other       = 1 if engine_type == "Other"          else 0
    EngineType_Petrol      = 1 if engine_type == "Petrol"         else 0
    EngineType_Revotorq    = 1 if engine_type == "Revotorq"       else 0
    EngineType_Revotron    = 1 if engine_type == "Revotron"       else 0
    EngineType_SmartStream = 1 if engine_type == "SmartStream"    else 0
    EngineType_TDI         = 1 if engine_type == "TDI"            else 0
    EngineType_TSI         = 1 if engine_type == "TSI"            else 0
    EngineType_ToyotaDiesel= 1 if engine_type == "Toyota Diesel"  else 0
    EngineType_TwinPower   = 1 if engine_type == "TwinPower"      else 0
    EngineType_VVT         = 1 if engine_type == "VVT"            else 0
    EngineType_iDTEC       = 1 if engine_type == "i-DTEC"         else 0
    EngineType_iVTEC       = 1 if engine_type == "i-VTEC"         else 0
    EngineType_mHawk       = 1 if engine_type == "mHawk"          else 0
    EngineType_mStallion   = 1 if engine_type == "mStallion"      else 0

    # ---------- Feature Array (exact model column order) -------------
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

    # ---------- Prediction -------------
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
    fuel   = request.form["fuel"]
    Fuel_Diesel = 1 if fuel == "Diesel" else 0
    Fuel_Petrol = 1 if fuel == "Petrol" else 0

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