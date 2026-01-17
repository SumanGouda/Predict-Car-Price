import joblib 
import pandas as pd 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "car_price_model.joblib"

app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
    print("✅ BACKEND: Model loaded successfully!")
except Exception as e:
    print(f"❌ BACKEND ERROR: Could not load model. Reason: {e}")
    model = None
    
# Define exactly what a "Car" looks like
class CarFeatures(BaseModel):
    year: int
    leather_interior: int  # Note: Pydantic usually uses underscores
    engine_volume: float
    mileage: float
    cylinders: float
    wheel: int
    airbags: int
    engine_type: int
    gearbox_automatic: int
    gearbox_manual: int
    gearbox_tiptronic: int
    gearbox_variator: int
    drive_fwd: int
    drive_rwd: int
    fuel_diesel: int
    fuel_hybrid: int
    fuel_hydrogen: int
    fuel_lpg: int
    fuel_petrol: int
    fuel_plugin: int
    engine_volume_log: float
    
@app.post("/predict")
def predict_price(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert to float to make it JSON serializable
        return {"prediction": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")