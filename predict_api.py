from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("glucose_predictor.pkl")

# Input schema
class GlucoseInput(BaseModel):
    current_glucose: float
    carbs: float
    insulin: float
    steps: int
    minutes_since_last_meal: int
    minutes_since_last_insulin: int
    time_of_day: str  # 'morning', 'afternoon', 'evening', 'night'

# Time of day mapping
time_map = {
    'morning': 0,
    'afternoon': 1,
    'evening': 2,
    'night': 3
}

@app.post("/predict")
def predict(input: GlucoseInput):
    data = input.dict()
    data["time_of_day"] = time_map.get(data["time_of_day"], 0)
    
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    
    return {"predicted_glucose": round(prediction, 2)}
