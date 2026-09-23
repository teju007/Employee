import joblib
import pandas as pd
from fastapi import FastAPI

MODEL_PATH = "models/attrition_pipeline.pkl"

app = FastAPI(title="Employee Attrition Prediction API")

model_pipeline = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Employee Attrition API is running"}


@app.post("/predict")
def predict_employee(employee_data: dict):

    data = pd.DataFrame([employee_data])

    prediction = model_pipeline.predict(data)[0]

    if prediction == "Yes":
        risk = "HIGH RISK"
        message = "Employee is predicted to leave."
    else:
        risk = "LOW RISK"
        message = "Employee is predicted to stay."

    return {
        "prediction": prediction,
        "risk": risk,
        "message": message
    }