from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib


try:
    model = joblib.load("pima.pkl")
except FileNotFoundError:
    raise RuntimeError("❌ Model file not found! Please train and save `pima.pkl` first.")

app = FastAPI(title="Pima Diabetes Prediction API")

class PatientData(BaseModel):
    pregnancies: Optional[int] = Field(0, ge=0, le=20)
    glucose: Optional[float] = Field(90.0, ge=40, le=600)
    blood_pressure: Optional[float] = Field(70.0, ge=40, le=200)
    skin_thickness: Optional[float] = Field(20.0, ge=0, le=99)
    insulin: Optional[float] = Field(10.0, ge=0, le=900)
    bmi: Optional[float] = Field(25.0, ge=10.0, le=70.0)
    dpf: Optional[float] = Field(0.5, ge=0.0, le=3.0)
    age: Optional[int] = Field(30, ge=1, le=100)

def engineer_features(data: PatientData):
    N1 = 1
    if (data.age <= 30 and data.glucose <= 120) or (30 < data.age < 48 and data.glucose <= 88) or (data.age >= 63 and data.glucose <= 142):
        N1 = 0

    N2 = 0 if data.bmi <= 30 else 1

    N3 = 1
    if (data.age <= 27 and data.pregnancies <= 6) or (data.age > 60 and data.pregnancies > 7.5):
        N3 = 0

    N6 = 1
    if (data.bmi < 30 and data.skin_thickness <= 20) or (data.bmi >= 30 and data.skin_thickness <= 20):
        N6 = 0

    N7 = 1
    if (data.glucose <= 105 and data.bmi <= 30) or (data.glucose <= 105 and data.bmi > 40):
        N7 = 0

    N9 = 1
    if data.insulin < 200:
        N9 = 0

    N10 = 1
    if data.blood_pressure < 80:
        N10 = 0

    N11 = 1
    if (data.pregnancies < 4 and data.pregnancies != 0):
        N11 = 0

    N0 = data.bmi * data.skin_thickness
    N8 = data.pregnancies / data.age if data.age != 0 else 0
    N13 = data.glucose / data.dpf if data.dpf != 0 else 0
    N12 = data.age * data.dpf
    N15 = 1 if N0 >= 1034 else 0

    features = pd.DataFrame([[
        data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness, data.insulin,
        data.bmi, data.dpf, data.age,
        N1, N2, N3, N6, N7, N9, N10, N11, N0, N8, N13, N12, N15
    ]], columns=[
        'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
        'BMI','DiabetesPedigreeFunction','Age',
        'N1','N2','N3','N6','N7','N9','N10','N11','N0','N8','N13','N12','N15'
    ])
    return features


def predict_diabetes(data: PatientData):
    if data.glucose < 100:
        return {"prediction": "Non-Diabetic (Normal Glucose)"}
    elif 100 <= data.glucose < 126:
        return {"prediction": "Prediabetic (Borderline Glucose)"}
    else:
        features = engineer_features(data)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        if prediction == 1:
            return {"prediction": "Diabetic", "confidence": round(proba*100, 1)}
        else:
            return {"prediction": "Non-Diabetic", "confidence": round((1-proba)*100, 1)}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pima Diabetes Prediction API. Use the /docs endpoint to get predictions."}

@app.get("/predict")
def get_predict():
    return {"message": "Please use a POST request with patient data to get a prediction."}

@app.post("/predict")
def predict(patient: PatientData):
    try:
        result = predict_diabetes(patient)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
