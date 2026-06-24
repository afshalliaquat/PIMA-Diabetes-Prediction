# PIMA Diabetes Prediction API

## Overview

This project predicts diabetes risk from patient health data using the PIMA Indians Diabetes Dataset, and serves the trained model through a FastAPI application with request validation and confidence-scored predictions.

Pipeline:
- Missing value handling with median imputation
- Feature engineering (N0 to N15)
- Exploratory data visualization (Plotly)
- Model training with Bagging + Decision Trees
- Evaluation using Confusion Matrix and Classification Report
- Model saved as pima.pkl with joblib
- Served via a FastAPI /predict endpoint with Pydantic request validation

## Dataset

Source: Kaggle, PIMA Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
Target: Outcome (1 = diabetic, 0 = non-diabetic)

## Requirements

pip install numpy pandas matplotlib seaborn scikit-learn plotly xgboost joblib fastapi uvicorn pydantic

## Usage

### 1. Train the model

Add your kaggle.json key to the project, then run the notebook:

jupyter notebook PIMA_Diabetes.ipynb

The trained model will be saved as pima.pkl.

### 2. Run the API

uvicorn main:app --reload

The API will be available at http://localhost:8000, with interactive docs at http://localhost:8000/docs.

## Model

Algorithm: BaggingClassifier with DecisionTreeClassifier

Parameters:
- n_estimators = 500
- bootstrap = True
- oob_score = True

### Results

Class 0 (Non-diabetic): Precision 0.95, Recall 0.91, F1-score 0.93, Support 103
Class 1 (Diabetic): Precision 0.84, Recall 0.90, F1-score 0.87, Support 51
Overall: Precision 0.91, Recall 0.91, F1-score 0.91, Support 154

## API Reference

### GET /

Returns a welcome message and points to /docs for interactive usage.

### GET /predict

Returns a usage hint to send a POST request with patient data.

### POST /predict

Accepts patient health data and returns a prediction.

Request body (PatientData):
- pregnancies: int, default 0, range 0 to 20
- glucose: float, default 90.0, range 40 to 600
- blood_pressure: float, default 70.0, range 40 to 200
- skin_thickness: float, default 20.0, range 0 to 99
- insulin: float, default 10.0, range 0 to 900
- bmi: float, default 25.0, range 10.0 to 70.0
- dpf: float, default 0.5, range 0.0 to 3.0
- age: int, default 30, range 1 to 100

All fields are optional and validated with Pydantic, using Field constraints to enforce realistic value ranges.

Prediction logic:
- If glucose is below 100, returns "Non-Diabetic (Normal Glucose)" directly, without calling the model
- If glucose is between 100 and 125, returns "Prediabetic (Borderline Glucose)" directly
- Otherwise, runs full feature engineering (N0 to N15) and passes the result through the trained model, returning "Diabetic" or "Non-Diabetic" with a confidence score as a percentage

Example request:

POST /predict
{
  "pregnancies": 2,
  "glucose": 150,
  "blood_pressure": 85,
  "skin_thickness": 30,
  "insulin": 120,
  "bmi": 32.5,
  "dpf": 0.6,
  "age": 45
}

Example response:

{
  "prediction": "Diabetic",
  "confidence": 78.3
}

## Model Loading

import joblib
model = joblib.load("pima.pkl")

## Project Structure

pima-diabetes-prediction/
- PIMA_Diabetes.ipynb: data exploration, feature engineering, model training
- main.py: FastAPI app containing the request schema, feature engineering, and /predict endpoint
- pima.pkl: trained BaggingClassifier model
- requirements.txt
- README.md

## Disclaimer

This project is for educational purposes only and is not a substitute for professional medical advice or diagnosis.

## License

MIT
