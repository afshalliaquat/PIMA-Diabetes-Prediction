# 🩺 PIMA Diabetes Prediction

## 📖 Overview

This project uses the **PIMA Indians Diabetes Dataset** to predict diabetes.
Steps included:

* Missing value handling with median imputation
* Feature engineering (`N0–N15`)
* Exploratory data visualization (Plotly)
* Model training with **Bagging + Decision Trees**
* Evaluation using **Confusion Matrix** and **Classification Report**
* Model saved as `pima.pkl` with `joblib`

## 📂 Dataset

* **Source:** [Kaggle – PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Target:** `Outcome` (1 = diabetic, 0 = non-diabetic)

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly xgboost joblib
```

## 🚀 Usage

1. Add your `kaggle.json` key to the project.
2. Run the notebook:

   ```bash
   jupyter notebook PIMA_.ipynb
   ```
3. The trained model will be saved as `pima.pkl`.

## 📊 Model

* **Algorithm:** BaggingClassifier with DecisionTreeClassifier
* **Parameters:**

  * n\_estimators = 500
  * bootstrap = True
  * oob\_score = True

### ✅ Results

* **Accuracy:** 91%
* **Precision / Recall / F1-score:**

| Class            | Precision | Recall   | F1-score | Support |
| ---------------- | --------- | -------- | -------- | ------- |
| 0 (Non-diabetic) | 0.95      | 0.91     | 0.93     | 103     |
| 1 (Diabetic)     | 0.84      | 0.90     | 0.87     | 51      |
| **Overall**      | **0.91**  | **0.91** | **0.91** | 154     |

## 💾 Model Saving

```python
import joblib
model = joblib.load("pima.pkl")

