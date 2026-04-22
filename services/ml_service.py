import joblib
import numpy as np
import pandas as pd

model_ml = joblib.load("models_saved/modele_tsa_ml.pkl")

def predict_ml(features):

    columns = [
        "A1", "A2", "A3", "A4", "A5",
        "A6", "A7", "A8", "A9", "A10",
        "Age_Years", "Sex", "Jaundice", "Family_mem_with_ASD"
    ]

    df = pd.DataFrame([features], columns=columns)

    prob = model_ml.predict_proba(df)[0][1]

    return float(prob)