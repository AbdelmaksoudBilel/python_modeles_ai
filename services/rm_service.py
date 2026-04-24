# RM SERVICE

import pandas as pd

import joblib
import numpy as np
import os


class RMService:

    def __init__(self):
        model_path = os.path.join(
            "models_saved",
            "modele_rm_ml.pkl"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Le modèle RM n'existe pas."
            )

        self.model = joblib.load(model_path)


    def predict(self, features: list):
        """
        features : liste des variables numériques
        """

        try:
            columns = [
                "PR_AGE1","PR_Q3D","PR_QF1A","PR_QG1A","PR_QH1A","PR_QH1B",
                "PR_QI1","PR_QJ1","PR_QK1","PR_QQ",
                "PR_QN1_A","PR_QN1_B","PR_QN1_C","PR_QN1_D","PR_QN1_E",
                "PR_QN1_F","PR_QN1_G","PR_QN1_H",
                "PR_QO1_A_COMBINE","PR_QO1_B_COMBINE",
                "PR_QO1_C_COMBINE","PR_QO1_D_COMBINE","PR_QO1_E_COMBINE"
            ]

            X = pd.DataFrame([features], columns=columns)
            print(f"Features reçues pour RM : {X}")

            score = self.model.decision_function(X)[0]
            print(f"Score d'anomalie RM : {score}")
            prediction = self.model.predict(X)[0]
            print(f"Prédiction RM : {prediction}")

            result = {
                "score_anomalie": float(score),
                "prediction": "RM_typique" if prediction == 1 else "Profil_atypique",
                "is_anomaly": bool(prediction == -1)
            }
            print(f"Résultat RM : {result}")
            return result

        except Exception as e:
            return {
                "error": str(e)
            }

def _main():
    rm_service = RMService()

    features = [2.0,2.0,2.0,1.0,5.0,1.0,1.0,1.0,5.0,4,3.0,3.0,3.0,3.0,3.0,3.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0]

    result = rm_service.predict(features)
    print(result)


if __name__ == "__main__":
    _main()