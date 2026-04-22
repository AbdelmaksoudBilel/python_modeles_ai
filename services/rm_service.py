# RM SERVICE

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
            X = np.array(features).reshape(1, -1)

            score = self.model.decision_function(X)[0]
            prediction = self.model.predict(X)[0]

            result = {
                "score_anomalie": float(score),
                "prediction": "RM_typique" if prediction == 1 else "Profil_atypique",
                "is_anomaly": bool(prediction == -1)
            }

            return result

        except Exception as e:
            return {
                "error": str(e)
            }