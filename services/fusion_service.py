import numpy as np

# Poids calculés mathématiquement
W_ML = 0.556
W_CNN = 0.444

def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fusion_prediction(prob_ml, prob_cnn):
    logit_fusion = (
        W_ML * logit(prob_ml) +
        W_CNN * logit(prob_cnn)
    )
    final_prob = sigmoid(logit_fusion)
    return float(final_prob)