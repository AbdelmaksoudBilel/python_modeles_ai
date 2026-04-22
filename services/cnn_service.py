import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

import tf_keras as keras
# On charge le DOSSIER
model_cnn = keras.models.load_model("models_saved/modele_tsa_cnn", compile=False)

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_cnn(image: Image.Image):
    img_array = preprocess_image(image)
    prob = model_cnn.predict(img_array, verbose=0)[0][0]
    return float(prob)