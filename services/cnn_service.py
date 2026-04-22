import numpy as np
from PIL import Image

import tf_keras as keras
import tensorflow as tf

# Correctif pour les arguments obsolètes de InputLayer
from tf_keras.src.layers import InputLayer
original_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    # On retire les arguments que le serveur ne comprend pas
    kwargs.pop('batch_shape', None)
    kwargs.pop('optional', None)
    return original_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

# Maintenant on tente le chargement
model_cnn = keras.models.load_model("models_saved/modele_tsa_cnn.h5", compile=False)

IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_cnn(image: Image.Image):
    img_array = preprocess_image(image)
    prob = model_cnn.predict(img_array, verbose=0)[0][0]
    return float(prob)