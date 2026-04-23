import tensorflow as tf
import numpy as np
from PIL import Image
import tf_keras as keras

IMG_SIZE = 224

# Chargement du modèle
model_cnn = keras.models.load_model("models_saved/modele_tsa_cnn", compile=False)

def preprocess_image(image: Image.Image):
    # S'assurer que l'image est en RGB (évite les erreurs avec PNG/RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def predict_cnn(image: Image.Image):
    img_array = preprocess_image(image)
    
    try:
        # Tentative standard
        prob = model_cnn.predict(img_array, verbose=0)[0][0]
    except AttributeError:
        # Solution de secours : Appel direct au graphe (SavedModel)
        infer = model_cnn.signatures["serving_default"]
        # On convertit l'array en tenseur
        input_tensor = tf.convert_to_tensor(img_array)
        # L'entrée du modèle MobileNetV2 s'appelle souvent 'input_layer' ou 'input_1'
        # On récupère le premier nom de clé d'entrée dynamiquement
        input_name = list(infer.structured_input_signature[1].keys())[0]
        output = infer(**{input_name: input_tensor})
        # On récupère la valeur de sortie
        prob = list(output.values())[0].numpy()[0][0]
        
    return float(prob)