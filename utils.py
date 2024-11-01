import numpy as np
from os import getenv, listdir
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from dotenv import load_dotenv, set_key
import matplotlib.pyplot as plt

load_dotenv("config") # Charger le fichier

def get_config(key):
    return getenv(key)

def edit_config(key, value):
    set_key("config", key, value)

# Charger et redimensionner une image et son masque
def load_image_mask(image_path, mask_path, target_size=(128, 128)):
    # Charger l'image et le masque
    image = load_img(image_path, target_size=target_size)
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")

    # Convertir en tableaux NumPy
    image = img_to_array(image) / 255.0  # Normaliser les pixels avec des valeurs entre 0 et 1 pour chaque pixel
    mask = img_to_array(mask) / 255.0  # Masque binaire
    #mask = (mask > 0.5).astype(np.uint8)  # Binarisation

    return image, mask


# Listes pour stocker les images et les masques
def load_images(target_size=(128, 128)):
    images = []
    masks = []

    # Boucle sur tous les fichiers de masque
    print("Chargement de candle_on")
    i = 0
    for nom_fichier in listdir("assets/candle_on/mask"):
        if nom_fichier.endswith(".png"):  # Vérifier l'extension du fichier
            i += 1
            # Charger l'image et son masque correspondant
            image_path = "assets/candle_on/" + nom_fichier[:-9] + ".jpg"
            mask_path = "assets/candle_on/mask/" + nom_fichier
            image, mask = load_image_mask(image_path, mask_path, target_size)

            images.append(image)
            masks.append(mask)
    print(i, " images chargées avec masque")

    print("Chargement de candle_off")
    for nom_fichier in listdir("assets/candle_off"):
        if i <= 0:
            break
        if nom_fichier.endswith(".jpg"):  # Vérifier l'extension du fichier
            i -= 1
            # Charger l'image et son masque correspondant
            image_path = "assets/candle_off/" + nom_fichier
            mask_path = "assets/no_candle.png"
            image, mask = load_image_mask(image_path, mask_path, target_size)

            images.append(image)
            masks.append(mask)
    #
    # for nom_fichier in listdir("assets/no_candle"):
    #     if nom_fichier.endswith(".jpg"):  # Vérifier l'extension du fichier
    #         # Charger l'image et son masque correspondant
    #         image_path = "assets/no_candle/" + nom_fichier[:-9] + ".jpg"
    #         mask_path = "assets/no_candle.png"
    #         image, mask = load_image_mask(image_path, mask_path, target_size)
    #
    #         images.append(image)
    #         masks.append(mask)

    # Convertir les listes en tableaux NumPy
    images = np.array(images)
    masks = np.array(masks)

    return images, masks

def show_model_stats(history):
    # Tracer la courbe de la perte
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()

    # Tracer la courbe de la précision (si elle est calculée)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
        plt.plot(history.history['val_accuracy'], label='Précision de validation')
        plt.xlabel('Époques')
        plt.ylabel('Précision')
        plt.legend()
        plt.show()