from os import listdir
import random as rd
from tabnanny import verbose

from PyQt5.sip import array
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from discord import image_to_discord
from utils import edit_config, get_config

seuil = 0.05
model_name = f"fuoco_0_0_{get_config('MODEL_VERSION')}.h5"
#model_name = "best_model_0_0_5.keras"


# Chargement du modèle
model = load_model('assets/model/'+model_name)

def prepare_image(image_path, target_size=(128, 128)):
    # Charger l'image
    image = load_img(image_path, target_size=target_size)
    # Convertir en tableau NumPy et normaliser
    image = img_to_array(image) / 255.0
    # Ajouter une dimension pour représenter le lot
    image = np.expand_dims(image, axis=0)  # Shape devient (1, 512, 512, 3)
    return image




# Chemin de l'image à prédire
img_folder = ["candle_on", "candle_off", "no_candle"]
folder = img_folder[rd.randint(0, len(img_folder)-1)]
img_list = listdir(f"assets/{folder}")

img = img_list[rd.randint(0, len(img_list))]
image_path = f"assets/{folder}/{img}"

#image_path = f"assets/candle_on/aug_0_0_1944.jpg"
while not img.endswith(".jpg"):
    img = img_list[rd.randint(0, len(img_list))]

prepared_image = prepare_image(image_path, (128, 128))

### Faire la prédiction
predicted_mask = model.predict(prepared_image, verbose=1)



# Charger l'image d'origine pour obtenir ses dimensions
original_image = load_img(image_path)
original_size = original_image.size  # Récupère les dimensions (largeur, hauteur)
original_image = img_to_array(original_image) / 255.0  # Normaliser l'image d'origine



# Pour obtenir un masque binaire, applique un seuil
thresholded_mask = (predicted_mask[0, :, :, 0] >= seuil).astype(np.uint8)  # Appliquer le seuil



# Redimensionner le masque prédit pour qu'il corresponde à l'image originale
predicted_mask_resized = Image.fromarray((thresholded_mask * 255).astype(np.uint8))  # Conversion en Image PIL
predicted_mask_resized = predicted_mask_resized.resize(original_size, Image.NEAREST)  # Redimensionnement

print("Valeur min:", predicted_mask.min())
print("Valeur max:", predicted_mask.max())



### Affichage des résultats

if False: # Afficher l'image d'origine et le masque prédit
    plt.figure(figsize=(12, 6))
    # Image d'origine
    plt.subplot(1, 2, 1)
    plt.title(f"Image d'origine ({img})")
    plt.imshow(load_img(image_path))
    plt.axis('off')

    # Masque prédit
    plt.subplot(1, 2, 2)
    plt.title("Masque prédit")
    #plt.imshow(thresholded_mask, cmap='gray')  # Afficher en niveaux de gris
    plt.imshow(predicted_mask_resized, cmap='gray')
    plt.axis('off')

    image_path = f"temp_{rd.randint(1, 9999)}.png"
    plt.savefig(image_path, dpi=300)
    image_to_discord(image_path, model_name)

if True:
    # Convertir le masque redimensionné en un tableau NumPy
    predicted_mask_resized = np.array(predicted_mask_resized)

    # Ici, on applique une couleur rouge au masque pour plus de visibilité
    colored_mask = np.zeros_like(original_image)  # Masque coloré de la même taille que l'image
    colored_mask[:, :, 0] = predicted_mask_resized  # On applique la couleur rouge là où le masque est présent

    # Afficher l'image d'origine et le masque superposé
    plt.figure(figsize=(10, 3))

    # Image d'origine
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Image d'origine ({img})")
    plt.axis('off')

    # Image avec le masque superposé
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)  # Affiche l'image d'origine
    plt.imshow(colored_mask, cmap='Reds', alpha=0.5)  # Superpose le masque avec transparence
    plt.title("Image avec masque superposé")
    plt.axis('off')

    image_path = f"temp_{rd.randint(1, 9999)}.png"
    plt.savefig(image_path, dpi=300)
    image_to_discord(image_path, model_name)

plt.show()
